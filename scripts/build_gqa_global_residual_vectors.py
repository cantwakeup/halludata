"""Build GQA global/shared and residual expert vectors.

The residuals remove the projection of each typed vector onto a global direction
computed from all GQA type-aware train activations:

    cat_res = cat - proj_global(cat)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-cache", default="data/gqa_typeaware_v1/activations/train.pt")
    parser.add_argument("--metadata", default="data/gqa_typeaware_v1/activations/train.meta.jsonl")
    parser.add_argument("--vector-path", default="data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt")
    parser.add_argument("--output", default="data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt")
    parser.add_argument("--report-output", default="data/gqa_typeaware_v1/steering/GLOBAL_RESIDUAL_REPORT.md")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_torch() -> Any:
    try:
        import torch
        return torch
    except Exception as exc:
        raise RuntimeError("build_gqa_global_residual_vectors requires torch.") from exc


def load_torch(path: Path) -> Mapping[str, Any]:
    torch = require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_float_tensor(value: Any) -> Any:
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def cosine(a: Any, b: Any) -> float | None:
    torch = require_torch()
    if a is None or b is None:
        return None
    av = a.float().reshape(-1)
    bv = b.float().reshape(-1)
    denom = float(av.norm().item() * bv.norm().item())
    if denom == 0.0:
        return None
    return float(torch.dot(av, bv).item() / denom)


def project_onto(vector: Any, basis: Any) -> Any:
    torch = require_torch()
    v = vector.float()
    g = basis.float()
    denom = torch.dot(g.reshape(-1), g.reshape(-1))
    if float(denom.item()) == 0.0:
        return torch.zeros_like(v)
    scale = torch.dot(v.reshape(-1), g.reshape(-1)) / denom
    return scale * g


def norm_summary(vector: Any) -> dict[str, float]:
    norms = vector.float().norm(dim=-1)
    return {
        "mean": float(norms.mean().item()),
        "max": float(norms.max().item()),
        "min": float(norms.min().item()),
    }


def topk_heads(vector: Any, layers: list[int], topk: int) -> list[tuple[int, int, float]]:
    scores = vector.float().norm(dim=-1)
    rows: list[tuple[int, int, float]] = []
    for layer_index, layer in enumerate(layers):
        for head in range(scores.shape[1]):
            rows.append((int(layer), int(head), float(scores[layer_index, head].item())))
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows[: int(topk)]


def topk_overlap(vectors: Mapping[str, Any], layers: list[int], topk: int) -> list[dict[str, Any]]:
    head_sets = {
        key: {(layer, head) for layer, head, _ in topk_heads(vector, layers, topk)}
        for key, vector in vectors.items()
    }
    rows: list[dict[str, Any]] = []
    for a, b in combinations(sorted(head_sets), 2):
        inter = len(head_sets[a] & head_sets[b])
        union = len(head_sets[a] | head_sets[b])
        rows.append(
            {
                "expert_a": a,
                "expert_b": b,
                "intersection": inter,
                "jaccard": inter / union if union else 0.0,
            }
        )
    return rows


def cosine_matrix(vectors: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a, b in combinations(sorted(vectors), 2):
        rows.append({"expert_a": a, "expert_b": b, "cosine": cosine(vectors[a], vectors[b])})
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def table(headers: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    activation_cache: Path,
    vector_path: Path,
    output: Path,
    layers: list[int],
    sample_counts: Mapping[str, int],
    vectors: Mapping[str, Any],
    topk: int,
) -> None:
    raw_vectors = {key: vectors[key] for key in ("global_all", "cat", "attr", "rel") if key in vectors}
    residual_vectors = {key: vectors[key] for key in ("cat_res", "attr_res", "rel_res") if key in vectors}
    norm_rows = [{"vector": key, **norm_summary(vector)} for key, vector in vectors.items()]
    text = [
        "# GQA Global Residual Vector Report",
        "",
        f"- Activation cache: `{activation_cache}`",
        f"- Base vector path: `{vector_path}`",
        f"- Output vector path: `{output}`",
        f"- Layers: `{layers}`",
        f"- Sample counts: `{dict(sample_counts)}`",
        "",
        "## Vector Norms",
        "",
        table(["vector", "mean", "max", "min"], norm_rows),
        "",
        "## Raw Cosine Matrix",
        "",
        table(["expert_a", "expert_b", "cosine"], cosine_matrix(raw_vectors)),
        "",
        "## Residual Cosine Matrix",
        "",
        table(["expert_a", "expert_b", "cosine"], cosine_matrix(residual_vectors)),
        "",
        f"## Top{topk} Head Overlap",
        "",
        table(["expert_a", "expert_b", "intersection", "jaccard"], topk_overlap(vectors, layers, topk)),
        "",
        "## Notes",
        "",
        "- `global_all` is mean(z_text - z_visual) over all train-vector rows.",
        "- Residual vectors subtract the flattened projection onto `global_all`.",
        "- `global_all+cat_res` style runs can be evaluated by enabling both expert keys; the steering controller sums enabled vectors.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    activation_cache = Path(args.activation_cache)
    metadata_path = Path(args.metadata)
    vector_path = Path(args.vector_path)
    output = Path(args.output)
    report_output = Path(args.report_output)
    try:
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace.")
        torch = require_torch()
        cache = load_torch(activation_cache)
        base_payload = load_torch(vector_path)
        metadata = read_jsonl(metadata_path)
        layers = [int(layer) for layer in base_payload["layers"]]
        z_text = as_float_tensor(cache.get("z_text", cache.get("z_pos")))
        z_visual = as_float_tensor(cache.get("z_visual", cache.get("z_neg")))
        if z_text.shape != z_visual.shape:
            raise ValueError("z_text and z_visual shapes differ")
        layer_index = torch.tensor(layers, dtype=torch.long)
        diff = z_text.index_select(1, layer_index) - z_visual.index_select(1, layer_index)
        global_all = diff.mean(dim=0).float()
        base_vectors = {key: as_float_tensor(base_payload["vectors"][key]) for key in ("cat", "attr", "rel")}
        vectors = {
            "global_all": global_all,
            **base_vectors,
        }
        for key in ("cat", "attr", "rel"):
            vectors[f"{key}_res"] = (base_vectors[key] - project_onto(base_vectors[key], global_all)).float()
        sample_counts = {
            key: sum(1 for row in metadata if str(row.get("hallucination_type")) == key)
            for key in ("cat", "attr", "rel")
        }
        payload = {
            "vectors": vectors,
            "layers": layers,
            "num_heads": int(base_payload["num_heads"]),
            "head_dim": int(base_payload["head_dim"]),
            "hidden_size": int(base_payload["hidden_size"]),
            "config": {
                "source": "gqa_typeaware_global_residual_v1",
                "activation_cache": str(activation_cache),
                "metadata": str(metadata_path),
                "base_vector_path": str(vector_path),
                "direction": "mean(z_text - z_visual)",
                "residual_definition": "typed_vector - proj_global(typed_vector)",
            },
            "stats": {
                "sample_counts_by_type": sample_counts,
                "vector_norms": {key: norm_summary(vector) for key, vector in vectors.items()},
                "raw_cosines": cosine_matrix({key: vectors[key] for key in ("global_all", "cat", "attr", "rel")}),
                "residual_cosines": cosine_matrix({key: vectors[key] for key in ("cat_res", "attr_res", "rel_res")}),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        write_report(
            report_output,
            activation_cache=activation_cache,
            vector_path=vector_path,
            output=output,
            layers=layers,
            sample_counts=sample_counts,
            vectors=vectors,
            topk=int(args.topk),
        )
        write_json(output.with_suffix(".stats.json"), payload["stats"])
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote GQA global residual vectors to {output}")
    print(f"Wrote report to {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
