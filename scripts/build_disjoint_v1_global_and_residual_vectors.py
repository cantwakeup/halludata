"""Build global and type-residual vectors for AFTER-template disjoint v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import read_jsonl

EXPERTS = ("cat", "attr", "rel")
VECTOR_ORDER = (
    "global_all",
    "cat",
    "attr",
    "rel",
    "cat_res",
    "attr_res",
    "rel_res",
    "global_plus_cat_res",
    "global_plus_attr_res",
    "global_plus_rel_res",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-cache", default="data/outputs_after_template_disjoint_v1/activations/train.pt")
    parser.add_argument("--metadata", default="data/outputs_after_template_disjoint_v1/activations/train.meta.jsonl")
    parser.add_argument("--typed-vector-path", default="data/outputs_after_template_disjoint_v1/steering/after_template_expert_vectors.pt")
    parser.add_argument("--output", default="data/outputs_after_template_disjoint_v1/steering/disjoint_v1_global_residual_vectors.pt")
    parser.add_argument("--report", default="data/outputs_after_template_disjoint_v1/steering/GLOBAL_RESIDUAL_VECTOR_REPORT.md")
    parser.add_argument("--stats-output", default="data/outputs_after_template_disjoint_v1/steering/disjoint_v1_global_residual_vectors.stats.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_torch() -> Any:
    """Import torch lazily."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("build_disjoint_v1_global_and_residual_vectors requires torch.") from exc


def load_torch(path: Path) -> dict[str, Any]:
    """Load a torch payload."""

    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return payload


def as_float_tensor(value: Any) -> Any:
    """Convert a tensor-like value to CPU float32."""

    torch = require_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def dot_flat(a: Any, b: Any) -> Any:
    """Flattened dot product."""

    return (a.flatten().float() * b.flatten().float()).sum()


def project_onto(vector: Any, basis: Any) -> Any:
    """Project vector onto basis with safe zero-norm handling."""

    denom = dot_flat(basis, basis)
    if float(denom.item()) <= 1e-12:
        return vector.detach().clone().zero_()
    return (dot_flat(vector, basis) / denom) * basis


def cosine_flat(a: Any, b: Any) -> float:
    """Cosine similarity between flattened tensors."""

    denom = float(a.flatten().float().norm().item() * b.flatten().float().norm().item())
    if denom <= 1e-12:
        return 0.0
    return float(dot_flat(a, b).item() / denom)


def norm_summary(vector: Any) -> dict[str, float]:
    """Summarize per-head norms."""

    norms = vector.float().norm(dim=-1)
    return {
        "mean": float(norms.mean().item()),
        "max": float(norms.max().item()),
        "min": float(norms.min().item()),
        "flat": float(vector.flatten().float().norm().item()),
    }


def top_heads(vector: Any, layers: list[int], top_k: int = 64) -> list[list[float | int]]:
    """Return top-K [layer, head, score] rows."""

    rows: list[list[float | int]] = []
    norms = vector.float().norm(dim=-1)
    for layer_index, layer in enumerate(layers):
        for head in range(int(norms.shape[1])):
            rows.append([int(layer), int(head), float(norms[layer_index, head].item())])
    rows.sort(key=lambda item: (-float(item[2]), int(item[0]), int(item[1])))
    return rows[:top_k]


def head_set(rows: list[list[float | int]]) -> set[tuple[int, int]]:
    """Convert head rows to a set."""

    return {(int(row[0]), int(row[1])) for row in rows}


def table(headers: list[str], rows: list[Mapping[str, Any]]) -> list[str]:
    """Render a markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    """Write text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_report(stats: Mapping[str, Any], report_path: Path, output_path: Path) -> str:
    """Render markdown report."""

    lines: list[str] = [
        "# Disjoint V1 Global / Residual Vector Report",
        "",
        "## Results First",
        "",
        "This file builds a shared `global_all` direction and type-residual directions:",
        "`cat_res`, `attr_res`, and `rel_res` are each computed by subtracting the projection onto `global_all`.",
        "",
        "## Vector Norms",
        "",
    ]
    norm_rows = [
        {
            "vector": key,
            "mean_norm": value["mean"],
            "max_norm": value["max"],
            "min_norm": value["min"],
            "flat_norm": value["flat"],
        }
        for key, value in stats.get("vector_norms", {}).items()
    ]
    lines.extend(table(["vector", "mean_norm", "max_norm", "min_norm", "flat_norm"], norm_rows))
    lines.extend(["", "## Cosine Matrix", ""])
    cosine_rows: list[dict[str, Any]] = []
    matrix = stats.get("cosine_matrix", {})
    vector_keys = list(matrix)
    for left in vector_keys:
        row = {"vector": left}
        row.update({right: matrix[left].get(right, "") for right in vector_keys})
        cosine_rows.append(row)
    if cosine_rows:
        lines.extend(table(["vector", *vector_keys], cosine_rows))
    else:
        lines.append("- No cosine matrix available.")
    lines.extend(["", "## Residual Similarity Diagnostics", ""])
    lines.extend(table(["pair", "cosine"], [{"pair": key, "cosine": value} for key, value in stats.get("residual_similarity", {}).items()]))
    lines.extend(["", "## Top64 Head Overlap", ""])
    lines.extend(table(["vector_a", "vector_b", "intersection", "jaccard"], stats.get("top64_head_overlap", [])))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Vector file: `{output_path}`",
            f"- Report: `{report_path}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    try:
        torch = require_torch()
        activation_path = resolve_project_path(args.activation_cache)
        metadata_path = resolve_project_path(args.metadata)
        typed_path = resolve_project_path(args.typed_vector_path)
        output_path = resolve_project_path(args.output)
        report_path = resolve_project_path(args.report)
        stats_path = resolve_project_path(args.stats_output)
        if (output_path.exists() or report_path.exists() or stats_path.exists()) and not args.overwrite:
            raise FileExistsError("Output exists. Pass --overwrite to replace global/residual outputs.")

        typed_payload = load_torch(typed_path)
        layers = [int(layer) for layer in typed_payload.get("layers", [])]
        if not layers:
            raise ValueError(f"Typed vector payload has no layers: {typed_path}")

        cache = load_torch(activation_path)
        metadata = read_jsonl(metadata_path)
        z_text = as_float_tensor(cache.get("z_text", cache.get("z_pos")))
        z_visual = as_float_tensor(cache.get("z_visual", cache.get("z_neg")))
        if z_text.ndim != 4 or tuple(z_text.shape) != tuple(z_visual.shape):
            raise ValueError(f"Unexpected activation shapes: {list(z_text.shape)} vs {list(z_visual.shape)}")
        if len(metadata) != int(z_text.shape[0]):
            raise ValueError("Metadata length does not match activation rows")
        if max(layers) >= int(z_text.shape[1]):
            raise ValueError(f"Layer {max(layers)} out of activation range {list(z_text.shape)}")

        layer_index = torch.tensor(layers, dtype=torch.long)
        diff = z_text.index_select(1, layer_index) - z_visual.index_select(1, layer_index)
        global_all = diff.float().mean(dim=0)

        typed_vectors = {
            expert: as_float_tensor(typed_payload["vectors"][expert])
            for expert in EXPERTS
            if expert in typed_payload.get("vectors", {})
        }
        for expert in EXPERTS:
            if expert not in typed_vectors:
                indices = [index for index, row in enumerate(metadata) if str(row.get("hallucination_type")) == expert]
                if not indices:
                    raise ValueError(f"No typed vector or metadata rows for expert: {expert}")
                typed_vectors[expert] = diff.index_select(0, torch.tensor(indices, dtype=torch.long)).mean(dim=0)

        residuals = {
            f"{expert}_res": typed_vectors[expert] - project_onto(typed_vectors[expert], global_all)
            for expert in EXPERTS
        }
        vectors = {
            "global_all": global_all,
            **typed_vectors,
            **residuals,
        }
        vectors["global_plus_cat_res"] = vectors["global_all"] + vectors["cat_res"]
        vectors["global_plus_attr_res"] = vectors["global_all"] + vectors["attr_res"]
        vectors["global_plus_rel_res"] = vectors["global_all"] + vectors["rel_res"]
        vectors = {key: vectors[key].float() for key in VECTOR_ORDER if key in vectors}

        cosine_matrix = {
            left: {right: cosine_flat(vectors[left], vectors[right]) for right in vectors}
            for left in vectors
        }
        top64 = {key: top_heads(vector, layers, top_k=64) for key, vector in vectors.items()}
        overlap_rows: list[dict[str, Any]] = []
        keys = list(top64)
        for index, left in enumerate(keys):
            left_set = head_set(top64[left])
            for right in keys[index + 1 :]:
                right_set = head_set(top64[right])
                intersection = len(left_set & right_set)
                union = len(left_set | right_set)
                overlap_rows.append(
                    {
                        "vector_a": left,
                        "vector_b": right,
                        "intersection": intersection,
                        "jaccard": (intersection / union) if union else 0.0,
                    }
                )

        stats = {
            "source": "after_template_disjoint_v1_global_residual",
            "activation_cache": str(activation_path),
            "metadata": str(metadata_path),
            "typed_vector_path": str(typed_path),
            "layers": layers,
            "shape": [len(layers), int(diff.shape[2]), int(diff.shape[3])],
            "vector_norms": {key: norm_summary(vector) for key, vector in vectors.items()},
            "cosine_matrix": cosine_matrix,
            "residual_similarity": {
                "cat_attr": cosine_flat(vectors["cat"], vectors["attr"]),
                "cat_rel": cosine_flat(vectors["cat"], vectors["rel"]),
                "attr_rel": cosine_flat(vectors["attr"], vectors["rel"]),
                "cat_res_attr_res": cosine_flat(vectors["cat_res"], vectors["attr_res"]),
                "cat_res_rel_res": cosine_flat(vectors["cat_res"], vectors["rel_res"]),
                "attr_res_rel_res": cosine_flat(vectors["attr_res"], vectors["rel_res"]),
            },
            "top64_heads": top64,
            "top64_head_overlap": overlap_rows,
            "notes": [
                "global_all = mean(z_text - z_visual) over all disjoint train samples.",
                "type_residual = type_vector - proj_global(type_vector), using flattened dot products.",
                "global_plus_* vectors are included as convenience summed keys.",
            ],
        }

        payload = {
            "vectors": vectors,
            "layers": layers,
            "num_heads": int(diff.shape[2]),
            "head_dim": int(diff.shape[3]),
            "hidden_size": int(diff.shape[2] * diff.shape[3]),
            "config": {
                "source": "after_template_disjoint_v1_global_residual",
                "activation_cache": str(activation_path),
                "metadata": str(metadata_path),
                "typed_vector_path": str(typed_path),
                "direction": "global_all and type residuals from mean(z_text - z_visual)",
            },
            "stats": stats,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        write_json(stats_path, stats)
        write_text(report_path, render_report(stats, report_path, output_path))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote global/residual vectors to {output_path}")
    print(f"Wrote global/residual stats to {stats_path}")
    print(f"Wrote global/residual report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
