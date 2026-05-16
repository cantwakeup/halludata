"""Build COCO/GQA/mixed cat-vector bundles for domain-transfer POPE tests.

The official POPE cat runner expects a vector file with ``vectors["cat"]``.
This utility takes an existing COCO-derived expert-vector file and a GQA-derived
expert-vector file, aligns their shared layers, and writes three compatible
bundles:

- ``coco_cat_as_cat.pt``
- ``gqa_cat_as_cat.pt``
- ``mixed_cat_as_cat.pt``

The mixed vector defaults to averaging flat-unit-normalized COCO and GQA cat
directions, then normalizing the result. This keeps one source from dominating
only because its raw vector has a larger norm.
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco-vector",
        default="data/outputs_after_template_disjoint_v2/steering/after_template_disjoint_v2_expert_vectors.pt",
        help="COCO-derived vector .pt containing vectors['cat'].",
    )
    parser.add_argument(
        "--gqa-vector",
        default="data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt",
        help="GQA-derived vector .pt containing vectors['cat'].",
    )
    parser.add_argument(
        "--output-dir",
        default="data/pope_cat_expert_eval/cat_domain_vector_comparison/vectors",
        help="Directory for compatible vector bundles and report.",
    )
    parser.add_argument(
        "--mix-mode",
        choices=("unit_mean", "raw_mean"),
        default="unit_mean",
        help="How to combine COCO/GQA cat vectors for the mixed direction.",
    )
    parser.add_argument(
        "--layers",
        default="intersection",
        help="Layer spec to keep, or 'intersection' for shared source layers.",
    )
    parser.add_argument("--report-output", default="", help="Optional Markdown report path.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("This script requires torch.") from exc


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_payload(path: Path) -> dict[str, Any]:
    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    if "vectors" not in payload or "cat" not in payload["vectors"]:
        raise ValueError(f"Vector file must contain vectors['cat']: {path}")
    return payload


def parse_layer_spec(text: str, available: list[int]) -> list[int]:
    text = str(text).strip()
    if not text or text == "intersection":
        return list(available)
    selected: set[int] = set()
    for chunk in text.replace(",", " ").split():
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending layer span: {chunk}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(chunk))
    missing = sorted(selected - set(available))
    if missing:
        raise ValueError(f"Requested layers not shared by both sources: {missing}")
    return sorted(selected)


def as_float_tensor(value: Any, name: str) -> Any:
    torch = require_torch()
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    tensor = value.detach().cpu().float()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be [layers, heads, head_dim], got {list(tensor.shape)}")
    return tensor


def layer_index(payload: Mapping[str, Any], vector: Any) -> list[int]:
    layers = payload.get("layers")
    if layers is None:
        return list(range(int(vector.shape[0])))
    layers = [int(layer) for layer in layers]
    if len(layers) != int(vector.shape[0]):
        raise ValueError(f"Layer metadata length {len(layers)} does not match vector shape {list(vector.shape)}")
    return layers


def select_layers(vector: Any, source_layers: list[int], target_layers: list[int]) -> Any:
    torch = require_torch()
    positions = {layer: index for index, layer in enumerate(source_layers)}
    indices = torch.tensor([positions[layer] for layer in target_layers], dtype=torch.long)
    return vector.index_select(0, indices).contiguous()


def flat_normalize(vector: Any) -> Any:
    norm = vector.flatten().float().norm().clamp_min(1e-12)
    return vector.float() / norm


def build_mixed(coco: Any, gqa: Any, mode: str) -> Any:
    if mode == "raw_mean":
        mixed = (coco.float() + gqa.float()) / 2.0
    elif mode == "unit_mean":
        mixed = (flat_normalize(coco) + flat_normalize(gqa)) / 2.0
    else:
        raise ValueError(f"Unsupported mix mode: {mode}")
    return flat_normalize(mixed)


def vector_summary(vector: Any) -> dict[str, float | list[int]]:
    per_head = vector.float().norm(dim=-1)
    return {
        "shape": list(vector.shape),
        "flat_norm": float(vector.flatten().float().norm().item()),
        "head_norm_mean": float(per_head.mean().item()),
        "head_norm_max": float(per_head.max().item()),
        "head_norm_min": float(per_head.min().item()),
    }


def cosine(a: Any, b: Any) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = float(a_flat.norm().item() * b_flat.norm().item())
    if denom <= 0.0:
        return 0.0
    return float((a_flat * b_flat).sum().item() / denom)


def top_heads(vector: Any, k: int = 64) -> list[tuple[int, int, float]]:
    scored: list[tuple[float, int, int]] = []
    norms = vector.float().norm(dim=-1)
    for layer_index_value in range(int(norms.shape[0])):
        for head in range(int(norms.shape[1])):
            scored.append((float(norms[layer_index_value, head].item()), int(layer_index_value), int(head)))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(layer, head, score) for score, layer, head in scored[:k]]


def overlap(a: list[tuple[int, int, float]], b: list[tuple[int, int, float]]) -> dict[str, float | int]:
    a_set = {(layer, head) for layer, head, _score in a}
    b_set = {(layer, head) for layer, head, _score in b}
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return {"intersection": inter, "jaccard": float(inter / union) if union else 0.0}


def output_payload(
    *,
    source_name: str,
    cat_vector: Any,
    coco_vector: Any,
    gqa_vector: Any,
    mixed_vector: Any,
    layers: list[int],
    args: argparse.Namespace,
    metadata_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "source": source_name,
        "created_by": "scripts/build_cat_domain_comparison_vectors.py",
        "coco_vector": str(resolve_path(args.coco_vector)),
        "gqa_vector": str(resolve_path(args.gqa_vector)),
        "mix_mode": str(args.mix_mode),
        "layers": layers,
    }
    if metadata_extra:
        metadata.update(dict(metadata_extra))
    return {
        "vectors": {
            "cat": cat_vector.float(),
            "coco_cat": coco_vector.float(),
            "gqa_cat": gqa_vector.float(),
            "mixed_cat": mixed_vector.float(),
        },
        "layers": layers,
        "num_heads": int(cat_vector.shape[1]),
        "head_dim": int(cat_vector.shape[2]),
        "hidden_size": int(cat_vector.shape[1] * cat_vector.shape[2]),
        "config": metadata,
        "stats": {
            "cat": vector_summary(cat_vector),
            "coco_cat": vector_summary(coco_vector),
            "gqa_cat": vector_summary(gqa_vector),
            "mixed_cat": vector_summary(mixed_vector),
        },
    }


def markdown_table(headers: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif isinstance(value, (list, tuple)):
                values.append(json.dumps(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    *,
    report_path: Path,
    args: argparse.Namespace,
    layers: list[int],
    vectors: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> None:
    names = ["coco_cat", "gqa_cat", "mixed_cat"]
    norm_rows = [{"vector": name, **vector_summary(vectors[name])} for name in names]
    cosine_rows = []
    for a_name in names:
        row: dict[str, Any] = {"vector": a_name}
        for b_name in names:
            row[b_name] = cosine(vectors[a_name], vectors[b_name])
        cosine_rows.append(row)

    top64 = {name: top_heads(vectors[name], 64) for name in names}
    overlap_rows = []
    for idx, a_name in enumerate(names):
        for b_name in names[idx + 1 :]:
            overlap_rows.append({"pair": f"{a_name} vs {b_name}", **overlap(top64[a_name], top64[b_name])})

    lines = [
        "# Cat Domain Comparison Vectors",
        "",
        "## Inputs",
        "",
        f"- COCO vector: `{resolve_path(args.coco_vector)}`",
        f"- GQA vector: `{resolve_path(args.gqa_vector)}`",
        f"- Mix mode: `{args.mix_mode}`",
        f"- Layers: `{layers}`",
        "",
        "## Outputs",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Vector Norms",
            "",
            markdown_table(
                ["vector", "shape", "flat_norm", "head_norm_mean", "head_norm_max", "head_norm_min"],
                norm_rows,
            ),
            "",
            "## Flat Cosine Matrix",
            "",
            markdown_table(["vector", *names], cosine_rows),
            "",
            "## Top64 Head Overlap",
            "",
            markdown_table(["pair", "intersection", "jaccard"], overlap_rows),
            "",
            "## Suggested Experiment",
            "",
            "Run the official POPE sweep three times with the same baseline config and only change `CAT_VECTOR_PATH`.",
            "If COCO-only helps MSCOCO more, GQA-only helps GQA more, and mixed is stable on both, that supports a shared-plus-domain-specific category direction interpretation.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        torch = require_torch()
        coco_path = resolve_path(args.coco_vector)
        gqa_path = resolve_path(args.gqa_vector)
        output_dir = resolve_path(args.output_dir)
        report_path = resolve_path(args.report_output) if args.report_output else output_dir / "REPORT.md"
        outputs = {
            "coco_cat": output_dir / "coco_cat_as_cat.pt",
            "gqa_cat": output_dir / "gqa_cat_as_cat.pt",
            "mixed_cat": output_dir / "mixed_cat_as_cat.pt",
        }
        if not args.overwrite:
            existing = [path for path in [*outputs.values(), report_path] if path.exists()]
            if existing:
                raise FileExistsError(f"Outputs already exist; pass --overwrite. First existing: {existing[0]}")

        coco_payload = load_payload(coco_path)
        gqa_payload = load_payload(gqa_path)
        coco_cat_full = as_float_tensor(coco_payload["vectors"]["cat"], "coco vectors['cat']")
        gqa_cat_full = as_float_tensor(gqa_payload["vectors"]["cat"], "gqa vectors['cat']")
        coco_layers = layer_index(coco_payload, coco_cat_full)
        gqa_layers = layer_index(gqa_payload, gqa_cat_full)
        shared_layers = sorted(set(coco_layers) & set(gqa_layers))
        if not shared_layers:
            raise ValueError("COCO and GQA vectors have no overlapping layer IDs")
        layers = parse_layer_spec(str(args.layers), shared_layers)
        coco_cat = select_layers(coco_cat_full, coco_layers, layers)
        gqa_cat = select_layers(gqa_cat_full, gqa_layers, layers)
        if list(coco_cat.shape) != list(gqa_cat.shape):
            raise ValueError(f"Aligned vector shapes differ: {list(coco_cat.shape)} vs {list(gqa_cat.shape)}")

        mixed_cat = build_mixed(coco_cat, gqa_cat, str(args.mix_mode))
        vectors = {"coco_cat": coco_cat, "gqa_cat": gqa_cat, "mixed_cat": mixed_cat}

        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            output_payload(
                source_name="coco_cat_as_cat",
                cat_vector=coco_cat,
                coco_vector=coco_cat,
                gqa_vector=gqa_cat,
                mixed_vector=mixed_cat,
                layers=layers,
                args=args,
            ),
            outputs["coco_cat"],
        )
        torch.save(
            output_payload(
                source_name="gqa_cat_as_cat",
                cat_vector=gqa_cat,
                coco_vector=coco_cat,
                gqa_vector=gqa_cat,
                mixed_vector=mixed_cat,
                layers=layers,
                args=args,
            ),
            outputs["gqa_cat"],
        )
        torch.save(
            output_payload(
                source_name="mixed_cat_as_cat",
                cat_vector=mixed_cat,
                coco_vector=coco_cat,
                gqa_vector=gqa_cat,
                mixed_vector=mixed_cat,
                layers=layers,
                args=args,
                metadata_extra={"mixed_definition": f"{args.mix_mode}(coco_cat, gqa_cat)"},
            ),
            outputs["mixed_cat"],
        )
        write_report(report_path=report_path, args=args, layers=layers, vectors=vectors, outputs=outputs)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote cat-domain vector bundles to {output_dir}")
    print(f"Wrote report to {report_path}")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
