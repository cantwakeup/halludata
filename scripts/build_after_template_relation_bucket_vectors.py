"""Build relation-bucket expert vectors from AFTER-template activations.

This diagnostic builder reuses an existing activation cache. It separates the
current broad relation expert into buckets such as horizontal, contact, and
interaction so each bucket can be evaluated without re-extracting activations.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import read_jsonl, write_json
from expert_data.steering import parse_layer_spec


RELATION_BUCKETS = (
    "horizontal",
    "vertical",
    "depth",
    "contact",
    "interaction",
    "semantic",
)
RELATION_VECTOR_KEYS = tuple(f"rel_{bucket}" for bucket in RELATION_BUCKETS) + (
    "rel_position_2d",
    "rel_position",
    "rel_contact_interaction",
)
TYPE_VECTOR_KEYS = ("cat", "attr", "rel")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-cache", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument(
        "--output",
        default="data/outputs_after_template_disjoint_v2/steering/after_template_disjoint_v2_relation_bucket_vectors.pt",
    )
    parser.add_argument(
        "--stats-output",
        default="data/outputs_after_template_disjoint_v2/steering/after_template_disjoint_v2_relation_bucket_vectors.stats.json",
    )
    parser.add_argument(
        "--report-output",
        default="data/outputs_after_template_disjoint_v2/steering/RELATION_BUCKET_VECTOR_REPORT.md",
    )
    parser.add_argument("--layers", default="5-25")
    parser.add_argument("--normalize", default="false")
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
        raise RuntimeError("build_after_template_relation_bucket_vectors requires torch.") from exc


def normalize_bool(value: str | bool) -> bool:
    """Parse a flexible boolean CLI value."""

    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean: {value}")


def load_torch(path: Path) -> dict[str, Any]:
    """Load a torch payload with compatibility for older torch versions."""

    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return payload


def as_float_tensor(value: Any, name: str) -> Any:
    """Convert tensor-like input to a CPU float32 tensor."""

    torch = require_torch()
    if value is None:
        raise ValueError(f"Activation cache is missing '{name}'")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def maybe_normalize(vector: Any, normalize: bool) -> Any:
    """Optionally L2-normalize every layer-head direction."""

    if not normalize:
        return vector.float()
    denom = vector.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vector.float() / denom


def vector_norm_summary(vector: Any | None) -> dict[str, float] | None:
    """Summarize per-head vector norms."""

    if vector is None:
        return None
    norms = vector.float().norm(dim=-1)
    return {
        "mean": float(norms.mean().item()),
        "max": float(norms.max().item()),
        "min": float(norms.min().item()),
    }


def cosine_flat(a: Any | None, b: Any | None) -> float | None:
    """Compute cosine similarity between flattened tensors."""

    if a is None or b is None:
        return None
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = float(a_flat.norm().item() * b_flat.norm().item())
    if denom == 0.0:
        return 0.0
    return float((a_flat * b_flat).sum().item() / denom)


def select_rows(tensor: Any, indices: list[int]) -> Any | None:
    """Mean-pool selected rows or return None for empty selections."""

    if not indices:
        return None
    torch = require_torch()
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return tensor.index_select(0, index_tensor).mean(dim=0).float()


def normalize_relation_name(value: str) -> str:
    """Normalize a raw relation label into a stable snake_case token."""

    text = str(value).strip().lower().replace("-", " ")
    return "_".join(part for part in text.split() if part)


def infer_relation_bucket(row: dict[str, Any]) -> str:
    """Infer the relation bucket from metadata, subtype, or relation text."""

    explicit = str(row.get("relation_bucket", "")).strip().lower()
    if explicit in RELATION_BUCKETS:
        return explicit

    subtype = str(row.get("subtype", "")).strip().lower()
    subtype_map = {
        "rel_position_horizontal": "horizontal",
        "rel_horizontal": "horizontal",
        "rel_left": "horizontal",
        "rel_right": "horizontal",
        "rel_position_vertical": "vertical",
        "rel_vertical": "vertical",
        "rel_above": "vertical",
        "rel_below": "vertical",
        "rel_position_depth": "depth",
        "rel_depth": "depth",
        "rel_contact": "contact",
        "rel_interaction": "interaction",
        "rel_semantic": "semantic",
    }
    if subtype in subtype_map:
        return subtype_map[subtype]

    relation = normalize_relation_name(row.get("true_relation", ""))
    if relation in {"left_of", "right_of"}:
        return "horizontal"
    if relation in {"above", "below"}:
        return "vertical"
    if relation in {"in_front_of", "behind"}:
        return "depth"
    if relation in {"on", "under", "near", "next_to", "touching", "direct_contact"}:
        return "contact"
    if relation in {"holding", "wearing", "riding", "eating", "watching", "carrying", "using"}:
        return "interaction"
    return "semantic" if relation else ""


def is_yes(row: dict[str, Any]) -> bool:
    """Return whether a metadata row has a yes label."""

    return str(row.get("label", "")).strip().lower() in {"yes", "y", "true", "1"}


def is_no(row: dict[str, Any]) -> bool:
    """Return whether a metadata row has a no label."""

    return str(row.get("label", "")).strip().lower() in {"no", "n", "false", "0"}


def layer_indices(cache_layers: list[int], requested_layers: list[int]) -> list[int]:
    """Map requested true layer ids to activation tensor dimension indices."""

    layer_to_index = {int(layer): index for index, layer in enumerate(cache_layers)}
    missing = [layer for layer in requested_layers if layer not in layer_to_index]
    if missing:
        raise ValueError(f"Requested layers are missing from activation cache: {missing}")
    return [layer_to_index[layer] for layer in requested_layers]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a compact Markdown table."""

    if not rows:
        return "_None._\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines) + "\n"


def build_report(stats: dict[str, Any]) -> str:
    """Build a human-readable Markdown report from stats."""

    count_rows = [[key, value] for key, value in stats["sample_counts"].items()]
    norm_rows = [
        [key, summary["mean"], summary["max"], summary["min"]]
        for key, summary in stats["vector_norms"].items()
        if summary is not None
    ]
    cosine_rows = [
        [key, value]
        for key, value in stats["cosine_diagnostics"].items()
        if value is not None
    ]
    answer_rows = [
        [key, value]
        for key, value in stats["answer_polarity_cosines"].items()
        if value is not None
    ]
    return "\n".join(
        [
            "# Relation Bucket Vector Report",
            "",
            "This report reuses an existing AFTER-template activation cache and rebuilds relation bucket vectors.",
            "",
            "## Config",
            "",
            f"- Activation cache: `{stats['activation_cache']}`",
            f"- Metadata: `{stats['metadata']}`",
            f"- Layers: `{stats['layers']}`",
            f"- Direction: `{stats['direction']}`",
            "",
            "## Sample Counts",
            "",
            markdown_table(["vector_key", "count"], count_rows).rstrip(),
            "",
            "## Vector Norms",
            "",
            markdown_table(["vector_key", "mean_norm", "max_norm", "min_norm"], norm_rows).rstrip(),
            "",
            "## Bucket Cosines",
            "",
            markdown_table(["cosine", "value"], cosine_rows).rstrip(),
            "",
            "## Answer-Polarity Cosines",
            "",
            markdown_table(["cosine", "value"], answer_rows).rstrip(),
            "",
            "## Notes",
            "",
            "- Use `--steer-router no_filter --steer-enabled-experts rel_contact` to test one bucket directly.",
            "- `rel_position_2d` means horizontal + vertical only.",
            "- `rel_position` means horizontal + vertical + depth.",
        ]
    ) + "\n"


def main() -> int:
    """Build and save relation bucket vectors."""

    args = parse_args()
    torch = require_torch()
    cache_path = resolve_project_path(args.activation_cache)
    metadata_path = resolve_project_path(args.metadata)
    output_path = resolve_project_path(args.output)
    stats_path = resolve_project_path(args.stats_output)
    report_path = resolve_project_path(args.report_output)
    if (output_path.exists() or stats_path.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError("Output exists. Pass --overwrite to replace relation bucket vector outputs.")

    cache = load_torch(cache_path)
    metadata_rows = read_jsonl(metadata_path)
    z_text = as_float_tensor(cache.get("z_text", cache.get("z_pos")), "z_text")
    z_visual = as_float_tensor(cache.get("z_visual", cache.get("z_neg")), "z_visual")
    if tuple(z_text.shape) != tuple(z_visual.shape):
        raise ValueError(f"Activation shapes differ: {list(z_text.shape)} vs {list(z_visual.shape)}")
    if z_text.ndim != 4:
        raise ValueError(f"Expected [N,L,H,D] activations, got {list(z_text.shape)}")
    if len(metadata_rows) != int(z_text.shape[0]):
        raise ValueError("Metadata row count does not match activation rows")

    requested_layers = parse_layer_spec(args.layers)
    cache_layers = [int(layer) for layer in cache.get("layers", list(range(int(z_text.shape[1]))))]
    selected_layer_indices = layer_indices(cache_layers, requested_layers)
    layer_index_tensor = torch.tensor(selected_layer_indices, dtype=torch.long)
    z_text_layers = z_text.index_select(1, layer_index_tensor)
    z_visual_layers = z_visual.index_select(1, layer_index_tensor)
    diff = z_text_layers - z_visual_layers
    normalize = normalize_bool(args.normalize)

    bucket_indices: dict[str, list[int]] = {bucket: [] for bucket in RELATION_BUCKETS}
    type_indices: dict[str, list[int]] = {expert: [] for expert in TYPE_VECTOR_KEYS}
    label_indices = {
        "all_yes": [],
        "all_no": [],
        "rel_yes": [],
        "rel_no": [],
    }
    for index, row in enumerate(metadata_rows):
        hallucination_type = str(row.get("hallucination_type", "")).strip()
        if hallucination_type in type_indices:
            type_indices[hallucination_type].append(index)
        if is_yes(row):
            label_indices["all_yes"].append(index)
            if hallucination_type == "rel":
                label_indices["rel_yes"].append(index)
        elif is_no(row):
            label_indices["all_no"].append(index)
            if hallucination_type == "rel":
                label_indices["rel_no"].append(index)
        if hallucination_type == "rel":
            bucket = infer_relation_bucket(row)
            if bucket in bucket_indices:
                bucket_indices[bucket].append(index)

    combined_indices = {
        "rel_position_2d": bucket_indices["horizontal"] + bucket_indices["vertical"],
        "rel_position": bucket_indices["horizontal"] + bucket_indices["vertical"] + bucket_indices["depth"],
        "rel_contact_interaction": bucket_indices["contact"] + bucket_indices["interaction"],
    }
    all_indices: dict[str, list[int]] = {
        **type_indices,
        **{f"rel_{bucket}": indices for bucket, indices in bucket_indices.items()},
        **combined_indices,
    }

    raw_vectors: dict[str, Any] = {}
    warnings: list[str] = []
    for key, indices in all_indices.items():
        vector = select_rows(diff, indices)
        if vector is None:
            warnings.append(f"No samples found for vector '{key}'; skipping vector")
            continue
        raw_vectors[key] = vector

    answer_vectors = {
        "answer_text_all": None,
        "answer_text_rel": None,
        "answer_diff_all": None,
        "answer_diff_rel": None,
    }
    if label_indices["all_yes"] and label_indices["all_no"]:
        answer_vectors["answer_text_all"] = select_rows(z_text_layers, label_indices["all_yes"]) - select_rows(
            z_text_layers, label_indices["all_no"]
        )
        answer_vectors["answer_diff_all"] = select_rows(diff, label_indices["all_yes"]) - select_rows(
            diff, label_indices["all_no"]
        )
    if label_indices["rel_yes"] and label_indices["rel_no"]:
        answer_vectors["answer_text_rel"] = select_rows(z_text_layers, label_indices["rel_yes"]) - select_rows(
            z_text_layers, label_indices["rel_no"]
        )
        answer_vectors["answer_diff_rel"] = select_rows(diff, label_indices["rel_yes"]) - select_rows(
            diff, label_indices["rel_no"]
        )

    payload_vectors = {key: maybe_normalize(vector, normalize) for key, vector in raw_vectors.items()}
    payload = {
        "vectors": payload_vectors,
        "layers": requested_layers,
        "num_heads": int(diff.shape[2]),
        "head_dim": int(diff.shape[3]),
        "hidden_size": int(diff.shape[2] * diff.shape[3]),
        "config": {
            "source": "after_template_disjoint_v2_relation_buckets",
            "activation_cache": str(cache_path),
            "metadata": str(metadata_path),
            "normalize": normalize,
            "direction": "mean(z_text - z_visual)",
            "bucket_source": "relation_bucket with subtype fallback",
        },
        "stats": {
            "sample_counts": {key: len(indices) for key, indices in all_indices.items()},
            "label_counts": {key: len(indices) for key, indices in label_indices.items()},
            "warnings": warnings,
        },
        "components": {
            "answer_polarity_vectors": {
                key: value.float() for key, value in answer_vectors.items() if value is not None
            }
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    selected_cosine_pairs = [
        ("rel_horizontal", "rel_vertical"),
        ("rel_horizontal", "rel_contact"),
        ("rel_horizontal", "rel_interaction"),
        ("rel_horizontal", "rel_semantic"),
        ("rel_vertical", "rel_depth"),
        ("rel_contact", "rel_interaction"),
        ("rel_depth", "rel_semantic"),
        ("rel", "rel_contact"),
        ("rel", "rel_position_2d"),
        ("rel", "rel_contact_interaction"),
        ("rel_position_2d", "rel_contact"),
        ("rel_position_2d", "rel_contact_interaction"),
    ]
    cosine_diagnostics = {
        f"{left}_{right}": cosine_flat(raw_vectors.get(left), raw_vectors.get(right))
        for left, right in selected_cosine_pairs
    }
    all_bucket_cosines = {
        f"{left}_{right}": cosine_flat(raw_vectors.get(left), raw_vectors.get(right))
        for left, right in itertools.combinations(RELATION_VECTOR_KEYS, 2)
        if left in raw_vectors and right in raw_vectors
    }
    answer_polarity_cosines = {}
    for vector_key in ("rel", "rel_horizontal", "rel_vertical", "rel_depth", "rel_contact", "rel_interaction", "rel_semantic"):
        for answer_key, answer_vector in answer_vectors.items():
            answer_polarity_cosines[f"{vector_key}_{answer_key}"] = cosine_flat(raw_vectors.get(vector_key), answer_vector)

    stats = {
        "source": "after_template_disjoint_v2_relation_buckets",
        "activation_cache": str(cache_path),
        "metadata": str(metadata_path),
        "output": str(output_path),
        "layers": requested_layers,
        "shape": [len(requested_layers), int(diff.shape[2]), int(diff.shape[3])],
        "direction": "mean(z_text - z_visual)",
        "sample_counts": {key: len(indices) for key, indices in all_indices.items()},
        "label_counts": {key: len(indices) for key, indices in label_indices.items()},
        "vector_norms": {key: vector_norm_summary(vector) for key, vector in raw_vectors.items()},
        "answer_vector_norms": {key: vector_norm_summary(vector) for key, vector in answer_vectors.items()},
        "cosine_diagnostics": cosine_diagnostics,
        "all_bucket_cosines": all_bucket_cosines,
        "answer_polarity_cosines": answer_polarity_cosines,
        "warnings": warnings,
        "notes": [
            "vectors are trusted factual text minus visual-query activation means",
            "relation bucket vectors reuse existing activations; no new extraction is performed",
            "rel_position_2d combines horizontal and vertical only",
            "rel_position combines horizontal, vertical, and depth",
        ],
    }
    write_json(stats_path, stats)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(stats), encoding="utf-8")
    print(f"Wrote relation bucket vectors to {output_path}")
    print(f"Wrote stats to {stats_path}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
