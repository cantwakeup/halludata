"""Build cat/attr/rel expert vectors from AFTER-template activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import read_jsonl, write_json
from expert_data.steering import parse_layer_spec


EXPERT_TYPES = ("cat", "attr", "rel")
SUBTYPES = (
    "cat_present",
    "cat_absent",
    "attr_count",
    "attr_color",
    "rel_spatial",
    "rel_left",
    "rel_right",
    "rel_above",
    "rel_below",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for AFTER-template vector construction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-cache", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="data/outputs_after_template_v1/steering/after_template_expert_vectors.pt")
    parser.add_argument("--stats-output", default="data/outputs_after_template_v1/steering/after_template_expert_vectors.stats.json")
    parser.add_argument("--layers", default="10-20")
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
    """Import torch lazily for vector construction."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("build_after_template_vectors requires torch.") from exc


def normalize_bool(value: str | bool) -> bool:
    """Parse a flexible boolean value."""

    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean: {value}")


def load_torch(path: Path) -> dict[str, Any]:
    """Load a torch cache file with compatibility for older torch versions."""

    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return payload


def as_float_tensor(value: Any) -> Any:
    """Convert tensor-like input to a CPU float32 torch tensor."""

    torch = require_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def maybe_normalize(vector: Any, normalize: bool) -> Any:
    """Optionally L2-normalize every layer-head vector."""

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
    """Compute cosine similarity between two flattened tensors."""

    if a is None or b is None:
        return None
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = float(a_flat.norm().item() * b_flat.norm().item())
    if denom == 0.0:
        return 0.0
    return float((a_flat * b_flat).sum().item() / denom)


def select_rows(diff: Any, indices: list[int]) -> Any | None:
    """Mean-pool selected rows or return None if absent."""

    if not indices:
        return None
    torch = require_torch()
    return diff.index_select(0, torch.tensor(indices, dtype=torch.long)).mean(dim=0)


def zero_vector_like(diff: Any) -> Any:
    """Return a zero vector with [L,H,D] shape."""

    torch = require_torch()
    return torch.zeros(tuple(diff.shape[1:]), dtype=torch.float32)


def main() -> int:
    """Build and save AFTER-template expert vectors."""

    args = parse_args()
    try:
        torch = require_torch()
        cache_path = resolve_project_path(args.activation_cache)
        metadata_path = resolve_project_path(args.metadata)
        output_path = resolve_project_path(args.output)
        stats_path = resolve_project_path(args.stats_output)
        if (output_path.exists() or stats_path.exists()) and not args.overwrite:
            raise FileExistsError("Output exists. Pass --overwrite to replace after_template_v1 vector outputs.")

        cache = load_torch(cache_path)
        metadata_rows = read_jsonl(metadata_path)
        z_text = as_float_tensor(cache.get("z_text", cache.get("z_pos")))
        z_visual = as_float_tensor(cache.get("z_visual", cache.get("z_neg")))
        if tuple(z_text.shape) != tuple(z_visual.shape):
            raise ValueError(f"Activation shapes differ: {list(z_text.shape)} vs {list(z_visual.shape)}")
        if z_text.ndim != 4:
            raise ValueError(f"Expected [N,L,H,D] activations, got {list(z_text.shape)}")
        if len(metadata_rows) != int(z_text.shape[0]):
            raise ValueError("Metadata row count does not match activation rows")

        layers = parse_layer_spec(args.layers)
        if max(layers) >= int(z_text.shape[1]):
            raise ValueError(f"Layer {max(layers)} is out of range for activation shape {list(z_text.shape)}")
        layer_index = torch.tensor(layers, dtype=torch.long)
        diff = z_text.index_select(1, layer_index) - z_visual.index_select(1, layer_index)
        normalize = normalize_bool(args.normalize)

        type_indices = {
            expert: [index for index, row in enumerate(metadata_rows) if str(row.get("hallucination_type")) == expert]
            for expert in EXPERT_TYPES
        }
        subtype_indices = {
            subtype: [index for index, row in enumerate(metadata_rows) if str(row.get("subtype")) == subtype]
            for subtype in SUBTYPES
        }
        raw_vectors: dict[str, Any] = {}
        warnings: list[str] = []
        for expert in EXPERT_TYPES:
            vector = select_rows(diff, type_indices[expert])
            if vector is None:
                warnings.append(f"No samples found for expert '{expert}'; using a zero vector")
                vector = zero_vector_like(diff)
            raw_vectors[expert] = vector.float()

        subtype_vectors = {
            subtype: select_rows(diff, indices)
            for subtype, indices in subtype_indices.items()
        }
        payload_vectors = {
            expert: maybe_normalize(vector, normalize)
            for expert, vector in raw_vectors.items()
        }
        payload = {
            "vectors": payload_vectors,
            "layers": layers,
            "num_heads": int(diff.shape[2]),
            "head_dim": int(diff.shape[3]),
            "hidden_size": int(diff.shape[2] * diff.shape[3]),
            "config": {
                "source": "after_template_v1",
                "activation_cache": str(cache_path),
                "metadata": str(metadata_path),
                "normalize": normalize,
                "direction": "mean(z_text - z_visual)",
            },
            "stats": {
                "sample_counts_by_type": {expert: len(indices) for expert, indices in type_indices.items()},
                "sample_counts_by_subtype": {subtype: len(indices) for subtype, indices in subtype_indices.items()},
                "warnings": warnings,
            },
            "components": {
                "subtype_vectors": {
                    subtype: vector.float() for subtype, vector in subtype_vectors.items() if vector is not None
                }
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)

        stats = {
            "source": "after_template_v1",
            "activation_cache": str(cache_path),
            "metadata": str(metadata_path),
            "layers": layers,
            "shape": [len(layers), int(diff.shape[2]), int(diff.shape[3])],
            "sample_counts_by_type": {expert: len(indices) for expert, indices in type_indices.items()},
            "sample_counts_by_subtype": {subtype: len(indices) for subtype, indices in subtype_indices.items()},
            "vector_norms": {expert: vector_norm_summary(vector) for expert, vector in raw_vectors.items()},
            "subtype_vector_norms": {subtype: vector_norm_summary(vector) for subtype, vector in subtype_vectors.items()},
            "cosine_diagnostics": {
                "cat_present_cat_absent": cosine_flat(subtype_vectors["cat_present"], subtype_vectors["cat_absent"]),
                "attr_count_attr_color": cosine_flat(subtype_vectors["attr_count"], subtype_vectors["attr_color"]),
                "rel_left_rel_right": cosine_flat(subtype_vectors["rel_left"], subtype_vectors["rel_right"]),
                "rel_above_rel_below": cosine_flat(subtype_vectors["rel_above"], subtype_vectors["rel_below"]),
                "cat_attr": cosine_flat(raw_vectors["cat"], raw_vectors["attr"]),
                "cat_rel": cosine_flat(raw_vectors["cat"], raw_vectors["rel"]),
                "attr_rel": cosine_flat(raw_vectors["attr"], raw_vectors["rel"]),
            },
            "warnings": warnings,
            "notes": [
                "vectors are trusted factual text minus visual-query activation means",
                "direction is mean(z_text - z_visual)",
                "negative cat_present/cat_absent cosine suggests object-existence polarity remains conditional",
                "this vector file is compatible with the existing ExpertSteeringController",
            ],
        }
        write_json(stats_path, stats)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote AFTER-template expert vectors to {output_path}")
    print(f"Wrote stats to {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
