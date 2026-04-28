"""Build relation-v2 vectors from AFTER-template z_text - z_visual activations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import read_jsonl, write_json
from expert_data.steering import parse_layer_spec

REL_KEYS = ("rel_all", "rel_left", "rel_right", "rel_above", "rel_below", "rel_horizontal", "rel_vertical")
RELATION_TO_KEY = {
    "left_of": "rel_left",
    "right_of": "rel_right",
    "above": "rel_above",
    "below": "rel_below",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-cache", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="data/outputs_after_template_rel_v2/steering/relation_v2_vectors.pt")
    parser.add_argument("--stats-output", default="data/outputs_after_template_rel_v2/steering/relation_v2_vectors.stats.json")
    parser.add_argument("--layers", default="all")
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
        raise RuntimeError("build_relation_v2_vectors requires torch.") from exc


def normalize_bool(value: str | bool) -> bool:
    """Parse a bool-like CLI value."""

    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean: {value}")


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
    """Convert tensor-like values to CPU float32 tensors."""

    torch = require_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def parse_layers(raw_layers: str, available_layers: list[int]) -> list[int]:
    """Parse `all` or a normal layer spec."""

    text = str(raw_layers).strip().lower()
    if text == "all":
        return list(available_layers)
    return parse_layer_spec(raw_layers)


def maybe_normalize(vector: Any, normalize: bool) -> Any:
    """Optionally normalize per layer/head vector rows."""

    if not normalize:
        return vector.float()
    denom = vector.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vector.float() / denom


def select_rows(diff: Any, indices: list[int]) -> Any | None:
    """Mean-pool selected rows."""

    if not indices:
        return None
    torch = require_torch()
    return diff.index_select(0, torch.tensor(indices, dtype=torch.long)).mean(dim=0).float()


def zero_vector_like(diff: Any) -> Any:
    """Return a zero [L,H,D] vector."""

    torch = require_torch()
    return torch.zeros(tuple(diff.shape[1:]), dtype=torch.float32)


def vector_norm_summary(vector: Any | None) -> dict[str, float] | None:
    """Summarize vector norm by layer/head."""

    if vector is None:
        return None
    norms = vector.float().norm(dim=-1)
    return {
        "mean": float(norms.mean().item()),
        "max": float(norms.max().item()),
        "min": float(norms.min().item()),
    }


def cosine_flat(a: Any | None, b: Any | None) -> float | None:
    """Cosine between flattened vectors."""

    if a is None or b is None:
        return None
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = float(a_flat.norm().item() * b_flat.norm().item())
    if denom == 0.0:
        return 0.0
    return float((a_flat * b_flat).sum().item() / denom)


def build_indices(metadata_rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Build vector group indices from relation metadata."""

    indices: dict[str, list[int]] = {key: [] for key in REL_KEYS}
    for index, row in enumerate(metadata_rows):
        true_relation = str(row.get("true_relation") or "")
        indices["rel_all"].append(index)
        key = RELATION_TO_KEY.get(true_relation)
        if key:
            indices[key].append(index)
        if true_relation in {"left_of", "right_of"}:
            indices["rel_horizontal"].append(index)
        if true_relation in {"above", "below"}:
            indices["rel_vertical"].append(index)
    return indices


def main() -> int:
    """Build and save relation-v2 vectors."""

    args = parse_args()
    try:
        torch = require_torch()
        cache_path = resolve_project_path(args.activation_cache)
        metadata_path = resolve_project_path(args.metadata)
        output_path = resolve_project_path(args.output)
        stats_path = resolve_project_path(args.stats_output)
        if (output_path.exists() or stats_path.exists()) and not args.overwrite:
            raise FileExistsError("Output exists. Pass --overwrite to replace relation v2 vector outputs.")

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
        available_layers = [int(layer) for layer in cache.get("layers", list(range(int(z_text.shape[1]))))]
        layers = parse_layers(str(args.layers), available_layers)
        layer_to_cache_index = {int(layer): index for index, layer in enumerate(available_layers)}
        missing = [layer for layer in layers if layer not in layer_to_cache_index]
        if missing:
            raise ValueError(f"Requested layers missing from cache: {missing}")
        layer_indices = torch.tensor([layer_to_cache_index[layer] for layer in layers], dtype=torch.long)
        diff = z_text.index_select(1, layer_indices) - z_visual.index_select(1, layer_indices)
        normalize = normalize_bool(args.normalize)
        group_indices = build_indices(metadata_rows)

        raw_vectors: dict[str, Any] = {}
        warnings: list[str] = []
        for key in REL_KEYS:
            vector = select_rows(diff, group_indices[key])
            if vector is None:
                warnings.append(f"No samples found for '{key}'; using a zero vector")
                vector = zero_vector_like(diff)
            raw_vectors[key] = vector.float()

        payload_vectors = {key: maybe_normalize(vector, normalize) for key, vector in raw_vectors.items()}
        payload_vectors["rel"] = payload_vectors["rel_all"]
        payload = {
            "vectors": payload_vectors,
            "layers": layers,
            "num_heads": int(diff.shape[2]),
            "head_dim": int(diff.shape[3]),
            "hidden_size": int(diff.shape[2] * diff.shape[3]),
            "config": {
                "source": "after_template_rel_v2",
                "activation_cache": str(cache_path),
                "metadata": str(metadata_path),
                "normalize": normalize,
                "direction": "mean(z_text - z_visual)",
                "layers": layers,
            },
            "stats": {
                "sample_counts_by_vector": {key: len(indices) for key, indices in group_indices.items()},
                "label_balance": dict(sorted(Counter(str(row.get("label", "")) for row in metadata_rows).items())),
                "warnings": warnings,
            },
            "components": {"relation_vectors": raw_vectors},
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)

        stats = {
            "source": "after_template_rel_v2",
            "activation_cache": str(cache_path),
            "metadata": str(metadata_path),
            "layers": layers,
            "shape": [len(layers), int(diff.shape[2]), int(diff.shape[3])],
            "sample_counts_by_vector": {key: len(indices) for key, indices in group_indices.items()},
            "sample_counts_by_true_relation": dict(sorted(Counter(str(row.get("true_relation", "")) for row in metadata_rows).items())),
            "label_balance": dict(sorted(Counter(str(row.get("label", "")) for row in metadata_rows).items())),
            "vector_norms": {key: vector_norm_summary(vector) for key, vector in raw_vectors.items()},
            "cosine_diagnostics": {
                "rel_left_rel_right": cosine_flat(raw_vectors["rel_left"], raw_vectors["rel_right"]),
                "rel_above_rel_below": cosine_flat(raw_vectors["rel_above"], raw_vectors["rel_below"]),
                "rel_horizontal_rel_vertical": cosine_flat(raw_vectors["rel_horizontal"], raw_vectors["rel_vertical"]),
            },
            "warnings": warnings,
            "notes": [
                "vectors are trusted factual relation text minus visual-query activation means",
                "rel is saved as an alias for rel_all for compatibility with force_rel steering",
            ],
        }
        write_json(stats_path, stats)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote relation v2 vectors to {output_path}")
    print(f"Wrote stats to {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
