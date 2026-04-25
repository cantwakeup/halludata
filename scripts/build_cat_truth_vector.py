"""Build a balanced category truthfulness steering vector from factual/counterfactual activations."""

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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for cat truth vector construction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-cache", required=True)
    parser.add_argument("--metadata", default="")
    parser.add_argument("--output", default="data/outputs/steering/cat_truth_vector.pt")
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
        raise RuntimeError("build_cat_truth_vector requires torch.") from exc


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
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_float_tensor(value: Any) -> Any:
    """Convert tensor-like input to CPU float32 torch tensor."""

    torch = require_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def vector_norm_summary(vector: Any) -> dict[str, float]:
    """Summarize per-head vector norms."""

    norms = vector.float().norm(dim=-1)
    return {
        "mean": float(norms.mean().item()),
        "max": float(norms.max().item()),
        "min": float(norms.min().item()),
    }


def cosine_flat(a: Any, b: Any) -> float:
    """Compute cosine similarity between two flattened tensors."""

    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = float(a_flat.norm().item() * b_flat.norm().item())
    if denom == 0.0:
        return 0.0
    return float((a_flat * b_flat).sum().item() / denom)


def maybe_normalize(vector: Any, normalize: bool) -> Any:
    """Optionally L2-normalize every layer-head vector."""

    if not normalize:
        return vector
    torch = require_torch()
    denom = vector.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(denom > 0, vector.float() / denom, vector.float())


def metadata_labels(cache: dict[str, Any], metadata_path: Path | None) -> list[str]:
    """Read row subtypes from metadata or cache payload."""

    if metadata_path is not None and metadata_path.exists():
        rows = read_jsonl(metadata_path)
        return [str(row.get("subtype", "")) for row in rows]
    return [str(item) for item in cache.get("subtypes", [])]


def main() -> int:
    """Build and save cat truthfulness vector."""

    args = parse_args()
    try:
        torch = require_torch()
        cache_path = resolve_project_path(args.activation_cache)
        output_path = resolve_project_path(args.output)
        stats_path = output_path.with_suffix(".stats.json")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace.")
        cache = load_torch(cache_path)
        z_factual = as_float_tensor(cache.get("z_factual", cache.get("z_pos")))
        z_counter = as_float_tensor(cache.get("z_counterfactual", cache.get("z_neg")))
        if tuple(z_factual.shape) != tuple(z_counter.shape):
            raise ValueError(f"Activation shapes differ: {list(z_factual.shape)} vs {list(z_counter.shape)}")
        if z_factual.ndim != 4:
            raise ValueError(f"Expected [N,L,H,D] activations, got {list(z_factual.shape)}")
        layers = parse_layer_spec(args.layers)
        if max(layers) >= int(z_factual.shape[1]):
            raise ValueError(f"Layer {max(layers)} is out of range for activation shape {list(z_factual.shape)}")
        layer_index = torch.tensor(layers, dtype=torch.long)
        diff = z_factual.index_select(1, layer_index) - z_counter.index_select(1, layer_index)
        meta_path = resolve_project_path(args.metadata) if str(args.metadata).strip() else cache_path.with_suffix(".meta.jsonl")
        subtypes = metadata_labels(cache, meta_path)
        if len(subtypes) != int(diff.shape[0]):
            raise ValueError("Metadata subtype count does not match activation rows")
        present_indices = [index for index, subtype in enumerate(subtypes) if subtype == "cat_truth_present"]
        absent_indices = [index for index, subtype in enumerate(subtypes) if subtype == "cat_truth_absent"]
        if not present_indices or not absent_indices:
            raise ValueError("Need both cat_truth_present and cat_truth_absent rows to build a balanced vector")
        present_vector = diff.index_select(0, torch.tensor(present_indices, dtype=torch.long)).mean(dim=0)
        absent_vector = diff.index_select(0, torch.tensor(absent_indices, dtype=torch.long)).mean(dim=0)
        combined_vector = diff.mean(dim=0)
        normalized = normalize_bool(args.normalize)
        payload_vector = maybe_normalize(combined_vector, normalized).float()
        payload = {
            "vectors": {"cat": payload_vector},
            "layers": layers,
            "num_heads": int(payload_vector.shape[1]),
            "head_dim": int(payload_vector.shape[2]),
            "hidden_size": int(payload_vector.shape[1] * payload_vector.shape[2]),
            "components": {
                "present_vector": present_vector.float(),
                "absent_vector": absent_vector.float(),
                "combined_vector": combined_vector.float(),
            },
            "config": {
                "activation_cache": str(cache_path),
                "metadata": str(meta_path),
                "normalize": normalized,
                "direction": "mean(z_factual - z_counterfactual)",
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        stats = {
            "activation_cache": str(cache_path),
            "metadata": str(meta_path),
            "layers": layers,
            "shape": [int(item) for item in payload_vector.shape],
            "sample_counts": {
                "present": len(present_indices),
                "absent": len(absent_indices),
                "total": int(diff.shape[0]),
            },
            "present_vector_norm": vector_norm_summary(present_vector),
            "absent_vector_norm": vector_norm_summary(absent_vector),
            "combined_vector_norm": vector_norm_summary(combined_vector),
            "present_absent_cosine": cosine_flat(present_vector, absent_vector),
            "notes": [
                "positive branch is factual answer",
                "negative branch is counterfactual answer",
                "high present_absent_cosine suggests a shared truthfulness direction",
                "negative cosine suggests existence polarity remains query-dependent",
            ],
        }
        write_json(stats_path, stats)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote cat truth vector to {output_path}")
    print(f"Wrote stats to {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
