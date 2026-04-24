"""Build typed expert steering vectors from positive/negative activation caches."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

from expert_data.activation_metrics import l2_normalize

DEFAULT_EXPERT_MAP = {
    "cat": "cat",
    "cnt": "attr",
    "col": "attr",
    "rel": "rel",
}
DEFAULT_EXPERT_ORDER = ("cat", "attr", "rel")


def _require_torch() -> Any:
    """Import torch lazily for tensor-based steering-vector construction."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("Expert steering vector construction requires a working torch installation.") from exc


def parse_layer_spec(layer_spec: str, num_layers: int | None = None) -> list[int]:
    """Parse a layer spec like `10-20` or `10,12,14` into sorted layer indices."""

    layers: set[int] = set()
    for chunk in str(layer_spec).split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise ValueError("At least one layer must be selected")
    ordered = sorted(layers)
    if min(ordered) < 0:
        raise ValueError("Layer indices must be non-negative")
    if num_layers is not None and max(ordered) >= int(num_layers):
        raise ValueError(f"Layer index {max(ordered)} is out of range for num_layers={num_layers}")
    return ordered


def normalize_bool(value: str | bool) -> bool:
    """Parse a flexible CLI boolean string."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value}")


def _activation_dict(cache_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the activations dictionary from a loaded cache payload or raw cache."""

    if "activations" in cache_payload and isinstance(cache_payload["activations"], Mapping):
        return cache_payload["activations"]
    return cache_payload


def _as_float_tensor(value: Any) -> Any:
    """Convert an activation array to a CPU float32 tensor."""

    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def _cache_tensors(cache_payload: Mapping[str, Any]) -> tuple[Any, Any, list[str]]:
    """Extract z_pos, z_neg, and row subtype labels from a cache payload."""

    activations = _activation_dict(cache_payload)
    required = ("z_pos", "z_neg", "subtypes")
    missing = [field for field in required if field not in activations]
    if missing:
        raise ValueError(f"Activation cache is missing required field(s): {', '.join(missing)}")
    z_pos = _as_float_tensor(activations["z_pos"])
    z_neg = _as_float_tensor(activations["z_neg"])
    subtypes = [str(item) for item in activations["subtypes"]]
    if tuple(z_pos.shape) != tuple(z_neg.shape):
        raise ValueError(f"z_pos and z_neg shapes differ: {list(z_pos.shape)} vs {list(z_neg.shape)}")
    if z_pos.ndim != 4:
        raise ValueError(f"Expected activation shape [N,L,H,D], got {list(z_pos.shape)}")
    if len(subtypes) != int(z_pos.shape[0]):
        raise ValueError("subtypes length does not match activation rows")
    return z_pos, z_neg, subtypes


def _sample_indices(indices: list[int], max_count: int, seed: int) -> list[int]:
    """Deterministically subsample row indices when a positive max count is configured."""

    if int(max_count) <= 0 or len(indices) <= int(max_count):
        return list(indices)
    rng = random.Random(int(seed))
    sampled = list(indices)
    rng.shuffle(sampled)
    return sorted(sampled[: int(max_count)])


def _norm_summary(tensor: Any) -> dict[str, float]:
    """Summarize per-head vector norms for one expert tensor."""

    norms = tensor.float().norm(dim=-1)
    return {
        "mean": float(norms.mean().item()),
        "max": float(norms.max().item()),
        "min": float(norms.min().item()),
    }


def _per_layer_mean_norm(tensor: Any, layers: list[int]) -> dict[str, float]:
    """Return mean vector norm for each selected source layer."""

    norms = tensor.float().norm(dim=-1)
    return {
        str(layer): float(norms[layer_index].mean().item())
        for layer_index, layer in enumerate(layers)
    }


def _top_heads_by_norm(vectors: Mapping[str, Any], layers: list[int], top_k: int = 20) -> list[dict[str, float | int | str]]:
    """Return the strongest layer-head vectors across experts by L2 norm."""

    rows: list[dict[str, float | int | str]] = []
    for expert, tensor in vectors.items():
        norms = tensor.float().norm(dim=-1)
        for layer_index, layer in enumerate(layers):
            for head in range(int(norms.shape[1])):
                rows.append(
                    {
                        "expert": str(expert),
                        "layer": int(layer),
                        "head": int(head),
                        "norm": float(norms[layer_index, head].item()),
                    }
                )
    rows.sort(key=lambda row: (-float(row["norm"]), str(row["expert"]), int(row["layer"]), int(row["head"])))
    return rows[: max(int(top_k), 0)]


def build_expert_vectors_from_cache(
    cache_payload: Mapping[str, Any],
    *,
    layers: list[int],
    expert_map: Mapping[str, str] | None = None,
    expert_order: tuple[str, ...] = DEFAULT_EXPERT_ORDER,
    max_samples_per_type: int = 0,
    normalize: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate factual-minus-counterfactual activation differences into expert vectors.

    The input cache is expected to contain paired `z_pos` and `z_neg` tensors with
    shape `[N, L, H, D]`. Each row is mapped from its original subtype to one of
    the typed experts, then averaged as `mean(z_pos - z_neg)`.
    """

    torch = _require_torch()
    z_pos, z_neg, row_subtypes = _cache_tensors(cache_payload)
    num_rows, num_layers, num_heads, head_dim = [int(item) for item in z_pos.shape]
    selected_layers = parse_layer_spec(",".join(str(layer) for layer in layers), num_layers=num_layers)
    subtype_to_expert = dict(DEFAULT_EXPERT_MAP)
    subtype_to_expert.update(dict(expert_map or {}))

    diff = z_pos.index_select(1, torch.tensor(selected_layers, dtype=torch.long)) - z_neg.index_select(
        1,
        torch.tensor(selected_layers, dtype=torch.long),
    )
    vectors: dict[str, Any] = {}
    sample_counts: dict[str, int] = {}
    subtype_counts_by_expert: dict[str, dict[str, int]] = {}
    skipped_sample_count = 0
    for expert in expert_order:
        matching_indices = [
            index
            for index, subtype in enumerate(row_subtypes)
            if subtype_to_expert.get(subtype) == expert
        ]
        sampled_indices = _sample_indices(matching_indices, max_samples_per_type, seed + len(expert))
        sample_counts[expert] = len(sampled_indices)
        subtype_counts_by_expert[expert] = {}
        for index in matching_indices:
            subtype = row_subtypes[index]
            subtype_counts_by_expert[expert][subtype] = subtype_counts_by_expert[expert].get(subtype, 0) + 1
        if not sampled_indices:
            skipped_sample_count += len(matching_indices)
            vectors[expert] = torch.zeros(len(selected_layers), num_heads, head_dim, dtype=torch.float32)
            continue
        index_tensor = torch.tensor(sampled_indices, dtype=torch.long)
        expert_vector = diff.index_select(0, index_tensor).mean(dim=0).float()
        if normalize:
            expert_vector = l2_normalize(expert_vector, dim=-1)
        vectors[expert] = expert_vector

    stats = {
        "source_num_rows": num_rows,
        "sample_counts": sample_counts,
        "source_subtype_counts_by_expert": subtype_counts_by_expert,
        "vector_norms": {
            expert: _norm_summary(vector)
            for expert, vector in vectors.items()
        },
        "per_layer_mean_norm": {
            expert: _per_layer_mean_norm(vector, selected_layers)
            for expert, vector in vectors.items()
        },
        "top_heads_by_norm": _top_heads_by_norm(vectors, selected_layers, top_k=20),
        "mean_diff_span_length": None,
        "fallback_to_last_token_ratio": None,
        "skipped_sample_count": skipped_sample_count,
        "notes": [
            "vectors are built from existing teacher-forced activation cache",
            "direction is factual z_pos minus counterfactual z_neg",
            "current cache positions are answer-last-token activations, not diff-span activations",
        ],
    }
    return {
        "vectors": vectors,
        "layers": selected_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": num_heads * head_dim,
        "stats": stats,
        "config": {
            "expert_map": subtype_to_expert,
            "expert_order": list(expert_order),
            "max_samples_per_type": int(max_samples_per_type),
            "normalize": bool(normalize),
            "seed": int(seed),
        },
    }


def expert_stats_json_path(output_path: str | Path) -> Path:
    """Return the companion human-readable stats path for an expert-vector file."""

    path = Path(output_path)
    return path.with_name(f"{path.stem}.stats.json")


def save_expert_vectors(output_path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Save expert vectors as a torch file and return the output path."""

    torch = _require_torch()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
    return path
