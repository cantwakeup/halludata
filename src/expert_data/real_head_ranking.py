"""Head-ranking utilities for real activation caches."""

from __future__ import annotations

import random
from statistics import mean, pstdev
from typing import Any, Mapping

from expert_data.activation_metrics import l2_normalize
from expert_data.real_prototypes import evaluate_prototypes


def _require_torch() -> Any:
    """Import torch lazily for real activation tensor work."""

    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("Real head ranking requires torch.") from exc


def _activation_dict(cache_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the activations dictionary from a loaded cache payload or raw cache."""

    if "activations" in cache_payload and isinstance(cache_payload["activations"], Mapping):
        return cache_payload["activations"]
    return cache_payload


def _as_float_tensor(value: Any) -> Any:
    """Convert activation storage to CPU float32 tensor."""

    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def _cache_tensors(cache_payload: Mapping[str, Any]) -> tuple[Any, Any, list[str]]:
    """Extract train tensors and subtype labels from a cache payload."""

    activations = _activation_dict(cache_payload)
    z_pos = _as_float_tensor(activations["z_pos"])
    z_neg = _as_float_tensor(activations["z_neg"])
    subtypes = [str(item) for item in activations["subtypes"]]
    if tuple(z_pos.shape) != tuple(z_neg.shape):
        raise ValueError("z_pos and z_neg shapes must match for head ranking")
    if z_pos.ndim != 4:
        raise ValueError(f"Expected [N,L,H,D] activations, got {list(z_pos.shape)}")
    if len(subtypes) != int(z_pos.shape[0]):
        raise ValueError("subtypes length does not match activation rows")
    return z_pos, z_neg, subtypes


def _subtype_indices(subtypes: list[str], subtype: str) -> list[int]:
    """Return cache row indices for one subtype."""

    return [index for index, row_subtype in enumerate(subtypes) if row_subtype == str(subtype)]


def compute_head_scores(
    train_cache: Mapping[str, Any],
    prototypes: Mapping[str, Any],
    eps: float = 1e-8,
) -> dict[str, list[dict[str, float | int]]]:
    """Compute per-head separation scores for each subtype."""

    torch = _require_torch()
    z_pos, z_neg, row_subtypes = _cache_tensors(train_cache)
    zhat_pos = l2_normalize(z_pos, dim=-1)
    zhat_neg = l2_normalize(z_neg, dim=-1)
    results: dict[str, list[dict[str, float | int]]] = {}
    proto_subtypes = [str(item) for item in prototypes["subtypes"]]
    for subtype_index, subtype in enumerate(proto_subtypes):
        indices = _subtype_indices(row_subtypes, subtype)
        if not indices:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long)
        subtype_pos = zhat_pos.index_select(0, index_tensor)
        subtype_neg = zhat_neg.index_select(0, index_tensor)
        mu_pos = prototypes["mu_pos"][subtype_index].float()
        mu_neg = prototypes["mu_neg"][subtype_index].float()
        sep = 1.0 - (mu_pos * mu_neg).sum(dim=-1)
        disp_pos = 1.0 - (subtype_pos * mu_pos.unsqueeze(0)).sum(dim=-1)
        disp_neg = 1.0 - (subtype_neg * mu_neg.unsqueeze(0)).sum(dim=-1)
        disp_pos_mean = disp_pos.mean(dim=0)
        disp_neg_mean = disp_neg.mean(dim=0)
        score = sep / (disp_pos_mean + disp_neg_mean + float(eps))
        rows: list[dict[str, float | int]] = []
        for layer in range(int(score.shape[0])):
            for head in range(int(score.shape[1])):
                rows.append(
                    {
                        "layer": int(layer),
                        "head": int(head),
                        "score": float(score[layer, head].item()),
                        "sep": float(sep[layer, head].item()),
                        "disp_pos": float(disp_pos_mean[layer, head].item()),
                        "disp_neg": float(disp_neg_mean[layer, head].item()),
                    }
                )
        results[subtype] = rank_heads(rows, top_k=len(rows))
    return results


def rank_heads(head_rows: list[dict[str, float | int]], top_k: int) -> list[dict[str, float | int]]:
    """Sort head rows by descending score and return the requested prefix."""

    ranked = list(head_rows)
    ranked.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["sep"]),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    return ranked[: max(int(top_k), 0)]


def _heads_for_subtype(head_scores: Mapping[str, list[dict[str, float | int]]], subtype: str, top_k: int) -> list[tuple[int, int]]:
    """Return top-K (layer, head) pairs for one subtype."""

    return [
        (int(row["layer"]), int(row["head"]))
        for row in list(head_scores.get(str(subtype), []))[: max(int(top_k), 0)]
    ]


def evaluate_topk_heads(
    prototypes: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
    head_scores: Mapping[str, list[dict[str, float | int]]],
    top_k_by_subtype: Mapping[str, int] | int,
    *,
    score_type: str = "two_proto",
    thresholds: Mapping[str, float] | None = None,
    tune_thresholds: bool = False,
) -> dict[str, Any]:
    """Evaluate prototypes using only each subtype's top-K ranked heads."""

    selected_heads: dict[str, list[tuple[int, int]]] = {}
    for subtype in prototypes["subtypes"]:
        top_k = int(top_k_by_subtype) if isinstance(top_k_by_subtype, int) else int(top_k_by_subtype[str(subtype)])
        selected_heads[str(subtype)] = _heads_for_subtype(head_scores, str(subtype), top_k)
    result = evaluate_prototypes(
        prototypes,
        cache_payload,
        score_type=score_type,
        thresholds=thresholds,
        selected_heads_by_subtype=selected_heads,
        tune_thresholds=tune_thresholds,
    )
    result["selected_topk"] = {
        subtype: len(heads)
        for subtype, heads in selected_heads.items()
    }
    return result


def _metric_mean_std(values: list[float]) -> dict[str, float]:
    """Summarize repeated random-baseline metric values."""

    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def random_topk_baseline(
    prototypes: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
    top_k: int,
    repeats: int = 100,
    seed: int = 42,
    *,
    score_type: str = "two_proto",
) -> dict[str, Any]:
    """Evaluate random Top-K head subsets as a baseline."""

    num_layers = int(prototypes["axis"].shape[1])
    num_heads = int(prototypes["axis"].shape[2])
    all_heads = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
    rng = random.Random(int(seed))
    metric_values: dict[str, dict[str, list[float]]] = {
        str(subtype): {"pairwise_acc": [], "auroc": [], "average_precision": []}
        for subtype in prototypes["subtypes"]
    }
    for _ in range(max(int(repeats), 1)):
        selected = {
            str(subtype): rng.sample(all_heads, min(int(top_k), len(all_heads)))
            for subtype in prototypes["subtypes"]
        }
        result = evaluate_prototypes(
            prototypes,
            cache_payload,
            score_type=score_type,
            selected_heads_by_subtype=selected,
            tune_thresholds=True,
        )
        for subtype, metrics in result["by_subtype"].items():
            metric_values[subtype]["pairwise_acc"].append(float(metrics["pairwise_acc"]))
            if metrics["auroc"] is not None:
                metric_values[subtype]["auroc"].append(float(metrics["auroc"]))
            if metrics["average_precision"] is not None:
                metric_values[subtype]["average_precision"].append(float(metrics["average_precision"]))
    return {
        subtype: {
            metric_name: _metric_mean_std(values)
            for metric_name, values in metrics.items()
        }
        for subtype, metrics in metric_values.items()
    }


def topk_overlap_matrix(
    head_scores: Mapping[str, list[dict[str, float | int]]],
    top_k: int,
) -> dict[str, Any]:
    """Compute Top-K Jaccard overlap between subtype head sets."""

    subtypes = sorted(head_scores)
    head_sets = {
        subtype: set(_heads_for_subtype(head_scores, subtype, top_k))
        for subtype in subtypes
    }
    matrix: list[list[float]] = []
    for row_subtype in subtypes:
        matrix_row: list[float] = []
        for col_subtype in subtypes:
            union = head_sets[row_subtype] | head_sets[col_subtype]
            intersection = head_sets[row_subtype] & head_sets[col_subtype]
            matrix_row.append(len(intersection) / len(union) if union else 0.0)
        matrix.append(matrix_row)
    return {"subtypes": subtypes, "top_k": int(top_k), "jaccard": matrix}
