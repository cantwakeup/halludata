"""Subtype prototype builders and evaluators for real activation caches."""

from __future__ import annotations

import random
from typing import Any, Mapping

from expert_data.activation_metrics import (
    bootstrap_ci,
    compute_binary_metrics,
    find_best_threshold,
    l2_normalize,
    pairwise_accuracy,
    safe_auc_ap,
)

DEFAULT_SUBTYPES = ("cat", "cnt", "col", "rel")


def _require_torch() -> Any:
    """Import torch only when real activation tensor work is requested."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("Real activation prototype analysis requires a working torch installation.") from exc


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
    """Extract z_pos, z_neg, and subtype rows from a cache payload."""

    activations = _activation_dict(cache_payload)
    z_pos = _as_float_tensor(activations["z_pos"])
    z_neg = _as_float_tensor(activations["z_neg"])
    subtypes = [str(item) for item in activations["subtypes"]]
    if tuple(z_pos.shape) != tuple(z_neg.shape):
        raise ValueError(f"z_pos and z_neg shapes differ: {list(z_pos.shape)} vs {list(z_neg.shape)}")
    if z_pos.ndim != 4:
        raise ValueError(f"Expected z_pos/z_neg shape [N,L,H,D], got {list(z_pos.shape)}")
    if len(subtypes) != int(z_pos.shape[0]):
        raise ValueError("Number of subtypes does not match activation rows")
    return z_pos, z_neg, subtypes


def _prototype_index(prototypes: Mapping[str, Any], subtype: str) -> int:
    """Return the prototype row index for a subtype."""

    subtypes = [str(item) for item in prototypes["subtypes"]]
    if str(subtype) not in subtypes:
        raise KeyError(f"Prototype subtype '{subtype}' is unavailable")
    return subtypes.index(str(subtype))


def _selected_head_indices(selected_heads: list[tuple[int, int]] | None, num_layers: int, num_heads: int) -> Any:
    """Convert selected (layer, head) pairs to a flat tensor index."""

    torch = _require_torch()
    if not selected_heads:
        return None
    indices = []
    for layer, head in selected_heads:
        if layer < 0 or layer >= num_layers or head < 0 or head >= num_heads:
            raise ValueError(f"Selected head out of range: layer={layer}, head={head}")
        indices.append(int(layer) * num_heads + int(head))
    return torch.tensor(indices, dtype=torch.long)


def _subtype_mask(subtypes: list[str], subtype: str) -> list[int]:
    """Return row indices matching a subtype."""

    return [index for index, row_subtype in enumerate(subtypes) if row_subtype == str(subtype)]


def build_subtype_prototypes_from_cache(
    cache_payload: Mapping[str, Any],
    subtypes: list[str] | None = None,
    *,
    shuffle_labels: bool = False,
    seed: int = 42,
    low_count_threshold: int = 100,
) -> dict[str, Any]:
    """Build normalized per-subtype mu_pos, mu_neg, and contrastive axes from a train cache."""

    torch = _require_torch()
    z_pos, z_neg, row_subtypes = _cache_tensors(cache_payload)
    requested_subtypes = list(subtypes or [subtype for subtype in DEFAULT_SUBTYPES if subtype in set(row_subtypes)])
    if not requested_subtypes:
        raise ValueError("No requested subtypes are present in the activation cache")
    zhat_pos = l2_normalize(z_pos, dim=-1)
    zhat_neg = l2_normalize(z_neg, dim=-1)
    rng = random.Random(int(seed))

    mu_pos_rows = []
    mu_neg_rows = []
    axis_rows = []
    counts: dict[str, int] = {}
    warnings: dict[str, list[str]] = {}
    for subtype in requested_subtypes:
        indices = _subtype_mask(row_subtypes, subtype)
        if not indices:
            raise ValueError(f"Subtype '{subtype}' has no rows in the activation cache")
        index_tensor = torch.tensor(indices, dtype=torch.long)
        subtype_pos = zhat_pos.index_select(0, index_tensor)
        subtype_neg = zhat_neg.index_select(0, index_tensor)
        if shuffle_labels:
            swap_mask = torch.tensor([rng.random() < 0.5 for _ in indices], dtype=torch.bool)
            shuffled_pos = subtype_pos.clone()
            shuffled_neg = subtype_neg.clone()
            shuffled_pos[swap_mask] = subtype_neg[swap_mask]
            shuffled_neg[swap_mask] = subtype_pos[swap_mask]
            subtype_pos, subtype_neg = shuffled_pos, shuffled_neg
        mu_pos = l2_normalize(subtype_pos.mean(dim=0), dim=-1)
        mu_neg = l2_normalize(subtype_neg.mean(dim=0), dim=-1)
        axis = l2_normalize(mu_pos - mu_neg, dim=-1)
        mu_pos_rows.append(mu_pos)
        mu_neg_rows.append(mu_neg)
        axis_rows.append(axis)
        counts[str(subtype)] = len(indices)
        warnings[str(subtype)] = []
        if len(indices) < int(low_count_threshold):
            warnings[str(subtype)].append("low_train_count")
    return {
        "subtypes": requested_subtypes,
        "mu_pos": torch.stack(mu_pos_rows, dim=0),
        "mu_neg": torch.stack(mu_neg_rows, dim=0),
        "axis": torch.stack(axis_rows, dim=0),
        "counts": counts,
        "warnings": warnings,
    }


def _score_tensor(
    z: Any,
    prototype_vector: Any,
    selected_heads: list[tuple[int, int]] | None = None,
) -> Any:
    """Score rows by mean cosine similarity over all or selected heads."""

    zhat = l2_normalize(z, dim=-1)
    head_scores = (zhat * prototype_vector.unsqueeze(0)).sum(dim=-1)
    num_layers = int(head_scores.shape[1])
    num_heads = int(head_scores.shape[2])
    flat_scores = head_scores.reshape(head_scores.shape[0], num_layers * num_heads)
    flat_indices = _selected_head_indices(selected_heads, num_layers, num_heads)
    if flat_indices is not None:
        flat_scores = flat_scores.index_select(1, flat_indices)
    return flat_scores.mean(dim=1)


def score_with_axis(
    z: Any,
    prototypes: Mapping[str, Any],
    subtype: str,
    selected_heads: list[tuple[int, int]] | None = None,
) -> Any:
    """Score activations by cosine similarity to a subtype contrastive axis."""

    index = _prototype_index(prototypes, subtype)
    return _score_tensor(_as_float_tensor(z), prototypes["axis"][index].float(), selected_heads)


def score_with_two_prototypes(
    z: Any,
    prototypes: Mapping[str, Any],
    subtype: str,
    selected_heads: list[tuple[int, int]] | None = None,
) -> Any:
    """Score activations by cosine to mu_pos minus cosine to mu_neg."""

    index = _prototype_index(prototypes, subtype)
    tensor_z = _as_float_tensor(z)
    pos_score = _score_tensor(tensor_z, prototypes["mu_pos"][index].float(), selected_heads)
    neg_score = _score_tensor(tensor_z, prototypes["mu_neg"][index].float(), selected_heads)
    return pos_score - neg_score


def _score_branch(
    z: Any,
    prototypes: Mapping[str, Any],
    subtype: str,
    score_type: str,
    selected_heads: list[tuple[int, int]] | None = None,
) -> Any:
    """Dispatch branch scoring by score type."""

    if score_type == "axis":
        return score_with_axis(z, prototypes, subtype, selected_heads)
    if score_type in {"two_proto", "two-proto", "two_prototypes"}:
        return score_with_two_prototypes(z, prototypes, subtype, selected_heads)
    raise ValueError(f"Unsupported score_type '{score_type}'")


def _bootstrap_pair_metric(
    score_pos: list[float],
    score_neg: list[float],
    metric_name: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float] | None:
    """Bootstrap a paired score metric over pair indices."""

    if n_bootstrap <= 0 or not score_pos:
        return None
    rng = random.Random(int(seed))
    estimates: list[float] = []
    for _ in range(int(n_bootstrap)):
        indices = [rng.randrange(len(score_pos)) for _ in score_pos]
        sampled_pos = [score_pos[index] for index in indices]
        sampled_neg = [score_neg[index] for index in indices]
        if metric_name == "pairwise_acc":
            estimates.append(pairwise_accuracy(sampled_pos, sampled_neg))
        else:
            labels = [1] * len(sampled_pos) + [0] * len(sampled_neg)
            scores = sampled_pos + sampled_neg
            auc_ap = safe_auc_ap(labels, scores)
            metric_key = "auroc" if metric_name == "auroc" else "average_precision"
            metric_value = auc_ap[metric_key]
            if metric_value is not None:
                estimates.append(float(metric_value))
    if not estimates:
        return None
    estimates.sort()
    lower_index = int(0.025 * (len(estimates) - 1))
    upper_index = int(0.975 * (len(estimates) - 1))
    if metric_name == "pairwise_acc":
        mean_value = pairwise_accuracy(score_pos, score_neg)
    else:
        labels = [1] * len(score_pos) + [0] * len(score_neg)
        scores = score_pos + score_neg
        metric_key = "auroc" if metric_name == "auroc" else "average_precision"
        mean_value = float(safe_auc_ap(labels, scores)[metric_key] or 0.0)
    return {
        "mean": float(mean_value),
        "lower_95": float(estimates[lower_index]),
        "upper_95": float(estimates[upper_index]),
    }


def _evaluate_one_subtype(
    z_pos: Any,
    z_neg: Any,
    prototypes: Mapping[str, Any],
    subtype: str,
    score_type: str,
    threshold: float | None,
    selected_heads: list[tuple[int, int]] | None,
    tune_threshold: bool,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one subtype using positive/negative branch scores."""

    score_pos_tensor = _score_branch(z_pos, prototypes, subtype, score_type, selected_heads)
    score_neg_tensor = _score_branch(z_neg, prototypes, subtype, score_type, selected_heads)
    score_pos = [float(value) for value in score_pos_tensor.detach().cpu().tolist()]
    score_neg = [float(value) for value in score_neg_tensor.detach().cpu().tolist()]
    labels = [1] * len(score_pos) + [0] * len(score_neg)
    scores = score_pos + score_neg
    if tune_threshold or threshold is None:
        threshold, binary_metrics = find_best_threshold(labels, scores)
    else:
        binary_metrics = compute_binary_metrics(labels, scores, threshold=threshold)
    pair_acc = pairwise_accuracy(score_pos, score_neg)
    auc_ap = safe_auc_ap(labels, scores)
    binary_metrics["auroc"] = auc_ap["auroc"]
    binary_metrics["average_precision"] = auc_ap["average_precision"]
    result = {
        "num_pairs": len(score_pos),
        "threshold": float(threshold),
        "pairwise_acc": pair_acc,
        "auroc": auc_ap["auroc"],
        "average_precision": auc_ap["average_precision"],
        "accuracy": binary_metrics["accuracy"],
        "balanced_accuracy": binary_metrics["balanced_accuracy"],
        "f1": binary_metrics["f1"],
        "mean_score_pos": sum(score_pos) / len(score_pos) if score_pos else 0.0,
        "mean_score_neg": sum(score_neg) / len(score_neg) if score_neg else 0.0,
        "mean_score_gap": (sum(score_pos) - sum(score_neg)) / len(score_pos) if score_pos else 0.0,
        "score_pos": score_pos,
        "score_neg": score_neg,
    }
    if bootstrap > 0:
        pair_indicators = [1.0 if pos > neg else 0.0 for pos, neg in zip(score_pos, score_neg)]
        result["bootstrap_ci"] = {
            "pairwise_acc": bootstrap_ci(pair_indicators, n_bootstrap=bootstrap, seed=seed),
            "auroc": _bootstrap_pair_metric(score_pos, score_neg, "auroc", bootstrap, seed),
            "average_precision": _bootstrap_pair_metric(score_pos, score_neg, "average_precision", bootstrap, seed),
        }
    return result


def evaluate_prototypes(
    prototypes: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
    *,
    score_type: str = "axis",
    thresholds: Mapping[str, float] | None = None,
    selected_heads_by_subtype: Mapping[str, list[tuple[int, int]]] | None = None,
    tune_thresholds: bool = False,
    bootstrap: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate subtype prototypes against an activation cache."""

    z_pos, z_neg, row_subtypes = _cache_tensors(cache_payload)
    available_subtypes = [subtype for subtype in prototypes["subtypes"] if subtype in set(row_subtypes)]
    by_subtype: dict[str, Any] = {}
    for subtype in available_subtypes:
        indices = _subtype_mask(row_subtypes, subtype)
        index_tensor = _require_torch().tensor(indices, dtype=_require_torch().long)
        selected_heads = dict(selected_heads_by_subtype or {}).get(subtype)
        by_subtype[subtype] = _evaluate_one_subtype(
            z_pos.index_select(0, index_tensor),
            z_neg.index_select(0, index_tensor),
            prototypes,
            subtype,
            score_type,
            threshold=dict(thresholds or {}).get(subtype),
            selected_heads=selected_heads,
            tune_threshold=tune_thresholds,
            bootstrap=bootstrap,
            seed=seed,
        )
    macro_pairwise = (
        sum(metrics["pairwise_acc"] for metrics in by_subtype.values()) / len(by_subtype)
        if by_subtype
        else 0.0
    )
    return {
        "score_type": score_type,
        "by_subtype": by_subtype,
        "macro_pairwise_acc": macro_pairwise,
        "thresholds": {subtype: metrics["threshold"] for subtype, metrics in by_subtype.items()},
    }


def compute_cross_subtype_matrix(
    prototypes: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
    *,
    score_type: str = "axis",
) -> dict[str, Any]:
    """Evaluate each prototype subtype against every eval subtype."""

    torch = _require_torch()
    z_pos, z_neg, row_subtypes = _cache_tensors(cache_payload)
    proto_subtypes = [str(item) for item in prototypes["subtypes"]]
    eval_subtypes = [subtype for subtype in proto_subtypes if subtype in set(row_subtypes)]
    pairwise_matrix: list[list[float]] = []
    auroc_matrix: list[list[float | None]] = []
    for proto_subtype in proto_subtypes:
        pairwise_row: list[float] = []
        auroc_row: list[float | None] = []
        for eval_subtype in eval_subtypes:
            indices = _subtype_mask(row_subtypes, eval_subtype)
            index_tensor = torch.tensor(indices, dtype=torch.long)
            score_pos = _score_branch(
                z_pos.index_select(0, index_tensor),
                prototypes,
                proto_subtype,
                score_type,
            ).detach().cpu().tolist()
            score_neg = _score_branch(
                z_neg.index_select(0, index_tensor),
                prototypes,
                proto_subtype,
                score_type,
            ).detach().cpu().tolist()
            pairwise_row.append(pairwise_accuracy(score_pos, score_neg))
            labels = [1] * len(score_pos) + [0] * len(score_neg)
            scores = [float(item) for item in score_pos + score_neg]
            auroc_row.append(safe_auc_ap(labels, scores)["auroc"])
        pairwise_matrix.append(pairwise_row)
        auroc_matrix.append(auroc_row)
    return {
        "score_type": score_type,
        "prototype_subtypes": proto_subtypes,
        "eval_subtypes": eval_subtypes,
        "pairwise_acc": pairwise_matrix,
        "auroc": auroc_matrix,
    }
