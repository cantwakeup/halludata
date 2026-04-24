"""Metrics for real activation prototype and detection audits."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence


def _import_torch_or_none() -> Any | None:
    """Import torch lazily, returning None when it is unavailable or broken."""

    try:
        import torch

        return torch
    except Exception:
        return None


def l2_normalize(x: Any, dim: int = -1, eps: float = 1e-12) -> Any:
    """L2-normalize a tensor-like object along one dimension."""

    torch = _import_torch_or_none()
    if torch is not None and isinstance(x, torch.Tensor):
        norm = torch.linalg.norm(x.float(), dim=dim, keepdim=True).clamp_min(float(eps))
        return x.float() / norm
    vector = [float(value) for value in x]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= eps:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def cosine_similarity(left: Any, right: Any, dim: int = -1, eps: float = 1e-12) -> Any:
    """Compute cosine similarity between tensor-like objects."""

    torch = _import_torch_or_none()
    if torch is not None and isinstance(left, torch.Tensor):
        left_norm = l2_normalize(left, dim=dim, eps=eps)
        right_norm = l2_normalize(right, dim=dim, eps=eps)
        return (left_norm * right_norm).sum(dim=dim)
    left_norm = l2_normalize(left, eps=eps)
    right_norm = l2_normalize(right, eps=eps)
    if len(left_norm) != len(right_norm):
        raise ValueError("cosine_similarity inputs must share the same dimensionality")
    return sum(left_value * right_value for left_value, right_value in zip(left_norm, right_norm))


def _to_float_list(values: Any) -> list[float]:
    """Convert a tensor/list-like object to a flat list of floats."""

    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def _to_int_list(values: Any) -> list[int]:
    """Convert a tensor/list-like object to a flat list of integers."""

    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def pairwise_accuracy(score_pos: Any, score_neg: Any) -> float:
    """Return the fraction of pairs where the positive branch scores higher."""

    pos_values = _to_float_list(score_pos)
    neg_values = _to_float_list(score_neg)
    if len(pos_values) != len(neg_values):
        raise ValueError("score_pos and score_neg must have the same length")
    if not pos_values:
        return 0.0
    return sum(1.0 for pos, neg in zip(pos_values, neg_values) if pos > neg) / len(pos_values)


def _rankdata(values: Sequence[float]) -> list[float]:
    """Compute average ranks with tie handling using one-based ranks."""

    indexed_values = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed_values):
        end = cursor + 1
        while end < len(indexed_values) and indexed_values[end][1] == indexed_values[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[indexed_values[index][0]] = average_rank
        cursor = end
    return ranks


def _fallback_auroc(labels: list[int], scores: list[float]) -> float | None:
    """Compute AUROC with a rank-sum fallback when sklearn is absent."""

    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    if positives == 0 or negatives == 0:
        return None
    ranks = _rankdata(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _fallback_average_precision(labels: list[int], scores: list[float]) -> float | None:
    """Compute average precision by ranking candidates from high to low score."""

    positives = sum(1 for label in labels if label == 1)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hit_count = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ranked, start=1):
        if label == 1:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / positives


def safe_auc_ap(labels: Any, scores: Any) -> dict[str, float | None]:
    """Compute AUROC and average precision, using sklearn if available and fallbacks otherwise."""

    label_values = _to_int_list(labels)
    score_values = _to_float_list(scores)
    if len(label_values) != len(score_values):
        raise ValueError("labels and scores must have the same length")
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "auroc": float(roc_auc_score(label_values, score_values)),
            "average_precision": float(average_precision_score(label_values, score_values)),
        }
    except Exception:
        return {
            "auroc": _fallback_auroc(label_values, score_values),
            "average_precision": _fallback_average_precision(label_values, score_values),
        }


def compute_binary_metrics(labels: Any, scores: Any, threshold: float = 0.0) -> dict[str, float | None]:
    """Compute thresholded binary metrics plus threshold-free AUROC/AP."""

    label_values = _to_int_list(labels)
    score_values = _to_float_list(scores)
    if len(label_values) != len(score_values):
        raise ValueError("labels and scores must have the same length")
    predictions = [1 if score >= float(threshold) else 0 for score in score_values]
    tp = sum(1 for label, pred in zip(label_values, predictions) if label == 1 and pred == 1)
    tn = sum(1 for label, pred in zip(label_values, predictions) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(label_values, predictions) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(label_values, predictions) if label == 1 and pred == 0)
    total = len(label_values)
    accuracy = (tp + tn) / total if total else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    auc_ap = safe_auc_ap(label_values, score_values)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auroc": auc_ap["auroc"],
        "average_precision": auc_ap["average_precision"],
        "threshold": float(threshold),
    }


def find_best_threshold(labels: Any, scores: Any) -> tuple[float, dict[str, float | None]]:
    """Find the threshold that maximizes balanced accuracy on a candidate set."""

    score_values = sorted(set(_to_float_list(scores)))
    if not score_values:
        return 0.0, compute_binary_metrics([], [], threshold=0.0)
    candidates = [score_values[0] - 1e-6]
    candidates.extend((left + right) / 2.0 for left, right in zip(score_values, score_values[1:]))
    candidates.append(score_values[-1] + 1e-6)
    best_threshold = candidates[0]
    best_metrics = compute_binary_metrics(labels, scores, threshold=best_threshold)
    for threshold in candidates[1:]:
        metrics = compute_binary_metrics(labels, scores, threshold=threshold)
        if float(metrics["balanced_accuracy"] or 0.0) > float(best_metrics["balanced_accuracy"] or 0.0):
            best_threshold = threshold
            best_metrics = metrics
    return float(best_threshold), best_metrics


def bootstrap_ci(
    values: Sequence[Any],
    statistic_fn: Callable[[list[Any]], float] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap a scalar statistic over sample values and return a 95% confidence interval."""

    value_list = list(values)
    if not value_list:
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    statistic = statistic_fn or (lambda sample: sum(float(item) for item in sample) / len(sample))
    rng = random.Random(int(seed))
    estimates: list[float] = []
    for _ in range(max(int(n_bootstrap), 1)):
        sample = [value_list[rng.randrange(len(value_list))] for _ in value_list]
        estimates.append(float(statistic(sample)))
    estimates.sort()
    lower_index = int(0.025 * (len(estimates) - 1))
    upper_index = int(0.975 * (len(estimates) - 1))
    return {
        "mean": float(statistic(value_list)),
        "lower_95": float(estimates[lower_index]),
        "upper_95": float(estimates[upper_index]),
    }
