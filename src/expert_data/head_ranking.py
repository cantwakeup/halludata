"""Offline layer-head ranking helpers for subtype-level pilot experiments."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Mapping

from expert_data.prototypes import normalize_vector


def _to_float_vector(vector: Any) -> list[float]:
    """Convert one vector-like input into a flat list of floats."""

    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in list(vector)]


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Compute the element-wise mean vector for a non-empty list of vectors."""

    if not vectors:
        raise ValueError("Cannot compute a mean vector from an empty list")
    dimension = len(vectors[0])
    totals = [0.0] * dimension
    for vector in vectors:
        if len(vector) != dimension:
            raise ValueError("All vectors must share the same dimensionality")
        for index, value in enumerate(vector):
            totals[index] += value
    return [value / len(vectors) for value in totals]


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    """Compute Euclidean distance between two vectors."""

    if len(left) != len(right):
        raise ValueError("Vector dimensionalities must match")
    return math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


def _mean_dispersion(vectors: list[list[float]], prototype: list[float]) -> float:
    """Compute average distance between vectors and their prototype center."""

    if not vectors:
        return 0.0
    return mean(_euclidean_distance(vector, prototype) for vector in vectors)


def compute_separation_metrics(pos_vectors: list[Any], neg_vectors: list[Any]) -> dict[str, float]:
    """Compute a simplified positive-vs-negative separation score for one layer-head."""

    normalized_pos = [normalize_vector(vector) for vector in pos_vectors]
    normalized_neg = [normalize_vector(vector) for vector in neg_vectors]
    mu_pos = _mean_vector(normalized_pos)
    mu_neg = _mean_vector(normalized_neg)
    sep = _euclidean_distance(mu_pos, mu_neg)
    disp_pos = _mean_dispersion(normalized_pos, mu_pos)
    disp_neg = _mean_dispersion(normalized_neg, mu_neg)
    score = sep / max(disp_pos + disp_neg, 1e-8)
    return {
        "sep": sep,
        "disp_pos": disp_pos,
        "disp_neg": disp_neg,
        "score": score,
    }


def rank_heads(score_matrix: Mapping[str, Mapping[str, float]], top_k: int) -> list[dict[str, float | str]]:
    """Rank heads by descending separation score and keep the requested prefix."""

    ranked_rows = [
        {
            "head": str(head_key),
            "score": float(metrics["score"]),
            "sep": float(metrics["sep"]),
            "disp_pos": float(metrics["disp_pos"]),
            "disp_neg": float(metrics["disp_neg"]),
        }
        for head_key, metrics in score_matrix.items()
    ]
    ranked_rows.sort(key=lambda row: (-float(row["score"]), -float(row["sep"]), str(row["head"])))
    return ranked_rows[: max(int(top_k), 0)]


def compute_head_ranking(
    features_by_subtype: Mapping[str, Mapping[str, Mapping[str, list[Any]]]],
    top_k: int,
) -> dict[str, dict[str, Any]]:
    """Compute per-subtype layer-head ranking statistics from mock or real activations."""

    ranking: dict[str, dict[str, Any]] = {}
    for subtype, head_features in features_by_subtype.items():
        score_matrix: dict[str, dict[str, float]] = {}
        example_pairs = 0
        for head_key, branches in head_features.items():
            pos_vectors = list(branches.get("pos", []))
            neg_vectors = list(branches.get("neg", []))
            if not pos_vectors or not neg_vectors:
                continue
            example_pairs = max(example_pairs, min(len(pos_vectors), len(neg_vectors)))
            score_matrix[str(head_key)] = compute_separation_metrics(pos_vectors, neg_vectors)
        top_heads = rank_heads(score_matrix, top_k=top_k)
        all_scores = [float(metrics["score"]) for metrics in score_matrix.values()]
        ranking[str(subtype)] = {
            "number_of_pairs": example_pairs,
            "top_heads": top_heads,
            "score_matrix": score_matrix,
            "score_stats": {
                "num_heads": len(score_matrix),
                "max_score": max(all_scores) if all_scores else 0.0,
                "mean_score": mean(all_scores) if all_scores else 0.0,
                "min_score": min(all_scores) if all_scores else 0.0,
            },
        }
    return ranking

