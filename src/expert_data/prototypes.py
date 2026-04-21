"""Prototype extraction helpers for subtype-level offline pilots."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _to_float_vector(vector: Any) -> list[float]:
    """Convert one vector-like input into a flat list of floats."""

    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in list(vector)]


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Compute the element-wise mean over a non-empty list of vectors."""

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


def normalize_vector(x: Any) -> list[float]:
    """Normalize one vector to unit length while remaining safe on zero vectors."""

    vector = _to_float_vector(x)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def compute_prototype(pos_vectors: list[Any]) -> list[float]:
    """Compute one normalized prototype vector from positive example features."""

    vectors = [_to_float_vector(vector) for vector in pos_vectors]
    return normalize_vector(_mean_vector(vectors))


def compute_contrastive_axis(mu_pos: Any, mu_neg: Any) -> list[float]:
    """Compute the normalized contrastive axis between positive and negative means."""

    pos_vector = _to_float_vector(mu_pos)
    neg_vector = _to_float_vector(mu_neg)
    if len(pos_vector) != len(neg_vector):
        raise ValueError("Positive and negative prototype vectors must share the same dimensionality")
    return normalize_vector([pos_value - neg_value for pos_value, neg_value in zip(pos_vector, neg_vector)])


def aggregate_prototypes(
    features_by_subtype: Mapping[str, Mapping[str, list[Any]]],
) -> dict[str, dict[str, list[float] | int]]:
    """Aggregate subtype-level positive and negative features into prototype statistics."""

    aggregated: dict[str, dict[str, list[float] | int]] = {}
    for subtype, branches in features_by_subtype.items():
        pos_vectors = list(branches.get("pos", []))
        neg_vectors = list(branches.get("neg", []))
        if not pos_vectors or not neg_vectors:
            continue
        mu_pos = compute_prototype(pos_vectors)
        mu_neg = compute_prototype(neg_vectors)
        aggregated[str(subtype)] = {
            "mu_pos": mu_pos,
            "mu_neg": mu_neg,
            "mu_axis": compute_contrastive_axis(mu_pos, mu_neg),
            "num_pos": len(pos_vectors),
            "num_neg": len(neg_vectors),
        }
    return aggregated

