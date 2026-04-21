"""Activation-adapter interfaces and mock implementations for offline pilots."""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from typing import Any


class BaseActivationAdapter(ABC):
    """Abstract interface for adapters that expose layer-head activation vectors."""

    @abstractmethod
    def encode_pair(
        self,
        image_id: str,
        question: str,
        response: str,
        *,
        pair_id: str,
        subtype: str,
        branch: str,
    ) -> dict[str, Any]:
        """Encode one image-question-response triple into layer-head activation vectors."""


def _stable_seed(text: str) -> int:
    """Hash text deterministically into a 32-bit pseudo-random seed."""

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _random_vector(seed: int, dim: int) -> list[float]:
    """Generate one deterministic pseudo-random vector of a given dimension."""

    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


class MockActivationAdapter(BaseActivationAdapter):
    """Generate repeatable pseudo-random activations for scaffold and test runs."""

    def __init__(self, num_layers: int = 8, num_heads: int = 8, vector_dim: int = 8) -> None:
        """Configure the mock adapter's layer-head grid and vector dimensionality."""

        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.vector_dim = int(vector_dim)

    def _head_keys(self) -> list[str]:
        """Return all supported layer-head keys in deterministic order."""

        return [f"l{layer}_h{head}" for layer in range(self.num_layers) for head in range(self.num_heads)]

    def _subtype_basis(self, subtype: str) -> list[float]:
        """Build a subtype-specific basis direction that separates branches reproducibly."""

        subtype_to_index = {"cat": 0, "cnt": 1, "col": 2, "rel": 3}
        subtype_index = subtype_to_index.get(str(subtype), 0)
        return [
            0.0 if dimension != (subtype_index % self.vector_dim) else 1.0
            for dimension in range(self.vector_dim)
        ]

    def encode_pair(
        self,
        image_id: str,
        question: str,
        response: str,
        *,
        pair_id: str,
        subtype: str,
        branch: str,
    ) -> dict[str, Any]:
        """Encode one pair branch into deterministic pseudo-random layer-head vectors."""

        branch_sign = 1.0 if str(branch) == "pos" else -1.0
        basis = self._subtype_basis(subtype)
        layer_head_vectors: dict[str, list[float]] = {}
        for head_index, head_key in enumerate(self._head_keys()):
            seed = _stable_seed(f"{pair_id}|{image_id}|{question}|{response}|{head_key}")
            base_vector = _random_vector(seed, self.vector_dim)
            strength = 0.4 + ((head_index % self.num_heads) / max(self.num_heads, 1))
            layer_head_vectors[head_key] = [
                base_value + (branch_sign * strength * basis_value)
                for base_value, basis_value in zip(base_vector, basis)
            ]
        return {"layer_head_vectors": layer_head_vectors}


def load_activation_adapter(name: str) -> BaseActivationAdapter:
    """Load one supported activation adapter by its short name."""

    normalized_name = str(name).strip().lower()
    if normalized_name == "mock":
        return MockActivationAdapter()
    if normalized_name == "custom":
        raise ValueError("Custom activation adapters are scaffold-only for now; implement BaseActivationAdapter first.")
    raise ValueError(f"Unsupported activation adapter '{name}'")


def flatten_layer_head_vectors(activation_dict: dict[str, Any]) -> list[float]:
    """Flatten sorted layer-head vectors into one concatenated feature vector."""

    layer_head_vectors = dict(activation_dict.get("layer_head_vectors", {}))
    flattened: list[float] = []
    for head_key in sorted(layer_head_vectors):
        flattened.extend(float(value) for value in layer_head_vectors[head_key])
    return flattened

