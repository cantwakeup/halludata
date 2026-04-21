"""Tests for real activation prototype utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch installed.
    torch = None

from expert_data.real_prototypes import (
    build_subtype_prototypes_from_cache,
    compute_cross_subtype_matrix,
    evaluate_prototypes,
)


def _separable_cache() -> dict[str, object]:
    """Build a small cache where positive and negative branches are clearly separable."""

    assert torch is not None
    num_pairs, num_layers, num_heads, head_dim = 10, 2, 3, 4
    z_pos = torch.zeros(num_pairs, num_layers, num_heads, head_dim)
    z_neg = torch.zeros(num_pairs, num_layers, num_heads, head_dim)
    z_pos[..., 0] = 2.0
    z_neg[..., 0] = -2.0
    z_pos[:, 0, 1, 1] = 8.0
    z_neg[:, 0, 1, 1] = -8.0
    subtypes = ["cat"] * 5 + ["cnt"] * 5
    return {
        "activations": {
            "pair_ids": [f"p{i}" for i in range(num_pairs)],
            "row_indices": list(range(num_pairs)),
            "image_ids": [str(i) for i in range(num_pairs)],
            "subtypes": subtypes,
            "z_pos": z_pos,
            "z_neg": z_neg,
        },
        "metadata": [],
        "manifest": {},
    }


@unittest.skipIf(torch is None, "torch is required for real prototype tests")
class RealPrototypeTest(unittest.TestCase):
    """Validate prototype construction and evaluation on mock tensors."""

    def test_prototype_shapes_and_separable_pairwise_accuracy(self) -> None:
        """A strongly separated cache should yield correct shapes and high pairwise accuracy."""

        cache = _separable_cache()
        prototypes = build_subtype_prototypes_from_cache(cache, subtypes=["cat", "cnt"])
        self.assertEqual(list(prototypes["mu_pos"].shape), [2, 2, 3, 4])
        result = evaluate_prototypes(prototypes, cache, score_type="two_proto", tune_thresholds=True)
        self.assertGreaterEqual(result["by_subtype"]["cat"]["pairwise_acc"], 0.99)
        self.assertGreaterEqual(result["by_subtype"]["cnt"]["pairwise_acc"], 0.99)

    def test_label_shuffle_control_weakens_signal(self) -> None:
        """A shuffled-label prototype should not preserve perfect separation on the mock cache."""

        cache = _separable_cache()
        shuffled = build_subtype_prototypes_from_cache(
            cache,
            subtypes=["cat", "cnt"],
            shuffle_labels=True,
            seed=1,
        )
        result = evaluate_prototypes(shuffled, cache, score_type="two_proto", tune_thresholds=True)
        self.assertLess(result["macro_pairwise_acc"], 1.0)

    def test_cross_subtype_matrix_shape_and_missing_subtype_error(self) -> None:
        """Cross-subtype matrices should expose prototype rows and eval columns."""

        cache = _separable_cache()
        prototypes = build_subtype_prototypes_from_cache(cache, subtypes=["cat", "cnt"])
        matrix = compute_cross_subtype_matrix(prototypes, cache, score_type="axis")
        self.assertEqual(matrix["prototype_subtypes"], ["cat", "cnt"])
        self.assertEqual(matrix["eval_subtypes"], ["cat", "cnt"])
        self.assertEqual(len(matrix["pairwise_acc"]), 2)
        with self.assertRaisesRegex(ValueError, "has no rows"):
            build_subtype_prototypes_from_cache(cache, subtypes=["rel"])


if __name__ == "__main__":
    unittest.main()
