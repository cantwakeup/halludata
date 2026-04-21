"""Tests for subtype-level prototype aggregation."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.prototypes import aggregate_prototypes, compute_contrastive_axis, compute_prototype, normalize_vector


class PrototypeTest(unittest.TestCase):
    """Validate prototype computations on lightweight mock vectors."""

    def test_normalize_vector_has_unit_norm(self) -> None:
        """Normalized vectors should have norm one when input is non-zero."""

        vector = normalize_vector([3.0, 4.0])
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0)

    def test_aggregate_prototypes_returns_expected_fields(self) -> None:
        """Subtype aggregation should expose positive, negative and axis prototypes."""

        prototypes = aggregate_prototypes(
            {
                "cat": {
                    "pos": [[1.0, 0.0], [0.8, 0.2]],
                    "neg": [[0.0, 1.0], [0.2, 0.8]],
                }
            }
        )
        self.assertIn("cat", prototypes)
        self.assertIn("mu_pos", prototypes["cat"])
        self.assertIn("mu_neg", prototypes["cat"])
        self.assertIn("mu_axis", prototypes["cat"])
        self.assertEqual(len(prototypes["cat"]["mu_pos"]), 2)
        self.assertEqual(len(prototypes["cat"]["mu_neg"]), 2)
        self.assertEqual(len(prototypes["cat"]["mu_axis"]), 2)

    def test_compute_axis_uses_difference_direction(self) -> None:
        """Contrastive axes should point along the positive-minus-negative direction."""

        axis = compute_contrastive_axis([1.0, 0.0], [0.0, 1.0])
        self.assertGreater(axis[0], 0.0)
        self.assertLess(axis[1], 0.0)
        self.assertEqual(len(compute_prototype([[1.0, 0.0], [1.0, 0.0]])), 2)


if __name__ == "__main__":
    unittest.main()

