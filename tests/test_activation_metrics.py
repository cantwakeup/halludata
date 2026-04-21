"""Tests for activation metric helpers."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_metrics import (
    bootstrap_ci,
    compute_binary_metrics,
    cosine_similarity,
    l2_normalize,
    pairwise_accuracy,
    safe_auc_ap,
)


class ActivationMetricsTest(unittest.TestCase):
    """Validate lightweight binary and vector metrics."""

    def test_l2_normalize_and_cosine(self) -> None:
        """Normalized vectors should have unit norm and expected cosine."""

        vector = l2_normalize([3.0, 4.0])
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_binary_metrics_and_auc_ap(self) -> None:
        """Separable scores should produce high binary and ranking metrics."""

        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        metrics = compute_binary_metrics(labels, scores, threshold=0.5)
        auc_ap = safe_auc_ap(labels, scores)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(auc_ap["auroc"], 1.0)
        self.assertEqual(auc_ap["average_precision"], 1.0)

    def test_pairwise_accuracy_and_bootstrap(self) -> None:
        """Pairwise accuracy and bootstrap CIs should be stable on simple samples."""

        self.assertEqual(pairwise_accuracy([0.9, 0.8], [0.1, 0.2]), 1.0)
        ci = bootstrap_ci([1.0, 1.0, 0.0, 1.0], n_bootstrap=20, seed=1)
        self.assertIn("lower_95", ci)
        self.assertGreaterEqual(ci["upper_95"], ci["lower_95"])


if __name__ == "__main__":
    unittest.main()
