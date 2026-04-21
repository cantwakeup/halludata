"""Tests for real activation head-ranking utilities."""

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

from expert_data.real_head_ranking import (
    compute_head_scores,
    evaluate_topk_heads,
    random_topk_baseline,
    topk_overlap_matrix,
)
from expert_data.real_prototypes import build_subtype_prototypes_from_cache


def _head_specific_cache() -> dict[str, object]:
    """Build a small cache where head (0, 1) carries the strongest separation."""

    assert torch is not None
    num_pairs, num_layers, num_heads, head_dim = 10, 2, 3, 4
    z_pos = torch.randn(num_pairs, num_layers, num_heads, head_dim) * 0.01
    z_neg = torch.randn(num_pairs, num_layers, num_heads, head_dim) * 0.01
    z_pos[:, 0, 1, 0] = 5.0
    z_neg[:, 0, 1, 0] = -5.0
    z_pos[:, :, :, 1] += 0.5
    z_neg[:, :, :, 1] -= 0.5
    return {
        "activations": {
            "pair_ids": [f"p{i}" for i in range(num_pairs)],
            "row_indices": list(range(num_pairs)),
            "image_ids": [str(i) for i in range(num_pairs)],
            "subtypes": ["cat"] * num_pairs,
            "z_pos": z_pos,
            "z_neg": z_neg,
        },
        "metadata": [],
        "manifest": {},
    }


@unittest.skipIf(torch is None, "torch is required for real head-ranking tests")
class RealHeadRankingTest(unittest.TestCase):
    """Validate per-head scoring and Top-K evaluation."""

    def test_head_ranking_finds_artificial_high_signal_head(self) -> None:
        """The manually separated head should rank first."""

        cache = _head_specific_cache()
        prototypes = build_subtype_prototypes_from_cache(cache, subtypes=["cat"])
        head_scores = compute_head_scores(cache, prototypes)
        self.assertEqual((head_scores["cat"][0]["layer"], head_scores["cat"][0]["head"]), (0, 1))

    def test_topk_eval_random_baseline_and_overlap(self) -> None:
        """Top-K evaluation, random baseline, and overlap matrices should be well-formed."""

        cache = _head_specific_cache()
        prototypes = build_subtype_prototypes_from_cache(cache, subtypes=["cat"])
        head_scores = compute_head_scores(cache, prototypes)
        result = evaluate_topk_heads(prototypes, cache, head_scores, 1, tune_thresholds=True)
        self.assertGreaterEqual(result["by_subtype"]["cat"]["pairwise_acc"], 0.9)
        baseline = random_topk_baseline(prototypes, cache, 1, repeats=3, seed=1)
        self.assertIn("pairwise_acc", baseline["cat"])
        overlap = topk_overlap_matrix(head_scores, 1)
        self.assertEqual(overlap["subtypes"], ["cat"])
        self.assertEqual(overlap["jaccard"], [[1.0]])


if __name__ == "__main__":
    unittest.main()
