"""Tests for the offline head-ranking scaffold."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.head_ranking import compute_head_ranking, rank_heads


class HeadRankingTest(unittest.TestCase):
    """Validate head ranking on a small synthetic score setup."""

    def test_rank_heads_prefers_higher_score(self) -> None:
        """Heads with larger separation scores should rank ahead of weaker heads."""

        ranked = rank_heads(
            {
                "l0_h0": {"sep": 2.0, "disp_pos": 0.5, "disp_neg": 0.5, "score": 2.0},
                "l0_h1": {"sep": 1.0, "disp_pos": 0.5, "disp_neg": 0.5, "score": 1.0},
            },
            top_k=1,
        )
        self.assertEqual(ranked[0]["head"], "l0_h0")

    def test_compute_head_ranking_returns_top_heads(self) -> None:
        """Subtype-level head ranking should emit a top-head list and score matrix."""

        ranking = compute_head_ranking(
            {
                "cat": {
                    "l0_h0": {
                        "pos": [[2.0, 0.0], [1.8, 0.0]],
                        "neg": [[0.0, 2.0], [0.0, 1.8]],
                    },
                    "l0_h1": {
                        "pos": [[1.0, 0.0], [1.0, 0.0]],
                        "neg": [[0.8, 0.2], [0.9, 0.1]],
                    },
                }
            },
            top_k=2,
        )
        self.assertIn("cat", ranking)
        self.assertTrue(ranking["cat"]["top_heads"])
        self.assertIn("score_matrix", ranking["cat"])
        self.assertEqual(ranking["cat"]["top_heads"][0]["head"], "l0_h0")


if __name__ == "__main__":
    unittest.main()

