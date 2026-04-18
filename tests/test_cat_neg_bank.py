"""Tests for the category hard-negative resource layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.negatives import build_manual_seed_cat_neg_bank, validate_cat_neg_bank


class CatNegativeBankTest(unittest.TestCase):
    """Validate the deterministic category hard-negative bank."""

    def test_cat_negative_entries_are_non_empty_and_unique(self) -> None:
        """Each category should provide distinct negatives that exclude itself."""

        negative_bank = build_manual_seed_cat_neg_bank()
        validate_cat_neg_bank(negative_bank)
        self.assertIn("cat", negative_bank)
        for category, entry in negative_bank.items():
            manual = entry["manual"]
            self.assertGreaterEqual(len(manual), 3)
            self.assertEqual(len(manual), len(set(manual)))
            self.assertNotIn(category, manual)


if __name__ == "__main__":
    unittest.main()
