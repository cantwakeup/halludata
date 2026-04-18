"""Tests for the category hard-negative resource layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.negatives import build_coco_cat_neg_bank, validate_cat_neg_bank
from expert_data.resources.coco80_neg_seed import COCO80_CATEGORY_NAMES


class CatNegativeBankTest(unittest.TestCase):
    """Validate the deterministic category hard-negative bank."""

    def test_coco80_negative_bank_covers_core_categories(self) -> None:
        """Each COCO category should expose a populated, deduplicated negative list."""

        negative_bank = build_coco_cat_neg_bank()
        validate_cat_neg_bank(negative_bank)
        self.assertTrue(set(COCO80_CATEGORY_NAMES).issubset(set(negative_bank)))
        for category in COCO80_CATEGORY_NAMES:
            entry = negative_bank[category]
            manual = entry["manual"]
            self.assertGreaterEqual(len(manual), 5)
            self.assertEqual(len(manual), len(set(manual)))
            self.assertNotIn(category, manual)
            self.assertTrue(entry["semantic_group"])
            self.assertTrue(entry["supercategory"])


if __name__ == "__main__":
    unittest.main()
