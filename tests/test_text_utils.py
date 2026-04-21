"""Tests for count-conditioned text helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.text_utils import count_conditioned_noun, pluralize_noun


class TextUtilsTest(unittest.TestCase):
    """Verify singular and plural noun forms used by count templates."""

    def test_irregular_people_forms(self) -> None:
        """The helper should render person counts as person/people."""

        self.assertEqual(count_conditioned_noun("person", 1), "person")
        self.assertEqual(count_conditioned_noun("person", 2), "people")

    def test_regular_chair_forms(self) -> None:
        """Regular nouns should switch between singular and plural naturally."""

        self.assertEqual(count_conditioned_noun("chair", 1), "chair")
        self.assertEqual(count_conditioned_noun("chair", 2), "chairs")

    def test_motorcycle_forms(self) -> None:
        """Motorcycle should stay singular for one and pluralize otherwise."""

        self.assertEqual(count_conditioned_noun("motorcycle", 1), "motorcycle")
        self.assertEqual(count_conditioned_noun("motorcycle", 2), "motorcycles")
        self.assertEqual(pluralize_noun("motorcycle"), "motorcycles")


if __name__ == "__main__":
    unittest.main()

