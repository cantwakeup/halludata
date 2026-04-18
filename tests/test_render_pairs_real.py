"""Tests for rendering real pairs from COCO-backed atomic facts."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACT_INDEX_PATH = PROJECT_ROOT / "data" / "outputs" / "fact_index_v0.jsonl"
UNBALANCED_PATH = PROJECT_ROOT / "data" / "outputs" / "pairs_unbalanced_v0.jsonl"
BALANCED_PATH = PROJECT_ROOT / "data" / "outputs" / "pairs_balanced_v0.jsonl"
STATS_PATH = PROJECT_ROOT / "data" / "outputs" / "pair_stats_v0.json"
VALID_RELATIONS = ("left of", "right of", "above", "below")


def load_json(path: Path) -> dict[str, object]:
    """Load one JSON document from disk for assertions."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load JSONL rows into a list of dictionaries for assertions."""

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


class RenderPairsRealTest(unittest.TestCase):
    """Validate pair rendering against the real resource-layer outputs."""

    def _run_script(self, relative_path: str) -> None:
        """Execute a project script and fail with captured output if it errors."""

        result = subprocess.run(
            [sys.executable, relative_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"{relative_path} failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    def test_render_pairs_from_real_resources(self) -> None:
        """Render pairs from built COCO resources and validate key invariants."""

        self._run_script("scripts/build_shell_bank.py")
        self._run_script("scripts/build_cat_neg_bank.py")
        self._run_script("scripts/build_coco_fact_index.py")
        self._run_script("scripts/render_pairs.py")

        self.assertTrue(UNBALANCED_PATH.exists())
        self.assertTrue(BALANCED_PATH.exists())
        self.assertTrue(STATS_PATH.exists())

        fact_index_rows = load_jsonl(FACT_INDEX_PATH)
        image_categories = {
            str(row["image_id"]): {str(obj["category"]) for obj in row["objects"]}
            for row in fact_index_rows
        }

        unbalanced_pairs = load_jsonl(UNBALANCED_PATH)
        balanced_pairs = load_jsonl(BALANCED_PATH)
        pair_stats = load_json(STATS_PATH)

        self.assertTrue({"cat", "cnt", "rel"}.issubset({pair["subtype"] for pair in unbalanced_pairs}))
        self.assertTrue({"cat", "cnt", "rel"}.issubset({pair["subtype"] for pair in balanced_pairs}))

        cat_pair = next(
            (
                pair
                for pair in unbalanced_pairs
                if pair["subtype"] == "cat"
                and pair["neg_label"] != pair["pos_label"]
                and pair["neg_label"] not in image_categories[str(pair["image_id"])]
            ),
            None,
        )
        self.assertIsNotNone(cat_pair, msg="Expected a cat pair with an injected unseen negative label")

        cnt_pair = next((pair for pair in unbalanced_pairs if pair["subtype"] == "cnt"), None)
        self.assertIsNotNone(cnt_pair, msg="Expected at least one count pair")
        pos_digits = re.findall(r"\d+", str(cnt_pair["response_pos"]))
        neg_digits = re.findall(r"\d+", str(cnt_pair["response_neg"]))
        self.assertNotEqual(pos_digits, neg_digits)
        self.assertEqual(
            re.sub(r"\d+", "<n>", str(cnt_pair["response_pos"])),
            re.sub(r"\d+", "<n>", str(cnt_pair["response_neg"])),
        )

        rel_pair = next((pair for pair in unbalanced_pairs if pair["subtype"] == "rel"), None)
        self.assertIsNotNone(rel_pair, msg="Expected at least one relation pair")
        normalized_pos = str(rel_pair["response_pos"])
        normalized_neg = str(rel_pair["response_neg"])
        for relation in VALID_RELATIONS:
            normalized_pos = normalized_pos.replace(relation, "<rel>")
            normalized_neg = normalized_neg.replace(relation, "<rel>")
        self.assertEqual(normalized_pos, normalized_neg)
        self.assertNotEqual(rel_pair["response_pos"], rel_pair["response_neg"])

        self.assertIn("counts_unbalanced", pair_stats)
        self.assertIn("counts_balanced", pair_stats)
        self.assertIn("template_usage", pair_stats)
        self.assertIn("dropped_by_reason", pair_stats)

        rendered_fact_ids = {str(pair["fact_id"]) for pair in unbalanced_pairs}
        self.assertNotIn("coco_101_rel_1", rendered_fact_ids, msg="Same-category relation should be filtered")
        self.assertNotIn("coco_101_cat_1", rendered_fact_ids, msg="Non-unique cat anchor should be filtered")
        self.assertNotIn("coco_101_cat_2", rendered_fact_ids, msg="Non-unique cat anchor should be filtered")
        self.assertFalse(any(pair["subtype"] == "col" for pair in unbalanced_pairs))

        dropped_by_reason = pair_stats["dropped_by_reason"]
        self.assertGreater(dropped_by_reason.get("same_category_relation", 0), 0)
        self.assertGreater(dropped_by_reason.get("ambiguous_cat_anchor", 0), 0)
        self.assertGreater(dropped_by_reason.get("missing_color", 0), 0)


if __name__ == "__main__":
    unittest.main()
