"""Tests for deterministic image-level pair-bank splits."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "make_pair_splits.py"
SPEC = importlib.util.spec_from_file_location("make_pair_splits_script", SCRIPT_PATH)
make_pair_splits_script = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(make_pair_splits_script)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write test rows as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, object]:
    """Read one JSON payload from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read JSONL rows from disk."""

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mock_pairs() -> list[dict[str, object]]:
    """Build a small pair bank with repeated image IDs and mixed template metadata."""

    rows: list[dict[str, object]] = []
    subtypes = ["cat", "cnt", "col", "rel"]
    for image_index in range(10):
        image_id = f"img_{image_index}"
        for pair_index in range(2):
            subtype = subtypes[(image_index + pair_index) % len(subtypes)]
            row: dict[str, object] = {
                "pair_id": f"{image_id}_pair_{pair_index}",
                "image_id": image_id,
                "subtype": subtype,
                "question": f"question {image_id} {pair_index}",
                "response_pos": f"positive {image_id} {pair_index}",
                "response_neg": f"negative {image_id} {pair_index}",
                "metadata": {"template_id": f"{subtype}_tpl"},
            }
            if image_index == 0 and pair_index == 0:
                row.pop("metadata")
            if image_index == 1 and pair_index == 0:
                row["metadata"] = {"shell_id": "legacy_shell"}
            rows.append(row)
    return rows


class MakePairSplitsTest(unittest.TestCase):
    """Validate leakage-safe split creation on lightweight mock pairs."""

    def test_image_ids_do_not_cross_splits_and_pair_ids_are_complete(self) -> None:
        """Every image should appear in one split and every pair_id should be preserved once."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "pairs.jsonl"
            out_dir = tmp_path / "splits"
            input_rows = _mock_pairs()
            _write_jsonl(input_path, input_rows)

            result = make_pair_splits_script.make_pair_splits(
                pairs_path=input_path,
                out_dir=out_dir,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=42,
            )

            rows_by_split = {
                split: _read_jsonl(out_dir / f"pairs_{split}.jsonl")
                for split in ("train", "val", "test")
            }
            image_to_splits: dict[str, set[str]] = {}
            for split, rows in rows_by_split.items():
                for row in rows:
                    image_id = str(row["image_id"])
                    image_to_splits.setdefault(image_id, set()).add(split)
            self.assertTrue(image_to_splits)
            for split_names in image_to_splits.values():
                self.assertEqual(len(split_names), 1)

            output_pair_ids = [
                str(row["pair_id"])
                for rows in rows_by_split.values()
                for row in rows
            ]
            self.assertCountEqual(output_pair_ids, [str(row["pair_id"]) for row in input_rows])
            self.assertEqual(len(output_pair_ids), len(set(output_pair_ids)))
            self.assertEqual(result["stats"]["leakage_checks"]["image_overlap_train_val"], 0)
            self.assertEqual(result["stats"]["leakage_checks"]["image_overlap_train_test"], 0)
            self.assertEqual(result["stats"]["leakage_checks"]["image_overlap_val_test"], 0)
            self.assertEqual(result["stats"]["leakage_checks"]["num_duplicate_pair_ids"], 0)
            self.assertEqual(result["stats"]["leakage_checks"]["num_missing_pair_ids"], 0)

    def test_same_seed_is_deterministic_and_different_seed_can_change_assignments(self) -> None:
        """Assignments should be reproducible for a fixed seed and seed-sensitive for tied groups."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "pairs.jsonl"
            _write_jsonl(input_path, _mock_pairs())

            first = make_pair_splits_script.make_pair_splits(
                pairs_path=input_path,
                out_dir=tmp_path / "split_seed_42_a",
                seed=42,
            )
            second = make_pair_splits_script.make_pair_splits(
                pairs_path=input_path,
                out_dir=tmp_path / "split_seed_42_b",
                seed=42,
            )
            third = make_pair_splits_script.make_pair_splits(
                pairs_path=input_path,
                out_dir=tmp_path / "split_seed_7",
                seed=7,
            )

            self.assertEqual(first["assignments"], second["assignments"])
            self.assertNotEqual(first["assignments"], third["assignments"])

    def test_missing_template_id_uses_sentinel(self) -> None:
        """Missing template metadata should not crash and should be counted explicitly."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "pairs.jsonl"
            _write_jsonl(input_path, _mock_pairs())
            make_pair_splits_script.make_pair_splits(
                pairs_path=input_path,
                out_dir=tmp_path / "splits",
                seed=42,
            )
            stats = _read_json(tmp_path / "splits" / "split_stats.json")
            template_counts: dict[str, int] = {}
            for split_stats in stats["splits"].values():
                for template_id, count in split_stats["template_counts"].items():
                    template_counts[template_id] = template_counts.get(template_id, 0) + count
            self.assertEqual(template_counts["__missing_template__"], 1)
            self.assertEqual(template_counts["legacy_shell"], 1)

    def test_invalid_ratios_raise_clear_error(self) -> None:
        """Ratios that do not sum to one should be rejected before writing files."""

        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            make_pair_splits_script.validate_ratios(0.7, 0.2, 0.2)


if __name__ == "__main__":
    unittest.main()
