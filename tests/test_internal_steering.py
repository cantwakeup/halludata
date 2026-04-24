"""Tests for internal steering evaluation with the mock scorer."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from temp_utils import TemporaryWorkspace


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL rows for steering subprocess tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, object]:
    """Read one JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mock_pairs(prefix: str, num_rows: int = 8) -> list[dict[str, object]]:
    """Build simple pair rows across two subtypes."""

    subtypes = ["cat", "cnt"]
    rows: list[dict[str, object]] = []
    for index in range(num_rows):
        subtype = subtypes[index % len(subtypes)]
        rows.append(
            {
                "pair_id": f"{prefix}_{index}",
                "image_id": str(1000 + index),
                "subtype": subtype,
                "question": f"What is shown in image {index}?",
                "response_pos": f"positive {subtype} answer",
                "response_neg": f"negative {subtype} answer",
            }
        )
    return rows


class InternalSteeringTest(unittest.TestCase):
    """Validate mock internal steering selection and held-out evaluation."""

    def test_mock_internal_steering_selects_improving_config(self) -> None:
        """Mock steering should improve pairwise accuracy over the zero-alpha baseline."""

        with TemporaryWorkspace() as tmp_dir:
            tmp_path = Path(tmp_dir)
            val_path = tmp_path / "pairs_val.jsonl"
            test_path = tmp_path / "pairs_test.jsonl"
            out_dir = tmp_path / "steering"
            _write_jsonl(val_path, _mock_pairs("val"))
            _write_jsonl(test_path, _mock_pairs("test"))

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_internal_steering_test.py",
                    "--val-pairs",
                    str(val_path),
                    "--test-pairs",
                    str(test_path),
                    "--out-dir",
                    str(out_dir),
                    "--adapter",
                    "mock",
                    "--topk",
                    "1",
                    "--alphas",
                    "-1",
                    "1",
                    "--signs",
                    "1",
                    "--overwrite",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            self.assertTrue((out_dir / "steering_config.json").exists())
            self.assertTrue((out_dir / "val_tuning.json").exists())
            self.assertTrue((out_dir / "test_eval.json").exists())
            test_eval = _read_json(out_dir / "test_eval.json")
            for subtype in ("cat", "cnt"):
                self.assertEqual(test_eval[subtype]["baseline"]["pairwise_acc"], 0.0)
                self.assertEqual(test_eval[subtype]["steered"]["pairwise_acc"], 1.0)
                self.assertGreater(test_eval[subtype]["delta_pairwise_acc"], 0.0)


if __name__ == "__main__":
    unittest.main()
