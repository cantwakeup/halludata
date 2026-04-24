"""Tests for the lightweight steered benchmark runner."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.steering import route_question_to_experts
from temp_utils import TemporaryWorkspace


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write benchmark rows for subprocess tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, object]:
    """Read one JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class SteeredBenchmarkTest(unittest.TestCase):
    """Validate mock benchmark plumbing and routing utilities."""

    def test_rule_router_maps_question_keywords(self) -> None:
        """The rule router should map obvious prompts to typed experts."""

        self.assertEqual(route_question_to_experts("How many dogs are there?", "rule", ("cat", "attr", "rel")), ("attr",))
        self.assertEqual(route_question_to_experts("Is the cup left of the plate?", "rule", ("cat", "attr", "rel")), ("rel",))
        self.assertEqual(route_question_to_experts("Does image contain a bus?", "rule", ("cat", "attr", "rel")), ("cat",))
        self.assertEqual(route_question_to_experts("Describe this.", "no_filter", ("cat", "attr", "rel")), ("cat", "attr", "rel"))

    def test_mock_benchmark_writes_baseline_and_steered_metrics(self) -> None:
        """Mock benchmark mode should produce predictions, metrics, and config files."""

        with TemporaryWorkspace() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_path = tmp_path / "pope_mock.jsonl"
            out_dir = tmp_path / "run"
            _write_jsonl(
                benchmark_path,
                [
                    {"question": "Is there a dog in the image?", "answer": "yes", "image_id": 1},
                    {"question": "Is there a cat in the image?", "answer": "no", "image_id": 2},
                    {"question": "How many buses are visible?", "answer": "yes", "image_id": 3},
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_steered_benchmark.py",
                    "--benchmark-data",
                    str(benchmark_path),
                    "--benchmark-name",
                    "pope_mock",
                    "--out-dir",
                    str(out_dir),
                    "--adapter",
                    "mock",
                    "--steer-enable",
                    "--limit",
                    "3",
                    "--overwrite",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            self.assertTrue((out_dir / "predictions.jsonl").exists())
            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "config.json").exists())
            metrics = _read_json(out_dir / "metrics.json")
            self.assertIn("baseline", metrics)
            self.assertIn("steered", metrics)
            self.assertEqual(metrics["baseline"]["num_samples"], 3)
            self.assertEqual(metrics["steered"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
