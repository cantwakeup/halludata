"""Smoke test for the mock fact-counterfact pair rendering pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNBALANCED_OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "pairs_unbalanced_v0.jsonl"
BALANCED_OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "pairs_balanced_v0.jsonl"
STATS_OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "pair_stats_v0.json"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load JSONL rows for smoke-test assertions."""

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


class RenderPairsSmokeTest(unittest.TestCase):
    """Verify the mock pipeline renders at least one pair per subtype."""

    def _run_script(self, relative_path: str) -> None:
        """Run a helper script and fail fast when it exits with an error."""

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

    def test_render_pairs_smoke(self) -> None:
        """Run the render script end-to-end and validate subtype coverage."""

        self._run_script("scripts/build_shell_bank.py")
        self._run_script("scripts/build_cat_neg_bank.py")
        self._run_script("scripts/build_coco_fact_index.py")

        result = subprocess.run(
            [sys.executable, "scripts/render_pairs.py", "--config", "configs/v0_mini.yaml"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"render_pairs.py failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertTrue(UNBALANCED_OUTPUT_PATH.exists(), msg=f"Missing output file: {UNBALANCED_OUTPUT_PATH}")
        self.assertTrue(BALANCED_OUTPUT_PATH.exists(), msg=f"Missing output file: {BALANCED_OUTPUT_PATH}")
        self.assertTrue(STATS_OUTPUT_PATH.exists(), msg=f"Missing output file: {STATS_OUTPUT_PATH}")

        records = load_jsonl(BALANCED_OUTPUT_PATH)
        self.assertGreaterEqual(len(records), 3)

        subtype_counts: dict[str, int] = {}
        for record in records:
            subtype = str(record["subtype"])
            subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
            self.assertIn("question", record)
            self.assertIn("response_pos", record)
            self.assertIn("response_neg", record)
            self.assertNotEqual(record["response_pos"], record["response_neg"])

        for subtype in ("cat", "cnt", "rel"):
            self.assertGreaterEqual(
                subtype_counts.get(subtype, 0),
                1,
                msg=f"Missing rendered pair for subtype '{subtype}'",
            )


if __name__ == "__main__":
    unittest.main()
