"""Tests for mock activation extraction and shard merging scripts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import load_activation_cache, tensor_shape


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to JSONL for subprocess-based script tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mock_pairs(num_rows: int = 6) -> list[dict[str, object]]:
    """Build valid mock pair rows for activation extraction."""

    subtypes = ["cat", "cnt", "col", "rel"]
    return [
        {
            "pair_id": f"pair_{index}",
            "image_id": str(100 + index),
            "subtype": subtypes[index % len(subtypes)],
            "question": f"Question {index}?",
            "response_pos": f"Positive answer {index}.",
            "response_neg": f"Negative answer {index}.",
        }
        for index in range(num_rows)
    ]


class ExtractActivationsMockTest(unittest.TestCase):
    """Validate extraction script behavior with the mock adapter."""

    def _run_script(self, *argv: str, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a project script and optionally assert success."""

        result = subprocess.run(
            [sys.executable, *argv],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if expect_success:
            self.assertEqual(
                result.returncode,
                0,
                msg=f"{' '.join(argv)} failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )
        return result

    def test_mock_extraction_writes_cache_with_expected_shape(self) -> None:
        """Mock extraction should write activations, metadata, and manifest with [N,L,H,D] shapes."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pairs_path = tmp_path / "pairs.jsonl"
            out_dir = tmp_path / "cache"
            _write_jsonl(pairs_path, _mock_pairs())

            self._run_script(
                "scripts/extract_activations.py",
                "--pairs",
                str(pairs_path),
                "--out-dir",
                str(out_dir),
                "--adapter",
                "mock",
                "--max-samples",
                "3",
                "--split",
                "train",
                "--overwrite",
            )

            self.assertTrue((out_dir / "activations.pt").exists())
            self.assertTrue((out_dir / "metadata.jsonl").exists())
            self.assertTrue((out_dir / "activation_manifest.json").exists())
            payload = load_activation_cache(out_dir)
            self.assertEqual(tensor_shape(payload["activations"]["z_pos"]), [3, 8, 8, 8])
            self.assertEqual(tensor_shape(payload["activations"]["z_neg"]), [3, 8, 8, 8])
            self.assertEqual(len(payload["metadata"]), 3)
            self.assertEqual(payload["manifest"]["num_pairs"], 3)

    def test_shards_have_disjoint_rows_and_merge_restores_all_pairs(self) -> None:
        """Modulo shards should be disjoint and merge into original row-index order."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pairs = _mock_pairs(6)
            pairs_path = tmp_path / "pairs.jsonl"
            shard0 = tmp_path / "shard_0"
            shard1 = tmp_path / "shard_1"
            merged = tmp_path / "merged"
            _write_jsonl(pairs_path, pairs)

            for shard_index, out_dir in [(0, shard0), (1, shard1)]:
                self._run_script(
                    "scripts/extract_activations.py",
                    "--pairs",
                    str(pairs_path),
                    "--out-dir",
                    str(out_dir),
                    "--adapter",
                    "mock",
                    "--num-shards",
                    "2",
                    "--shard-index",
                    str(shard_index),
                    "--overwrite",
                )

            shard0_rows = set(load_activation_cache(shard0)["activations"]["row_indices"])
            shard1_rows = set(load_activation_cache(shard1)["activations"]["row_indices"])
            self.assertFalse(shard0_rows & shard1_rows)

            self._run_script(
                "scripts/merge_activation_shards.py",
                "--shard-dirs",
                str(shard0),
                str(shard1),
                "--out-dir",
                str(merged),
                "--overwrite",
            )
            merged_payload = load_activation_cache(merged)
            self.assertEqual(merged_payload["activations"]["row_indices"], list(range(6)))
            self.assertEqual(merged_payload["activations"]["pair_ids"], [row["pair_id"] for row in pairs])
            self.assertEqual(tensor_shape(merged_payload["activations"]["z_pos"]), [6, 8, 8, 8])

    def test_missing_required_field_fails_clearly(self) -> None:
        """Rows missing required pair fields should fail before extraction."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pairs_path = tmp_path / "bad_pairs.jsonl"
            bad_row = _mock_pairs(1)[0]
            bad_row.pop("question")
            _write_jsonl(pairs_path, [bad_row])

            result = self._run_script(
                "scripts/extract_activations.py",
                "--pairs",
                str(pairs_path),
                "--out-dir",
                str(tmp_path / "cache"),
                "--adapter",
                "mock",
                "--dry-run",
                expect_success=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing required field", result.stderr)


if __name__ == "__main__":
    unittest.main()
