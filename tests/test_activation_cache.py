"""Tests for activation-cache save/load helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from expert_data.activation_cache import load_activation_cache, save_activation_cache, tensor_shape, utc_now_iso
from temp_utils import TemporaryWorkspace


class ActivationCacheTest(unittest.TestCase):
    """Validate activation cache persistence with lightweight mock payloads."""

    def test_save_and_load_mock_cache(self) -> None:
        """Cache helpers should preserve IDs, row indices, metadata, and manifest fields."""

        with TemporaryWorkspace() as tmp_dir:
            out_dir = Path(tmp_dir) / "cache"
            cache_dict = {
                "pair_ids": ["p0", "p1"],
                "row_indices": [0, 1],
                "image_ids": ["101", "102"],
                "subtypes": ["cat", "cnt"],
                "z_pos": [[[[1.0, 2.0]]], [[[3.0, 4.0]]]],
                "z_neg": [[[[-1.0, -2.0]]], [[[-3.0, -4.0]]]],
            }
            metadata_rows = [
                {"row_index": 0, "pair_id": "p0", "image_id": "101", "subtype": "cat"},
                {"row_index": 1, "pair_id": "p1", "image_id": "102", "subtype": "cnt"},
            ]
            manifest = {
                "adapter": "mock",
                "model_id": "mock",
                "pairs_path": "pairs.jsonl",
                "pairs_sha256": "abc",
                "num_pairs": 2,
                "created_at": utc_now_iso(),
            }

            save_activation_cache(out_dir, cache_dict, metadata_rows, manifest)
            loaded = load_activation_cache(out_dir)

            self.assertEqual(loaded["activations"]["pair_ids"], ["p0", "p1"])
            self.assertEqual(loaded["activations"]["row_indices"], [0, 1])
            self.assertEqual(len(loaded["metadata"]), 2)
            self.assertEqual(loaded["manifest"]["adapter"], "mock")
            self.assertEqual(tensor_shape(loaded["activations"]["z_pos"]), [2, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
