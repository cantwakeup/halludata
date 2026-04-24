"""Tests for typed expert steering-vector construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
except Exception:  # pragma: no cover - exercised only without a working torch install.
    torch = None

from expert_data.expert_vectors import build_expert_vectors_from_cache, parse_layer_spec


class ExpertVectorsTest(unittest.TestCase):
    """Validate typed expert vector aggregation on small mock activations."""

    def test_parse_layer_spec(self) -> None:
        """Layer specs should support ranges and comma-separated indices."""

        self.assertEqual(parse_layer_spec("1-3"), [1, 2, 3])
        self.assertEqual(parse_layer_spec("0,2,4"), [0, 2, 4])
        with self.assertRaisesRegex(ValueError, "out of range"):
            parse_layer_spec("3", num_layers=3)

    @unittest.skipIf(torch is None, "torch is required for expert vector tensor tests")
    def test_build_vectors_maps_count_and_color_to_attr(self) -> None:
        """cat/cnt/col/rel rows should aggregate into cat/attr/rel expert vectors."""

        assert torch is not None
        z_pos = torch.zeros(6, 3, 2, 4)
        z_neg = torch.zeros(6, 3, 2, 4)
        z_pos[0:2] = 1.0
        z_pos[2:5] = 2.0
        z_pos[5:6] = 3.0
        cache = {
            "activations": {
                "z_pos": z_pos,
                "z_neg": z_neg,
                "subtypes": ["cat", "cat", "cnt", "col", "cnt", "rel"],
            }
        }

        payload = build_expert_vectors_from_cache(cache, layers=[1, 2], max_samples_per_type=0)

        self.assertEqual(payload["layers"], [1, 2])
        self.assertEqual(list(payload["vectors"]["cat"].shape), [2, 2, 4])
        self.assertEqual(payload["stats"]["sample_counts"], {"cat": 2, "attr": 3, "rel": 1})
        self.assertAlmostEqual(float(payload["vectors"]["cat"].mean().item()), 1.0)
        self.assertAlmostEqual(float(payload["vectors"]["attr"].mean().item()), 2.0)
        self.assertAlmostEqual(float(payload["vectors"]["rel"].mean().item()), 3.0)


if __name__ == "__main__":
    unittest.main()
