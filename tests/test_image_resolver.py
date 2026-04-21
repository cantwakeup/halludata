"""Tests for COCO image path resolution."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.image_resolver import CocoImageResolver


class ImageResolverTest(unittest.TestCase):
    """Validate COCO image ID to file path resolution."""

    def test_instances_json_file_name_resolution_accepts_int_and_str(self) -> None:
        """Resolver should use images[].file_name for both int and string image IDs."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_root = root / "images"
            image_root.mkdir()
            image_path = image_root / "nested_name.jpg"
            image_path.write_bytes(b"fake image")
            instances_path = root / "instances.json"
            instances_path.write_text(
                json.dumps({"images": [{"id": 101, "file_name": "nested_name.jpg"}]}),
                encoding="utf-8",
            )

            resolver = CocoImageResolver(image_root=image_root, instances_json=instances_path)
            self.assertEqual(resolver.resolve(101), str(image_path))
            self.assertEqual(resolver.resolve("101"), str(image_path))

    def test_fallback_uses_12_digit_coco_jpg(self) -> None:
        """Resolver should fall back to a 12-digit COCO jpg when no instances JSON is provided."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_root = Path(tmp_dir)
            image_path = image_root / "000000000007.jpg"
            image_path.write_bytes(b"fake image")

            resolver = CocoImageResolver(image_root=image_root)
            self.assertEqual(resolver.resolve(7), str(image_path))
            self.assertEqual(resolver.resolve("7"), str(image_path))

    def test_missing_image_raises_clear_error(self) -> None:
        """Missing images should raise FileNotFoundError with the image ID in the message."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            resolver = CocoImageResolver(image_root=tmp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "image_id=123"):
                resolver.resolve(123)


if __name__ == "__main__":
    unittest.main()
