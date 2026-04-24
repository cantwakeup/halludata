"""Tests for the lightweight color-estimation helpers."""

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

from expert_data.colors import (
    COLOR_VOCAB,
    encode_panoptic_segment_id,
    estimate_annotation_colors_for_image,
    estimate_dominant_color,
    extract_mask_pixels_from_panoptic,
)
from temp_utils import TemporaryWorkspace


def _write_ppm(path: Path, width: int, height: int, pixels: list[tuple[int, int, int]]) -> None:
    """Write a tiny ASCII PPM image for dependency-light color tests."""

    rows: list[str] = []
    for row_index in range(height):
        row_pixels = pixels[row_index * width : (row_index + 1) * width]
        rows.append("   ".join(f"{r} {g} {b}" for r, g, b in row_pixels))
    path.write_text("P3\n" + f"{width} {height}\n255\n" + "\n".join(rows) + "\n", encoding="ascii")


class ColorsTest(unittest.TestCase):
    """Verify the dominant-color helpers on small synthetic inputs."""

    def test_estimate_dominant_color_returns_vocab_token(self) -> None:
        """Synthetic RGB clusters should map into the supported base-color vocabulary."""

        for pixels in (
            [(255, 0, 0)] * 4,
            [(0, 0, 255)] * 4,
            [(0, 200, 0)] * 4,
            [(240, 230, 40)] * 4,
            [(10, 10, 10)] * 4,
        ):
            color = estimate_dominant_color(pixels)
            self.assertIn(color, COLOR_VOCAB)

    def test_extract_mask_pixels_and_annotation_lookup_with_ppm_fallback(self) -> None:
        """The PPM fallback should support mask extraction and per-annotation color lookup."""

        with TemporaryWorkspace() as temp_dir:
            temp_root = Path(temp_dir)
            image_path = temp_root / "image.ppm"
            mask_path = temp_root / "mask.ppm"

            image_pixels = [
                (255, 0, 0),
                (255, 0, 0),
                (0, 0, 255),
                (0, 0, 255),
            ]
            _write_ppm(image_path, 2, 2, image_pixels)

            seg1 = encode_panoptic_segment_id(1)
            seg2 = encode_panoptic_segment_id(2)
            mask_pixels = [seg1, seg1, seg2, seg2]
            _write_ppm(mask_path, 2, 2, mask_pixels)

            segment_pixels = extract_mask_pixels_from_panoptic(mask_path, image_path, 1)
            self.assertIsNotNone(segment_pixels)
            self.assertEqual(len(segment_pixels or []), 2)
            self.assertEqual(estimate_dominant_color(segment_pixels), "red")

            color_lookup, debug_rows = estimate_annotation_colors_for_image(
                image_info={"id": 1, "file_name": image_path.name},
                annotations=[
                    {"id": 11, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 1},
                    {"id": 12, "category_id": 2, "bbox": [1, 0, 1, 1], "area": 1},
                ],
                panoptic_annotation={
                    "file_name": mask_path.name,
                    "segments_info": [
                        {"id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 1},
                        {"id": 2, "category_id": 2, "bbox": [1, 0, 1, 1], "area": 1},
                    ],
                },
                panoptic_root=temp_root,
                image_root=temp_root,
            )
            self.assertEqual(color_lookup[11], "red")
            self.assertEqual(color_lookup[12], "blue")
            self.assertEqual(len(debug_rows), 2)

    def test_missing_paths_fail_closed(self) -> None:
        """The color lookup helper should safely return empty outputs when files are unavailable."""

        lookup, debug_rows = estimate_annotation_colors_for_image(
            image_info={"id": 1, "file_name": "missing.jpg"},
            annotations=[{"id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 1}],
            panoptic_annotation={"file_name": "missing.png", "segments_info": []},
            panoptic_root=PROJECT_ROOT / "does_not_exist",
            image_root=PROJECT_ROOT / "does_not_exist",
        )
        self.assertEqual(lookup, {})
        self.assertEqual(debug_rows, [])


if __name__ == "__main__":
    unittest.main()
