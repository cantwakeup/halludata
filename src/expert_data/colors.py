"""Optional color utilities for panoptic-guided dominant-color estimation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

COLOR_VOCAB = ("black", "white", "red", "yellow", "green", "blue", "brown", "orange")

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]


def _decode_panoptic_rgb(rgb_triplet: tuple[int, int, int]) -> int:
    """Decode a COCO panoptic RGB triplet into an integer segment id."""

    red, green, blue = rgb_triplet
    return int(red) + (256 * int(green)) + (256 * 256 * int(blue))


def extract_mask_pixels_from_panoptic(
    panoptic_mask_path: str | Path,
    image_path: str | Path,
    segment_id: int,
) -> list[tuple[int, int, int]] | None:
    """Extract RGB pixels for one segment id from a panoptic mask and source image."""

    if Image is None:
        return None

    mask_path = Path(panoptic_mask_path)
    rgb_image_path = Path(image_path)
    if not mask_path.exists() or not rgb_image_path.exists():
        return None

    try:
        with Image.open(mask_path) as panoptic_mask:
            mask_image = panoptic_mask.convert("RGB")
            mask_pixels = list(mask_image.getdata())
            width, height = mask_image.size

        with Image.open(rgb_image_path) as rgb_image:
            color_pixels = list(rgb_image.convert("RGB").getdata())
    except OSError:
        return None

    if len(color_pixels) != len(mask_pixels):
        return None

    selected_pixels: list[tuple[int, int, int]] = []
    for index, pixel in enumerate(mask_pixels):
        if _decode_panoptic_rgb(pixel) == int(segment_id):
            selected_pixels.append(color_pixels[index])

    if not selected_pixels:
        return None
    return selected_pixels


def _bucket_channel_value(value: float) -> str:
    """Bucket a single RGB channel into low or high intensity bands."""

    return "high" if value >= 160 else "low"


def estimate_dominant_color(
    pixels: Iterable[tuple[int, int, int]] | None,
) -> str | None:
    """Estimate a coarse dominant color token from RGB pixels."""

    if pixels is None:
        return None

    collected_pixels = list(pixels)
    if not collected_pixels:
        return None

    red_mean = sum(pixel[0] for pixel in collected_pixels) / len(collected_pixels)
    green_mean = sum(pixel[1] for pixel in collected_pixels) / len(collected_pixels)
    blue_mean = sum(pixel[2] for pixel in collected_pixels) / len(collected_pixels)

    if red_mean < 50 and green_mean < 50 and blue_mean < 50:
        return "black"
    if red_mean > 205 and green_mean > 205 and blue_mean > 205:
        return "white"

    channel_order = Counter(
        {
            "red": red_mean,
            "green": green_mean,
            "blue": blue_mean,
        }
    ).most_common()
    strongest_channel = channel_order[0][0]
    weakest_channel = channel_order[-1][0]

    if strongest_channel == "red":
        if green_mean > 140 and blue_mean < 120:
            return "yellow" if red_mean > 170 else "orange"
        if green_mean > 90 and blue_mean < 90:
            return "brown"
        return "red"
    if strongest_channel == "green":
        return "green"
    if strongest_channel == "blue":
        return "blue"

    bucket_signature = (
        _bucket_channel_value(red_mean),
        _bucket_channel_value(green_mean),
        _bucket_channel_value(blue_mean),
        weakest_channel,
    )
    if bucket_signature[0] == bucket_signature[1] == "high":
        return "yellow"
    return None
