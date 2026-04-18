"""Optional color utilities for panoptic-guided dominant-color estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from expert_data.filters import bbox_center, bbox_iou

COLOR_VOCAB = ("black", "white", "red", "yellow", "green", "blue", "brown", "orange")

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]


def encode_panoptic_segment_id(segment_id: int) -> tuple[int, int, int]:
    """Encode a COCO panoptic integer segment id into its RGB triplet form."""

    return (
        int(segment_id) % 256,
        (int(segment_id) // 256) % 256,
        (int(segment_id) // (256 * 256)) % 256,
    )


def _decode_panoptic_rgb(rgb_triplet: tuple[int, int, int]) -> int:
    """Decode a COCO panoptic RGB triplet into an integer segment id."""

    red, green, blue = rgb_triplet
    return int(red) + (256 * int(green)) + (256 * 256 * int(blue))


def _load_ascii_ppm(path: str | Path) -> tuple[int, int, list[tuple[int, int, int]]] | None:
    """Load a plain-text P3 PPM image for lightweight local smoke tests."""

    image_path = Path(path)
    try:
        with image_path.open("r", encoding="ascii") as handle:
            tokens: list[str] = []
            for raw_line in handle:
                line = raw_line.split("#", 1)[0].strip()
                if line:
                    tokens.extend(line.split())
    except (OSError, UnicodeDecodeError):
        return None

    if len(tokens) < 4 or tokens[0] != "P3":
        return None
    try:
        width = int(tokens[1])
        height = int(tokens[2])
        max_value = int(tokens[3])
        if max_value <= 0:
            return None
        channel_values = [int(token) for token in tokens[4:]]
    except ValueError:
        return None

    if len(channel_values) != width * height * 3:
        return None

    pixels: list[tuple[int, int, int]] = []
    for index in range(0, len(channel_values), 3):
        pixels.append(
            (
                channel_values[index],
                channel_values[index + 1],
                channel_values[index + 2],
            )
        )
    return width, height, pixels


def _load_rgb_raster(path: str | Path) -> tuple[int, int, list[tuple[int, int, int]]] | None:
    """Load an RGB raster via Pillow when available, then fall back to ASCII PPM."""

    image_path = Path(path)
    if Image is not None:
        try:
            with Image.open(image_path) as handle:
                rgb_image = handle.convert("RGB")
                return rgb_image.size[0], rgb_image.size[1], list(rgb_image.getdata())
        except OSError:
            pass
    return _load_ascii_ppm(image_path)


def match_annotation_to_segment_id(
    annotation: Mapping[str, Any],
    segments_info: Iterable[Mapping[str, Any]],
) -> int | None:
    """Match one instance annotation to the most plausible panoptic segment id."""

    category_id = int(annotation.get("category_id", -1))
    candidate_segments = [
        segment
        for segment in segments_info
        if int(segment.get("category_id", -2)) == category_id
    ]
    if not candidate_segments:
        return None

    explicit_segment_id = annotation.get("segment_id")
    if explicit_segment_id is not None:
        explicit_id = int(explicit_segment_id)
        if any(int(segment.get("id", -1)) == explicit_id for segment in candidate_segments):
            return explicit_id

    if len(candidate_segments) == 1:
        return int(candidate_segments[0]["id"])

    annotation_bbox = list(annotation.get("bbox", []))
    annotation_area = float(annotation.get("area", 0.0))
    annotation_center = bbox_center(annotation_bbox) if len(annotation_bbox) == 4 else None

    best_segment_id: int | None = None
    best_score = float("-inf")
    for segment in candidate_segments:
        score = 0.0
        segment_bbox = list(segment.get("bbox", []))
        if len(annotation_bbox) == 4 and len(segment_bbox) == 4:
            overlap_score = bbox_iou(annotation_bbox, segment_bbox)
            score += overlap_score * 10.0

            if annotation_center is not None:
                center_x, center_y = annotation_center
                seg_x, seg_y, seg_w, seg_h = [float(value) for value in segment_bbox]
                if seg_x <= center_x <= seg_x + seg_w and seg_y <= center_y <= seg_y + seg_h:
                    score += 1.0

        segment_area = float(segment.get("area", 0.0))
        if annotation_area > 0.0 and segment_area > 0.0:
            normalized_gap = abs(segment_area - annotation_area) / max(annotation_area, segment_area)
            score += max(0.0, 1.0 - normalized_gap)

        if score > best_score:
            best_score = score
            best_segment_id = int(segment["id"])

    return best_segment_id


def extract_mask_pixels_from_panoptic(
    panoptic_mask_path: str | Path,
    image_path: str | Path,
    segment_id: int,
) -> list[tuple[int, int, int]] | None:
    """Extract RGB pixels for one segment id from a panoptic mask and source image."""

    mask_raster = _load_rgb_raster(panoptic_mask_path)
    image_raster = _load_rgb_raster(image_path)
    if mask_raster is None or image_raster is None:
        return None

    mask_width, mask_height, mask_pixels = mask_raster
    image_width, image_height, color_pixels = image_raster
    if mask_width != image_width or mask_height != image_height:
        return None

    selected_pixels: list[tuple[int, int, int]] = []
    for index, pixel in enumerate(mask_pixels):
        if _decode_panoptic_rgb(pixel) == int(segment_id):
            selected_pixels.append(color_pixels[index])

    if not selected_pixels:
        return None
    return selected_pixels


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
    brightness = (red_mean + green_mean + blue_mean) / 3.0

    if brightness < 45:
        return "black"
    if brightness > 225 and max(red_mean, green_mean, blue_mean) - min(red_mean, green_mean, blue_mean) < 25:
        return "white"

    if red_mean > 150 and green_mean > 110 and blue_mean < 120:
        if red_mean > 185 and green_mean > 170:
            return "yellow"
        if red_mean > 165 and green_mean < 150:
            return "orange"
        return "brown"

    if red_mean > 120 and green_mean > 70 and blue_mean < 90 and brightness < 170:
        return "brown"
    if red_mean > green_mean + 35 and red_mean > blue_mean + 35:
        return "red"
    if green_mean > red_mean + 25 and green_mean > blue_mean + 25:
        return "green"
    if blue_mean > red_mean + 25 and blue_mean > green_mean + 25:
        return "blue"
    if red_mean > 170 and green_mean > 150 and blue_mean < 110:
        return "yellow"
    if red_mean > 175 and green_mean > 95 and blue_mean < 90:
        return "orange"
    if brightness < 95:
        return "brown"
    return None


def estimate_annotation_colors_for_image(
    image_info: Mapping[str, Any],
    annotations: list[Mapping[str, Any]],
    panoptic_annotation: Mapping[str, Any] | None,
    panoptic_root: str | Path | None,
    image_root: str | Path | None,
) -> tuple[dict[int, str | None], list[dict[str, Any]]]:
    """Estimate colors for all annotations in one image and return debug-friendly rows."""

    if panoptic_annotation is None or panoptic_root is None or image_root is None:
        return {}, []

    panoptic_file_name = panoptic_annotation.get("file_name")
    image_file_name = image_info.get("file_name")
    if panoptic_file_name is None or image_file_name is None:
        return {}, []

    segments_info = panoptic_annotation.get("segments_info", [])
    if not isinstance(segments_info, list):
        return {}, []

    panoptic_mask_path = Path(panoptic_root) / str(panoptic_file_name)
    image_path = Path(image_root) / str(image_file_name)
    if not panoptic_mask_path.exists() or not image_path.exists():
        return {}, []

    image_id = str(image_info.get("id"))
    color_lookup: dict[int, str | None] = {}
    debug_rows: list[dict[str, Any]] = []
    for annotation in annotations:
        annotation_id = int(annotation["id"])
        segment_id = match_annotation_to_segment_id(annotation, segments_info)
        pixels = None
        if segment_id is not None:
            pixels = extract_mask_pixels_from_panoptic(
                panoptic_mask_path=panoptic_mask_path,
                image_path=image_path,
                segment_id=segment_id,
            )
        dominant_color = estimate_dominant_color(pixels)
        color_lookup[annotation_id] = dominant_color
        debug_rows.append(
            {
                "image_id": image_id,
                "object_id": f"{image_id}_{annotation_id}",
                "annotation_id": annotation_id,
                "segment_id": segment_id,
                "mask_pixels": len(pixels) if pixels is not None else 0,
                "dominant_color": dominant_color,
            }
        )
    return color_lookup, debug_rows

