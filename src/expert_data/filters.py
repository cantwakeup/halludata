"""Geometry helpers for filtering COCO objects and inferring simple relations."""

from __future__ import annotations

from typing import Sequence


def is_valid_object(area_ratio: float, min_area_ratio: float) -> bool:
    """Return whether an object passes the minimum normalized area threshold."""

    return area_ratio >= min_area_ratio


def compute_area_ratio(bbox: Sequence[float], width: float, height: float) -> float:
    """Compute object box area divided by image area for a COCO-style bbox."""

    if len(bbox) != 4 or width <= 0 or height <= 0:
        return 0.0
    box_width = max(float(bbox[2]), 0.0)
    box_height = max(float(bbox[3]), 0.0)
    return (box_width * box_height) / (float(width) * float(height))


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    """Return the center point of a COCO-style bbox."""

    if len(bbox) != 4:
        raise ValueError("bbox must contain four values: x, y, width, height")
    x, y, width, height = (float(value) for value in bbox)
    return x + (width / 2.0), y + (height / 2.0)


def bbox_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute IoU between two COCO-style bboxes."""

    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1, y1, w1, h1 = (float(value) for value in box1)
    x2, y2, w2, h2 = (float(value) for value in box2)
    x1_max = x1 + max(w1, 0.0)
    y1_max = y1 + max(h1, 0.0)
    x2_max = x2 + max(w2, 0.0)
    y2_max = y2 + max(h2, 0.0)

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area1 = max(w1, 0.0) * max(h1, 0.0)
    area2 = max(w2, 0.0) * max(h2, 0.0)
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def infer_spatial_relation(
    box1: Sequence[float],
    box2: Sequence[float],
    dx_thresh: float,
    dy_thresh: float,
    iou_thresh: float,
) -> str | None:
    """Infer a coarse pairwise spatial relation from two bboxes."""

    if bbox_iou(box1, box2) > iou_thresh:
        return None

    center1_x, center1_y = bbox_center(box1)
    center2_x, center2_y = bbox_center(box2)
    avg_width = max((float(box1[2]) + float(box2[2])) / 2.0, 1e-6)
    avg_height = max((float(box1[3]) + float(box2[3])) / 2.0, 1e-6)

    dx_norm = (center2_x - center1_x) / avg_width
    dy_norm = (center2_y - center1_y) / avg_height

    if abs(dx_norm) >= dx_thresh and abs(dy_norm) <= dy_thresh:
        return "left of" if dx_norm > 0 else "right of"
    if abs(dy_norm) >= dx_thresh and abs(dx_norm) <= dy_thresh:
        return "above" if dy_norm > 0 else "below"
    return None
