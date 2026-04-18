"""Lightweight loaders for COCO instance and panoptic annotation JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from expert_data.io_utils import read_json


def load_coco_instances(
    path: str | Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, str], dict[int, list[dict[str, Any]]]]:
    """Load a minimal COCO instances file without relying on pycocotools."""

    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"COCO instances file at {path} must contain a JSON object")

    raw_images = payload.get("images", [])
    raw_annotations = payload.get("annotations", [])
    raw_categories = payload.get("categories", [])
    if not isinstance(raw_images, list) or not isinstance(raw_annotations, list) or not isinstance(raw_categories, list):
        raise ValueError("COCO instances payload must contain list-valued images, annotations, and categories")

    images_by_id: dict[int, dict[str, Any]] = {}
    for image in raw_images:
        if not isinstance(image, dict):
            raise ValueError("Each image entry must be a JSON object")
        image_id = int(image["id"])
        images_by_id[image_id] = dict(image)

    categories_by_id: dict[int, str] = {}
    for category in raw_categories:
        if not isinstance(category, dict):
            raise ValueError("Each category entry must be a JSON object")
        category_id = int(category["id"])
        categories_by_id[category_id] = str(category["name"])

    annotations_by_image: dict[int, list[dict[str, Any]]] = {
        image_id: [] for image_id in images_by_id
    }
    for annotation in raw_annotations:
        if not isinstance(annotation, dict):
            raise ValueError("Each annotation entry must be a JSON object")
        image_id = int(annotation["image_id"])
        annotations_by_image.setdefault(image_id, []).append(dict(annotation))

    return images_by_id, categories_by_id, annotations_by_image


def load_coco_panoptic(
    path: str | Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Load a minimal COCO panoptic JSON file keyed by image id and category id."""

    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"COCO panoptic file at {path} must contain a JSON object")

    raw_annotations = payload.get("annotations", [])
    raw_categories = payload.get("categories", [])
    if not isinstance(raw_annotations, list) or not isinstance(raw_categories, list):
        raise ValueError("COCO panoptic payload must contain list-valued annotations and categories")

    annotations_by_image: dict[int, dict[str, Any]] = {}
    for annotation in raw_annotations:
        if not isinstance(annotation, dict):
            raise ValueError("Each panoptic annotation entry must be a JSON object")
        annotations_by_image[int(annotation["image_id"])] = dict(annotation)

    categories_by_id: dict[int, dict[str, Any]] = {}
    for category in raw_categories:
        if not isinstance(category, dict):
            raise ValueError("Each panoptic category entry must be a JSON object")
        categories_by_id[int(category["id"])] = dict(category)

    return annotations_by_image, categories_by_id
