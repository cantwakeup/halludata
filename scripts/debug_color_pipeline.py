"""Debug the panoptic-guided dominant-color pipeline on a small image sample."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.coco_io import load_coco_instances, load_coco_panoptic
from expert_data.colors import estimate_annotation_colors_for_image
from expert_data.io_utils import read_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the color debug utility."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cloud_smoke.yaml", help="Path to the YAML config file.")
    parser.add_argument("--instances-json", default=None, help="Optional COCO instances JSON override.")
    parser.add_argument("--panoptic-json", default=None, help="Optional panoptic JSON override.")
    parser.add_argument("--panoptic-root", default=None, help="Optional panoptic mask root override.")
    parser.add_argument("--image-root", default=None, help="Optional source image root override.")
    parser.add_argument("--max-images", type=int, default=None, help="Process at most this many images.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_optional_project_path(raw_path: str | Path | None) -> Path | None:
    """Resolve an optional project-relative path and treat blanks as missing."""

    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    return resolve_project_path(text)


def _select_image_ids(image_ids: list[int], max_images: int | None) -> list[int]:
    """Select a deterministic image prefix for lightweight debugging."""

    if max_images is None or max_images <= 0 or max_images >= len(image_ids):
        return list(image_ids)
    return list(image_ids[:max_images])


def main() -> int:
    """Run the color debug utility and print a compact summary report."""

    args = parse_args()
    config = read_yaml(resolve_project_path(args.config))
    coco_cfg = dict(config.get("coco", {}))

    instances_json = resolve_optional_project_path(args.instances_json or coco_cfg.get("instances_json"))
    panoptic_json = resolve_optional_project_path(args.panoptic_json or coco_cfg.get("panoptic_json"))
    panoptic_root = resolve_optional_project_path(args.panoptic_root or coco_cfg.get("panoptic_root"))
    image_root = resolve_optional_project_path(args.image_root or coco_cfg.get("image_root"))
    max_images = args.max_images if args.max_images is not None else int(coco_cfg.get("max_images", 5))

    if instances_json is None:
        raise ValueError("An instances JSON path is required for color debugging")

    images_by_id, categories_by_id, annotations_by_image = load_coco_instances(instances_json)
    panoptic_by_image: dict[int, dict[str, Any]] = {}
    if panoptic_json is not None and panoptic_json.exists():
        panoptic_by_image, _ = load_coco_panoptic(panoptic_json)

    report = {
        "checked_images": 0,
        "checked_objects": 0,
        "objects_with_mask": 0,
        "objects_with_color": 0,
        "per_category_color_examples": {},
    }
    sample_rows: list[dict[str, Any]] = []
    examples_by_category: dict[str, list[str]] = defaultdict(list)

    for image_id in _select_image_ids(sorted(images_by_id), max_images):
        image_info = images_by_id[image_id]
        annotations = annotations_by_image.get(image_id, [])
        report["checked_images"] += 1
        report["checked_objects"] += len(annotations)

        color_lookup, debug_rows = estimate_annotation_colors_for_image(
            image_info=image_info,
            annotations=annotations,
            panoptic_annotation=panoptic_by_image.get(image_id),
            panoptic_root=panoptic_root,
            image_root=image_root,
        )

        for annotation, debug_row in zip(annotations, debug_rows):
            if int(debug_row["mask_pixels"]) > 0:
                report["objects_with_mask"] += 1
            color_value = color_lookup.get(int(annotation["id"]))
            if color_value:
                report["objects_with_color"] += 1
                category = categories_by_id[int(annotation["category_id"])]
                if len(examples_by_category[category]) < 3:
                    examples_by_category[category].append(str(color_value))
            if int(debug_row["mask_pixels"]) > 0 and len(sample_rows) < 5:
                sample_rows.append(
                    {
                        "image_id": debug_row["image_id"],
                        "object_id": debug_row["object_id"],
                        "category": categories_by_id[int(annotation["category_id"])],
                        "mask_pixels": debug_row["mask_pixels"],
                        "dominant_color": color_value,
                    }
                )

    report["per_category_color_examples"] = {
        category: colors
        for category, colors in sorted(examples_by_category.items())
    }

    print(
        "Color debug summary: "
        + ", ".join(f"{key}={value}" for key, value in report.items() if key != "per_category_color_examples")
    )
    print(f"Per-category color examples: {report['per_category_color_examples']}")
    for sample in sample_rows:
        print(
            f"sample image_id={sample['image_id']} object_id={sample['object_id']} "
            f"category={sample['category']} mask_pixels={sample['mask_pixels']} "
            f"dominant_color={sample['dominant_color']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

