"""Build balanced present/absent category truthfulness pairs from COCO instances."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.coco_io import load_coco_instances
from expert_data.io_utils import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for category truthfulness pair construction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-ann", required=True, help="COCO instances JSON file.")
    parser.add_argument("--image-root", required=True, help="Image root recorded for provenance.")
    parser.add_argument("--output", default="data/outputs/pair_banks/cat_truthfulness_train.jsonl")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--positives-per-image", type=int, default=1)
    parser.add_argument("--negatives-per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def article_for(noun: str) -> str:
    """Return a simple English indefinite article for a category name."""

    normalized = str(noun).strip().lower()
    if not normalized:
        return "a"
    if normalized.startswith(("hour", "honest")):
        return "an"
    return "an" if normalized[0] in "aeiou" else "a"


def question_for(category: str) -> str:
    """Build a POPE-style object-existence question for one category."""

    return f"Is there {article_for(category)} {category} in the image?"


def derive_split_paths(train_path: Path) -> dict[str, Path]:
    """Derive train/val/test sibling paths from the requested train output path."""

    name = train_path.name
    if "train" in name:
        return {
            "train": train_path,
            "val": train_path.with_name(name.replace("train", "val", 1)),
            "test": train_path.with_name(name.replace("train", "test", 1)),
        }
    return {
        "train": train_path,
        "val": train_path.with_name(f"{train_path.stem}_val{train_path.suffix}"),
        "test": train_path.with_name(f"{train_path.stem}_test{train_path.suffix}"),
    }


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate split ratios."""

    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if min(train_ratio, val_ratio, test_ratio) < 0.0:
        raise ValueError("Split ratios must be non-negative")


def image_categories(
    categories_by_id: dict[int, str],
    annotations: list[dict[str, Any]],
) -> list[str]:
    """Return sorted unique category names present in one image's annotations."""

    names = {
        categories_by_id[int(annotation["category_id"])]
        for annotation in annotations
        if int(annotation.get("category_id", -1)) in categories_by_id
    }
    return sorted(names)


def build_pairs_for_image(
    *,
    image: dict[str, Any],
    present_categories: list[str],
    all_categories: list[str],
    positives_per_image: int,
    negatives_per_image: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build balanced present/absent truthfulness rows for one image."""

    image_id = int(image["id"])
    file_name = str(image.get("file_name") or f"{image_id:012d}.jpg")
    absent_categories = sorted(set(all_categories) - set(present_categories))
    rows: list[dict[str, Any]] = []
    present_sample = rng.sample(present_categories, k=min(int(positives_per_image), len(present_categories)))
    absent_sample = rng.sample(absent_categories, k=min(int(negatives_per_image), len(absent_categories)))
    for category in present_sample:
        pair_id = f"cat_truth_present_{image_id}_{category.replace(' ', '_')}"
        rows.append(
            {
                "pair_id": pair_id,
                "image": file_name,
                "image_id": image_id,
                "question": question_for(category),
                "factual_answer": "Yes.",
                "counterfactual_answer": "No.",
                "response_pos": "Yes.",
                "response_neg": "No.",
                "label": "yes",
                "object": category,
                "subtype": "cat_truth_present",
            }
        )
    for category in absent_sample:
        pair_id = f"cat_truth_absent_{image_id}_{category.replace(' ', '_')}"
        rows.append(
            {
                "pair_id": pair_id,
                "image": file_name,
                "image_id": image_id,
                "question": question_for(category),
                "factual_answer": "No.",
                "counterfactual_answer": "Yes.",
                "response_pos": "No.",
                "response_neg": "Yes.",
                "label": "no",
                "object": category,
                "subtype": "cat_truth_absent",
            }
        )
    return rows


def split_image_ids(image_ids: list[int], train_ratio: float, val_ratio: float) -> dict[str, set[int]]:
    """Split shuffled image ids into train/val/test sets."""

    n = len(image_ids)
    train_end = int(round(n * float(train_ratio)))
    val_end = train_end + int(round(n * float(val_ratio)))
    return {
        "train": set(image_ids[:train_end]),
        "val": set(image_ids[train_end:val_end]),
        "test": set(image_ids[val_end:]),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize row counts by subtype and label."""

    return {
        "num_rows": len(rows),
        "num_images": len({row["image_id"] for row in rows}),
        "subtype_counts": dict(Counter(str(row["subtype"]) for row in rows)),
        "label_counts": dict(Counter(str(row["label"]) for row in rows)),
    }


def main() -> int:
    """Build balanced category truthfulness pair-bank splits."""

    args = parse_args()
    try:
        validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
        rng = random.Random(int(args.seed))
        images_by_id, categories_by_id, annotations_by_image = load_coco_instances(resolve_project_path(args.coco_ann))
        all_categories = sorted(categories_by_id.values())
        eligible_images: list[tuple[int, list[str]]] = []
        for image_id, image in images_by_id.items():
            present = image_categories(categories_by_id, annotations_by_image.get(image_id, []))
            if present and len(set(all_categories) - set(present)) > 0:
                eligible_images.append((int(image_id), present))
        rng.shuffle(eligible_images)
        selected = eligible_images[: int(args.num_images)]
        selected_ids = [image_id for image_id, _present in selected]
        split_ids = split_image_ids(selected_ids, args.train_ratio, args.val_ratio)
        output_paths = derive_split_paths(resolve_project_path(args.output))
        for path in output_paths.values():
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace.")

        rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        present_by_image = dict(selected)
        for split, image_ids in split_ids.items():
            for image_id in sorted(image_ids):
                rows_by_split[split].extend(
                    build_pairs_for_image(
                        image=images_by_id[image_id],
                        present_categories=present_by_image[image_id],
                        all_categories=all_categories,
                        positives_per_image=args.positives_per_image,
                        negatives_per_image=args.negatives_per_image,
                        rng=rng,
                    )
                )
        for split, rows in rows_by_split.items():
            write_jsonl(output_paths[split], rows)
        manifest = {
            "source": str(resolve_project_path(args.coco_ann)),
            "image_root": str(resolve_project_path(args.image_root)),
            "num_selected_images": len(selected_ids),
            "seed": int(args.seed),
            "outputs": {split: str(path) for split, path in output_paths.items()},
            "splits": {split: summarize_rows(rows) for split, rows in rows_by_split.items()},
            "notes": [
                "balanced present/absent category truthfulness pairs",
                "image-level split",
                "factual answers are response_pos and counterfactual answers are response_neg",
            ],
        }
        write_json(output_paths["train"].with_name("cat_truthfulness_manifest.json"), manifest)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote cat truthfulness pairs: {manifest['outputs']}")
    print(f"Summary: {manifest['splits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
