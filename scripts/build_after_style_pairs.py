"""Build lightweight AFTER-style factual/counterfactual COCO pair banks."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.coco_io import load_coco_instances
from expert_data.filters import bbox_center, bbox_iou
from expert_data.io_utils import write_json, write_jsonl
from expert_data.text_utils import count_conditioned_noun, pluralize_noun

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]


COLOR_VOCAB = (
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "gray",
    "brown",
    "orange",
    "purple",
    "pink",
)
OPPOSITE_RELATION = {
    "left of": "right of",
    "right of": "left of",
    "above": "below",
    "below": "above",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for AFTER-style pair construction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-instances", required=True, help="COCO instances JSON.")
    parser.add_argument("--image-root", required=True, help="COCO image directory.")
    parser.add_argument("--output-dir", default="data/after_style_v1/pairs")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", default="0.6,0.2,0.2")
    parser.add_argument("--max-pairs-per-image", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def article_for(noun: str) -> str:
    """Return a simple English indefinite article for a category phrase."""

    normalized = str(noun).strip().lower()
    if not normalized:
        return "a"
    if normalized.startswith(("hour", "honest")):
        return "an"
    return "an" if normalized[0] in "aeiou" else "a"


def relation_phrase(relation: str) -> str:
    """Render a relation as a natural phrase for answer text."""

    if relation == "left of":
        return "to the left of"
    if relation == "right of":
        return "to the right of"
    return relation


def parse_split_ratio(raw_ratio: str) -> tuple[float, float, float]:
    """Parse and validate a comma-separated train,val,test split ratio."""

    pieces = [float(piece.strip()) for piece in str(raw_ratio).split(",") if piece.strip()]
    if len(pieces) != 3:
        raise ValueError("--split-ratio must contain exactly three comma-separated numbers")
    total = sum(pieces)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--split-ratio must sum to 1.0, got {total}")
    if min(pieces) < 0.0:
        raise ValueError("--split-ratio values must be non-negative")
    return pieces[0], pieces[1], pieces[2]


def valid_annotations(
    annotations: Iterable[Mapping[str, Any]],
    categories_by_id: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Return non-crowd annotations with usable boxes and known categories."""

    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        category_id = int(annotation.get("category_id", -1))
        bbox = list(annotation.get("bbox", []))
        if category_id not in categories_by_id:
            continue
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        if len(bbox) != 4 or float(bbox[2]) <= 1.0 or float(bbox[3]) <= 1.0:
            continue
        row = dict(annotation)
        row["category_name"] = categories_by_id[category_id]
        rows.append(row)
    return rows


def pair_id(image_id: int, subtype: str, suffix: str) -> str:
    """Build a stable pair id."""

    safe_suffix = suffix.replace(" ", "_").replace("/", "_")
    return f"after_style_v1_{subtype}_{image_id}_{safe_suffix}"


def base_row(
    *,
    image: Mapping[str, Any],
    question: str,
    factual_answer: str,
    counterfactual_answer: str,
    hallucination_type: str,
    subtype: str,
    objects: list[str],
    factual_fact: str,
    counterfactual_fact: str,
    suffix: str,
) -> dict[str, Any]:
    """Create one common AFTER-style pair row."""

    image_id = int(image["id"])
    identifier = pair_id(image_id, subtype, suffix)
    return {
        "id": identifier,
        "pair_id": identifier,
        "image": str(image.get("file_name") or f"{image_id:012d}.jpg"),
        "image_id": image_id,
        "question": question,
        "factual_answer": factual_answer,
        "counterfactual_answer": counterfactual_answer,
        "response_pos": factual_answer,
        "response_neg": counterfactual_answer,
        "hallucination_type": hallucination_type,
        "subtype": subtype,
        "objects": objects,
        "factual_fact": factual_fact,
        "counterfactual_fact": counterfactual_fact,
        "source": "after_style_v1",
    }


def build_cat_pairs(
    image: Mapping[str, Any],
    present_categories: list[str],
    all_categories: list[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build one present and one absent object-existence pair."""

    rows: list[dict[str, Any]] = []
    image_id = int(image["id"])
    if present_categories:
        category = rng.choice(present_categories)
        article = article_for(category)
        rows.append(
            base_row(
                image=image,
                question=f"Is there {article} {category} in the image?",
                factual_answer=f"Yes. There is {article} {category} in the image.",
                counterfactual_answer=f"No. There is no {category} in the image.",
                hallucination_type="cat",
                subtype="cat_present",
                objects=[category],
                factual_fact=f"{category} is present in the image",
                counterfactual_fact=f"{category} is absent from the image",
                suffix=category,
            )
        )
    absent_categories = sorted(set(all_categories) - set(present_categories))
    if absent_categories:
        category = rng.choice(absent_categories)
        article = article_for(category)
        rows.append(
            base_row(
                image=image,
                question=f"Is there {article} {category} in the image?",
                factual_answer=f"No. There is no {category} in the image.",
                counterfactual_answer=f"Yes. There is {article} {category} in the image.",
                hallucination_type="cat",
                subtype="cat_absent",
                objects=[category],
                factual_fact=f"{category} is absent from the image",
                counterfactual_fact=f"{category} is present in the image",
                suffix=f"{category}_absent_{image_id}",
            )
        )
    return rows


def count_answer(category: str, count: int) -> str:
    """Render a natural count answer."""

    noun = count_conditioned_noun(category, count)
    verb = "is" if int(count) == 1 else "are"
    return f"There {verb} {int(count)} {noun} in the image."


def choose_wrong_count(true_count: int, rng: random.Random) -> int:
    """Choose a nearby count that differs from the true count."""

    candidates = []
    if true_count > 1:
        candidates.append(true_count - 1)
    candidates.append(true_count + 1)
    if true_count == 1:
        candidates.append(3)
    candidates = sorted({candidate for candidate in candidates if candidate >= 0 and candidate != true_count})
    return rng.choice(candidates)


def build_count_pair(
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any] | None:
    """Build one count factual/counterfactual pair from category counts."""

    counts = Counter(str(annotation["category_name"]) for annotation in annotations)
    candidates = [(category, count) for category, count in counts.items() if count > 0]
    if not candidates:
        return None
    category, true_count = rng.choice(candidates)
    wrong_count = choose_wrong_count(int(true_count), rng)
    question_noun = pluralize_noun(category)
    return base_row(
        image=image,
        question=f"How many {question_noun} are there in the image?",
        factual_answer=count_answer(category, int(true_count)),
        counterfactual_answer=count_answer(category, int(wrong_count)),
        hallucination_type="attr",
        subtype="attr_count",
        objects=[category],
        factual_fact=f"count({category}) = {int(true_count)}",
        counterfactual_fact=f"count({category}) = {int(wrong_count)}",
        suffix=f"{category}_count",
    )


def estimate_color_from_bbox(image_path: Path, bbox: Iterable[float]) -> str | None:
    """Estimate a coarse dominant color from a COCO bbox crop."""

    if Image is None or not image_path.exists():
        return None
    try:
        with Image.open(image_path) as handle:
            rgb = handle.convert("RGB")
            x, y, width, height = [float(value) for value in bbox]
            left = max(0, int(round(x)))
            upper = max(0, int(round(y)))
            right = min(rgb.size[0], int(round(x + width)))
            lower = min(rgb.size[1], int(round(y + height)))
            if right <= left or lower <= upper:
                return None
            pixels = list(rgb.crop((left, upper, right, lower)).getdata())
    except OSError:
        return None
    if not pixels:
        return None

    red = sum(pixel[0] for pixel in pixels) / len(pixels)
    green = sum(pixel[1] for pixel in pixels) / len(pixels)
    blue = sum(pixel[2] for pixel in pixels) / len(pixels)
    brightness = (red + green + blue) / 3.0
    spread = max(red, green, blue) - min(red, green, blue)

    if brightness < 45:
        return "black"
    if brightness > 220 and spread < 30:
        return "white"
    if spread < 22:
        return "gray"
    if red > 170 and blue > 140 and green < 135:
        return "pink" if brightness > 150 else "purple"
    if red > green + 35 and red > blue + 35:
        if green > 105 and blue < 110:
            return "orange"
        return "red"
    if green > red + 30 and green > blue + 30:
        return "green"
    if blue > red + 30 and blue > green + 30:
        return "blue"
    if red > 165 and green > 145 and blue < 120:
        return "yellow"
    if red > 95 and green > 55 and blue < 85:
        return "brown"
    return None


def build_color_pair(
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    image_root: Path,
    rng: random.Random,
    skipped: Counter[str],
) -> dict[str, Any] | None:
    """Build one color pair when a stable bbox color can be estimated."""

    image_path = image_root / str(image.get("file_name") or "")
    shuffled = list(annotations)
    rng.shuffle(shuffled)
    for annotation in shuffled:
        category = str(annotation["category_name"])
        true_color = estimate_color_from_bbox(image_path, annotation.get("bbox", []))
        if true_color is None:
            continue
        wrong_color = rng.choice([color for color in COLOR_VOCAB if color != true_color])
        return base_row(
            image=image,
            question=f"What color is the {category} in the image?",
            factual_answer=f"The {category} is {true_color}.",
            counterfactual_answer=f"The {category} is {wrong_color}.",
            hallucination_type="attr",
            subtype="attr_color",
            objects=[category],
            factual_fact=f"color({category}) = {true_color}",
            counterfactual_fact=f"color({category}) = {wrong_color}",
            suffix=f"{category}_{true_color}",
        )
    skipped["no_stable_color"] += 1
    return None


def infer_relation(
    bbox_a: Iterable[float],
    bbox_b: Iterable[float],
    image_width: float,
    image_height: float,
) -> str | None:
    """Infer a stable left/right/above/below relation from two bboxes."""

    box_a = list(bbox_a)
    box_b = list(bbox_b)
    if bbox_iou(box_a, box_b) > 0.20:
        return None
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    dx = ax - bx
    dy = ay - by
    min_dx = max(float(image_width) * 0.10, 1.0)
    min_dy = max(float(image_height) * 0.10, 1.0)
    if abs(dx) >= abs(dy) and abs(dx) >= min_dx:
        return "left of" if dx < 0 else "right of"
    if abs(dy) > abs(dx) and abs(dy) >= min_dy:
        return "above" if dy < 0 else "below"
    return None


def build_relation_pair(
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    rng: random.Random,
    skipped: Counter[str],
) -> dict[str, Any] | None:
    """Build one bbox-derived spatial relation pair."""

    pairs = list(combinations(annotations, 2))
    rng.shuffle(pairs)
    width = float(image.get("width", 0.0) or 0.0)
    height = float(image.get("height", 0.0) or 0.0)
    for left, right in pairs:
        category_a = str(left["category_name"])
        category_b = str(right["category_name"])
        if category_a == category_b:
            continue
        if rng.random() < 0.5:
            ann_a, ann_b = left, right
            category_a, category_b = category_a, category_b
        else:
            ann_a, ann_b = right, left
            category_a, category_b = category_b, category_a
        relation = infer_relation(ann_a.get("bbox", []), ann_b.get("bbox", []), width, height)
        if relation is None:
            continue
        counter_relation = OPPOSITE_RELATION[relation]
        factual_relation = relation_phrase(relation)
        counter_relation_phrase = relation_phrase(counter_relation)
        factual = f"The {category_a} is {factual_relation} the {category_b}."
        counterfactual = f"The {category_a} is {counter_relation_phrase} the {category_b}."
        return base_row(
            image=image,
            question=f"Where is the {category_a} relative to the {category_b}?",
            factual_answer=factual,
            counterfactual_answer=counterfactual,
            hallucination_type="rel",
            subtype="rel_spatial",
            objects=[category_a, category_b],
            factual_fact=f"{category_a} {factual_relation} {category_b}",
            counterfactual_fact=f"{category_a} {counter_relation_phrase} {category_b}",
            suffix=f"{category_a}_{relation}_{category_b}",
        )
    skipped["no_clear_relation"] += 1
    return None


def build_pairs_for_image(
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    all_categories: list[str],
    image_root: Path,
    max_pairs_per_image: int,
    rng: random.Random,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    """Build all lightweight AFTER-style pairs for one image."""

    present_categories = sorted({str(annotation["category_name"]) for annotation in annotations})
    rows: list[dict[str, Any]] = []
    rows.extend(build_cat_pairs(image, present_categories, all_categories, rng))
    count_pair = build_count_pair(image, annotations, rng)
    if count_pair is not None:
        rows.append(count_pair)
    color_pair = build_color_pair(image, annotations, image_root, rng, skipped)
    if color_pair is not None:
        rows.append(color_pair)
    relation_pair = build_relation_pair(image, annotations, rng, skipped)
    if relation_pair is not None:
        rows.append(relation_pair)
    return rows[: max(0, int(max_pairs_per_image))]


def split_image_ids(image_ids: list[int], ratios: tuple[float, float, float]) -> dict[str, set[int]]:
    """Split image ids into train/val/test sets."""

    n = len(image_ids)
    train_end = int(round(n * ratios[0]))
    val_end = train_end + int(round(n * ratios[1]))
    return {
        "train": set(image_ids[:train_end]),
        "val": set(image_ids[train_end:val_end]),
        "test": set(image_ids[val_end:]),
    }


def summarize_rows(rows_by_split: Mapping[str, list[dict[str, Any]]], skipped: Counter[str]) -> dict[str, Any]:
    """Summarize generated rows and skip reasons."""

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    return {
        "total_pairs": len(all_rows),
        "splits": {
            split: {
                "num_pairs": len(rows),
                "num_images": len({row["image_id"] for row in rows}),
                "hallucination_type_counts": dict(Counter(str(row["hallucination_type"]) for row in rows)),
                "subtype_counts": dict(Counter(str(row["subtype"]) for row in rows)),
            }
            for split, rows in rows_by_split.items()
        },
        "hallucination_type_counts": dict(Counter(str(row["hallucination_type"]) for row in all_rows)),
        "subtype_counts": dict(Counter(str(row["subtype"]) for row in all_rows)),
        "cat_present_absent": {
            "cat_present": sum(1 for row in all_rows if row["subtype"] == "cat_present"),
            "cat_absent": sum(1 for row in all_rows if row["subtype"] == "cat_absent"),
        },
        "attribute_counts": {
            "attr_count": sum(1 for row in all_rows if row["subtype"] == "attr_count"),
            "attr_color": sum(1 for row in all_rows if row["subtype"] == "attr_color"),
        },
        "relation_subtype_counts": dict(Counter(str(row["subtype"]) for row in all_rows if row["hallucination_type"] == "rel")),
        "skipped_reason_counts": dict(skipped),
    }


def main() -> int:
    """Build and save AFTER-style pair splits."""

    args = parse_args()
    try:
        output_dir = resolve_project_path(args.output_dir)
        output_paths = {split: output_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
        stats_path = output_dir / "stats.json"
        if not args.overwrite:
            existing = [path for path in [*output_paths.values(), stats_path] if path.exists()]
            if existing:
                raise FileExistsError(f"Output already exists: {existing[0]}. Pass --overwrite to replace after_style_v1 outputs.")
        rng = random.Random(int(args.seed))
        ratios = parse_split_ratio(args.split_ratio)
        image_root = resolve_project_path(args.image_root)
        images_by_id, categories_by_id, annotations_by_image = load_coco_instances(resolve_project_path(args.coco_instances))
        all_categories = sorted(categories_by_id.values())
        skipped: Counter[str] = Counter()

        candidate_ids = list(images_by_id)
        rng.shuffle(candidate_ids)
        rows_by_image: dict[int, list[dict[str, Any]]] = {}
        for image_id in candidate_ids:
            image = images_by_id[image_id]
            annotations = valid_annotations(annotations_by_image.get(image_id, []), categories_by_id)
            if not annotations:
                skipped["no_valid_annotations"] += 1
                continue
            rows = build_pairs_for_image(
                image=image,
                annotations=annotations,
                all_categories=all_categories,
                image_root=image_root,
                max_pairs_per_image=args.max_pairs_per_image,
                rng=rng,
                skipped=skipped,
            )
            if not rows:
                skipped["no_pairs_for_image"] += 1
                continue
            rows_by_image[int(image_id)] = rows
            if len(rows_by_image) >= int(args.num_images):
                break

        selected_ids = list(rows_by_image)
        split_ids = split_image_ids(selected_ids, ratios)
        rows_by_split = {
            split: [row for image_id in selected_ids if image_id in image_ids for row in rows_by_image[image_id]]
            for split, image_ids in split_ids.items()
        }
        for split, rows in rows_by_split.items():
            write_jsonl(output_paths[split], rows)
        stats = {
            "source": "after_style_v1",
            "coco_instances": str(resolve_project_path(args.coco_instances)),
            "image_root": str(image_root),
            "num_requested_images": int(args.num_images),
            "num_selected_images": len(selected_ids),
            "split_ratio": list(ratios),
            "max_pairs_per_image": int(args.max_pairs_per_image),
            "seed": int(args.seed),
            "outputs": {split: str(path) for split, path in output_paths.items()},
            **summarize_rows(rows_by_split, skipped),
        }
        write_json(stats_path, stats)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote AFTER-style pairs to {output_dir}")
    print(f"Summary: total_pairs={stats['total_pairs']}, selected_images={stats['num_selected_images']}, counts={stats['hallucination_type_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
