"""Build high-confidence MME-style relation yes/no AFTER-template pairs.

This relation-only v2 bank is intentionally isolated from the earlier
``after_template_v1`` data. It uses yes/no spatial relation questions plus
trusted factual text with inverse-relation explanations.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.coco_io import load_coco_instances
from expert_data.filters import bbox_center, bbox_iou, compute_area_ratio
from expert_data.io_utils import write_json, write_jsonl

RELATIONS = ("left_of", "right_of", "above", "below")
OPPOSITE = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
}
SUBTYPE = {
    "left_of": "rel_left",
    "right_of": "rel_right",
    "above": "rel_above",
    "below": "rel_below",
}
SIDE_PHRASE = {
    "left_of": "left side of",
    "right_of": "right side of",
    "above": "above",
    "below": "below",
}
QUERY_PHRASE = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "above": "above",
    "below": "below",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-instances", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", default="data/after_template_rel_v2/pairs")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", default="0.6,0.2,0.2")
    parser.add_argument("--max-pairs-per-image", type=int, default=4)
    parser.add_argument("--template-variant", choices=["basic", "inverse", "contrastive_inverse"], default="contrastive_inverse")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_split_ratio(raw_ratio: str) -> tuple[float, float, float]:
    """Parse a train,val,test split ratio."""

    pieces = [float(piece.strip()) for piece in str(raw_ratio).split(",") if piece.strip()]
    if len(pieces) != 3:
        raise ValueError("--split-ratio must contain exactly three comma-separated values")
    total = sum(pieces)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--split-ratio must sum to 1.0, got {total}")
    return pieces[0], pieces[1], pieces[2]


def split_image_ids(image_ids: list[int], ratios: tuple[float, float, float]) -> dict[str, set[int]]:
    """Split image IDs with image-level isolation."""

    train_end = int(round(len(image_ids) * ratios[0]))
    val_end = train_end + int(round(len(image_ids) * ratios[1]))
    return {
        "train": set(image_ids[:train_end]),
        "val": set(image_ids[train_end:val_end]),
        "test": set(image_ids[val_end:]),
    }


def clean_annotations(
    annotations: list[Mapping[str, Any]],
    categories_by_id: Mapping[int, str],
    image_width: float,
    image_height: float,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    """Filter annotations by crowd flag, known category, and bbox area."""

    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        category_id = int(annotation.get("category_id", -1))
        if category_id not in categories_by_id:
            continue
        if int(annotation.get("iscrowd", 0)) != 0:
            skipped["crowd"] += 1
            continue
        bbox = list(annotation.get("bbox", []))
        if len(bbox) != 4 or float(bbox[2]) <= 1.0 or float(bbox[3]) <= 1.0:
            skipped["invalid_bbox"] += 1
            continue
        area_ratio = compute_area_ratio(bbox, image_width, image_height)
        if area_ratio <= 0.005:
            skipped["too_small"] += 1
            continue
        if area_ratio >= 0.5:
            skipped["too_large"] += 1
            continue
        row = dict(annotation)
        row["category_name"] = categories_by_id[category_id]
        row["area_ratio"] = area_ratio
        rows.append(row)
    return rows


def largest_representatives(annotations: list[dict[str, Any]], skipped: Counter[str]) -> list[dict[str, Any]]:
    """Prefer unique categories; otherwise use the largest instance per category."""

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        by_category[str(annotation["category_name"])].append(annotation)
    unique = [items[0] for items in by_category.values() if len(items) == 1]
    if len(unique) >= 2:
        return unique
    representatives: list[dict[str, Any]] = []
    for items in by_category.values():
        if len(items) > 1:
            skipped["multi_instance_used"] += 1
        representatives.append(max(items, key=lambda item: float(item.get("area", 0.0))))
    return representatives


def infer_high_confidence_relation(
    ann_a: Mapping[str, Any],
    ann_b: Mapping[str, Any],
    image_width: float,
    image_height: float,
    skipped: Counter[str],
) -> str | None:
    """Infer a high-confidence directional relation from two bboxes."""

    box_a = list(ann_a.get("bbox", []))
    box_b = list(ann_b.get("bbox", []))
    if bbox_iou(box_a, box_b) > 0.05:
        skipped["overlap"] += 1
        return None
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    dx = ax - bx
    dy = ay - by
    horizontal = abs(dx) > 0.25 * float(image_width) and abs(dx) > 2.0 * abs(dy)
    vertical = abs(dy) > 0.25 * float(image_height) and abs(dy) > 2.0 * abs(dx)
    if horizontal:
        return "left_of" if dx < 0.0 else "right_of"
    if vertical:
        return "above" if dy < 0.0 else "below"
    if abs(dx) >= abs(dy):
        skipped["ambiguous_horizontal"] += 1
    else:
        skipped["ambiguous_vertical"] += 1
    return None


def inverse_fact(object_a: str, object_b: str, relation: str) -> str:
    """Render the inverse-relation explanation sentence."""

    inverse = OPPOSITE[relation]
    return f"This means the {object_b} is {SIDE_PHRASE[inverse]} the {object_a}."


def fact_sentence(object_a: str, object_b: str, relation: str) -> str:
    """Render the true spatial fact sentence."""

    if relation in {"left_of", "right_of"}:
        return f"The {object_a} is located on the {SIDE_PHRASE[relation]} the {object_b} in the image."
    return f"The {object_a} is located {SIDE_PHRASE[relation]} the {object_b} in the image."


def trusted_text(object_a: str, object_b: str, true_relation: str, queried_relation: str, label: str, variant: str) -> str:
    """Render trusted factual text for one relation query."""

    basic = f"The {object_a} is {QUERY_PHRASE[true_relation]} the {object_b}."
    if variant == "basic":
        return basic
    fact = fact_sentence(object_a, object_b, true_relation)
    if variant == "inverse" or label == "yes":
        return f"{fact} {inverse_fact(object_a, object_b, true_relation)}"
    if variant == "contrastive_inverse":
        not_clause = f"not {QUERY_PHRASE[queried_relation]} the {object_b}"
        return f"{fact}, {not_clause}. {inverse_fact(object_a, object_b, true_relation)}"
    return f"{fact} {inverse_fact(object_a, object_b, true_relation)}"


def render_visual_prompt(question: str) -> str:
    """Render the image-query side in a fuller AFTER-style format."""

    return f"Question: {question}\nPlease answer the question based on the image."


def render_trusted_prompt(trusted_factual_text: str, question: str) -> str:
    """Render the text-only trusted side close to AFTER's factual-description prompt."""

    return (
        f"The given image depicts the following scene: {trusted_factual_text}\n"
        "Please directly answer the following question from the image description, "
        f"without guessing or reasoning. Question: {question}"
    )


def relation_row(
    *,
    image: Mapping[str, Any],
    ann_a: Mapping[str, Any],
    ann_b: Mapping[str, Any],
    true_relation: str,
    queried_relation: str,
    label: str,
    template_variant: str,
    suffix: str,
) -> dict[str, Any]:
    """Build one MME-style relation yes/no row."""

    object_a = str(ann_a["category_name"])
    object_b = str(ann_b["category_name"])
    question = f"Is the {object_a} {QUERY_PHRASE[queried_relation]} the {object_b} in the image?"
    trusted_factual_text = trusted_text(object_a, object_b, true_relation, queried_relation, label, template_variant)
    image_id = int(image["id"])
    identifier = f"after_template_rel_v2_{image_id}_{suffix}_{label}_{queried_relation}"
    return {
        "id": identifier,
        "pair_id": identifier,
        "image": str(image.get("file_name") or f"{image_id:012d}.jpg"),
        "image_id": image_id,
        "object_a": object_a,
        "object_b": object_b,
        "bbox_a": [float(value) for value in ann_a.get("bbox", [])],
        "bbox_b": [float(value) for value in ann_b.get("bbox", [])],
        "true_relation": true_relation,
        "queried_relation": queried_relation,
        "label": label,
        "question": question,
        "visual_prompt": render_visual_prompt(question),
        "trusted_factual_text": trusted_factual_text,
        "trusted_prompt": render_trusted_prompt(trusted_factual_text, question),
        "hallucination_type": "rel",
        "subtype": SUBTYPE[true_relation],
        "template_variant": template_variant,
        "prompt_style": "after_fas_complete_v1",
        "objects": [object_a, object_b],
        "factual_fact": f"{object_a} {QUERY_PHRASE[true_relation]} {object_b}",
        "source": "after_template_rel_v2",
    }


def build_pairs_for_image(
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    max_pairs_per_image: int,
    rng: random.Random,
    skipped: Counter[str],
    template_variant: str,
) -> list[dict[str, Any]]:
    """Build balanced yes/no relation rows for one image."""

    width = float(image.get("width", 0.0) or 0.0)
    height = float(image.get("height", 0.0) or 0.0)
    representatives = largest_representatives(annotations, skipped)
    if len(representatives) < 2:
        skipped["multi_instance_ambiguous"] += 1
        return []
    pairs = list(combinations(representatives, 2))
    rng.shuffle(pairs)
    rows: list[dict[str, Any]] = []
    max_rows = max(0, int(max_pairs_per_image))
    if max_rows % 2 == 1:
        max_rows -= 1
    if max_rows <= 0:
        return []
    for ann_a, ann_b in pairs:
        if str(ann_a["category_name"]) == str(ann_b["category_name"]):
            skipped["multi_instance_ambiguous"] += 1
            continue
        if rng.random() < 0.5:
            ann_a, ann_b = ann_b, ann_a
        relation = infer_high_confidence_relation(ann_a, ann_b, width, height, skipped)
        if relation is None:
            continue
        suffix = f"{ann_a['category_name']}_{relation}_{ann_b['category_name']}".replace(" ", "_")
        rows.append(
            relation_row(
                image=image,
                ann_a=ann_a,
                ann_b=ann_b,
                true_relation=relation,
                queried_relation=relation,
                label="yes",
                template_variant=template_variant,
                suffix=suffix,
            )
        )
        rows.append(
            relation_row(
                image=image,
                ann_a=ann_a,
                ann_b=ann_b,
                true_relation=relation,
                queried_relation=OPPOSITE[relation],
                label="no",
                template_variant=template_variant,
                suffix=suffix,
            )
        )
        if len(rows) >= max_rows:
            return rows[:max_rows]
    if not rows:
        skipped["no_valid_relation"] += 1
    return rows[:max_rows]


def summarize(rows_by_split: Mapping[str, list[dict[str, Any]]], skipped: Counter[str]) -> dict[str, Any]:
    """Summarize relation v2 rows."""

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    return {
        "total_pairs": len(all_rows),
        "train_pairs": len(rows_by_split.get("train", [])),
        "val_pairs": len(rows_by_split.get("val", [])),
        "test_pairs": len(rows_by_split.get("test", [])),
        "num_images": len({int(row["image_id"]) for row in all_rows}),
        "split_image_counts": {
            split: len({int(row["image_id"]) for row in rows})
            for split, rows in rows_by_split.items()
        },
        "label_counts": dict(Counter(str(row["label"]) for row in all_rows)),
        "true_relation_counts": dict(Counter(str(row["true_relation"]) for row in all_rows)),
        "queried_relation_counts": dict(Counter(str(row["queried_relation"]) for row in all_rows)),
        "subtype_counts": dict(Counter(str(row["subtype"]) for row in all_rows)),
        "skipped": dict(skipped),
    }


def main() -> int:
    """Build relation v2 pair splits."""

    args = parse_args()
    try:
        output_dir = resolve_project_path(args.output_dir)
        output_paths = {split: output_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
        stats_path = output_dir / "stats.json"
        if not args.overwrite:
            existing = [path for path in [*output_paths.values(), stats_path] if path.exists()]
            if existing:
                raise FileExistsError(f"Output already exists: {existing[0]}. Pass --overwrite to replace.")

        rng = random.Random(int(args.seed))
        ratios = parse_split_ratio(args.split_ratio)
        images_by_id, categories_by_id, annotations_by_image = load_coco_instances(resolve_project_path(args.coco_instances))
        skipped: Counter[str] = Counter()
        candidate_ids = list(images_by_id)
        rng.shuffle(candidate_ids)

        rows_by_image: dict[int, list[dict[str, Any]]] = {}
        for image_id in candidate_ids:
            image = images_by_id[image_id]
            width = float(image.get("width", 0.0) or 0.0)
            height = float(image.get("height", 0.0) or 0.0)
            annotations = clean_annotations(
                annotations_by_image.get(image_id, []),
                categories_by_id,
                width,
                height,
                skipped,
            )
            if len(annotations) < 2:
                skipped["no_valid_objects"] += 1
                continue
            rows = build_pairs_for_image(
                image=image,
                annotations=annotations,
                max_pairs_per_image=int(args.max_pairs_per_image),
                rng=rng,
                skipped=skipped,
                template_variant=str(args.template_variant),
            )
            if not rows:
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
            "source": "after_template_rel_v2",
            "coco_instances": str(resolve_project_path(args.coco_instances)),
            "image_root": str(resolve_project_path(args.image_root)),
            "num_requested_images": int(args.num_images),
            "num_selected_images": len(selected_ids),
            "split_ratio": list(ratios),
            "max_pairs_per_image": int(args.max_pairs_per_image),
            "template_variant": str(args.template_variant),
            "seed": int(args.seed),
            "outputs": {split: str(path) for split, path in output_paths.items()},
            **summarize(rows_by_split, skipped),
        }
        write_json(stats_path, stats)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote relation v2 pairs to {output_dir}")
    print(f"Summary: total_pairs={stats['total_pairs']}, selected_images={stats['num_selected_images']}, labels={stats['label_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
