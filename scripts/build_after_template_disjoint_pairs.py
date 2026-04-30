"""Build image-disjoint AFTER-template cat/attr/rel pair banks.

This builder fixes a leakage issue in the earlier AFTER-template data: the
same COCO image could contribute cat, attr, and rel samples. Here each selected
image is assigned to exactly one hallucination type before pairs are built.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.coco_io import load_coco_instances
from expert_data.io_utils import write_json, write_jsonl

EXPERT_TYPES = ("cat", "attr", "rel")


def _load_script_module(module_name: str, script_name: str) -> Any:
    """Load a sibling script as a module without relying on package imports."""

    helper_path = Path(__file__).resolve().with_name(script_name)
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE = _load_script_module("_halludata_after_template_pairs", "build_after_template_pairs.py")
RELATION_V2 = _load_script_module("_halludata_after_template_relation_v2", "build_after_template_relation_v2.py")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-instances", required=True, help="COCO instances JSON.")
    parser.add_argument("--image-root", required=True, help="COCO image directory.")
    parser.add_argument("--output-dir", default="data/after_template_disjoint_v1/pairs")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument(
        "--type-image-ratio",
        default="cat=0.3,attr=0.3,rel=0.4",
        help="Image allocation ratio, either cat=0.3,attr=0.3,rel=0.4 or 0.3,0.3,0.4.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", default="0.6,0.2,0.2")
    parser.add_argument("--max-cat-pairs-per-image", type=int, default=2)
    parser.add_argument("--max-attr-pairs-per-image", type=int, default=2)
    parser.add_argument("--max-rel-pairs-per-image", type=int, default=4)
    parser.add_argument(
        "--rel-template-variant",
        choices=["basic", "inverse", "contrastive_inverse"],
        default="contrastive_inverse",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_type_ratio(raw_ratio: str) -> dict[str, float]:
    """Parse and validate cat/attr/rel image allocation ratios."""

    text = str(raw_ratio).strip()
    if "=" in text:
        ratios: dict[str, float] = {}
        for piece in text.split(","):
            if not piece.strip():
                continue
            name, value = piece.split("=", 1)
            key = name.strip()
            if key not in EXPERT_TYPES:
                raise ValueError(f"Unknown type in --type-image-ratio: {key}")
            ratios[key] = float(value.strip())
    else:
        values = [float(piece.strip()) for piece in text.split(",") if piece.strip()]
        if len(values) != len(EXPERT_TYPES):
            raise ValueError("--type-image-ratio must provide cat,attr,rel ratios")
        ratios = dict(zip(EXPERT_TYPES, values))
    missing = [expert for expert in EXPERT_TYPES if expert not in ratios]
    if missing:
        raise ValueError(f"--type-image-ratio missing types: {missing}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--type-image-ratio must sum to 1.0, got {total}")
    if min(ratios.values()) < 0.0:
        raise ValueError("--type-image-ratio values must be non-negative")
    return ratios


def allocation_counts(num_images: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Convert ratios to integer counts using largest remainders."""

    exact = {expert: float(num_images) * float(ratios[expert]) for expert in EXPERT_TYPES}
    counts = {expert: int(math.floor(value)) for expert, value in exact.items()}
    remaining = int(num_images) - sum(counts.values())
    order = sorted(EXPERT_TYPES, key=lambda expert: (exact[expert] - counts[expert], expert), reverse=True)
    for expert in order[:remaining]:
        counts[expert] += 1
    return counts


def retag_rows(rows: list[dict[str, Any]], expert: str) -> list[dict[str, Any]]:
    """Mark rows as coming from the disjoint builder while preserving metadata."""

    retagged: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        old_id = str(item.get("id") or item.get("pair_id") or "")
        item["original_source"] = item.get("source")
        item["original_pair_id"] = old_id
        new_id = f"after_template_disjoint_v1_{expert}_{old_id}"
        item["id"] = new_id
        item["pair_id"] = new_id
        item["source"] = "after_template_disjoint_v1"
        item["type_image_bucket"] = expert
        retagged.append(item)
    return retagged


def build_cat_rows(
    *,
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    all_categories: list[str],
    max_pairs: int,
    rng: random.Random,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    """Build only category/existence rows for one image."""

    present_categories = sorted({str(annotation["category_name"]) for annotation in annotations})
    rows = TEMPLATE.STYLE.build_cat_pairs(image, present_categories, all_categories, rng)
    if not rows:
        skipped["cat_no_pairs_for_image"] += 1
    return rows[: max(0, int(max_pairs))]


def build_attr_rows(
    *,
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    image_root: Path,
    max_pairs: int,
    rng: random.Random,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    """Build only attribute rows for one image."""

    rows: list[dict[str, Any]] = []
    count_pair = TEMPLATE.STYLE.build_count_pair(image, annotations, rng)
    if count_pair is not None:
        rows.append(count_pair)
    else:
        skipped["attr_no_count"] += 1
    if len(rows) < int(max_pairs):
        color_pair = TEMPLATE.STYLE.build_color_pair(image, annotations, image_root, rng, skipped)
        if color_pair is not None:
            rows.append(color_pair)
    if not rows:
        skipped["attr_no_pairs_for_image"] += 1
    return rows[: max(0, int(max_pairs))]


def build_rel_rows(
    *,
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    categories_by_id: Mapping[int, str],
    max_pairs: int,
    rng: random.Random,
    skipped: Counter[str],
    template_variant: str,
) -> list[dict[str, Any]]:
    """Build only relation-v2 rows for one image."""

    width = float(image.get("width", 0.0) or 0.0)
    height = float(image.get("height", 0.0) or 0.0)
    clean = RELATION_V2.clean_annotations(annotations, categories_by_id, width, height, skipped)
    if len(clean) < 2:
        skipped["rel_no_valid_objects"] += 1
        return []
    rows = RELATION_V2.build_pairs_for_image(
        image=image,
        annotations=clean,
        max_pairs_per_image=int(max_pairs),
        rng=rng,
        skipped=skipped,
        template_variant=template_variant,
    )
    if not rows:
        skipped["rel_no_pairs_for_image"] += 1
    return rows


def split_type_images(
    image_ids_by_type: Mapping[str, list[int]],
    ratios: tuple[float, float, float],
) -> dict[str, set[int]]:
    """Split image ids per type, then merge splits to preserve type ratios."""

    merged = {"train": set(), "val": set(), "test": set()}
    for image_ids in image_ids_by_type.values():
        split_ids = TEMPLATE.STYLE.split_image_ids(list(image_ids), ratios)
        for split, ids in split_ids.items():
            merged[split].update(ids)
    return merged


def summarize(
    *,
    rows_by_split: Mapping[str, list[dict[str, Any]]],
    image_ids_by_type: Mapping[str, list[int]],
    target_counts: Mapping[str, int],
    skipped: Counter[str],
) -> dict[str, Any]:
    """Summarize disjoint rows and verify image overlap."""

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    image_sets = {expert: set(ids) for expert, ids in image_ids_by_type.items()}
    overlaps = {
        f"{left}_{right}": len(image_sets[left] & image_sets[right])
        for index, left in enumerate(EXPERT_TYPES)
        for right in EXPERT_TYPES[index + 1 :]
    }
    split_type_image_counts = {
        split: {
            expert: len({int(row["image_id"]) for row in rows if str(row["hallucination_type"]) == expert})
            for expert in EXPERT_TYPES
        }
        for split, rows in rows_by_split.items()
    }
    return {
        "total_pairs": len(all_rows),
        "train_pairs": len(rows_by_split.get("train", [])),
        "val_pairs": len(rows_by_split.get("val", [])),
        "test_pairs": len(rows_by_split.get("test", [])),
        "num_images": len({int(row["image_id"]) for row in all_rows}),
        "target_type_image_counts": dict(target_counts),
        "actual_type_image_counts": {expert: len(ids) for expert, ids in image_ids_by_type.items()},
        "cross_type_image_overlap": overlaps,
        "type_counts": dict(Counter(str(row["hallucination_type"]) for row in all_rows)),
        "subtype_counts": dict(Counter(str(row["subtype"]) for row in all_rows)),
        "split_image_counts": {
            split: len({int(row["image_id"]) for row in rows})
            for split, rows in rows_by_split.items()
        },
        "split_type_image_counts": split_type_image_counts,
        "splits": {
            split: {
                "num_pairs": len(rows),
                "num_images": len({int(row["image_id"]) for row in rows}),
                "type_counts": dict(Counter(str(row["hallucination_type"]) for row in rows)),
                "subtype_counts": dict(Counter(str(row["subtype"]) for row in rows)),
            }
            for split, rows in rows_by_split.items()
        },
        "skipped": dict(skipped),
    }


def main() -> int:
    """Build and save image-disjoint AFTER-template pair splits."""

    args = parse_args()
    try:
        output_dir = resolve_project_path(args.output_dir)
        output_paths = {split: output_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
        stats_path = output_dir / "stats.json"
        assignments_path = output_dir / "image_assignments.json"
        if not args.overwrite:
            existing = [path for path in [*output_paths.values(), stats_path, assignments_path] if path.exists()]
            if existing:
                raise FileExistsError(f"Output already exists: {existing[0]}. Pass --overwrite to replace.")

        rng = random.Random(int(args.seed))
        ratios = TEMPLATE.STYLE.parse_split_ratio(args.split_ratio)
        type_ratios = parse_type_ratio(args.type_image_ratio)
        target_counts = allocation_counts(int(args.num_images), type_ratios)
        image_root = resolve_project_path(args.image_root)
        images_by_id, categories_by_id, annotations_by_image = load_coco_instances(resolve_project_path(args.coco_instances))
        all_categories = sorted(categories_by_id.values())
        skipped: Counter[str] = Counter()

        builders: dict[str, Callable[[int], list[dict[str, Any]]]] = {}

        def annotations_for(image_id: int) -> list[dict[str, Any]]:
            return TEMPLATE.STYLE.valid_annotations(annotations_by_image.get(image_id, []), categories_by_id)

        def cat_builder(image_id: int) -> list[dict[str, Any]]:
            image = images_by_id[image_id]
            annotations = annotations_for(image_id)
            if not annotations:
                skipped["cat_no_valid_objects"] += 1
                return []
            return build_cat_rows(
                image=image,
                annotations=annotations,
                all_categories=all_categories,
                max_pairs=int(args.max_cat_pairs_per_image),
                rng=rng,
                skipped=skipped,
            )

        def attr_builder(image_id: int) -> list[dict[str, Any]]:
            image = images_by_id[image_id]
            annotations = annotations_for(image_id)
            if not annotations:
                skipped["attr_no_valid_objects"] += 1
                return []
            return build_attr_rows(
                image=image,
                annotations=annotations,
                image_root=image_root,
                max_pairs=int(args.max_attr_pairs_per_image),
                rng=rng,
                skipped=skipped,
            )

        def rel_builder(image_id: int) -> list[dict[str, Any]]:
            image = images_by_id[image_id]
            return build_rel_rows(
                image=image,
                annotations=annotations_by_image.get(image_id, []),
                categories_by_id=categories_by_id,
                max_pairs=int(args.max_rel_pairs_per_image),
                rng=rng,
                skipped=skipped,
                template_variant=str(args.rel_template_variant),
            )

        builders.update({"cat": cat_builder, "attr": attr_builder, "rel": rel_builder})

        candidate_ids = list(images_by_id)
        rng.shuffle(candidate_ids)
        assigned_ids: set[int] = set()
        rows_by_image: dict[int, list[dict[str, Any]]] = {}
        image_ids_by_type: dict[str, list[int]] = {expert: [] for expert in EXPERT_TYPES}

        # Fill harder relation images first so the requested rel quota is easier to satisfy.
        for expert in ("rel", "attr", "cat"):
            for image_id in candidate_ids:
                if image_id in assigned_ids:
                    continue
                if len(image_ids_by_type[expert]) >= target_counts[expert]:
                    break
                rows = retag_rows(builders[expert](image_id), expert)
                if not rows:
                    continue
                rows_by_image[int(image_id)] = rows
                image_ids_by_type[expert].append(int(image_id))
                assigned_ids.add(int(image_id))

        selected_ids = [image_id for expert in EXPERT_TYPES for image_id in image_ids_by_type[expert]]
        split_ids = split_type_images(image_ids_by_type, ratios)
        rows_by_split = {
            split: [row for image_id in selected_ids if image_id in image_ids for row in rows_by_image[image_id]]
            for split, image_ids in split_ids.items()
        }

        for split, rows in rows_by_split.items():
            write_jsonl(output_paths[split], rows)
        stats = {
            "source": "after_template_disjoint_v1",
            "coco_instances": str(resolve_project_path(args.coco_instances)),
            "image_root": str(image_root),
            "num_requested_images": int(args.num_images),
            "num_selected_images": len(selected_ids),
            "type_image_ratio": type_ratios,
            "split_ratio": list(ratios),
            "max_pairs_per_image": {
                "cat": int(args.max_cat_pairs_per_image),
                "attr": int(args.max_attr_pairs_per_image),
                "rel": int(args.max_rel_pairs_per_image),
            },
            "rel_template_variant": str(args.rel_template_variant),
            "seed": int(args.seed),
            "outputs": {split: str(path) for split, path in output_paths.items()},
            **summarize(
                rows_by_split=rows_by_split,
                image_ids_by_type=image_ids_by_type,
                target_counts=target_counts,
                skipped=skipped,
            ),
        }
        write_json(stats_path, stats)
        write_json(assignments_path, {expert: image_ids_by_type[expert] for expert in EXPERT_TYPES})
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote image-disjoint AFTER-template pairs to {output_dir}")
    print(
        "Summary: "
        f"total_pairs={stats['total_pairs']}, "
        f"selected_images={stats['num_selected_images']}, "
        f"type_images={stats['actual_type_image_counts']}, "
        f"overlap={stats['cross_type_image_overlap']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
