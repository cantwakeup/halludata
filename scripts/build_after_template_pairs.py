"""Build AFTER-template factual-text COCO pair banks.

This is the template-FAS variant:

- untrusted side: image + visual prompt
- trusted side: factual text + question

The output is intentionally isolated under data/after_template_v1 by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.coco_io import load_coco_instances
from expert_data.io_utils import write_json, write_jsonl


def _load_after_style_helpers() -> Any:
    """Load the sibling after_style builder by file path, avoiding package-name collisions."""

    helper_path = Path(__file__).resolve().with_name("build_after_style_pairs.py")
    spec = importlib.util.spec_from_file_location("_halludata_after_style_pairs", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STYLE = _load_after_style_helpers()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for AFTER-template pair construction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-instances", required=True, help="COCO instances JSON.")
    parser.add_argument("--image-root", required=True, help="COCO image directory.")
    parser.add_argument("--output-dir", default="data/after_template_v1/pairs")
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


def template_pair_id(image_id: int, subtype: str, suffix: str) -> str:
    """Build a stable AFTER-template pair id."""

    safe_suffix = suffix.replace(" ", "_").replace("/", "_")
    return f"after_template_v1_{subtype}_{image_id}_{safe_suffix}"


def strip_yes_no_prefix(answer: str) -> str:
    """Remove a leading Yes./No. while keeping the factual sentence itself."""

    text = str(answer).strip()
    lowered = text.lower()
    for prefix in ("yes. ", "no. "):
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


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
    """Create one AFTER-template row while preserving compatibility fields."""

    image_id = int(image["id"])
    identifier = template_pair_id(image_id, subtype, suffix)
    trusted_factual_text = strip_yes_no_prefix(factual_answer)
    visual_prompt = render_visual_prompt(question)
    trusted_prompt = render_trusted_prompt(trusted_factual_text, question)
    return {
        "id": identifier,
        "pair_id": identifier,
        "image": str(image.get("file_name") or f"{image_id:012d}.jpg"),
        "image_id": image_id,
        "question": question,
        "visual_prompt": visual_prompt,
        "trusted_factual_text": trusted_factual_text,
        "trusted_prompt": trusted_prompt,
        "factual_answer": factual_answer,
        "counterfactual_answer": counterfactual_answer,
        "hallucination_type": hallucination_type,
        "subtype": subtype,
        "objects": objects,
        "factual_fact": factual_fact,
        "counterfactual_fact": counterfactual_fact,
        "prompt_style": "after_fas_complete_v1",
        "source": "after_template_v1",
    }


# Reuse the validated COCO construction logic but swap its row renderer.
STYLE.base_row = base_row


def summarize_rows(rows_by_split: Mapping[str, list[dict[str, Any]]], skipped: Counter[str]) -> dict[str, Any]:
    """Summarize generated rows and skip reasons in the requested template schema."""

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    skipped_counts = dict(skipped)
    skipped_counts.setdefault("no_valid_objects", int(skipped.get("no_valid_objects", 0)))
    skipped_counts.setdefault("no_absent_category", int(skipped.get("no_absent_category", 0)))
    skipped_counts.setdefault("no_color", int(skipped.get("no_stable_color", 0)))
    skipped_counts.setdefault("ambiguous_relation", int(skipped.get("no_clear_relation", 0)))
    split_image_counts = {
        split: len({int(row["image_id"]) for row in rows})
        for split, rows in rows_by_split.items()
    }
    return {
        "total_pairs": len(all_rows),
        "train_pairs": len(rows_by_split.get("train", [])),
        "val_pairs": len(rows_by_split.get("val", [])),
        "test_pairs": len(rows_by_split.get("test", [])),
        "type_counts": dict(Counter(str(row["hallucination_type"]) for row in all_rows)),
        "subtype_counts": dict(Counter(str(row["subtype"]) for row in all_rows)),
        "num_images": len({int(row["image_id"]) for row in all_rows}),
        "split_image_counts": split_image_counts,
        "splits": {
            split: {
                "num_pairs": len(rows),
                "num_images": split_image_counts[split],
                "type_counts": dict(Counter(str(row["hallucination_type"]) for row in rows)),
                "subtype_counts": dict(Counter(str(row["subtype"]) for row in rows)),
            }
            for split, rows in rows_by_split.items()
        },
        "skipped": skipped_counts,
        "raw_skipped": dict(skipped),
    }


def main() -> int:
    """Build and save AFTER-template pair splits."""

    args = parse_args()
    try:
        output_dir = resolve_project_path(args.output_dir)
        output_paths = {split: output_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
        stats_path = output_dir / "stats.json"
        if not args.overwrite:
            existing = [path for path in [*output_paths.values(), stats_path] if path.exists()]
            if existing:
                raise FileExistsError(f"Output already exists: {existing[0]}. Pass --overwrite to replace after_template_v1 outputs.")

        rng = random.Random(int(args.seed))
        ratios = STYLE.parse_split_ratio(args.split_ratio)
        image_root = resolve_project_path(args.image_root)
        images_by_id, categories_by_id, annotations_by_image = load_coco_instances(resolve_project_path(args.coco_instances))
        all_categories = sorted(categories_by_id.values())
        skipped: Counter[str] = Counter()

        candidate_ids = list(images_by_id)
        rng.shuffle(candidate_ids)
        rows_by_image: dict[int, list[dict[str, Any]]] = {}
        for image_id in candidate_ids:
            image = images_by_id[image_id]
            annotations = STYLE.valid_annotations(annotations_by_image.get(image_id, []), categories_by_id)
            if not annotations:
                skipped["no_valid_objects"] += 1
                continue
            rows = STYLE.build_pairs_for_image(
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
        split_ids = STYLE.split_image_ids(selected_ids, ratios)
        rows_by_split = {
            split: [row for image_id in selected_ids if image_id in image_ids for row in rows_by_image[image_id]]
            for split, image_ids in split_ids.items()
        }
        for split, rows in rows_by_split.items():
            write_jsonl(output_paths[split], rows)
        stats = {
            "source": "after_template_v1",
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
    print(f"Wrote AFTER-template pairs to {output_dir}")
    print(f"Summary: total_pairs={stats['total_pairs']}, selected_images={stats['num_selected_images']}, counts={stats['type_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
