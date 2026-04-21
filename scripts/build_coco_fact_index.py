"""Build COCO-backed fact-index and atomic-fact outputs for local or cloud runs."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.coco_io import load_coco_instances, load_coco_panoptic
from expert_data.colors import estimate_annotation_colors_for_image
from expert_data.facts import (
    build_atomic_facts_for_image,
    build_counts_for_image,
    build_objects_for_image,
    build_relations_for_image,
)
from expert_data.io_utils import read_jsonl, read_yaml, write_jsonl
from expert_data.schemas import FactRecord, RelationInfo

DEFAULT_INSTANCES_JSON = "data/mock/mock_coco_instances.json"
DEFAULT_PANOPTIC_JSON = "data/mock/mock_coco_panoptic.json"
DEFAULT_PANOPTIC_ROOT = "data/mock/panoptic"
DEFAULT_IMAGE_ROOT = "data/mock/images"
DEFAULT_FACT_INDEX_PATH = "data/outputs/fact_index_v0.jsonl"
DEFAULT_ATOMIC_FACTS_PATH = "data/outputs/atomic_facts_v0.jsonl"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for fact-index construction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v0_mini.yaml", help="Path to the YAML config file.")
    parser.add_argument("--instances-json", default=None, help="Path to a COCO instances JSON file.")
    parser.add_argument("--panoptic-json", default=None, help="Optional COCO panoptic JSON path.")
    parser.add_argument("--panoptic-root", default=None, help="Optional directory containing panoptic PNG masks.")
    parser.add_argument("--image-root", default=None, help="Optional directory containing source RGB images.")
    parser.add_argument("--max-images", type=int, default=None, help="Process at most this many images.")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for image sampling.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output files when possible.")
    parser.add_argument("--output-fact-index", default=None, help="Output path for aggregated fact-index JSONL.")
    parser.add_argument("--output-atomic-facts", default=None, help="Output path for atomic facts JSONL.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_optional_project_path(raw_path: str | Path | None) -> Path | None:
    """Resolve optional project-relative paths and treat empty strings as absent."""

    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    return resolve_project_path(text)


def _serialize_object_entry(object_entry: dict[str, Any]) -> dict[str, Any]:
    """Serialize one filtered object entry for the aggregated fact index."""

    return {
        "object_id": str(object_entry["object_info"].object_id),
        "annotation_id": int(object_entry["annotation_id"]),
        "category": str(object_entry["object_info"].category),
        "bbox": list(object_entry["bbox"]),
        "area_ratio": float(object_entry["area_ratio"]),
        "dominant_color": object_entry["dominant_color"],
    }


def _build_fact_index_record_for_image(
    image_info: Mapping[str, Any],
    objects: list[dict[str, Any]],
    counts: dict[str, int],
    relations: list[RelationInfo],
) -> dict[str, Any]:
    """Build one aggregated per-image fact-index row."""

    return {
        "image_id": str(image_info["id"]),
        "height": int(image_info["height"]),
        "width": int(image_info["width"]),
        "objects": [_serialize_object_entry(item) for item in objects],
        "counts": {str(category): int(count) for category, count in counts.items()},
        "relations": [relation.to_dict() for relation in relations],
        "meta": {"source": "coco_instance"},
    }


def _select_image_ids(
    image_ids: list[int],
    max_images: int | None,
    sample_images_randomly: bool,
    random_seed: int,
) -> list[int]:
    """Select image ids according to the configured smoke-run sampling policy."""

    if max_images is None or max_images <= 0 or max_images >= len(image_ids):
        return list(image_ids)
    if sample_images_randomly:
        rng = random.Random(random_seed)
        return sorted(rng.sample(image_ids, max_images))
    return list(image_ids[:max_images])


def _load_resume_rows(path: Path) -> list[dict[str, Any]]:
    """Load existing JSONL rows when a resume checkpoint is available."""

    if not path.exists():
        return []
    return read_jsonl(path)


def _materialize_resume_atomic_rows(rows: list[dict[str, Any]]) -> list[FactRecord]:
    """Convert previously written atomic-fact rows back into typed fact records."""

    return [FactRecord.from_dict(row) for row in rows]


def _periodic_write(
    fact_index_rows: list[dict[str, Any]],
    atomic_fact_rows: list[FactRecord],
    fact_index_output_path: Path,
    atomic_facts_output_path: Path,
) -> None:
    """Write checkpoint outputs for long-running COCO fact-index jobs."""

    write_jsonl(fact_index_output_path, fact_index_rows)
    write_jsonl(atomic_facts_output_path, atomic_fact_rows)


def build_coco_fact_outputs(
    instances_json_path: str | Path,
    filters_cfg: dict[str, Any],
    coco_cfg: dict[str, Any] | None = None,
    run_cfg: dict[str, Any] | None = None,
    fact_index_output_path: str | Path | None = None,
    atomic_facts_output_path: str | Path | None = None,
    resume: bool = False,
) -> tuple[list[dict[str, Any]], list[FactRecord], dict[str, Any]]:
    """Build aggregated fact-index rows and atomic fact rows from COCO resources."""

    coco_cfg = dict(coco_cfg or {})
    run_cfg = dict(run_cfg or {})
    images_by_id, categories_by_id, annotations_by_image = load_coco_instances(instances_json_path)

    panoptic_annotations_by_image: dict[int, dict[str, Any]] = {}
    if coco_cfg.get("use_panoptic"):
        panoptic_json_path = resolve_optional_project_path(coco_cfg.get("panoptic_json"))
        if panoptic_json_path is not None and panoptic_json_path.exists():
            panoptic_annotations_by_image, _ = load_coco_panoptic(panoptic_json_path)

    image_ids = sorted(images_by_id)
    max_images = coco_cfg.get("max_images")
    selected_image_ids = _select_image_ids(
        image_ids=image_ids,
        max_images=int(max_images) if max_images not in {None, ""} else None,
        sample_images_randomly=bool(coco_cfg.get("sample_images_randomly", False)),
        random_seed=int(coco_cfg.get("random_seed", 42)),
    )

    fact_index_rows: list[dict[str, Any]] = []
    atomic_fact_rows: list[FactRecord] = []
    resumed_image_ids: set[str] = set()
    resolved_fact_index_output = resolve_optional_project_path(fact_index_output_path)
    resolved_atomic_output = resolve_optional_project_path(atomic_facts_output_path)
    if resume and resolved_fact_index_output is not None and resolved_atomic_output is not None:
        existing_fact_index_rows = _load_resume_rows(resolved_fact_index_output)
        existing_atomic_rows = _load_resume_rows(resolved_atomic_output)
        fact_index_rows.extend(existing_fact_index_rows)
        atomic_fact_rows.extend(_materialize_resume_atomic_rows(existing_atomic_rows))
        resumed_image_ids = {str(row["image_id"]) for row in existing_fact_index_rows}

    min_area_ratio = float(filters_cfg.get("object_min_area_ratio", 0.01))
    log_every = max(int(run_cfg.get("log_every", 20)), 1)
    save_every = max(int(run_cfg.get("save_every", 100)), 1)
    panoptic_root = resolve_optional_project_path(coco_cfg.get("panoptic_root"))
    image_root = resolve_optional_project_path(coco_cfg.get("image_root"))
    use_images_for_color = bool(coco_cfg.get("use_images_for_color", False))

    processed_count = 0
    skipped_for_resume = 0
    color_hits = 0
    for image_id in selected_image_ids:
        image_key = str(image_id)
        if image_key in resumed_image_ids:
            skipped_for_resume += 1
            continue

        image_info = images_by_id[image_id]
        annotations = annotations_by_image.get(image_id, [])
        dominant_colors_by_annotation_id: dict[int, str | None] = {}
        if use_images_for_color:
            try:
                dominant_colors_by_annotation_id, _ = estimate_annotation_colors_for_image(
                    image_info=image_info,
                    annotations=annotations,
                    panoptic_annotation=panoptic_annotations_by_image.get(image_id),
                    panoptic_root=panoptic_root,
                    image_root=image_root,
                )
            except Exception:
                dominant_colors_by_annotation_id = {}

        objects = build_objects_for_image(
            image_info=image_info,
            annotations=annotations,
            category_names=categories_by_id,
            min_area_ratio=min_area_ratio,
            dominant_colors_by_annotation_id=dominant_colors_by_annotation_id,
        )
        counts = build_counts_for_image(objects)
        relations = build_relations_for_image(objects, filters_cfg)
        fact_index_rows.append(
            _build_fact_index_record_for_image(
                image_info=image_info,
                objects=objects,
                counts=counts,
                relations=relations,
            )
        )
        image_atomic_rows = build_atomic_facts_for_image(
            image_info=image_info,
            objects=objects,
            counts=counts,
            relations=relations,
        )
        atomic_fact_rows.extend(image_atomic_rows)

        color_hits += sum(1 for item in objects if item["dominant_color"] is not None)
        processed_count += 1
        if processed_count % log_every == 0:
            print(
                f"[build_coco_fact_index] processed {processed_count}/{len(selected_image_ids)} selected images"
            )
        if (
            processed_count % save_every == 0
            and resolved_fact_index_output is not None
            and resolved_atomic_output is not None
        ):
            _periodic_write(
                fact_index_rows=fact_index_rows,
                atomic_fact_rows=atomic_fact_rows,
                fact_index_output_path=resolved_fact_index_output,
                atomic_facts_output_path=resolved_atomic_output,
            )

    summary = {
        "total_images_available": len(image_ids),
        "selected_images": len(selected_image_ids),
        "processed_images": processed_count,
        "skipped_for_resume": skipped_for_resume,
        "object_count": sum(len(row["objects"]) for row in fact_index_rows),
        "relation_count": sum(len(row["relations"]) for row in fact_index_rows),
        "colored_objects": color_hits,
        "atomic_counts": dict(sorted(Counter(row.subtype for row in atomic_fact_rows).items())),
    }
    return fact_index_rows, atomic_fact_rows, summary


def _resolve_instances_json(cli_args: argparse.Namespace, coco_cfg: Mapping[str, Any]) -> Path:
    """Resolve the active instances JSON path, falling back to the mock file when unavailable."""

    configured_path = cli_args.instances_json or coco_cfg.get("instances_json") or DEFAULT_INSTANCES_JSON
    resolved_path = resolve_optional_project_path(configured_path)
    if resolved_path is None:
        return resolve_project_path(DEFAULT_INSTANCES_JSON)
    if not resolved_path.exists():
        return resolve_project_path(DEFAULT_INSTANCES_JSON)
    return resolved_path


def _path_is_unavailable(raw_path: str | Path | None) -> bool:
    """Return whether an optional configured path is empty or missing on this machine."""

    resolved_path = resolve_optional_project_path(raw_path)
    return resolved_path is None or not resolved_path.exists()


def _with_mock_color_fallback(instances_json_path: Path, coco_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Use mock panoptic and image roots when mock instances are active but cloud paths are absent."""

    updated_cfg = dict(coco_cfg)
    mock_instances_path = resolve_project_path(DEFAULT_INSTANCES_JSON).resolve()
    if instances_json_path.resolve() != mock_instances_path:
        return updated_cfg

    if updated_cfg.get("use_panoptic") and _path_is_unavailable(updated_cfg.get("panoptic_json")):
        updated_cfg["panoptic_json"] = DEFAULT_PANOPTIC_JSON
    if updated_cfg.get("use_images_for_color"):
        if _path_is_unavailable(updated_cfg.get("panoptic_root")):
            updated_cfg["panoptic_root"] = DEFAULT_PANOPTIC_ROOT
        if _path_is_unavailable(updated_cfg.get("image_root")):
            updated_cfg["image_root"] = DEFAULT_IMAGE_ROOT
    return updated_cfg


def main() -> int:
    """Run the CLI entry point for local or cloud COCO fact-index construction."""

    args = parse_args()
    config = read_yaml(resolve_project_path(args.config))
    coco_cfg = dict(config.get("coco", {}))
    run_cfg = dict(config.get("run", {}))
    filters_cfg = dict(config.get("filters", {}))

    if args.panoptic_json is not None:
        coco_cfg["panoptic_json"] = args.panoptic_json
        coco_cfg["use_panoptic"] = True
    if args.panoptic_root is not None:
        coco_cfg["panoptic_root"] = args.panoptic_root
    if args.image_root is not None:
        coco_cfg["image_root"] = args.image_root
    if args.max_images is not None:
        coco_cfg["max_images"] = args.max_images
    if args.random_seed is not None:
        coco_cfg["random_seed"] = args.random_seed

    resume_enabled = bool(args.resume or run_cfg.get("resume", False))
    instances_json_path = _resolve_instances_json(args, coco_cfg)
    coco_cfg = _with_mock_color_fallback(instances_json_path, coco_cfg)
    fact_index_output_path = args.output_fact_index or DEFAULT_FACT_INDEX_PATH
    atomic_facts_output_path = args.output_atomic_facts or DEFAULT_ATOMIC_FACTS_PATH

    fact_index_rows, atomic_fact_rows, summary = build_coco_fact_outputs(
        instances_json_path=instances_json_path,
        filters_cfg=filters_cfg,
        coco_cfg=coco_cfg,
        run_cfg=run_cfg,
        fact_index_output_path=fact_index_output_path,
        atomic_facts_output_path=atomic_facts_output_path,
        resume=resume_enabled,
    )

    resolved_fact_index_output = resolve_project_path(fact_index_output_path)
    resolved_atomic_output = resolve_project_path(atomic_facts_output_path)
    write_jsonl(resolved_fact_index_output, fact_index_rows)
    write_jsonl(resolved_atomic_output, atomic_fact_rows)

    print(f"Instances source: {instances_json_path}")
    if coco_cfg.get("use_panoptic"):
        print(f"Panoptic source: {resolve_optional_project_path(coco_cfg.get('panoptic_json'))}")
    if coco_cfg.get("use_images_for_color"):
        print(f"Image root for color: {resolve_optional_project_path(coco_cfg.get('image_root'))}")
    print(f"Wrote {len(fact_index_rows)} fact-index rows to {resolved_fact_index_output}")
    print(f"Wrote {len(atomic_fact_rows)} atomic facts to {resolved_atomic_output}")
    print(
        "Summary: "
        + ", ".join(f"{key}={value}" for key, value in summary.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
