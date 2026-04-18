"""Helpers for converting COCO instance annotations into typed fact records."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from expert_data.filters import (
    bbox_center,
    bbox_iou,
    compute_area_ratio,
    infer_spatial_relation,
    is_valid_object,
)
from expert_data.schemas import FactRecord, ObjectInfo, RelationInfo

__all__ = [
    "build_objects_for_image",
    "build_counts_for_image",
    "build_relations_for_image",
    "build_fact_record",
    "build_atomic_facts_for_image",
]


def build_objects_for_image(
    image_info: Mapping[str, Any],
    annotations: list[Mapping[str, Any]],
    category_names: Mapping[int, str],
    min_area_ratio: float,
    dominant_colors_by_annotation_id: Mapping[int, str | None] | None = None,
) -> list[dict[str, Any]]:
    """Build filtered per-object records for one image from COCO instance annotations."""

    image_id = int(image_info["id"])
    width = float(image_info["width"])
    height = float(image_info["height"])
    objects: list[dict[str, Any]] = []

    for annotation in annotations:
        bbox = list(annotation.get("bbox", []))
        area_ratio = compute_area_ratio(bbox, width, height)
        if not is_valid_object(area_ratio, min_area_ratio):
            continue

        annotation_id = int(annotation["id"])
        category_id = int(annotation["category_id"])
        category_name = str(category_names[category_id])
        dominant_color = None
        if dominant_colors_by_annotation_id is not None:
            dominant_color = dominant_colors_by_annotation_id.get(annotation_id)
        object_info = ObjectInfo(
            object_id=f"{image_id}_{annotation_id}",
            name=category_name,
            category=category_name,
            color=dominant_color,
            aliases=[],
        )
        objects.append(
            {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area_ratio": area_ratio,
                "dominant_color": dominant_color,
                "object_info": object_info,
            }
        )

    return objects


def build_counts_for_image(objects: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate filtered objects into per-category counts for one image."""

    category_counts = Counter(str(item["object_info"].category) for item in objects)
    return {category_name: int(count) for category_name, count in sorted(category_counts.items())}


def build_relations_for_image(
    objects: list[dict[str, Any]],
    cfg: Mapping[str, Any],
) -> list[RelationInfo]:
    """Infer at most one coarse spatial relation for each object pair in an image."""

    dx_thresh = float(cfg.get("rel_min_abs_dx", 0.75))
    dy_thresh = float(cfg.get("rel_max_abs_dy", 1.0))
    iou_thresh = float(cfg.get("rel_max_iou", 0.1))

    relations: list[RelationInfo] = []
    for left_index in range(len(objects)):
        for right_index in range(left_index + 1, len(objects)):
            subject_object = objects[left_index]
            object_object = objects[right_index]
            predicate = infer_spatial_relation(
                subject_object["bbox"],
                object_object["bbox"],
                dx_thresh=dx_thresh,
                dy_thresh=dy_thresh,
                iou_thresh=iou_thresh,
            )
            if predicate is None:
                continue
            subject_info = subject_object["object_info"]
            object_info = object_object["object_info"]
            subject_center_x, subject_center_y = bbox_center(subject_object["bbox"])
            object_center_x, object_center_y = bbox_center(object_object["bbox"])
            avg_width = max(
                (float(subject_object["bbox"][2]) + float(object_object["bbox"][2])) / 2.0,
                1e-6,
            )
            avg_height = max(
                (float(subject_object["bbox"][3]) + float(object_object["bbox"][3])) / 2.0,
                1e-6,
            )
            dx_norm = (object_center_x - subject_center_x) / avg_width
            dy_norm = (object_center_y - subject_center_y) / avg_height
            relations.append(
                RelationInfo(
                    subject_id=subject_info.object_id,
                    predicate=predicate,
                    object_id=object_info.object_id,
                    subject_category=subject_info.category,
                    object_category=object_info.category,
                    dx=dx_norm,
                    dy=dy_norm,
                    iou=bbox_iou(subject_object["bbox"], object_object["bbox"]),
                )
            )
    return relations


def build_fact_record(
    fact_id: str,
    image_id: int | str,
    subtype: str,
    subject: ObjectInfo,
    positive_value: Any,
    object_info: ObjectInfo | None = None,
    relation: RelationInfo | None = None,
    negative_candidates: list[Any] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> FactRecord:
    """Build a schema-aligned fact record with COCO instance source metadata."""

    metadata = {"source": "coco_instance"}
    if meta:
        metadata.update(dict(meta))
    return FactRecord(
        fact_id=str(fact_id),
        image_id=str(image_id),
        subtype=str(subtype),
        subject=subject,
        object=object_info,
        relation=relation,
        positive_value=positive_value,
        negative_candidates=list(negative_candidates or []),
        meta=metadata,
    )


def _build_count_subject(image_id: int | str, category_name: str) -> ObjectInfo:
    """Create a synthetic subject object used by count facts."""

    return ObjectInfo(
        object_id=f"{image_id}_count_{category_name}",
        name=category_name,
        category=category_name,
        color=None,
        aliases=[],
    )


def build_atomic_facts_for_image(
    image_info: Mapping[str, Any],
    objects: list[dict[str, Any]],
    counts: Mapping[str, int],
    relations: list[RelationInfo],
) -> list[FactRecord]:
    """Build atomic cat/cnt/rel/optional-col fact records for one image."""

    image_id = int(image_info["id"])
    object_lookup = {
        item["object_info"].object_id: item["object_info"]
        for item in objects
    }

    fact_records: list[FactRecord] = []
    for item in objects:
        object_info = item["object_info"]
        fact_records.append(
            build_fact_record(
                fact_id=f"coco_{image_id}_cat_{item['annotation_id']}",
                image_id=image_id,
                subtype="cat",
                subject=object_info,
                positive_value=object_info.category,
                meta={
                    "annotation_id": item["annotation_id"],
                    "bbox": list(item["bbox"]),
                    "area_ratio": item["area_ratio"],
                },
            )
        )
        fact_records.append(
            build_fact_record(
                fact_id=f"coco_{image_id}_col_{item['annotation_id']}",
                image_id=image_id,
                subtype="col",
                subject=object_info,
                positive_value=item["dominant_color"],
                meta={"annotation_id": item["annotation_id"]},
            )
        )

    for category_name, count in counts.items():
        fact_records.append(
            build_fact_record(
                fact_id=f"coco_{image_id}_cnt_{category_name}",
                image_id=image_id,
                subtype="cnt",
                subject=_build_count_subject(image_id, category_name),
                positive_value=int(count),
            )
        )

    for index, relation in enumerate(relations, start=1):
        subject_info = object_lookup.get(relation.subject_id)
        object_info = object_lookup.get(relation.object_id)
        if subject_info is None or object_info is None:
            raise KeyError(
                f"Relation references unknown object ids: {relation.subject_id}, {relation.object_id}"
            )
        fact_records.append(
            build_fact_record(
                fact_id=f"coco_{image_id}_rel_{index}",
                image_id=image_id,
                subtype="rel",
                subject=subject_info,
                object_info=object_info,
                relation=relation,
                positive_value=relation.predicate,
            )
        )

    return fact_records
