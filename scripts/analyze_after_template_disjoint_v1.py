"""Analyze the image-disjoint AFTER-template v1 pair bank.

This is a dataset diagnostic script only. It does not build activations,
vectors, routers, or benchmark runs.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = "data/outputs_after_template_disjoint_v1"
SPLITS = ("train", "val", "test")
TYPES = ("cat", "attr", "rel")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", default="data/after_template_disjoint_v1/pairs")
    parser.add_argument("--activation-meta", default="", help="Optional train.meta.jsonl from activation extraction.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object when absent."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""

    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_pair_rows(pair_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load train/val/test JSONL rows from one pair directory."""

    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split in SPLITS:
        split_rows = read_jsonl(pair_dir / f"{split}.jsonl")
        split_counts[split] = len(split_rows)
        for row in split_rows:
            item = dict(row)
            item["_split"] = split
            rows.append(item)
    if not rows:
        for path in sorted(pair_dir.glob("*.jsonl")):
            split_rows = read_jsonl(path)
            split_counts[path.stem] = len(split_rows)
            for row in split_rows:
                item = dict(row)
                item["_split"] = path.stem
                rows.append(item)
    return rows, split_counts


def normalize_type(row: Mapping[str, Any]) -> str:
    """Return the hallucination type, preferring the disjoint bucket when present."""

    value = str(row.get("type_image_bucket") or row.get("hallucination_type") or "").strip()
    if value in TYPES:
        return value
    return "unknown"


def normalize_subtype(row: Mapping[str, Any]) -> str:
    """Return a stable subtype string."""

    return str(row.get("subtype") or "unknown").strip() or "unknown"


def normalize_label(row: Mapping[str, Any]) -> str:
    """Infer a yes/no label when one exists; otherwise return unknown."""

    for key in ("label", "answer", "truth", "target", "factual_answer"):
        value = row.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip().lower()
        if text in {"yes", "y", "true", "1"} or text.startswith("yes."):
            return "yes"
        if text in {"no", "n", "false", "0"} or text.startswith("no."):
            return "no"
    subtype = normalize_subtype(row)
    if subtype == "cat_present":
        return "yes"
    if subtype == "cat_absent":
        return "no"
    return "unknown"


def words(text: Any) -> list[str]:
    """Split a question into simple word tokens."""

    return re.findall(r"[A-Za-z0-9']+", str(text or ""))


def relation_axis(value: Any) -> str:
    """Map a relation name or phrase to horizontal/vertical/unknown."""

    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"left_of", "right_of", "left", "right", "to_the_left_of", "to_the_right_of"}:
        return "horizontal"
    if text in {"above", "below", "top", "bottom"}:
        return "vertical"
    return "unknown"


def row_objects(row: Mapping[str, Any]) -> list[str]:
    """Return object/category mentions stored in a row."""

    objects = row.get("objects", [])
    if not isinstance(objects, list):
        objects = [objects]
    output = [str(value).strip() for value in objects if str(value).strip()]
    for key in ("object", "object_a", "object_b"):
        value = str(row.get(key) or "").strip()
        if value:
            output.append(value)
    return output


def counter_to_rows(counter: Counter[str], top_k: int | None = None) -> list[dict[str, Any]]:
    """Convert a Counter to sorted row dictionaries."""

    items = counter.most_common(top_k)
    return [{"key": key, "count": count} for key, count in items]


def stats_for_numbers(values: Iterable[int | float]) -> dict[str, Any]:
    """Return compact descriptive statistics."""

    data = [float(value) for value in values]
    if not data:
        return {"count": 0, "mean": None, "min": None, "max": None, "median": None}
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "min": min(data),
        "max": max(data),
        "median": statistics.median(data),
    }


def table(headers: list[str], rows: list[Mapping[str, Any]]) -> list[str]:
    """Render a markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def nested_count_rows(counts: Mapping[str, Mapping[str, int]]) -> list[dict[str, Any]]:
    """Flatten nested counts for markdown."""

    rows: list[dict[str, Any]] = []
    for outer, inner in sorted(counts.items()):
        for key, count in sorted(inner.items()):
            rows.append({"group": outer, "key": key, "count": count})
    return rows


def build_warnings(stats: Mapping[str, Any]) -> list[str]:
    """Build warnings for imbalance, overlap, or suspicious distribution shifts."""

    warnings: list[str] = []
    overlaps = stats.get("cross_type_image_overlap", {})
    if isinstance(overlaps, Mapping):
        for key, value in overlaps.items():
            if int(value) > 0:
                warnings.append(f"Image leakage detected: {key} overlap has {value} images.")

    image_counts = stats.get("image_count_by_type", {})
    if isinstance(image_counts, Mapping):
        known_counts = [int(image_counts.get(name, 0)) for name in TYPES if int(image_counts.get(name, 0)) > 0]
        if len(known_counts) >= 2 and max(known_counts) / max(min(known_counts), 1) > 2.0:
            warnings.append(f"Large image-count imbalance across types: {dict(image_counts)}.")

    subtype_counts = stats.get("subtype_counts", {})
    if isinstance(subtype_counts, Mapping):
        cat_present = int(subtype_counts.get("cat_present", 0))
        cat_absent = int(subtype_counts.get("cat_absent", 0))
        if cat_present and cat_absent:
            ratio = max(cat_present, cat_absent) / max(min(cat_present, cat_absent), 1)
            if ratio > 1.25:
                warnings.append(f"cat_present/cat_absent imbalance ratio is {ratio:.2f}.")
        attr_count = int(subtype_counts.get("attr_count", 0))
        attr_color = int(subtype_counts.get("attr_color", 0))
        if attr_count and attr_color:
            ratio = max(attr_count, attr_color) / max(min(attr_count, attr_color), 1)
            if ratio > 2.0:
                warnings.append(f"attr_count/attr_color imbalance ratio is {ratio:.2f}.")
        if attr_count and not attr_color:
            warnings.append("attr_color is missing while attr_count exists.")

    rel_axis = stats.get("rel_axis_counts_by_true_relation", {})
    if isinstance(rel_axis, Mapping):
        horizontal = int(rel_axis.get("horizontal", 0))
        vertical = int(rel_axis.get("vertical", 0))
        if horizontal and vertical:
            ratio = max(horizontal, vertical) / max(min(horizontal, vertical), 1)
            if ratio > 2.0:
                warnings.append(f"Relation horizontal/vertical imbalance ratio is {ratio:.2f}.")

    label_counts = stats.get("label_counts_by_type", {})
    if isinstance(label_counts, Mapping):
        for type_name, counts in label_counts.items():
            if not isinstance(counts, Mapping):
                continue
            yes = int(counts.get("yes", 0))
            no = int(counts.get("no", 0))
            if yes and no:
                ratio = max(yes, no) / max(min(yes, no), 1)
                if ratio > 1.5:
                    warnings.append(f"{type_name} yes/no imbalance ratio is {ratio:.2f}.")

    return warnings


def analyze(pair_dir: Path, activation_meta: Path | None, top_k: int) -> dict[str, Any]:
    """Analyze the pair bank and return JSON-serializable stats."""

    rows, split_counts = load_pair_rows(pair_dir)
    source_stats = read_json(pair_dir / "stats.json")
    meta_rows = read_jsonl(activation_meta) if activation_meta is not None and activation_meta.exists() else []

    type_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    image_ids_by_type: dict[str, set[str]] = defaultdict(set)
    labels_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    labels_by_subtype: dict[str, Counter[str]] = defaultdict(Counter)
    question_lengths_by_type: dict[str, list[int]] = defaultdict(list)
    objects_by_type_image: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    object_counter: Counter[str] = Counter()
    subject_counter: Counter[str] = Counter()
    object_position_counter: Counter[str] = Counter()
    pair_counter: Counter[str] = Counter()
    true_relation_counter: Counter[str] = Counter()
    queried_relation_counter: Counter[str] = Counter()
    true_axis_counter: Counter[str] = Counter()
    queried_axis_counter: Counter[str] = Counter()

    for row in rows:
        type_name = normalize_type(row)
        subtype = normalize_subtype(row)
        image_id = str(row.get("image_id") or row.get("image") or "")
        label = normalize_label(row)
        type_counts[type_name] += 1
        subtype_counts[subtype] += 1
        if image_id:
            image_ids_by_type[type_name].add(image_id)
        labels_by_type[type_name][label] += 1
        labels_by_subtype[subtype][label] += 1
        question_lengths_by_type[type_name].append(len(words(row.get("question", ""))))

        objects = row_objects(row)
        for obj in objects:
            object_counter[obj] += 1
            if image_id:
                objects_by_type_image[type_name][image_id].add(obj)
        object_a = str(row.get("object_a") or (objects[0] if objects else "")).strip()
        object_b = str(row.get("object_b") or (objects[1] if len(objects) > 1 else "")).strip()
        if object_a:
            subject_counter[object_a] += 1
        if object_b:
            object_position_counter[object_b] += 1
        if object_a and object_b:
            pair_counter[f"{object_a} -> {object_b}"] += 1

        if type_name == "rel":
            true_relation = str(row.get("true_relation") or "").strip()
            queried_relation = str(row.get("queried_relation") or "").strip()
            if true_relation:
                true_relation_counter[true_relation] += 1
                true_axis_counter[relation_axis(true_relation)] += 1
            if queried_relation:
                queried_relation_counter[queried_relation] += 1
                queried_axis_counter[relation_axis(queried_relation)] += 1

    image_sets = {type_name: set(ids) for type_name, ids in image_ids_by_type.items()}
    overlap = {
        f"{left}_{right}": len(image_sets.get(left, set()) & image_sets.get(right, set()))
        for index, left in enumerate(TYPES)
        for right in TYPES[index + 1 :]
    }
    average_objects_per_image_by_type = {
        type_name: statistics.fmean([len(objects) for objects in by_image.values()]) if by_image else 0.0
        for type_name, by_image in objects_by_type_image.items()
    }
    question_length_stats = {
        type_name: stats_for_numbers(values)
        for type_name, values in sorted(question_lengths_by_type.items())
    }
    output: dict[str, Any] = {
        "source": "after_template_disjoint_v1_analysis",
        "pair_dir": str(pair_dir),
        "activation_meta": str(activation_meta) if activation_meta else "",
        "source_stats_present": bool(source_stats),
        "split_pair_counts": split_counts,
        "total_pairs": len(rows),
        "total_images": len({str(row.get("image_id") or row.get("image") or "") for row in rows if row.get("image_id") or row.get("image")}),
        "pair_count_by_type": dict(type_counts),
        "subtype_counts": dict(subtype_counts),
        "image_count_by_type": {type_name: len(image_sets.get(type_name, set())) for type_name in sorted(image_sets)},
        "label_counts_by_type": {key: dict(value) for key, value in labels_by_type.items()},
        "label_counts_by_subtype": {key: dict(value) for key, value in labels_by_subtype.items()},
        "cat_present_absent": {
            "cat_present": int(subtype_counts.get("cat_present", 0)),
            "cat_absent": int(subtype_counts.get("cat_absent", 0)),
        },
        "attr_count_color": {
            "attr_count": int(subtype_counts.get("attr_count", 0)),
            "attr_color": int(subtype_counts.get("attr_color", 0)),
        },
        "rel_true_relation_counts": dict(true_relation_counter),
        "rel_queried_relation_counts": dict(queried_relation_counter),
        "rel_axis_counts_by_true_relation": dict(true_axis_counter),
        "rel_axis_counts_by_queried_relation": dict(queried_axis_counter),
        "top_categories": counter_to_rows(object_counter, top_k),
        "top_subject_categories": counter_to_rows(subject_counter, top_k),
        "top_object_categories": counter_to_rows(object_position_counter, top_k),
        "top_category_pairs": counter_to_rows(pair_counter, top_k),
        "average_distinct_mentioned_objects_per_image_by_type": average_objects_per_image_by_type,
        "question_length_stats_by_type": question_length_stats,
        "cross_type_image_overlap": overlap,
        "activation_meta_summary": {
            "present": bool(meta_rows),
            "num_rows": len(meta_rows),
            "type_counts": dict(Counter(normalize_type(row) for row in meta_rows)),
            "subtype_counts": dict(Counter(normalize_subtype(row) for row in meta_rows)),
        },
        "source_stats": source_stats,
    }
    output["warnings"] = build_warnings(output)
    if meta_rows and len(meta_rows) != len([row for row in rows if row.get("_split") == "train"]):
        output["warnings"].append(
            f"Activation metadata rows ({len(meta_rows)}) do not match train split pair count ({split_counts.get('train', 0)})."
        )
    return output


def render_report(stats: Mapping[str, Any]) -> str:
    """Render the markdown data report."""

    lines: list[str] = [
        "# AFTER-Template Disjoint V1 Data Report",
        "",
        "This is a diagnostic report for the image-disjoint AFTER-template pair bank.",
        "It does not train or evaluate a router.",
        "",
        "## Overview",
        "",
        f"- Pair directory: `{stats.get('pair_dir', '')}`",
        f"- Total pairs: {stats.get('total_pairs', 0)}",
        f"- Total images: {stats.get('total_images', 0)}",
        f"- Activation metadata rows: {stats.get('activation_meta_summary', {}).get('num_rows', 0)}",
        "",
        "## Split Pair Counts",
        "",
    ]
    lines.extend(table(["split", "count"], [{"split": key, "count": value} for key, value in sorted(stats.get("split_pair_counts", {}).items())]))
    lines.extend(["", "## Pair Counts By Type", ""])
    lines.extend(table(["key", "count"], counter_to_rows(Counter(stats.get("pair_count_by_type", {})))))
    lines.extend(["", "## Pair Counts By Subtype", ""])
    lines.extend(table(["key", "count"], counter_to_rows(Counter(stats.get("subtype_counts", {})))))
    lines.extend(["", "## Image Counts By Type", ""])
    lines.extend(table(["key", "count"], counter_to_rows(Counter(stats.get("image_count_by_type", {})))))
    lines.extend(["", "## Yes/No Counts By Type", ""])
    lines.extend(table(["group", "key", "count"], nested_count_rows(stats.get("label_counts_by_type", {}))))
    lines.extend(["", "## Yes/No Counts By Subtype", ""])
    lines.extend(table(["group", "key", "count"], nested_count_rows(stats.get("label_counts_by_subtype", {}))))
    lines.extend(["", "## Cat Balance", ""])
    lines.extend(table(["key", "count"], counter_to_rows(Counter(stats.get("cat_present_absent", {})))))
    lines.extend(["", "## Attribute Balance", ""])
    lines.extend(table(["key", "count"], counter_to_rows(Counter(stats.get("attr_count_color", {})))))
    lines.extend(["", "## Relation Counts", ""])
    lines.extend(["### True Relation", ""])
    lines.extend(table(["key", "count"], counter_to_rows(Counter(stats.get("rel_true_relation_counts", {})))))
    lines.extend(["", "### Queried Relation", ""])
    lines.extend(table(["key", "count"], counter_to_rows(Counter(stats.get("rel_queried_relation_counts", {})))))
    lines.extend(["", "### Horizontal/Vertical By True Relation", ""])
    lines.extend(table(["key", "count"], counter_to_rows(Counter(stats.get("rel_axis_counts_by_true_relation", {})))))
    lines.extend(["", "### Horizontal/Vertical By Queried Relation", ""])
    lines.extend(table(["key", "count"], counter_to_rows(Counter(stats.get("rel_axis_counts_by_queried_relation", {})))))
    lines.extend(["", "## Top Categories", ""])
    lines.extend(table(["key", "count"], stats.get("top_categories", [])))
    lines.extend(["", "## Top Subject Categories", ""])
    lines.extend(table(["key", "count"], stats.get("top_subject_categories", [])))
    lines.extend(["", "## Top Object Categories", ""])
    lines.extend(table(["key", "count"], stats.get("top_object_categories", [])))
    lines.extend(["", "## Top Category Pairs", ""])
    lines.extend(table(["key", "count"], stats.get("top_category_pairs", [])))
    lines.extend(["", "## Average Distinct Mentioned Objects Per Image By Type", ""])
    avg_rows = [{"key": key, "count": value} for key, value in sorted(stats.get("average_distinct_mentioned_objects_per_image_by_type", {}).items())]
    lines.extend(table(["key", "count"], avg_rows))
    lines.extend(["", "## Question Length Statistics By Type", ""])
    q_rows = []
    for type_name, values in sorted(stats.get("question_length_stats_by_type", {}).items()):
        row = {"type": type_name}
        if isinstance(values, Mapping):
            row.update(values)
        q_rows.append(row)
    lines.extend(table(["type", "count", "mean", "min", "median", "max"], q_rows))
    lines.extend(["", "## Leakage Check", ""])
    lines.extend(table(["key", "count"], counter_to_rows(Counter(stats.get("cross_type_image_overlap", {})))))
    lines.extend(["", "## Warnings", ""])
    warnings = list(stats.get("warnings", []))
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No obvious imbalance or leakage warnings were detected.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    try:
        pair_dir = resolve_project_path(args.pair_dir)
        activation_meta = resolve_project_path(args.activation_meta) if str(args.activation_meta).strip() else None
        output_dir = resolve_project_path(args.output_dir)
        stats = analyze(pair_dir, activation_meta, int(args.top_k))
        stats_path = output_dir / "disjoint_data_stats.json"
        report_path = output_dir / "DISJOINT_DATA_REPORT.md"
        write_json(stats_path, stats)
        write_text(report_path, render_report(stats))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote disjoint data stats to {stats_path}")
    print(f"Wrote disjoint data report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
