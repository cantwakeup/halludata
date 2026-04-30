"""Prepare AMBER yes/no hallucination subsets for steering eval.

The public AMBER dumps are not always laid out identically, so this script is
intentionally permissive: it scans JSON/JSONL/CSV/TSV files under an AMBER root
and keeps rows that contain a yes/no label, a question/query, and an image path.
The output JSONL files are compatible with ``scripts/run_steered_benchmark.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CATEGORIES = ("existence", "attribute", "relation")
EXPERT_BY_CATEGORY = {
    "existence": "cat",
    "attribute": "attr",
    "relation": "rel",
}
QUESTION_KEYS = ("question", "query", "prompt", "text", "instruction")
LABEL_KEYS = ("answer", "label", "gt_answer", "ground_truth", "target", "gt")
IMAGE_KEYS = ("image", "image_path", "img", "file_name", "filename", "path")
CATEGORY_KEYS = (
    "category",
    "type",
    "task",
    "hallucination_type",
    "subtype",
    "question_type",
    "eval_type",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amber-root", required=True, help="Root directory of the AMBER dataset.")
    parser.add_argument(
        "--image-root",
        default="",
        help="Image root used by the benchmark runner. Defaults to --amber-root.",
    )
    parser.add_argument("--out-dir", default="data/benchmarks/amber_hallucination")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="Canonical categories to export: existence attribute relation other.",
    )
    parser.add_argument("--input-files", nargs="*", default=[], help="Optional explicit annotation files.")
    parser.add_argument("--max-rows-per-category", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def first_present(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """Return the first non-empty value under candidate keys."""

    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def normalize_label(value: Any) -> str | None:
    """Normalize yes/no labels."""

    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"} or text.startswith("yes"):
        return "yes"
    if text in {"no", "n", "false", "0"} or text.startswith("no"):
        return "no"
    return None


def normalize_category(row: Mapping[str, Any]) -> str:
    """Map AMBER type/category names to existence/attribute/relation/other."""

    raw_values = [str(first_present(row, CATEGORY_KEYS))]
    for key in CATEGORY_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            raw_values.append(str(value))
    blob = " ".join(raw_values).lower()
    if any(token in blob for token in ("exist", "object", "chair")):
        return "existence"
    if any(token in blob for token in ("attr", "attribute", "color", "colour", "count", "number")):
        return "attribute"
    if any(token in blob for token in ("relation", "spatial", "position", "left", "right", "above", "below")):
        return "relation"
    return "other"


def flatten_records(payload: Any) -> list[dict[str, Any]]:
    """Flatten common JSON containers into a list of dictionaries."""

    if isinstance(payload, dict):
        for key in ("data", "samples", "questions", "annotations", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return flatten_records(value)
        return [payload]
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                rows.extend(flatten_records(item))
        return rows
    return []


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON or line-delimited JSON records."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return flatten_records(payload)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    payload["_source_line"] = line_number
                    rows.extend(flatten_records(payload))
        return rows


def read_table(path: Path) -> list[dict[str, Any]]:
    """Read CSV/TSV rows."""

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def candidate_annotation_files(amber_root: Path, explicit_files: list[str]) -> list[Path]:
    """Return annotation files to inspect."""

    if explicit_files:
        return [resolve_project_path(path) for path in explicit_files]
    suffixes = {".json", ".jsonl", ".csv", ".tsv"}
    files = [
        path for path in amber_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    return sorted(files, key=lambda path: (len(path.parts), str(path)))


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read one candidate annotation file."""

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return read_json_or_jsonl(path)
    if suffix in {".csv", ".tsv"}:
        return read_table(path)
    return []


def safe_sample_id(category: str, index: int, source_stem: str) -> str:
    """Build a stable sample id."""

    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_stem)[:60]
    return f"amber_{category}_{index:05d}_{safe_stem or 'sample'}"


def build_image_index(amber_root: Path) -> dict[str, Path]:
    """Index image files by basename for datasets that store only filenames."""

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    index: dict[str, Path] = {}
    for path in amber_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            index.setdefault(path.name, path)
    return index


def normalize_image_path(raw_value: Any, amber_root: Path, image_root: Path, image_index: Mapping[str, Path]) -> str:
    """Normalize an AMBER image path for the benchmark runner."""

    text = str(raw_value).strip()
    if not text:
        return ""
    path = Path(text)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([image_root / path, amber_root / path])
    for candidate in candidates:
        if candidate.exists():
            try:
                return str(candidate.relative_to(image_root))
            except ValueError:
                return str(candidate)
    indexed = image_index.get(path.name)
    if indexed is not None and indexed.exists():
        try:
            return str(indexed.relative_to(image_root))
        except ValueError:
            return str(indexed)
    return text


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_output_ready(out_dir: Path, categories: set[str], overwrite: bool) -> None:
    """Validate output paths before writing."""

    targets = [out_dir / "all.jsonl", out_dir / "stats.json", *(out_dir / f"{category}.jsonl" for category in categories)]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Prepared AMBER outputs already exist: {existing[0]}. Pass --overwrite to replace.")
    out_dir.mkdir(parents=True, exist_ok=True)


def load_official_amber_annotations(amber_root: Path) -> dict[int, dict[str, Any]]:
    """Load official AMBER annotations keyed by question id if present."""

    annotation_path = amber_root / "data" / "annotations.json"
    if not annotation_path.exists():
        return {}
    rows = read_json_or_jsonl(annotation_path)
    annotations: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            question_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        annotations[question_id] = row
    return annotations


def official_query_file(amber_root: Path, category: str) -> Path:
    """Return the official AMBER discriminative query file for a category."""

    return amber_root / "data" / "query" / f"query_discriminative-{category}.json"


def build_official_amber_samples(
    *,
    amber_root: Path,
    image_root: Path,
    categories: set[str],
    image_index: Mapping[str, Path],
    max_rows_per_category: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Build samples from the official AMBER query/annotation split schema."""

    annotations_by_id = load_official_amber_annotations(amber_root)
    if not annotations_by_id:
        return None
    available_files = {
        category: official_query_file(amber_root, category)
        for category in DEFAULT_CATEGORIES
        if official_query_file(amber_root, category).exists()
    }
    if not available_files:
        return None

    samples: list[dict[str, Any]] = []
    per_category_counts: Counter[str] = Counter()
    answer_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_file_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for category in sorted(categories):
        query_path = available_files.get(category)
        if query_path is None:
            skipped[f"missing_query_file_{category}"] += 1
            continue
        queries = read_json_or_jsonl(query_path)
        for local_index, query_row in enumerate(queries):
            if int(max_rows_per_category) > 0 and per_category_counts[category] >= int(max_rows_per_category):
                continue
            try:
                question_id = int(query_row.get("id"))
            except (TypeError, ValueError):
                skipped["missing_id"] += 1
                continue
            annotation = annotations_by_id.get(question_id)
            if annotation is None:
                skipped["missing_annotation"] += 1
                continue
            label = normalize_label(annotation.get("truth"))
            if label is None:
                skipped["missing_yes_no_label"] += 1
                continue
            question = str(query_row.get("query", "")).strip()
            if not question:
                skipped["missing_question"] += 1
                continue
            image_path = normalize_image_path(query_row.get("image", ""), amber_root, image_root, image_index)
            if not image_path:
                skipped["missing_image"] += 1
                continue
            category_index = per_category_counts[category]
            sample = {
                "sample_id": safe_sample_id(category, category_index, f"{query_path.stem}_{question_id}"),
                "image": image_path,
                "image_path": image_path,
                "question": question,
                "label": label,
                "answer": "Yes" if label == "yes" else "No",
                "category": category,
                "amber_category": category,
                "raw_category": str(annotation.get("type", "")),
                "expert": EXPERT_BY_CATEGORY.get(category, ""),
                "source": "amber_official_discriminative",
                "source_file": str(query_path),
                "source_row": local_index,
                "annotation_id": question_id,
                "annotation_type": str(annotation.get("type", "")),
            }
            samples.append(sample)
            per_category_counts[category] += 1
            answer_counts[category][label] += 1
            source_file_counts[str(query_path)] += 1

    stats = {
        "source": "amber_official_discriminative",
        "amber_root": str(amber_root),
        "image_root": str(image_root),
        "categories": sorted(categories),
        "total_samples": len(samples),
        "category_counts": dict(sorted(per_category_counts.items())),
        "answer_counts": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(answer_counts.items())
        },
        "expert_counts": dict(sorted(Counter(sample["expert"] for sample in samples).items())),
        "source_file_counts": dict(source_file_counts.most_common()),
        "skipped": dict(sorted(skipped.items())),
        "schema": "official_query_join_annotations_by_id",
    }
    return sorted(samples, key=lambda item: (str(item["category"]), str(item["sample_id"]))), stats


def build_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build prepared AMBER yes/no samples and stats."""

    amber_root = resolve_project_path(args.amber_root)
    image_root = resolve_project_path(args.image_root) if str(args.image_root).strip() else amber_root
    out_dir = resolve_project_path(args.out_dir)
    categories = {str(category).strip().lower() for category in args.categories}
    ensure_output_ready(out_dir, categories, bool(args.overwrite))

    image_index = build_image_index(amber_root)
    if not list(args.input_files):
        official = build_official_amber_samples(
            amber_root=amber_root,
            image_root=image_root,
            categories=categories,
            image_index=image_index,
            max_rows_per_category=int(args.max_rows_per_category),
        )
        if official is not None and official[1].get("total_samples", 0) > 0:
            samples, stats = official
            stats["out_dir"] = str(out_dir)
            stats["jsonl_files"] = {
                "all": str(out_dir / "all.jsonl"),
                **{category: str(out_dir / f"{category}.jsonl") for category in sorted(categories)},
            }
            return samples, stats

    files = candidate_annotation_files(amber_root, list(args.input_files))
    samples: list[dict[str, Any]] = []
    per_category_counts: Counter[str] = Counter()
    answer_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_file_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for file_path in files:
        try:
            records = read_records(file_path)
        except Exception:
            skipped["unreadable_file"] += 1
            continue
        for local_index, row in enumerate(records):
            question = first_present(row, QUESTION_KEYS)
            label = normalize_label(first_present(row, LABEL_KEYS))
            image_value = first_present(row, IMAGE_KEYS)
            category = normalize_category(row)
            if category not in categories:
                skipped[f"category_{category}"] += 1
                continue
            if int(args.max_rows_per_category) > 0 and per_category_counts[category] >= int(args.max_rows_per_category):
                continue
            if not question:
                skipped["missing_question"] += 1
                continue
            if label is None:
                skipped["missing_yes_no_label"] += 1
                continue
            image_path = normalize_image_path(image_value, amber_root, image_root, image_index)
            if not image_path:
                skipped["missing_image"] += 1
                continue
            category_index = per_category_counts[category]
            sample_id = safe_sample_id(category, category_index, f"{file_path.stem}_{local_index}")
            sample = {
                "sample_id": sample_id,
                "image": image_path,
                "image_path": image_path,
                "question": str(question).strip(),
                "label": label,
                "answer": "Yes" if label == "yes" else "No",
                "category": category,
                "amber_category": category,
                "raw_category": str(first_present(row, CATEGORY_KEYS)),
                "expert": EXPERT_BY_CATEGORY.get(category, ""),
                "source": "amber",
                "source_file": str(file_path),
                "source_row": local_index,
            }
            samples.append(sample)
            per_category_counts[category] += 1
            answer_counts[category][label] += 1
            source_file_counts[str(file_path)] += 1

    samples.sort(key=lambda item: (str(item["category"]), str(item["sample_id"])))
    stats = {
        "source": "amber",
        "amber_root": str(amber_root),
        "image_root": str(image_root),
        "out_dir": str(out_dir),
        "categories": sorted(categories),
        "total_samples": len(samples),
        "category_counts": dict(sorted(per_category_counts.items())),
        "answer_counts": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(answer_counts.items())
        },
        "expert_counts": dict(sorted(Counter(sample["expert"] for sample in samples).items())),
        "source_file_counts": dict(source_file_counts.most_common()),
        "skipped": dict(sorted(skipped.items())),
        "jsonl_files": {
            "all": str(out_dir / "all.jsonl"),
            **{category: str(out_dir / f"{category}.jsonl") for category in sorted(categories)},
        },
    }
    return samples, stats


def main() -> int:
    """Prepare AMBER yes/no subsets."""

    args = parse_args()
    try:
        samples, stats = build_samples(args)
        out_dir = resolve_project_path(args.out_dir)
        write_jsonl(out_dir / "all.jsonl", samples)
        for category in stats["categories"]:
            write_jsonl(out_dir / f"{category}.jsonl", [sample for sample in samples if sample["category"] == category])
        write_json(out_dir / "stats.json", stats)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote prepared AMBER samples to {resolve_project_path(args.out_dir)}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
