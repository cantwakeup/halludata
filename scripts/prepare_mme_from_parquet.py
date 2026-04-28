"""Prepare MME parquet yes/no hallucination subsets for local steering eval.

The Hugging Face MME parquet dump stores images inline as bytes. This script
extracts the yes/no hallucination categories we care about, writes images to
disk, and emits JSONL files compatible with ``scripts/run_steered_benchmark.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CATEGORIES = ("existence", "count", "color", "position")
EXPERT_BY_CATEGORY = {
    "existence": "cat",
    "count": "attr",
    "color": "attr",
    "position": "rel",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True, help="Directory containing MME parquet shards.")
    parser.add_argument("--out-dir", default="data/benchmarks/mme_hallucination", help="Prepared output directory.")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), help="MME categories to export.")
    parser.add_argument("--max-rows-per-category", type=int, default=0, help="Optional smoke-test cap; 0 means all rows.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting generated files.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_label(value: Any) -> str | None:
    """Normalize MME answers to lowercase yes/no labels."""

    text = str(value).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def safe_stem(value: Any, fallback: str) -> str:
    """Return a conservative filename stem."""

    raw = str(value or fallback).replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(raw).stem or fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:80] or fallback


def image_suffix(value: Any) -> str:
    """Infer a safe image suffix from the original image path."""

    suffix = Path(str(value or "")).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return suffix
    return ".jpg"


def extract_image_payload(value: Any) -> tuple[bytes, str]:
    """Extract raw image bytes and source filename from a parquet image field."""

    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = str(value.get("path") or value.get("file_name") or "")
    else:
        image_bytes = getattr(value, "bytes", None)
        image_path = str(getattr(value, "path", "") or getattr(value, "filename", "") or "")
    if image_bytes is None and isinstance(value, (bytes, bytearray, memoryview)):
        image_bytes = value
    if image_bytes is None:
        raise ValueError(f"Unsupported MME image field: {type(value)!r}")
    if isinstance(image_bytes, memoryview):
        image_bytes = image_bytes.tobytes()
    if isinstance(image_bytes, bytearray):
        image_bytes = bytes(image_bytes)
    if not isinstance(image_bytes, bytes):
        raise ValueError(f"Unsupported image bytes type: {type(image_bytes)!r}")
    return image_bytes, image_path


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-empty value from common parquet column names."""

    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


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


def ensure_output_ready(out_dir: Path, overwrite: bool) -> None:
    """Validate generated output directory before writing."""

    generated_targets = [
        out_dir / "all.jsonl",
        out_dir / "stats.json",
        *(out_dir / f"{category}.jsonl" for category in DEFAULT_CATEGORIES),
    ]
    existing = [path for path in generated_targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Prepared MME outputs already exist under {out_dir}. Pass --overwrite to replace generated files."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)


def read_parquet_rows(parquet_dir: Path, target_categories: set[str]) -> list[dict[str, Any]]:
    """Read all parquet shards into a list of dictionaries."""

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - depends on runtime deps.
        raise RuntimeError("Preparing MME parquet files requires pandas and a parquet engine such as pyarrow.") from exc

    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {parquet_dir}")
    rows: list[dict[str, Any]] = []
    for parquet_file in files:
        category_frame = pd.read_parquet(parquet_file, columns=["category"])
        present_categories = {
            str(category).strip().lower()
            for category in category_frame["category"].dropna().unique().tolist()
        }
        if not (present_categories & target_categories):
            continue
        try:
            frame = pd.read_parquet(parquet_file, columns=["image", "question", "answer", "category"])
        except Exception:
            frame = pd.read_parquet(parquet_file)
        for local_index, row in enumerate(frame.to_dict("records")):
            row["_parquet_file"] = parquet_file.name
            row["_parquet_row"] = local_index
            rows.append(row)
    return rows


def build_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build prepared benchmark rows and stats from MME parquet rows."""

    parquet_dir = resolve_project_path(args.parquet_dir)
    out_dir = resolve_project_path(args.out_dir)
    categories = {str(category).strip().lower() for category in args.categories}
    ensure_output_ready(out_dir, bool(args.overwrite))
    rows = read_parquet_rows(parquet_dir, categories)

    samples: list[dict[str, Any]] = []
    per_category_counts: Counter[str] = Counter()
    answer_counts: dict[str, Counter[str]] = defaultdict(Counter)
    skipped: Counter[str] = Counter()

    for global_index, row in enumerate(rows):
        category = str(row.get("category", "")).strip().lower()
        if category not in categories:
            continue
        if int(args.max_rows_per_category) > 0 and per_category_counts[category] >= int(args.max_rows_per_category):
            continue
        question = first_present(row, ("question", "text", "query", "prompt"))
        label = normalize_label(first_present(row, ("answer", "label", "gt_answer", "ground_truth", "target")))
        if not question:
            skipped["missing_question"] += 1
            continue
        if label is None:
            skipped["missing_yes_no_label"] += 1
            continue
        image_value = first_present(row, ("image", "img", "picture"))
        try:
            image_bytes, source_image_path = extract_image_payload(image_value)
        except Exception:
            skipped["missing_image_bytes"] += 1
            continue

        local_index = per_category_counts[category]
        source_stem = safe_stem(source_image_path, f"{global_index:06d}")
        suffix = image_suffix(source_image_path)
        image_name = f"mme_{category}_{local_index:05d}_{source_stem}{suffix}"
        relative_image_path = f"{category}/{image_name}"
        image_path = out_dir / "images" / relative_image_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if image_path.exists() and not bool(args.overwrite):
            raise FileExistsError(f"Image already exists: {image_path}. Pass --overwrite to replace it.")
        image_path.write_bytes(image_bytes)

        sample = {
            "sample_id": f"mme_{category}_{local_index:05d}",
            "image": relative_image_path,
            "image_path": relative_image_path,
            "question": str(question).strip(),
            "label": label,
            "answer": "Yes" if label == "yes" else "No",
            "category": category,
            "mme_category": category,
            "expert": EXPERT_BY_CATEGORY.get(category, ""),
            "source": "mme_parquet",
            "source_image_path": source_image_path,
            "parquet_file": row.get("_parquet_file", ""),
            "parquet_row": row.get("_parquet_row", ""),
        }
        samples.append(sample)
        per_category_counts[category] += 1
        answer_counts[category][label] += 1

    samples.sort(key=lambda item: (str(item["category"]), str(item["sample_id"])))
    stats = {
        "source": "mme_parquet",
        "parquet_dir": str(parquet_dir),
        "out_dir": str(out_dir),
        "categories": sorted(categories),
        "total_samples": len(samples),
        "category_counts": dict(sorted(per_category_counts.items())),
        "answer_counts": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(answer_counts.items())
        },
        "expert_counts": dict(sorted(Counter(sample["expert"] for sample in samples).items())),
        "skipped": dict(sorted(skipped.items())),
        "image_root": str(out_dir / "images"),
        "jsonl_files": {
            "all": str(out_dir / "all.jsonl"),
            **{category: str(out_dir / f"{category}.jsonl") for category in sorted(categories)},
        },
    }
    return samples, stats


def main() -> int:
    """Prepare MME JSONL and image files."""

    args = parse_args()
    try:
        out_dir = resolve_project_path(args.out_dir)
        samples, stats = build_samples(args)
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            by_category[str(sample["category"])].append(sample)
        write_jsonl(out_dir / "all.jsonl", samples)
        for category, rows in sorted(by_category.items()):
            write_jsonl(out_dir / f"{category}.jsonl", rows)
        write_json(out_dir / "stats.json", stats)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote prepared MME benchmark to {resolve_project_path(args.out_dir)}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
