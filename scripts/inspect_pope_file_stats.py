"""Inspect POPE annotation files and optional Regular prediction outputs.

This is read-only diagnostics. It does not run model inference. It reports
file hashes, label/object/image distributions, first examples, and, when raw
Regular predictions are available, FP/TP/TN/FN counts for the matching
dataset/setting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import read_jsonl, write_json


DEFAULT_POPE_FILES = [
    "/home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json",
    "/home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_popular.json",
    "/home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_adversarial.json",
    "/home/huiwei/sy/benchmarks/POPE/output/seem/gqa/gqa_pope_seem_random.json",
    "/home/huiwei/sy/benchmarks/POPE/output/seem/gqa/gqa_pope_seem_popular.json",
    "/home/huiwei/sy/benchmarks/POPE/output/seem/gqa/gqa_pope_seem_adversarial.json",
]

DEFAULT_PREDICTION_ROOTS = [
    "data/pope_cat_expert_eval/official_llava_regular_full",
    "data/pope_cat_expert_eval/full_alpha_sweep",
]

SETTINGS = ("random", "popular", "adversarial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pope-files", nargs="*", default=DEFAULT_POPE_FILES)
    parser.add_argument("--extra-globs", nargs="*", default=[])
    parser.add_argument("--prediction-roots", nargs="*", default=DEFAULT_PREDICTION_ROOTS)
    parser.add_argument("--octopus-root", default="/home/huiwei/sy/Octopus-master")
    parser.add_argument("--output-dir", default="data/pope_cat_expert_eval/alignment_debug/pope_file_stats")
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "samples", "questions", "annotations"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def first_present(row: Mapping[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"} or text.startswith("yes"):
        return "yes"
    if text in {"no", "n", "false", "0"} or text.startswith("no"):
        return "no"
    return "unknown"


def infer_dataset_setting(path: Path) -> tuple[str, str]:
    text = str(path).lower()
    dataset = "GQA" if "gqa" in text else "MSCOCO" if "coco" in text or "mscoco" in text else "unknown"
    setting = next((item for item in SETTINGS if item in text), "unknown")
    return dataset, setting


def extract_object(question: Any) -> str:
    text = str(question).strip()
    patterns = [
        r"\bIs there (?:a|an|the|any)?\s*(.+?)\s+in the image\??",
        r"\bAre there (?:a|an|the|any)?\s*(.+?)\s+in the image\??",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"[?.!]+$", "", match.group(1).strip().lower())
    return ""


def image_key(row: Mapping[str, Any]) -> str:
    value = first_present(row, ("image_id", "image", "image_path", "filename", "file_name", "img", "coco_id"), "")
    return str(value)


def sample_signature(row: Mapping[str, Any]) -> str:
    question = str(first_present(row, ("question", "text", "query", "prompt"), ""))
    label = normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target"), ""))
    obj = first_present(row, ("object", "obj", "category", "class"), "") or extract_object(question)
    return json.dumps({"image": image_key(row), "question": question, "label": label, "object": obj}, sort_keys=True, ensure_ascii=False)


def signature_hash(rows: list[Mapping[str, Any]], limit: int | None = None) -> str:
    selected = rows if limit is None else rows[:limit]
    joined = "\n".join(sample_signature(row) for row in selected)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def metrics_from_predictions(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = invalid = pred_yes = 0
    for row in rows:
        label = normalize_label(row.get("label", ""))
        pred = str(row.get("pred", "invalid")).strip().lower()
        if pred == "yes":
            pred_yes += 1
        if pred not in {"yes", "no"}:
            invalid += 1
        if label == "yes" and pred == "yes":
            tp += 1
        elif label == "no" and pred == "no":
            tn += 1
        elif label == "no" and pred == "yes":
            fp += 1
        elif label == "yes" and pred == "no":
            fn += 1
        elif label == "yes":
            fn += 1
        elif label == "no":
            fp += 1
    total = len(rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": total,
        "accuracy": ((tp + tn) / total * 100.0) if total else 0.0,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "yes_rate": (pred_yes / total * 100.0) if total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid": invalid,
    }


def find_prediction_files(prediction_roots: list[str]) -> dict[tuple[str, str], list[Path]]:
    found: dict[tuple[str, str], list[Path]] = {}
    for root_text in prediction_roots:
        root = resolve(root_text)
        if not root.exists():
            continue
        for path in sorted(root.glob("**/*regular*.jsonl")):
            if not path.is_file():
                continue
            dataset, setting = infer_dataset_setting(path)
            if dataset == "unknown" or setting == "unknown":
                continue
            found.setdefault((dataset, setting), []).append(path)
    return found


def collect_pope_paths(args: argparse.Namespace) -> list[Path]:
    paths = [resolve(path) for path in args.pope_files]
    for pattern in args.extra_globs:
        paths.extend(sorted(resolve(".").glob(pattern)))
    octopus_root = resolve(args.octopus_root)
    if octopus_root.exists():
        for path in octopus_root.glob("**/*pope*.json*"):
            lowered = str(path).lower()
            if ("coco" in lowered or "gqa" in lowered) and any(setting in lowered for setting in SETTINGS):
                paths.append(path)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def summarize_pope_file(path: Path, pred_files: dict[tuple[str, str], list[Path]], top_k: int) -> dict[str, Any]:
    dataset, setting = infer_dataset_setting(path)
    if not path.exists():
        return {"path": str(path), "exists": False, "dataset": dataset, "setting": setting}
    rows = read_json_or_jsonl(path)
    labels = Counter()
    images = set()
    objects = Counter()
    positive_objects = Counter()
    negative_objects = Counter()
    examples = []
    for row in rows:
        label = normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target"), ""))
        question = first_present(row, ("question", "text", "query", "prompt"), "")
        obj = str(first_present(row, ("object", "obj", "category", "class"), "") or extract_object(question))
        labels[label] += 1
        if image_key(row):
            images.add(image_key(row))
        if obj:
            objects[obj] += 1
            if label == "yes":
                positive_objects[obj] += 1
            elif label == "no":
                negative_objects[obj] += 1
        if len(examples) < 5:
            examples.append(
                {
                    "image_id": image_key(row),
                    "question": question,
                    "label": label,
                    "object": obj,
                    "raw": dict(row),
                }
            )

    prediction_summaries = []
    for pred_path in pred_files.get((dataset, setting), []):
        pred_rows = list(read_jsonl(pred_path))
        prediction_summaries.append({"path": str(pred_path), **metrics_from_predictions(pred_rows)})

    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "dataset": dataset,
        "setting": setting,
        "num_samples": len(rows),
        "label_counts": dict(labels),
        "num_unique_images": len(images),
        "num_unique_objects": len(objects),
        "top_objects": objects.most_common(top_k),
        "top_positive_objects": positive_objects.most_common(top_k),
        "top_negative_objects": negative_objects.most_common(top_k),
        "all_sample_signature_sha256": signature_hash(rows),
        "first20_signature_sha256": signature_hash(rows, 20),
        "first5_examples": examples,
        "prediction_summaries": prediction_summaries,
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def write_csv(path: Path, summaries: list[Mapping[str, Any]]) -> None:
    fields = [
        "dataset",
        "setting",
        "exists",
        "path",
        "sha256",
        "num_samples",
        "yes",
        "no",
        "unknown",
        "num_unique_images",
        "num_unique_objects",
        "first20_signature_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            labels = item.get("label_counts", {}) if isinstance(item.get("label_counts"), dict) else {}
            writer.writerow(
                {
                    "dataset": item.get("dataset", ""),
                    "setting": item.get("setting", ""),
                    "exists": item.get("exists", False),
                    "path": item.get("path", ""),
                    "sha256": item.get("sha256", ""),
                    "num_samples": item.get("num_samples", ""),
                    "yes": labels.get("yes", ""),
                    "no": labels.get("no", ""),
                    "unknown": labels.get("unknown", ""),
                    "num_unique_images": item.get("num_unique_images", ""),
                    "num_unique_objects": item.get("num_unique_objects", ""),
                    "first20_signature_sha256": item.get("first20_signature_sha256", ""),
                }
            )


def write_report(path: Path, summaries: list[Mapping[str, Any]]) -> None:
    overview_rows = []
    pred_rows = []
    for item in summaries:
        labels = item.get("label_counts", {}) if isinstance(item.get("label_counts"), dict) else {}
        overview_rows.append(
            {
                "dataset": item.get("dataset", ""),
                "setting": item.get("setting", ""),
                "exists": item.get("exists", False),
                "n": item.get("num_samples", ""),
                "yes": labels.get("yes", ""),
                "no": labels.get("no", ""),
                "images": item.get("num_unique_images", ""),
                "objects": item.get("num_unique_objects", ""),
                "sha256": item.get("sha256", ""),
                "path": item.get("path", ""),
            }
        )
        for pred in item.get("prediction_summaries", []) or []:
            row = {
                "dataset": item.get("dataset", ""),
                "setting": item.get("setting", ""),
                "pred_n": pred.get("n", ""),
                "acc": pred.get("accuracy", ""),
                "f1": pred.get("f1", ""),
                "yes_rate": pred.get("yes_rate", ""),
                "tp": pred.get("tp", ""),
                "tn": pred.get("tn", ""),
                "fp": pred.get("fp", ""),
                "fn": pred.get("fn", ""),
                "invalid": pred.get("invalid", ""),
                "pred_path": pred.get("path", ""),
            }
            pred_rows.append(row)

    lines = [
        "# POPE File Stats",
        "",
        "## Overview",
        "",
        markdown_table(["dataset", "setting", "exists", "n", "yes", "no", "images", "objects", "sha256", "path"], overview_rows),
        "",
    ]
    if pred_rows:
        lines.extend(
            [
                "## Existing Regular Prediction Metrics",
                "",
                markdown_table(["dataset", "setting", "pred_n", "acc", "f1", "yes_rate", "tp", "tn", "fp", "fn", "invalid", "pred_path"], pred_rows),
                "",
            ]
        )
    for item in summaries:
        lines.extend(
            [
                f"## {item.get('dataset')} {item.get('setting')}",
                "",
                f"- Path: `{item.get('path')}`",
                f"- Exists: `{item.get('exists')}`",
                f"- SHA256: `{item.get('sha256', '')}`",
                f"- First20 signature SHA256: `{item.get('first20_signature_sha256', '')}`",
                "",
                "Top queried objects:",
                "",
                markdown_table(["object", "count"], [{"object": obj, "count": count} for obj, count in (item.get("top_objects") or [])[:10]]),
                "",
                "Top negative objects:",
                "",
                markdown_table(["object", "count"], [{"object": obj, "count": count} for obj, count in (item.get("top_negative_objects") or [])[:10]]),
                "",
                "First 5 examples:",
                "",
                "```json",
                json.dumps(item.get("first5_examples", []), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_files = find_prediction_files(list(args.prediction_roots))
    paths = collect_pope_paths(args)
    summaries = [summarize_pope_file(path, pred_files, int(args.top_k)) for path in paths]
    write_json(output_dir / "pope_file_stats.json", {"summaries": summaries})
    write_csv(output_dir / "pope_file_stats.csv", summaries)
    write_report(output_dir / "REPORT.md", summaries)
    print(f"Wrote POPE file stats to {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
