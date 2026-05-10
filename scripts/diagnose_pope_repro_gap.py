"""Read-only diagnostics for POPE reproduction gaps.

This script does not run model inference and does not modify existing
experiment outputs. It inspects the POPE files, raw Regular predictions,
prompt/decode configuration, parser behavior, and local model metadata used by
``run_pope_cat_expert_eval.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from expert_data.activation_cache import read_jsonl
except Exception:  # pragma: no cover - fallback for standalone report generation
    read_jsonl = None  # type: ignore[assignment]

try:
    from expert_data.steering import build_llava_prefix_prompt
except Exception:  # pragma: no cover

    def build_llava_prefix_prompt(question: str) -> str:
        return f"USER: <image>\n{question}\nASSISTANT:"

try:
    from run_pope_cat_expert_eval import PROMPT_SUFFIX
except Exception:  # pragma: no cover
    PROMPT_SUFFIX = "Please answer this question in one word."


DMAS_BASELINES: dict[tuple[str, str], dict[str, float]] = {
    ("MSCOCO", "random"): {"Accuracy": 83.29, "Precision": 92.13, "Recall": 72.80, "F1 Score": 81.33},
    ("MSCOCO", "popular"): {"Accuracy": 81.88, "Precision": 88.93, "Recall": 72.80, "F1 Score": 80.06},
    ("MSCOCO", "adversarial"): {"Accuracy": 78.96, "Precision": 83.06, "Recall": 72.75, "F1 Score": 77.57},
    ("GQA", "random"): {"Accuracy": 83.73, "Precision": 87.16, "Recall": 79.12, "F1 Score": 82.95},
    ("GQA", "popular"): {"Accuracy": 78.17, "Precision": 77.64, "Recall": 79.12, "F1 Score": 78.37},
    ("GQA", "adversarial"): {"Accuracy": 75.08, "Precision": 73.19, "Recall": 79.16, "F1 Score": 76.06},
}

DATASETS = ("MSCOCO", "GQA")
SETTINGS = ("random", "popular", "adversarial")
OBJECT_STOPWORDS = {"a", "an", "the", "any"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="data/pope_cat_expert_eval/full_alpha_sweep")
    parser.add_argument("--output", default="data/pope_cat_expert_eval/diagnostics/REPRO_GAP_REPORT.md")
    parser.add_argument("--pope-root", default="", help="Optional POPE root override; otherwise read from config.json.")
    parser.add_argument("--model-path", default="", help="Optional model path override; otherwise read from config.json.")
    parser.add_argument("--coco-instances", default="", help="Optional COCO instances annotation for co-occurrence proxy.")
    parser.add_argument("--gqa-scene-graphs", default="", help="Optional GQA scene graph JSON for co-occurrence proxy.")
    parser.add_argument("--max-inventory-files", type=int, default=200)
    parser.add_argument("--max-status-lines", type=int, default=120)
    parser.add_argument("--raw-sample-size", type=int, default=100)
    return parser.parse_args()


def run_git_status(max_lines: int) -> tuple[list[str], bool]:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        lines = proc.stdout.splitlines()
        truncated = len(lines) > max_lines
        return lines[:max_lines], truncated
    except Exception as exc:
        return [f"git status failed: {exc}"], False


def safe_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = safe_load_json(path)
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
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    if read_jsonl is not None:
        return list(read_jsonl(path))
    return read_json_or_jsonl(path)


def file_hashes(path: Path) -> dict[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


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


def metric_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(rows: Iterable[Mapping[str, Any]], parser: str = "stored") -> dict[str, Any]:
    tp = tn = fp = fn = invalid = pred_yes = 0
    rows = list(rows)
    for row in rows:
        label = normalize_label(row.get("label", ""))
        if parser == "stored":
            pred = str(row.get("pred", "invalid")).strip().lower()
        elif parser == "first_token":
            pred = parse_first_token(row.get("raw_output", ""))
        else:
            pred = parse_contains_earliest(row.get("raw_output", ""))
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
    precision = metric_div(tp, tp + fp)
    recall = metric_div(tp, tp + fn)
    f1 = metric_div(2 * precision * recall, precision + recall)
    accuracy = metric_div(tp + tn, len(rows))
    yes_rate = metric_div(pred_yes, len(rows))
    return {
        "N": len(rows),
        "Accuracy": accuracy * 100.0,
        "Precision": precision * 100.0,
        "Recall": recall * 100.0,
        "F1 Score": f1 * 100.0,
        "Yes Rate": yes_rate * 100.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Invalid": invalid,
    }


def parse_contains_earliest(text: Any) -> str:
    matches = [(match.start(), match.group(1)) for match in re.finditer(r"\b(yes|no)\b", str(text).lower())]
    if not matches:
        return "invalid"
    return sorted(matches)[0][1]


def parse_first_token(text: Any) -> str:
    stripped = str(text).strip().lower()
    if not stripped:
        return "invalid"
    match = re.match(r"^[\s\"'`*\(\[]*([a-z]+)", stripped)
    if not match:
        return "invalid"
    token = match.group(1)
    if token == "yes":
        return "yes"
    if token == "no":
        return "no"
    return "invalid"


def ensure_prompt(question: str) -> str:
    question = str(question).strip()
    if question.lower().endswith(PROMPT_SUFFIX.lower()):
        return question
    return f"{question} {PROMPT_SUFFIX}"


def extract_object(row: Mapping[str, Any]) -> str:
    for key in ("object", "obj", "category", "entity", "target_object"):
        value = row.get(key)
        if value not in (None, ""):
            return normalize_object_name(str(value))
    question = str(first_present(row, ("question", "text", "query", "prompt"), ""))
    patterns = [
        r"\b(?:is|are)\s+there\s+(?:a|an|the|any)?\s*(.+?)\s+in\s+(?:the\s+)?image\b",
        r"\b(?:is|are)\s+there\s+(?:a|an|the|any)?\s*(.+?)\?",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return normalize_object_name(match.group(1))
    return ""


def normalize_object_name(text: str) -> str:
    text = re.sub(r"[?.!,;:]+$", "", str(text).strip().lower())
    text = re.sub(r"\s+", " ", text)
    parts = text.split()
    while parts and parts[0] in OBJECT_STOPWORDS:
        parts = parts[1:]
    return " ".join(parts)


def extract_image_id(row: Mapping[str, Any]) -> str:
    image_id = first_present(row, ("image_id", "id", "coco_id", "img_id"), "")
    if image_id not in (None, ""):
        return str(image_id)
    image = str(first_present(row, ("image", "image_path", "filename", "file_name"), ""))
    stem = Path(image).stem
    match = re.search(r"(\d+)$", stem)
    return match.group(1) if match else stem


def summarize_annotation_file(path: Path) -> dict[str, Any]:
    rows = read_json_or_jsonl(path)
    labels = Counter(normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target"), "")) for row in rows)
    first_rows = []
    for row in rows[:5]:
        first_rows.append(
            {
                "image_id": extract_image_id(row),
                "image": first_present(row, ("image", "image_path", "filename", "file_name"), ""),
                "question": first_present(row, ("question", "text", "query", "prompt"), ""),
                "label": normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target"), "")),
                "object": extract_object(row),
            }
        )
    hashes = file_hashes(path)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "md5": hashes["md5"],
        "sha256": hashes["sha256"],
        "N": len(rows),
        "yes": labels.get("yes", 0),
        "no": labels.get("no", 0),
        "unknown": labels.get("unknown", 0),
        "first_rows": first_rows,
    }


def collect_annotation_paths(config: Mapping[str, Any], pope_root_override: str) -> dict[tuple[str, str], Path]:
    groups = dict(config.get("pope_manifest", {}).get("groups", {}))
    paths: dict[tuple[str, str], Path] = {}
    for dataset in DATASETS:
        for setting in SETTINGS:
            key = f"{dataset}_{setting}"
            info = groups.get(key, {})
            source_file = info.get("source_file")
            if source_file:
                paths[(dataset, setting)] = Path(source_file)
    if len(paths) == 6:
        return paths

    root_text = pope_root_override or str(config.get("pope_root", ""))
    pope_root = Path(root_text)
    aliases = {"MSCOCO": ("coco", "mscoco"), "GQA": ("gqa",)}
    for dataset in DATASETS:
        for setting in SETTINGS:
            if (dataset, setting) in paths:
                continue
            candidates = []
            if pope_root.exists():
                for path in pope_root.rglob("*"):
                    if path.suffix.lower() not in {".json", ".jsonl"}:
                        continue
                    full = str(path).lower()
                    if setting in full and "pope" in full and any(alias in full for alias in aliases[dataset]):
                        candidates.append(path)
            if candidates:
                paths[(dataset, setting)] = sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]
    return paths


def load_regular_predictions(runs_root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    raw_dir = runs_root / "raw"
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not raw_dir.exists():
        return groups
    for path in sorted(raw_dir.glob("*_regular.jsonl")):
        rows = read_raw_jsonl(path)
        for row in rows:
            if str(row.get("method", "")) != "Regular":
                continue
            key = (str(row.get("dataset", "")), str(row.get("setting", "")))
            groups.setdefault(key, []).append(row)
    return groups


def top_object_table(rows: list[Mapping[str, Any]], label: str, n: int = 30) -> list[dict[str, Any]]:
    counts = Counter()
    for row in rows:
        row_label = normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target"), ""))
        if row_label != label:
            continue
        obj = extract_object(row)
        if obj:
            counts[obj] += 1
    return [{"object": key, "count": value} for key, value in counts.most_common(n)]


def label_object_sets(rows: list[Mapping[str, Any]], label: str) -> Counter[str]:
    counts = Counter()
    for row in rows:
        row_label = normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target"), ""))
        if row_label == label:
            obj = extract_object(row)
            if obj:
                counts[obj] += 1
    return counts


def jaccard_top(counter_a: Counter[str], counter_b: Counter[str], k: int = 30) -> float:
    a = {item for item, _ in counter_a.most_common(k)}
    b = {item for item, _ in counter_b.most_common(k)}
    return metric_div(len(a & b), len(a | b))


def discover_pope_inventory(
    *,
    annotation_paths: Mapping[tuple[str, str], Path],
    pope_root: str,
    max_files: int,
) -> list[dict[str, Any]]:
    roots: list[Path] = []
    for raw_root in (pope_root, "data", "datasets", "reference"):
        if raw_root:
            root = Path(raw_root)
            if not root.is_absolute():
                root = PROJECT_ROOT / root
            if root.exists() and root not in roots:
                roots.append(root)
    for path in annotation_paths.values():
        for parent in path.parents[:3]:
            if parent.exists() and parent not in roots:
                roots.append(parent)

    candidates: list[Path] = []
    for root in roots:
        try:
            for path in root.rglob("*"):
                if len(candidates) >= max_files:
                    break
                if path.suffix.lower() not in {".json", ".jsonl"}:
                    continue
                full = str(path).lower()
                name = path.name.lower()
                if "pope" in full or "coco_pope" in name or "gqa_pope" in name:
                    candidates.append(path)
        except OSError:
            continue
    unique = sorted({path.resolve(): path for path in candidates}.values(), key=lambda p: str(p))[:max_files]
    used_by_sha = {}
    for key, path in annotation_paths.items():
        if path.exists():
            used_by_sha[file_hashes(path)["sha256"]] = f"{key[0]} {key[1]}"

    inventory: list[dict[str, Any]] = []
    for path in unique:
        try:
            rows = read_json_or_jsonl(path)
            hashes = file_hashes(path)
            first_ids = [extract_image_id(row) for row in rows[:10]]
            first_questions = [str(first_present(row, ("question", "text", "query", "prompt"), "")) for row in rows[:3]]
            labels = Counter(normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target"), "")) for row in rows)
            neg_top = label_object_sets(rows, "no").most_common(10)
            inventory.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "N": len(rows),
                    "yes": labels.get("yes", 0),
                    "no": labels.get("no", 0),
                    "sha256": hashes["sha256"],
                    "used_as": used_by_sha.get(hashes["sha256"], ""),
                    "first10_image_ids": first_ids,
                    "first_questions": first_questions,
                    "negative_top10": neg_top,
                }
            )
        except Exception as exc:
            inventory.append({"path": str(path), "error": str(exc)})
    return inventory


def load_coco_image_objects(path: Path) -> dict[str, set[str]]:
    data = safe_load_json(path)
    categories = {int(cat["id"]): normalize_object_name(str(cat["name"])) for cat in data.get("categories", [])}
    image_objects: dict[str, set[str]] = defaultdict(set)
    for ann in data.get("annotations", []):
        image_id = str(ann.get("image_id", ""))
        category = categories.get(int(ann.get("category_id", -1)), "")
        if image_id and category:
            image_objects[image_id].add(category)
    return dict(image_objects)


def load_gqa_image_objects(path: Path) -> dict[str, set[str]]:
    data = safe_load_json(path)
    image_objects: dict[str, set[str]] = {}
    for image_id, item in data.items():
        objects = set()
        for obj in item.get("objects", {}).values():
            name = normalize_object_name(str(obj.get("name", "")))
            if name:
                objects.add(name)
        image_objects[str(image_id)] = objects
    return image_objects


def cooccurrence_proxy(
    *,
    annotation_rows: list[Mapping[str, Any]],
    image_objects: Mapping[str, set[str]],
    label: str = "no",
) -> dict[str, Any]:
    if not image_objects:
        return {"available": False, "reason": "missing image object annotations"}
    vocab = sorted({obj for objects in image_objects.values() for obj in objects})
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for objects in image_objects.values():
        sorted_objects = sorted(objects)
        for index, left in enumerate(sorted_objects):
            for right in sorted_objects[index + 1 :]:
                pair_counts[(left, right)] += 1
                pair_counts[(right, left)] += 1

    ranks = []
    max_counts = []
    examples = []
    missing_gt = 0
    for row in annotation_rows:
        row_label = normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target"), ""))
        if row_label != label:
            continue
        image_id = extract_image_id(row)
        neg_obj = extract_object(row)
        gt_objects = set(image_objects.get(str(int(image_id)) if image_id.isdigit() else image_id, set()))
        if not gt_objects:
            gt_objects = set(image_objects.get(image_id, set()))
        if not gt_objects or not neg_obj:
            missing_gt += 1
            continue
        scores = []
        for candidate in vocab:
            if candidate in gt_objects:
                continue
            score = max((pair_counts.get((candidate, gt), 0) for gt in gt_objects), default=0)
            scores.append((score, candidate))
        scores.sort(reverse=True)
        neg_score = max((pair_counts.get((neg_obj, gt), 0) for gt in gt_objects), default=0)
        rank = next((idx + 1 for idx, (score, candidate) in enumerate(scores) if candidate == neg_obj), None)
        if rank is not None:
            ranks.append(rank)
        max_counts.append(neg_score)
        if len(examples) < 8:
            examples.append(
                {
                    "image_id": image_id,
                    "negative_object": neg_obj,
                    "gt_objects": sorted(gt_objects)[:15],
                    "max_cooccur_count": neg_score,
                    "cooccur_rank": rank,
                }
            )
    return {
        "available": True,
        "num_negative_with_gt": len(max_counts),
        "missing_gt": missing_gt,
        "mean_max_cooccur_count": statistics.mean(max_counts) if max_counts else 0.0,
        "median_max_cooccur_count": statistics.median(max_counts) if max_counts else 0.0,
        "mean_rank": statistics.mean(ranks) if ranks else None,
        "median_rank": statistics.median(ranks) if ranks else None,
        "examples": examples,
    }


def model_metadata(model_path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"model_path": str(model_path), "exists": model_path.exists()}
    if not model_path.exists():
        return info
    config_path = model_path / "config.json"
    if config_path.exists():
        config = safe_load_json(config_path)
        info["config_architectures"] = config.get("architectures")
        info["model_type"] = config.get("model_type")
        info["_name_or_path"] = config.get("_name_or_path")
        info["vision_tower"] = config.get("mm_vision_tower") or config.get("vision_tower")
        info["mm_projector_type"] = config.get("mm_projector_type")
        text_config = config.get("text_config", {})
        if isinstance(text_config, dict):
            info["text_config_model_type"] = text_config.get("model_type")
            info["text_config_name_or_path"] = text_config.get("_name_or_path")
        vision_config = config.get("vision_config", {})
        if isinstance(vision_config, dict):
            info["vision_config_model_type"] = vision_config.get("model_type")
            info["vision_config_image_size"] = vision_config.get("image_size")
            info["vision_config_patch_size"] = vision_config.get("patch_size")

    for filename in ("preprocessor_config.json", "image_processor_config.json", "processor_config.json"):
        path = model_path / filename
        if path.exists():
            payload = safe_load_json(path)
            info[filename] = {
                key: payload.get(key)
                for key in (
                    "do_center_crop",
                    "crop_size",
                    "size",
                    "image_mean",
                    "image_std",
                    "rescale_factor",
                    "processor_class",
                    "image_processor_type",
                )
                if key in payload
            }
    tokenizer_path = model_path / "tokenizer_config.json"
    if tokenizer_path.exists():
        tokenizer = safe_load_json(tokenizer_path)
        info["tokenizer_config"] = {
            key: tokenizer.get(key)
            for key in ("tokenizer_class", "model_max_length", "padding_side", "truncation_side", "name_or_path")
            if key in tokenizer
        }
    special_path = model_path / "special_tokens_map.json"
    if special_path.exists():
        info["special_tokens_map"] = safe_load_json(special_path)

    try:
        import torch

        info["torch_version"] = torch.__version__
    except Exception as exc:
        info["torch_version_error"] = str(exc)
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except Exception as exc:
        info["transformers_version_error"] = str(exc)
    try:
        import llava  # type: ignore

        info["llava_module"] = str(getattr(llava, "__file__", ""))
    except Exception as exc:
        info["llava_import"] = f"not importable: {exc}"
    return info


def parser_diagnostics(prediction_groups: Mapping[tuple[str, str], list[dict[str, Any]]], sample_size: int) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for key, rows in prediction_groups.items():
        raw_outputs = Counter(str(row.get("raw_output", "")).strip() for row in rows)
        parser_metrics = {
            "stored_current": compute_metrics(rows, parser="stored"),
            "first_token": compute_metrics(rows, parser="first_token"),
            "contains_earliest": compute_metrics(rows, parser="contains_earliest"),
        }
        sample = [
            {
                "label": row.get("label"),
                "question": row.get("question"),
                "raw_output": row.get("raw_output"),
                "stored_pred": row.get("pred"),
                "first_token_pred": parse_first_token(row.get("raw_output", "")),
                "contains_pred": parse_contains_earliest(row.get("raw_output", "")),
            }
            for row in rows[:sample_size]
        ]
        diagnostics[f"{key[0]}_{key[1]}"] = {
            "raw_output_distribution_top30": raw_outputs.most_common(30),
            "parser_metrics": parser_metrics,
            "sample_rows": sample,
        }
    return diagnostics


def markdown_table(headers: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.2f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(item) for item in value)
    if value is None:
        return ""
    text = str(value).replace("\n", "<br>")
    return text


def json_block(payload: Any) -> str:
    return "```json\n" + json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n```"


def code_block(lines: Iterable[str]) -> str:
    return "```text\n" + "\n".join(lines) + "\n```"


def write_report(
    *,
    output: Path,
    start_status: tuple[list[str], bool],
    end_status: tuple[list[str], bool],
    runs_root: Path,
    config: Mapping[str, Any],
    annotation_summaries: Mapping[tuple[str, str], dict[str, Any]],
    annotation_rows: Mapping[tuple[str, str], list[dict[str, Any]]],
    prediction_groups: Mapping[tuple[str, str], list[dict[str, Any]]],
    inventory: list[dict[str, Any]],
    cooccurrence: Mapping[tuple[str, str], dict[str, Any]],
    model_info: Mapping[str, Any],
    parser_info: Mapping[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.extend(
        [
            "# POPE Reproduction Gap Diagnostic Report",
            "",
            "This is a read-only diagnostic report. It inspects existing files and raw outputs; it does not rerun the full benchmark.",
            "",
            "## Git Status Before Diagnostics",
            "",
            code_block(start_status[0] + (["... truncated ..."] if start_status[1] else [])),
            "",
            "## Current Vs DMAS Baseline",
            "",
        ]
    )

    gap_rows = []
    side_rows = []
    for dataset in DATASETS:
        for setting in SETTINGS:
            rows = prediction_groups.get((dataset, setting), [])
            ours = compute_metrics(rows) if rows else {}
            dmas = DMAS_BASELINES[(dataset, setting)]
            gap_rows.append(
                {
                    "Dataset": dataset,
                    "Setting": setting,
                    "Ours Acc": ours.get("Accuracy"),
                    "DMAS Acc": dmas["Accuracy"],
                    "Acc Gap": (ours.get("Accuracy", 0.0) - dmas["Accuracy"]) if ours else None,
                    "Ours Prec": ours.get("Precision"),
                    "DMAS Prec": dmas["Precision"],
                    "Prec Gap": (ours.get("Precision", 0.0) - dmas["Precision"]) if ours else None,
                    "Ours Recall": ours.get("Recall"),
                    "DMAS Recall": dmas["Recall"],
                    "Recall Gap": (ours.get("Recall", 0.0) - dmas["Recall"]) if ours else None,
                    "Ours F1": ours.get("F1 Score"),
                    "DMAS F1": dmas["F1 Score"],
                    "F1 Gap": (ours.get("F1 Score", 0.0) - dmas["F1 Score"]) if ours else None,
                }
            )
            side_rows.append(
                {
                    "Dataset": dataset,
                    "Setting": setting,
                    "TP": ours.get("TP"),
                    "TN": ours.get("TN"),
                    "FP": ours.get("FP"),
                    "FN": ours.get("FN"),
                    "Recall": ours.get("Recall"),
                    "Precision": ours.get("Precision"),
                    "Interpretation": side_interpretation(ours, dmas) if ours else "missing raw Regular predictions",
                }
            )
    lines.append(markdown_table(["Dataset", "Setting", "Ours Acc", "DMAS Acc", "Acc Gap", "Ours Prec", "DMAS Prec", "Prec Gap", "Ours Recall", "DMAS Recall", "Recall Gap", "Ours F1", "DMAS F1", "F1 Gap"], gap_rows))
    lines.extend(["", "## Positive/Negative Side Analysis", ""])
    lines.append(markdown_table(["Dataset", "Setting", "TP", "TN", "FP", "FN", "Recall", "Precision", "Interpretation"], side_rows))

    lines.extend(["", "## Actual POPE Annotation Files", ""])
    file_rows = []
    for dataset in DATASETS:
        for setting in SETTINGS:
            summary = annotation_summaries.get((dataset, setting), {})
            file_rows.append(
                {
                    "Dataset": dataset,
                    "Setting": setting,
                    "Path": summary.get("path", "missing"),
                    "N": summary.get("N", ""),
                    "Yes": summary.get("yes", ""),
                    "No": summary.get("no", ""),
                    "MD5": summary.get("md5", ""),
                    "SHA256": summary.get("sha256", ""),
                }
            )
    lines.append(markdown_table(["Dataset", "Setting", "Path", "N", "Yes", "No", "MD5", "SHA256"], file_rows))
    for key, summary in annotation_summaries.items():
        lines.extend(["", f"### First 5 Rows: {key[0]} {key[1]}", "", json_block(summary.get("first_rows", []))])

    lines.extend(["", "## POPE File Inventory", ""])
    lines.append(
        markdown_table(
            ["path", "size_bytes", "N", "yes", "no", "used_as", "sha256"],
            inventory,
        )
    )
    lines.extend(["", "## Negative Sample Diagnostics", ""])
    for dataset in DATASETS:
        for setting in SETTINGS:
            rows = annotation_rows.get((dataset, setting), [])
            lines.extend(["", f"### {dataset} {setting}", ""])
            lines.append("Positive object top 30:")
            lines.append(markdown_table(["object", "count"], top_object_table(rows, "yes", 30)))
            lines.append("")
            lines.append("Negative object top 30:")
            lines.append(markdown_table(["object", "count"], top_object_table(rows, "no", 30)))
            lines.append("")
            lines.append("Co-occurrence proxy for label=no:")
            lines.append(json_block(cooccurrence.get((dataset, setting), {})))

    lines.extend(["", "### Negative Object Top-30 Jaccard Across Settings", ""])
    jaccard_rows = []
    for dataset in DATASETS:
        counters = {setting: label_object_sets(annotation_rows.get((dataset, setting), []), "no") for setting in SETTINGS}
        for left, right in (("random", "popular"), ("random", "adversarial"), ("popular", "adversarial")):
            jaccard_rows.append({"Dataset": dataset, "Compare": f"{left} vs {right}", "Top30 Jaccard": jaccard_top(counters[left], counters[right], 30)})
    lines.append(markdown_table(["Dataset", "Compare", "Top30 Jaccard"], jaccard_rows))

    lines.extend(["", "## Prompt And Conversation Template", ""])
    lines.extend(
        [
            f"- Prompt suffix: `{PROMPT_SUFFIX}`",
            "- LLaVA prefix builder: `USER: <image>\\n{question}\\nASSISTANT:`",
            "- System prompt: none in this repository prompt builder.",
            "- Contains `<image>`: yes.",
            "",
        ]
    )
    for dataset in DATASETS:
        for setting in SETTINGS:
            rows = annotation_rows.get((dataset, setting), [])[:3]
            prompt_rows = []
            for row in rows:
                question = str(first_present(row, ("question", "text", "query", "prompt"), ""))
                final_question = ensure_prompt(question)
                prompt_rows.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "original_question": question,
                        "final_question": final_question,
                        "full_prompt": build_llava_prefix_prompt(final_question),
                    }
                )
            lines.extend([f"### Prompt Samples: {dataset} {setting}", "", json_block(prompt_rows), ""])

    lines.extend(["", "## Decode Parameters", ""])
    lines.append(
        json_block(
            {
                "runner_decode_config": config.get("decode", {}),
                "observed_generate_call": {
                    "do_sample": False,
                    "num_beams": 1,
                    "temperature": 0.0,
                    "max_new_tokens": config.get("decode", {}).get("max_new_tokens", 5),
                    "use_cache": True,
                    "top_p": "not explicitly passed by run_pope_cat_expert_eval.py",
                    "stopping_criteria": "not explicitly passed by run_pope_cat_expert_eval.py",
                },
                "regular_hook_status": "Regular rows are generated before any ExpertSteeringController is attached; no vector/hook is used.",
            }
        )
    )

    lines.extend(["", "## Model Checkpoint Metadata", "", json_block(model_info)])

    lines.extend(["", "## Answer Parser Diagnostics", ""])
    for key, info in parser_info.items():
        lines.extend([f"### {key}", ""])
        parser_metrics = []
        for parser_name, metrics in info.get("parser_metrics", {}).items():
            row = {"Parser": parser_name}
            row.update(metrics)
            parser_metrics.append(row)
        lines.append(markdown_table(["Parser", "N", "Accuracy", "Precision", "Recall", "F1 Score", "Yes Rate", "TP", "TN", "FP", "FN", "Invalid"], parser_metrics))
        lines.extend(["", "Raw output distribution top 30:", ""])
        lines.append(markdown_table(["raw_output", "count"], [{"raw_output": out, "count": count} for out, count in info.get("raw_output_distribution_top30", [])]))
        lines.extend(["", "Sample raw outputs:", "", json_block(info.get("sample_rows", [])[:20])])

    lines.extend(["", "## Ranked Likely Causes", ""])
    lines.extend(rank_likely_causes(prediction_groups, annotation_summaries, parser_info, model_info))
    lines.extend(
        [
            "",
            "## Next Minimal Verification Experiments",
            "",
            "1. Compare the six used POPE file hashes and first image/question rows against the exact POPE files used by DMAS or their released scripts.",
            "2. Rerun a 50-sample Regular-only smoke with the official LLaVA/POPE prompt template if DMAS used a different conversation wrapper.",
            "3. Rerun the same 50 samples with `top_p=1.0` explicitly set and no steering hooks to isolate decode config differences.",
            "4. If hashes/prompt/decode match, test the exact DMAS checkpoint or official `liuhaotian/llava-v1.5-7b` loader on the same 50 samples.",
            "",
            "## Git Status After Diagnostics",
            "",
            code_block(end_status[0] + (["... truncated ..."] if end_status[1] else [])),
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def side_interpretation(ours: Mapping[str, Any], dmas: Mapping[str, float]) -> str:
    recall_gap = float(ours.get("Recall", 0.0)) - float(dmas["Recall"])
    precision_gap = float(ours.get("Precision", 0.0)) - float(dmas["Precision"])
    if abs(recall_gap) <= 3.0 and precision_gap >= 5.0:
        return "Recall is close but Precision/TN are much higher: negative side is the main gap."
    if recall_gap > 3.0 and precision_gap >= 5.0:
        return "Both positive and negative sides are stronger, but high Precision/TN still points to easier negatives or conservative outputs."
    if recall_gap > 3.0:
        return "Positive-side recall is higher; prompt/model/checkpoint may differ too."
    return "Mixed gap; inspect files/parser/prompt."


def rank_likely_causes(
    prediction_groups: Mapping[tuple[str, str], list[dict[str, Any]]],
    annotation_summaries: Mapping[tuple[str, str], dict[str, Any]],
    parser_info: Mapping[str, Any],
    model_info: Mapping[str, Any],
) -> list[str]:
    lines = []
    high_precision_gap = False
    recall_close = False
    for key, rows in prediction_groups.items():
        if key not in DMAS_BASELINES or not rows:
            continue
        ours = compute_metrics(rows)
        dmas = DMAS_BASELINES[key]
        if ours["Precision"] - dmas["Precision"] > 5.0:
            high_precision_gap = True
        if abs(ours["Recall"] - dmas["Recall"]) <= 3.5:
            recall_close = True

    parser_same = True
    for info in parser_info.values():
        metrics = info.get("parser_metrics", {})
        stored = metrics.get("stored_current", {})
        contains = metrics.get("contains_earliest", {})
        first = metrics.get("first_token", {})
        if abs(float(stored.get("Accuracy", 0.0)) - float(contains.get("Accuracy", 0.0))) > 0.01:
            parser_same = False
        if abs(float(stored.get("Accuracy", 0.0)) - float(first.get("Accuracy", 0.0))) > 0.5:
            parser_same = False

    if high_precision_gap and recall_close:
        lines.append("1. POPE annotation or negative sampling version mismatch is most likely: Recall is close to DMAS but Precision/TN are substantially higher, especially from very low FP on negative questions.")
    else:
        lines.append("1. POPE annotation or negative sampling version mismatch remains a leading candidate; inspect file hashes and negative object distributions above.")
    lines.append("2. Prompt/conversation template mismatch is likely: this runner uses `USER: <image>\\n{question} Please answer this question in one word.\\nASSISTANT:`; DMAS may use a different official LLaVA eval template or answer instruction.")
    model_note = str(model_info.get("_name_or_path") or model_info.get("model_path") or "")
    lines.append(f"3. Model checkpoint/image preprocessing mismatch is plausible: current metadata reports `{model_note}`; verify this is the exact LLaVA-v1.5-7B checkpoint and processor used by DMAS.")
    if parser_same:
        lines.append("4. Answer parser mismatch is less likely from existing raw outputs: stored parser and contains-earliest parser agree or nearly agree in this report.")
    else:
        lines.append("4. Answer parser mismatch may contribute: parser tables show metric differences across parsing rules.")
    lines.append("5. Decode mismatch is a smaller but real check item: `temperature=0` is set, but `top_p=1` is not explicitly passed in the current generate call.")
    if not all(summary.get("exists", False) for summary in annotation_summaries.values()):
        lines.append("6. Some POPE annotation files were not found by diagnostics; resolve missing paths before interpreting gaps.")
    return lines


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output = Path(args.output)
    config_path = runs_root / "config.json"

    start_status = run_git_status(int(args.max_status_lines))
    try:
        config = safe_load_json(config_path) if config_path.exists() else {}
        if not config:
            print(f"Warning: missing config.json at {config_path}; diagnostics will be partial.", file=sys.stderr)

        if args.pope_root:
            config = dict(config)
            config["pope_root"] = args.pope_root
        if args.model_path:
            config = dict(config)
            config["model_path"] = args.model_path

        annotation_paths = collect_annotation_paths(config, args.pope_root)
        annotation_summaries: dict[tuple[str, str], dict[str, Any]] = {}
        annotation_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for key in [(dataset, setting) for dataset in DATASETS for setting in SETTINGS]:
            path = annotation_paths.get(key)
            if path and path.exists():
                annotation_summaries[key] = summarize_annotation_file(path)
                annotation_rows[key] = read_json_or_jsonl(path)
            else:
                annotation_summaries[key] = {"path": str(path) if path else "missing", "exists": False}
                annotation_rows[key] = []

        prediction_groups = load_regular_predictions(runs_root)
        inventory = discover_pope_inventory(
            annotation_paths=annotation_paths,
            pope_root=str(config.get("pope_root", "")),
            max_files=int(args.max_inventory_files),
        )

        coco_objects = {}
        gqa_objects = {}
        coco_path = Path(args.coco_instances) if args.coco_instances else Path("")
        gqa_path = Path(args.gqa_scene_graphs) if args.gqa_scene_graphs else Path("")
        if coco_path.exists():
            coco_objects = load_coco_image_objects(coco_path)
        if gqa_path.exists():
            gqa_objects = load_gqa_image_objects(gqa_path)

        cooccurrence: dict[tuple[str, str], dict[str, Any]] = {}
        for dataset in DATASETS:
            objects = coco_objects if dataset == "MSCOCO" else gqa_objects
            for setting in SETTINGS:
                cooccurrence[(dataset, setting)] = cooccurrence_proxy(
                    annotation_rows=annotation_rows.get((dataset, setting), []),
                    image_objects=objects,
                    label="no",
                )

        model_path = Path(str(config.get("model_path", "")))
        model_info = model_metadata(model_path) if str(model_path) else {"model_path": "", "exists": False}
        parser_info = parser_diagnostics(prediction_groups, int(args.raw_sample_size))
        end_status = run_git_status(int(args.max_status_lines))

        write_report(
            output=output,
            start_status=start_status,
            end_status=end_status,
            runs_root=runs_root,
            config=config,
            annotation_summaries=annotation_summaries,
            annotation_rows=annotation_rows,
            prediction_groups=prediction_groups,
            inventory=inventory,
            cooccurrence=cooccurrence,
            model_info=model_info,
            parser_info=parser_info,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote POPE reproduction gap diagnostics to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
