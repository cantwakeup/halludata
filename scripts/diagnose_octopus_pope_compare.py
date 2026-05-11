"""Compare Octopus POPE evaluation code with this repo's HF POPE runner.

This is a diagnostics-only script. It does not train, does not run full
benchmarks, and does not modify the formal POPE evaluation scripts. The goal is
to make reproduction gaps auditable: POPE file hashes, prompt/template choices,
model-loading code paths, parser behavior, decode settings, and available
small-sample/raw-output evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OCTOPUS_REPORTED: dict[str, dict[str, float]] = {
    "random": {"Accuracy": 83.77, "F1 Score": 81.94},
    "popular": {"Accuracy": 82.57, "F1 Score": 80.86},
    "adversarial": {"Accuracy": 79.77, "F1 Score": 78.47},
    "ALL": {"Accuracy": 82.04, "F1 Score": 80.42},
}

OUR_REPORTED: dict[str, dict[str, float]] = {
    "random": {"Accuracy": 86.50, "F1 Score": 84.69, "Precision": 97.82, "Recall": 74.67},
    "popular": {"Accuracy": 85.67, "F1 Score": 83.90, "Precision": 95.73, "Recall": 74.67},
    "adversarial": {"Accuracy": 83.70, "F1 Score": 82.08, "Precision": 91.13, "Recall": 74.67},
}

OUR_POPE_HASHES = {
    "random": "ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994",
    "popular": "72c1a8ad45d0c13514f5f22598261df41d3b533854d29682e924db50ed8aa753",
    "adversarial": "420b3407db1fa9f1187a805dca41cb7b97fd91504e6c2179706188c107fb8ef8",
}

SETTINGS = ("random", "popular", "adversarial")
TEXT_EXTENSIONS = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json", ".jsonl"}
SEARCH_KEYWORDS = [
    "pope",
    "POPE",
    "MSCOCO",
    "coco_pope",
    "random",
    "popular",
    "adversarial",
    "Please answer this question in one word",
    "Please answer this question with one word",
    "llava",
    "llava-v1.5",
    "LLaVA-1.5",
    "conv_mode",
    "conv_templates",
    "temperature",
    "top_p",
    "max_new_tokens",
    "num_beams",
    "accuracy",
    "f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--octopus-root", default="", help="Path to Octopus repo; auto-detected if omitted.")
    parser.add_argument("--pope-root", default="/home/huiwei/sy/benchmarks/POPE", help="Our POPE benchmark root.")
    parser.add_argument("--our-runs-root", default="data/pope_cat_expert_eval/full_alpha_sweep")
    parser.add_argument("--output-dir", default="data/pope_cat_expert_eval/octopus_compare")
    parser.add_argument("--max-search-hits", type=int, default=240)
    parser.add_argument("--sample-limit", type=int, default=50, help="Small raw-output comparison size; no inference is run.")
    parser.add_argument("--octopus-answer-file", default="", help="Optional Octopus raw answer jsonl for side-by-side metrics.")
    parser.add_argument("--octopus-gt-file", default="", help="Optional Octopus GT POPE file for --octopus-answer-file.")
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return proc.stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_octopus_root(user_path: str) -> Path:
    candidates: list[Path] = []
    if user_path:
        candidates.append(resolve_path(user_path))
    candidates.extend(
        [
            Path("/home/huiwei/sy/Octopus-master"),
            Path("/home/huiwei/sy/Octopus"),
            Path("/home/huiwei/sy/Octopus_READONLY"),
            PROJECT_ROOT / "reference" / "Octopus-master",
            PROJECT_ROOT / "third_party" / "Octopus_READONLY",
        ]
    )
    for candidate in candidates:
        if (candidate / "experiments" / "eval").exists() and (candidate / "README.md").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Octopus repo. Pass --octopus-root, e.g. /home/huiwei/sy/Octopus-master."
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def line_snippet(path: Path, start: int, end: int) -> str:
    lines = read_text(path).splitlines()
    start = max(1, start)
    end = min(len(lines), end)
    return "\n".join(f"{idx:4}: {lines[idx - 1]}" for idx in range(start, end + 1))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
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


def extract_object(question: Any) -> str:
    text = str(question)
    patterns = [
        r"\bIs there (?:a|an|the|any)?\s*(.+?)\s+in the image\??",
        r"\bAre there (?:a|an|the|any)?\s*(.+?)\s+in the image\??",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"[?.!]+$", "", match.group(1).strip().lower())
    return ""


def summarize_pope_file(path: Path) -> dict[str, Any]:
    rows = read_json_or_jsonl(path)
    labels = Counter(normalize_label(first_present(row, ("label", "answer", "gt_answer"), "")) for row in rows)
    negative_objects = Counter()
    positive_objects = Counter()
    previews: list[dict[str, Any]] = []
    for row in rows:
        label = normalize_label(first_present(row, ("label", "answer", "gt_answer"), ""))
        question = first_present(row, ("question", "text", "query", "prompt"), "")
        obj = first_present(row, ("object", "obj", "category", "class"), "") or extract_object(question)
        if label == "no":
            negative_objects[str(obj)] += 1
        elif label == "yes":
            positive_objects[str(obj)] += 1
        if len(previews) < 5:
            previews.append(
                {
                    "question_id": first_present(row, ("question_id", "id", "qid"), ""),
                    "image_id": first_present(row, ("image_id", "image", "filename", "file_name"), ""),
                    "question": question,
                    "label": label,
                    "object": obj,
                }
            )
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "num_samples": len(rows),
        "label_yes": labels.get("yes", 0),
        "label_no": labels.get("no", 0),
        "first5": previews,
        "negative_top30": negative_objects.most_common(30),
        "positive_top30": positive_objects.most_common(30),
        "first20_keys": [
            {
                "question_id": first_present(row, ("question_id", "id", "qid"), ""),
                "image_id": first_present(row, ("image_id", "image", "filename", "file_name"), ""),
                "question": first_present(row, ("question", "text", "query", "prompt"), ""),
                "label": normalize_label(first_present(row, ("label", "answer", "gt_answer"), "")),
            }
            for row in rows[:20]
        ],
    }


def find_our_pope_file(pope_root: Path, setting: str) -> Path | None:
    candidates = [
        pope_root / "output" / "coco" / f"coco_pope_{setting}.json",
        pope_root / "output" / "coco" / f"coco_pope_{setting}.jsonl",
        pope_root / "coco" / f"coco_pope_{setting}.json",
        pope_root / f"coco_pope_{setting}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if pope_root.exists():
        matches = [
            p
            for p in pope_root.rglob("*")
            if p.suffix.lower() in {".json", ".jsonl"}
            and "pope" in str(p).lower()
            and "coco" in str(p).lower()
            and setting in p.name.lower()
        ]
        return sorted(matches, key=lambda p: (len(str(p)), str(p)))[0] if matches else None
    return None


def find_octopus_pope_file(octopus_root: Path, setting: str) -> Path | None:
    candidates = [
        octopus_root / "data" / "POPE" / f"coco_pope_{setting}.json",
        octopus_root / "data" / "POPE" / f"coco_pope_{setting}.jsonl",
        octopus_root / "data" / "pope" / f"coco_pope_{setting}.json",
        octopus_root / "data" / f"coco_pope_{setting}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = [
        p
        for p in octopus_root.rglob("*")
        if p.suffix.lower() in {".json", ".jsonl"}
        and "pope" in str(p).lower()
        and "coco" in str(p).lower()
        and setting in p.name.lower()
    ]
    return sorted(matches, key=lambda p: (len(str(p)), str(p)))[0] if matches else None


def compare_first20(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    if not a or not b:
        return {"available": False, "same_first20": None, "num_mismatches": None, "examples": []}
    a_rows = a.get("first20_keys", [])
    b_rows = b.get("first20_keys", [])
    mismatches = []
    for idx, (left, right) in enumerate(zip(a_rows, b_rows)):
        if left != right:
            mismatches.append({"index": idx, "ours": left, "octopus": right})
    return {
        "available": True,
        "same_first20": len(mismatches) == 0 and len(a_rows) == len(b_rows),
        "num_mismatches": len(mismatches),
        "examples": mismatches[:5],
    }


def search_octopus(octopus_root: Path, max_hits: int) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    lowered = [(kw, kw.lower()) for kw in SEARCH_KEYWORDS]
    skip_parts = {".git", "__pycache__", "assets"}

    def scan_file(path: Path) -> None:
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                lower = line.lower()
                matched = [kw for kw, kw_lower in lowered if kw_lower in lower]
                if matched:
                    hits.append(
                        {
                            "file": str(path.relative_to(octopus_root)),
                            "line": line_number,
                            "keywords": ", ".join(sorted(set(matched))),
                            "text": line.strip()[:220],
                        }
                    )
        except Exception:
            return

    core_paths = [
        octopus_root / "experiments" / "eval" / "object_hallucination_vqa_llava.py",
        octopus_root / "experiments" / "eval" / "eval_pope.py",
        octopus_root / "README.md",
        octopus_root / "Octopus.yaml",
    ]
    scanned = set()
    for path in core_paths:
        if path.exists():
            scan_file(path)
            scanned.add(path)

    for path in sorted(octopus_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if path in scanned:
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        if path.stat().st_size > 2_000_000 and path.suffix.lower() in {".json", ".jsonl"}:
            continue
        scan_file(path)
        if len(hits) >= max_hits:
            return hits[:max_hits]
    return hits


def detect_octopus_settings(octopus_root: Path) -> dict[str, Any]:
    gen_file = octopus_root / "experiments" / "eval" / "object_hallucination_vqa_llava.py"
    eval_file = octopus_root / "experiments" / "eval" / "eval_pope.py"
    readme = octopus_root / "README.md"
    env_file = octopus_root / "Octopus.yaml"
    gen_text = read_text(gen_file) if gen_file.exists() else ""
    eval_text = read_text(eval_file) if eval_file.exists() else ""
    env_text = read_text(env_file) if env_file.exists() else ""
    return {
        "generation_entry": str(gen_file) if gen_file.exists() else "missing",
        "metric_entry": str(eval_file) if eval_file.exists() else "missing",
        "readme": str(readme) if readme.exists() else "missing",
        "uses_official_llava_loader": "load_pretrained_model" in gen_text,
        "uses_conv_templates": "conv_templates" in gen_text,
        "uses_tokenizer_image_token": "tokenizer_image_token" in gen_text,
        "uses_hf_llava_for_conditional_generation": "LlavaForConditionalGeneration" in gen_text,
        "uses_auto_processor": "AutoProcessor" in gen_text,
        "monkey_patches_generation": "evolve_avisc_sampling()" in gen_text,
        "conv_mode_default": regex_default(gen_text, r'--conv-mode".*?default="([^"]+)"'),
        "model_path_default": regex_default(gen_text, r'--model-path".*?default="([^"]+)"'),
        "temperature_default": regex_default(gen_text, r'--temperature".*?default=([0-9.]+)'),
        "top_p_default": regex_default(gen_text, r'--top_p".*?default=([0-9.]+)'),
        "top_k_default": regex_default(gen_text, r'--top_k".*?default=([^),\n]+)'),
        "do_sample_literal": "do_sample=True" if "do_sample=True" in gen_text else ("do_sample=False" if "do_sample=False" in gen_text else "not found"),
        "max_new_tokens_literal": regex_default(gen_text, r"max_new_tokens=([0-9]+)"),
        "uses_stopping_criteria": "KeywordsStoppingCriteria" in gen_text,
        "passes_stopping_criteria_to_generate": "stopping_criteria" in re.sub(r"#.*", "", gen_text).split("model.generate", 1)[-1],
        "prompt_suffix": detect_prompt_suffix(gen_text),
        "image_token": "DEFAULT_IMAGE_TOKEN + '\\n' + qs" if "DEFAULT_IMAGE_TOKEN + '\\n' + qs" in gen_text else "see generation snippet",
        "image_preprocess": "image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]"
        if "image_processor.preprocess" in gen_text
        else "not found",
        "metric_parser": detect_metric_parser(eval_text),
        "metric_default_gt": regex_default(eval_text, r'--gt_files".*?default="([^"]+)"'),
        "metric_default_gen": regex_default(eval_text, r'--gen_files".*?default="([^"]+)"'),
        "environment_transformers": regex_default(env_text, r"transformers==([0-9.]+)"),
        "environment_torch": regex_default(env_text, r"torch==([0-9.]+)"),
        "generation_snippet": line_snippet(gen_file, 28, 96) if gen_file.exists() else "",
        "args_snippet": line_snippet(gen_file, 107, 133) if gen_file.exists() else "",
        "metric_snippet": line_snippet(eval_file, 17, 66) if eval_file.exists() else "",
    }


def regex_default(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else "not found"


def detect_prompt_suffix(text: str) -> str:
    if "Please answer this question with one word." in text:
        return "Please answer this question with one word."
    if "Please answer this question in one word." in text:
        return "Please answer this question in one word."
    return "not found"


def detect_metric_parser(text: str) -> str:
    if "if 'yes' in gen_answer" in text and "if 'no' in gen_answer" in text:
        return "contains-substring parser: yes if generated text contains 'yes'; no if it contains 'no'"
    return "not found"


def git_info(octopus_root: Path) -> dict[str, Any]:
    if not (octopus_root / ".git").exists():
        return {
            "is_git_repo": False,
            "remote": "not a standalone git repo at this path",
            "head": "not available",
            "status": "not available",
        }
    return {
        "is_git_repo": True,
        "remote": run_command(["git", "remote", "-v"], cwd=octopus_root),
        "head": run_command(["git", "rev-parse", "HEAD"], cwd=octopus_root),
        "status": run_command(["git", "status", "--short"], cwd=octopus_root),
    }


def load_our_config(our_runs_root: Path) -> dict[str, Any]:
    config_path = our_runs_root / "config.json"
    if not config_path.exists():
        return {"available": False, "path": str(config_path)}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["available"] = True
        config["path"] = str(config_path)
        return config
    except Exception as exc:
        return {"available": False, "path": str(config_path), "error": str(exc)}


def load_summary_rows(our_runs_root: Path) -> list[dict[str, Any]]:
    summary_path = our_runs_root / "summary.csv"
    if not summary_path.exists():
        return []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_our_regular_raw(our_runs_root: Path, setting: str) -> Path | None:
    raw_dir = our_runs_root / "raw"
    if not raw_dir.exists():
        return None
    patterns = [
        f"mscoco_{setting}_regular.jsonl",
        f"coco_{setting}_regular.jsonl",
        f"*{setting}*regular*.jsonl",
    ]
    for pattern in patterns:
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def parse_contains_earliest(text: Any) -> str:
    matches = [(m.start(), m.group(1)) for m in re.finditer(r"\b(yes|no)\b", str(text).lower())]
    return sorted(matches)[0][1] if matches else "invalid"


def parse_octopus_contains(text: Any, label: str) -> str:
    generated = str(text).lower().strip()
    if label == "yes":
        return "yes" if "yes" in generated else "no"
    if label == "no":
        return "no" if "no" in generated else "yes"
    return "invalid"


def metric_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(rows: Iterable[Mapping[str, Any]], pred_key: str = "pred") -> dict[str, Any]:
    tp = tn = fp = fn = invalid = pred_yes = 0
    total = 0
    for row in rows:
        total += 1
        label = normalize_label(first_present(row, ("label", "answer", "gt_answer"), ""))
        pred = str(row.get(pred_key, "")).strip().lower()
        if pred not in {"yes", "no"}:
            raw = first_present(row, ("raw_output", "text", "output", "answer"), "")
            pred = parse_contains_earliest(raw)
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
    accuracy = metric_div(tp + tn, total)
    return {
        "N": total,
        "Accuracy": accuracy * 100.0,
        "Precision": precision * 100.0,
        "Recall": recall * 100.0,
        "F1": f1 * 100.0,
        "Yes Rate": metric_div(pred_yes, total) * 100.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Invalid": invalid,
    }


def summarize_existing_small_outputs(args: argparse.Namespace, our_runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for setting in SETTINGS:
        raw = find_our_regular_raw(our_runs_root, setting)
        if not raw:
            continue
        raw_rows = read_json_or_jsonl(raw)[: args.sample_limit]
        metrics = compute_metrics(raw_rows)
        metrics.update({"Runner": "OurHFRunner-Regular", "Setting": setting, "Raw": str(raw)})
        rows.append(metrics)
    if args.octopus_answer_file and args.octopus_gt_file:
        answer_path = resolve_path(args.octopus_answer_file)
        gt_path = resolve_path(args.octopus_gt_file)
        if answer_path.exists() and gt_path.exists():
            answer_rows = read_json_or_jsonl(answer_path)[: args.sample_limit]
            gt_rows = read_json_or_jsonl(gt_path)[: args.sample_limit]
            joined: list[dict[str, Any]] = []
            for gt, ans in zip(gt_rows, answer_rows):
                text = first_present(ans, ("text", "raw_output", "output", "answer"), "")
                label = normalize_label(first_present(gt, ("label", "answer", "gt_answer"), ""))
                joined.append({"label": label, "pred": parse_octopus_contains(text, label), "raw_output": text})
            metrics = compute_metrics(joined)
            metrics.update({"Runner": "OctopusOfficial-Regular", "Setting": "provided", "Raw": str(answer_path)})
            rows.append(metrics)
    return rows


def markdown_table(headers: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = [fmt(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if value is None:
        return ""
    return str(value).replace("\n", "<br>")


def code_block(text: Any, lang: str = "text") -> str:
    return f"```{lang}\n{text}\n```"


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def build_diff_rows(octopus: Mapping[str, Any], config: Mapping[str, Any], pope_same: str) -> list[dict[str, Any]]:
    our_prompt = "USER: <image>\\n{question} Please answer this question in one word.\\nASSISTANT:"
    our_decode = "do_sample=False, temperature=0.0, num_beams=1, max_new_tokens=5, top_p not explicit"
    oct_prompt = (
        "llava_v1 conversation; DEFAULT_IMAGE_TOKEN + question + "
        f"`{octopus.get('prompt_suffix', 'not found')}`"
    )
    oct_decode = (
        f"{octopus.get('do_sample_literal')}, temperature default {octopus.get('temperature_default')}, "
        f"top_p default {octopus.get('top_p_default')}, max_new_tokens={octopus.get('max_new_tokens_literal')}, "
        f"stopping criteria constructed={octopus.get('uses_stopping_criteria')}, passed={octopus.get('passes_stopping_criteria_to_generate')}"
    )
    rows = [
        {
            "Item": "Model loader",
            "Our HFRunner": "HF LlavaForConditionalGeneration via run_steered_benchmark/LlavaBenchmarkGenerator",
            "Octopus Official": "official LLaVA load_pretrained_model" if octopus.get("uses_official_llava_loader") else "not found",
            "Same?": "no",
            "Risk": "high",
        },
        {
            "Item": "Checkpoint",
            "Our HFRunner": str(config.get("model_path", "/home/huiwei/sy/models/llava-1.5-7b-hf")),
            "Octopus Official": str(octopus.get("model_path_default", "path/checkpoints/llava-v1.5-7b")),
            "Same?": "unknown",
            "Risk": "high",
        },
        {
            "Item": "Conversation template",
            "Our HFRunner": "repo build_llava_prefix_prompt, USER/ASSISTANT style",
            "Octopus Official": f"conv_templates[{octopus.get('conv_mode_default')}]",
            "Same?": "likely similar but not byte-identical",
            "Risk": "medium",
        },
        {
            "Item": "Prompt suffix",
            "Our HFRunner": "Please answer this question in one word.",
            "Octopus Official": str(octopus.get("prompt_suffix")),
            "Same?": "no",
            "Risk": "medium",
        },
        {
            "Item": "Image token handling",
            "Our HFRunner": our_prompt,
            "Octopus Official": str(octopus.get("image_token")),
            "Same?": "unknown",
            "Risk": "medium",
        },
        {
            "Item": "Image preprocessing",
            "Our HFRunner": "HF AutoProcessor path in LlavaBenchmarkGenerator",
            "Octopus Official": str(octopus.get("image_preprocess")),
            "Same?": "no/unknown",
            "Risk": "high",
        },
        {
            "Item": "POPE file hash",
            "Our HFRunner": "known hashes from /home/huiwei/sy/benchmarks/POPE/output/coco",
            "Octopus Official": pope_same,
            "Same?": "see POPE comparison table",
            "Risk": "high",
        },
        {
            "Item": "Decode params",
            "Our HFRunner": our_decode,
            "Octopus Official": oct_decode,
            "Same?": "no",
            "Risk": "high",
        },
        {
            "Item": "Parser",
            "Our HFRunner": "first explicit yes/no; raw mostly exact Yes/No",
            "Octopus Official": str(octopus.get("metric_parser")),
            "Same?": "no",
            "Risk": "medium",
        },
        {
            "Item": "Metric calculation",
            "Our HFRunner": "TP/TN/FP/FN from parsed pred, invalid counted wrong",
            "Octopus Official": "contains-substring parser directly accumulates TP/TN/FP/FN",
            "Same?": "mostly same formula, different parser",
            "Risk": "medium",
        },
        {
            "Item": "Environment",
            "Our HFRunner": "current hallu_llava env, transformers observed 4.57.6 in prior check",
            "Octopus Official": f"Octopus.yaml torch=={octopus.get('environment_torch')}, transformers=={octopus.get('environment_transformers')}",
            "Same?": "no",
            "Risk": "high",
        },
    ]
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report(
    output: Path,
    octopus_root: Path,
    git: Mapping[str, Any],
    octopus: Mapping[str, Any],
    search_hits: list[dict[str, Any]],
    our_config: Mapping[str, Any],
    pope_rows: list[dict[str, Any]],
    diff_rows: list[dict[str, Any]],
    small_rows: list[dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    reported_rows = []
    for setting in ["random", "popular", "adversarial", "ALL"]:
        oct_vals = OCTOPUS_REPORTED[setting]
        our_vals = OUR_REPORTED.get(setting, {})
        reported_rows.append(
            {
                "Setting": setting,
                "Octopus Acc": oct_vals.get("Accuracy", ""),
                "Octopus F1": oct_vals.get("F1 Score", ""),
                "Our Acc": our_vals.get("Accuracy", ""),
                "Our F1": our_vals.get("F1 Score", ""),
                "Acc Gap": (our_vals.get("Accuracy", 0) - oct_vals.get("Accuracy", 0)) if our_vals else "",
                "F1 Gap": (our_vals.get("F1 Score", 0) - oct_vals.get("F1 Score", 0)) if our_vals else "",
            }
        )

    search_file_rows = [
        {"file": file, "hits": count}
        for file, count in Counter(str(hit.get("file", "")) for hit in search_hits).most_common(40)
    ]
    preferred_files = {
        "experiments/eval/object_hallucination_vqa_llava.py",
        "experiments/eval/eval_pope.py",
        "README.md",
        "Octopus.yaml",
    }
    preferred_hits = [
        hit for hit in search_hits if str(hit.get("file", "")).replace("\\", "/") in preferred_files
    ]
    search_rows = preferred_hits[:60] if preferred_hits else search_hits[:40]
    raw_snippets = [
        "## Octopus Generation Code Snippet",
        code_block(octopus.get("generation_snippet", ""), "python"),
        "## Octopus Argument Defaults Snippet",
        code_block(octopus.get("args_snippet", ""), "python"),
        "## Octopus Metric Parser Snippet",
        code_block(octopus.get("metric_snippet", ""), "python"),
    ]

    text = [
        "# Octopus vs Our POPE.MSCOCO Baseline Diagnostic",
        "",
        "This report is diagnostics-only. It inspects code/files and optional existing small raw outputs; it does not run full benchmarks or train Octopus.",
        "",
        "## Octopus Repo",
        "",
        f"- Path: `{octopus_root}`",
        f"- Standalone git repo: `{git.get('is_git_repo')}`",
        f"- Remote: `{git.get('remote')}`",
        f"- HEAD: `{git.get('head')}`",
        "",
        "Git status:",
        code_block(git.get("status", "")),
        "",
        "## Reported Baselines",
        "",
        markdown_table(["Setting", "Octopus Acc", "Octopus F1", "Our Acc", "Our F1", "Acc Gap", "F1 Gap"], reported_rows),
        "",
        "## Octopus Code Settings",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                {"Field": "Generation entry", "Value": octopus.get("generation_entry")},
                {"Field": "Metric entry", "Value": octopus.get("metric_entry")},
                {"Field": "Model loader", "Value": "official LLaVA load_pretrained_model" if octopus.get("uses_official_llava_loader") else "not found"},
                {"Field": "Uses conv_templates", "Value": octopus.get("uses_conv_templates")},
                {"Field": "Uses tokenizer_image_token", "Value": octopus.get("uses_tokenizer_image_token")},
                {"Field": "HF LlavaForConditionalGeneration", "Value": octopus.get("uses_hf_llava_for_conditional_generation")},
                {"Field": "conv_mode default", "Value": octopus.get("conv_mode_default")},
                {"Field": "prompt suffix", "Value": octopus.get("prompt_suffix")},
                {"Field": "do_sample", "Value": octopus.get("do_sample_literal")},
                {"Field": "temperature default", "Value": octopus.get("temperature_default")},
                {"Field": "top_p default", "Value": octopus.get("top_p_default")},
                {"Field": "max_new_tokens", "Value": octopus.get("max_new_tokens_literal")},
                {"Field": "KeywordsStoppingCriteria constructed", "Value": octopus.get("uses_stopping_criteria")},
                {"Field": "KeywordsStoppingCriteria passed to generate", "Value": octopus.get("passes_stopping_criteria_to_generate")},
                {"Field": "Image preprocessing", "Value": octopus.get("image_preprocess")},
                {"Field": "Metric parser", "Value": octopus.get("metric_parser")},
                {"Field": "Generation monkey patch", "Value": octopus.get("monkey_patches_generation")},
                {"Field": "Octopus env torch", "Value": octopus.get("environment_torch")},
                {"Field": "Octopus env transformers", "Value": octopus.get("environment_transformers")},
            ],
        ),
        "",
        "## POPE File Comparison",
        "",
        markdown_table(
            [
                "Setting",
                "Our path",
                "Our sha256",
                "Octopus path",
                "Octopus sha256",
                "Hash same?",
                "N ours",
                "N octopus",
                "First20 same?",
                "Neg top5 ours",
                "Neg top5 octopus",
            ],
            pope_rows,
        ),
        "",
        "## Pipeline Diff",
        "",
        markdown_table(["Item", "Our HFRunner", "Octopus Official", "Same?", "Risk"], diff_rows),
        "",
        "## Existing Small Raw-Output Comparison",
        "",
    ]

    if small_rows:
        text.append(markdown_table(["Runner", "Setting", "N", "Accuracy", "Precision", "Recall", "F1", "Yes Rate", "TP", "TN", "FP", "FN", "Invalid", "Raw"], small_rows))
    else:
        text.append(
            "No side-by-side Octopus raw output was available. This script did not run inference. "
            "Pass `--octopus-answer-file` and `--octopus-gt-file` after running an Octopus Regular smoke test to fill this table."
        )

    text.extend(
        [
            "",
            "## Search Hits",
            "",
            "Full search hits are written beside this report.",
            "",
            "Matched files by hit count:",
            "",
            markdown_table(["file", "hits"], search_file_rows),
            "",
            f"Showing {len(search_rows)} high-signal/core hits below.",
            "",
            markdown_table(["file", "line", "keywords", "text"], search_rows),
            "",
            *raw_snippets,
            "",
            "## Most Likely Explanations For Higher Our Baseline",
            "",
            "1. Runner/model stack mismatch is currently the highest-risk difference: our HF runner uses `LlavaForConditionalGeneration`, while Octopus uses the official LLaVA loader plus its own generation monkey patch infrastructure.",
            "2. Decode/prompt mismatch is high-risk: Octopus code uses `do_sample=True`, `max_new_tokens=1024`, `top_p=1`, and the suffix `Please answer this question with one word.`, while our formal runner uses greedy short generation with `max_new_tokens=5` and `in one word`.",
            "3. POPE file mismatch remains high-risk until hashes are proven identical on the cloud machine. The local Octopus checkout may not include `data/POPE`, so the report must compare against `/home/huiwei/sy/Octopus-master/data/POPE` if present.",
            "",
            "## Next Minimal Verification Steps",
            "",
            "1. On the cloud, run this script first and inspect whether Octopus `data/POPE/coco_pope_*.json` exists and whether hashes match our three files.",
            "2. If hashes match, run only MSCOCO adversarial first 50 with Octopus `object_hallucination_vqa_llava.py` Regular, but ensure raw outputs are real `Yes`/`No` before computing metrics.",
            "3. If Octopus small Regular is closer to 79.77/78.47, align our runner one variable at a time: official LLaVA loader, then prompt suffix, then decode length/sampling, then parser.",
            "4. If Octopus small Regular is also high, the likely gap is POPE file/version or paper-side setting not represented in the released code.",
            "",
            "## Our Config",
            "",
            code_block(json.dumps(our_config, indent=2, ensure_ascii=False), "json"),
        ]
    )

    output.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    report_path = output_dir / "OCTOPUS_COMPARE_REPORT.md"
    search_path = output_dir / "octopus_search_hits.json"
    metadata_path = output_dir / "octopus_compare_metadata.json"

    try:
        octopus_root = find_octopus_root(args.octopus_root)
        git = git_info(octopus_root)
        octopus = detect_octopus_settings(octopus_root)
        search_hits = search_octopus(octopus_root, args.max_search_hits)
        our_runs_root = resolve_path(args.our_runs_root)
        our_config = load_our_config(our_runs_root)
        pope_root = resolve_path(args.pope_root)

        pope_rows: list[dict[str, Any]] = []
        detailed_pope: dict[str, Any] = {}
        hash_statuses: list[str] = []
        for setting in SETTINGS:
            our_file = find_our_pope_file(pope_root, setting)
            oct_file = find_octopus_pope_file(octopus_root, setting)
            our_summary = summarize_pope_file(our_file) if our_file else None
            oct_summary = summarize_pope_file(oct_file) if oct_file else None
            detailed_pope[setting] = {"ours": our_summary, "octopus": oct_summary, "first20": compare_first20(our_summary, oct_summary)}
            hash_same = bool(our_summary and oct_summary and our_summary["sha256"] == oct_summary["sha256"])
            if our_summary and oct_summary:
                hash_statuses.append(f"{setting}: {'same' if hash_same else 'different'}")
            else:
                hash_statuses.append(f"{setting}: missing")
            pope_rows.append(
                {
                    "Setting": setting,
                    "Our path": our_summary["path"] if our_summary else "missing",
                    "Our sha256": our_summary["sha256"] if our_summary else OUR_POPE_HASHES.get(setting, "missing"),
                    "Octopus path": oct_summary["path"] if oct_summary else "missing",
                    "Octopus sha256": oct_summary["sha256"] if oct_summary else "missing",
                    "Hash same?": yes_no(hash_same) if our_summary and oct_summary else "unknown",
                    "N ours": our_summary["num_samples"] if our_summary else "missing",
                    "N octopus": oct_summary["num_samples"] if oct_summary else "missing",
                    "First20 same?": detailed_pope[setting]["first20"].get("same_first20"),
                    "Neg top5 ours": our_summary["negative_top30"][:5] if our_summary else "",
                    "Neg top5 octopus": oct_summary["negative_top30"][:5] if oct_summary else "",
                }
            )

        diff_rows = build_diff_rows(octopus, our_config, "; ".join(hash_statuses))
        small_rows = summarize_existing_small_outputs(args, our_runs_root)

        write_json(search_path, search_hits)
        write_json(
            metadata_path,
            {
                "octopus_root": str(octopus_root),
                "git": git,
                "octopus_settings": {k: v for k, v in octopus.items() if not str(k).endswith("snippet")},
                "pope": detailed_pope,
                "small_rows": small_rows,
            },
        )
        write_report(report_path, octopus_root, git, octopus, search_hits, our_config, pope_rows, diff_rows, small_rows)
        print(f"Wrote Octopus comparison report to {report_path}")
        print(f"Wrote search hits to {search_path}")
        print(f"Wrote metadata to {metadata_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
