"""Official-LLaVA POPE Regular-only diagnostic.

This script intentionally does not import or modify the repository's HF POPE
runner. It uses the official LLaVA loading/conversation utilities to run a
small Regular baseline on the same POPE annotation files, then compares the
result with the existing HF-runner raw predictions on the same first N rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from expert_data.activation_cache import read_jsonl, write_json, write_jsonl
except Exception:  # pragma: no cover - keeps --help usable in bare envs
    read_jsonl = None  # type: ignore[assignment]
    write_json = None  # type: ignore[assignment]
    write_jsonl = None  # type: ignore[assignment]


POPE_FILES = {
    ("MSCOCO", "random"): "/home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json",
    ("MSCOCO", "popular"): "/home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_popular.json",
    ("MSCOCO", "adversarial"): "/home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_adversarial.json",
    ("GQA", "random"): "/home/huiwei/sy/benchmarks/POPE/output/seem/gqa/gqa_pope_seem_random.json",
    ("GQA", "popular"): "/home/huiwei/sy/benchmarks/POPE/output/seem/gqa/gqa_pope_seem_popular.json",
    ("GQA", "adversarial"): "/home/huiwei/sy/benchmarks/POPE/output/seem/gqa/gqa_pope_seem_adversarial.json",
}

HF_RAW_NAMES = {
    ("MSCOCO", "random"): "mscoco_random_regular.jsonl",
    ("MSCOCO", "popular"): "mscoco_popular_regular.jsonl",
    ("MSCOCO", "adversarial"): "mscoco_adversarial_regular.jsonl",
    ("GQA", "random"): "gqa_random_regular.jsonl",
    ("GQA", "popular"): "gqa_popular_regular.jsonl",
    ("GQA", "adversarial"): "gqa_adversarial_regular.jsonl",
}

DATASETS = ("MSCOCO", "GQA")
SETTINGS = ("random", "popular", "adversarial")
DEFAULT_PROMPT_SUFFIX = "Please answer this question in one word."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Official LLaVA-v1.5-7B model path or HF id.")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--llava-repo-path", default="", help="Optional path to official LLaVA repo to prepend to sys.path.")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    parser.add_argument("--coco-image-root", default="/home/huiwei/sy/sy_data/COCO2014/val2014")
    parser.add_argument("--gqa-image-root", default="/home/huiwei/sy/sy_data/GQA/raw/images/images")
    parser.add_argument("--hf-runs-root", default="data/pope_cat_expert_eval/full_alpha_sweep")
    parser.add_argument("--output-dir", default="data/pope_cat_expert_eval/official_llava_diagnostics")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--drop-cache-position",
        action="store_true",
        help="Compatibility patch for old official LLaVA code under newer transformers generation APIs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("data", "samples", "questions", "annotations"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list or JSONL file: {path}")
        return [dict(item) for item in payload if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def read_jsonl_fallback(path: Path) -> list[dict[str, Any]]:
    if read_jsonl is not None:
        return list(read_jsonl(path))
    return read_json_or_jsonl(path)


def write_jsonl_fallback(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_jsonl is not None:
        write_jsonl(path, rows)
        return
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json_fallback(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_json is not None:
        write_json(path, payload)
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    raise ValueError(f"Could not normalize label: {value!r}")


def parse_first_yes_no(text: Any) -> str:
    matches = [(match.start(), match.group(1)) for match in re.finditer(r"\b(yes|no)\b", str(text).lower())]
    if not matches:
        return "invalid"
    return sorted(matches)[0][1]


def metric_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = invalid = pred_yes = 0
    rows = list(rows)
    for row in rows:
        label = str(row.get("label", "")).lower()
        pred = str(row.get("pred", "invalid")).lower()
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
    return {
        "N": len(rows),
        "Accuracy": metric_div(tp + tn, len(rows)) * 100.0,
        "Precision": precision * 100.0,
        "Recall": recall * 100.0,
        "F1 Score": f1 * 100.0,
        "Yes Rate": metric_div(pred_yes, len(rows)) * 100.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Invalid": invalid,
    }


def format_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return "" if value is None else str(value)


def markdown_table(headers: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_float(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def make_question(question: str, suffix: str) -> str:
    question = str(question).strip()
    suffix = str(suffix).strip()
    if not suffix:
        return question
    if question.lower().endswith(suffix.lower()):
        return question
    return f"{question} {suffix}"


def resolve_image_path(dataset: str, row: Mapping[str, Any], image_root: Path) -> str:
    candidates = []
    for key in ("image_path", "image", "filename", "file_name", "img"):
        value = row.get(key)
        if value not in (None, ""):
            candidates.append(str(value))
    image_id = first_present(row, ("image_id", "id", "coco_id"), "")
    if image_id not in (None, ""):
        image_id_text = str(image_id)
        candidates.append(image_id_text)
        if dataset == "MSCOCO":
            try:
                candidates.append(f"COCO_val2014_{int(image_id_text):012d}.jpg")
            except Exception:
                pass
        if dataset == "GQA":
            candidates.extend([f"{image_id_text}.jpg", f"{image_id_text}.png"])
    for candidate in candidates:
        path = Path(candidate)
        if path.is_absolute() and path.exists():
            return str(path)
        joined = image_root / candidate
        if joined.exists():
            return str(joined)
        if path.name != candidate:
            joined_name = image_root / path.name
            if joined_name.exists():
                return str(joined_name)
    return ""


def load_samples(args: argparse.Namespace) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    manifest = {}
    for dataset in args.datasets:
        image_root = Path(args.coco_image_root if dataset == "MSCOCO" else args.gqa_image_root)
        for setting in args.settings:
            path = Path(POPE_FILES[(dataset, setting)])
            if not path.exists():
                raise FileNotFoundError(f"Missing POPE file for {dataset} {setting}: {path}")
            raw_rows = read_json_or_jsonl(path)
            rows = []
            missing = []
            for index, row in enumerate(raw_rows[: int(args.limit)]):
                question = str(first_present(row, ("question", "text", "query", "prompt"), "")).strip()
                label = normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target"), ""))
                image_path = resolve_image_path(dataset, row, image_root)
                image_id = str(first_present(row, ("image_id", "id", "coco_id"), Path(str(first_present(row, ("image", "image_path", "filename", "file_name"), index))).stem))
                sample = {
                    "dataset": dataset,
                    "setting": setting,
                    "index": index,
                    "image_id": image_id,
                    "image_path": image_path,
                    "question": question,
                    "label": label,
                    "source_file": str(path),
                    "raw": dict(row),
                }
                if not image_path:
                    missing.append(sample)
                rows.append(sample)
            if missing:
                preview = [{"index": item["index"], "image_id": item["image_id"], "raw_image": item["raw"].get("image")} for item in missing[:10]]
                raise FileNotFoundError(f"Missing images for {dataset} {setting}: {len(missing)}/{len(rows)}. First missing: {preview}")
            groups[(dataset, setting)] = rows
            manifest[f"{dataset}_{setting}"] = {
                "source_file": str(path),
                "image_root": str(image_root),
                "num_samples": len(rows),
                "label_counts": dict(Counter(item["label"] for item in rows)),
            }
    return groups, manifest


def import_official_llava(llava_repo_path: str) -> dict[str, Any]:
    if llava_repo_path:
        sys.path.insert(0, str(Path(llava_repo_path).expanduser().resolve()))
    try:
        import torch
        from PIL import Image
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
    except Exception as exc:
        raise ImportError(
            "Could not import official LLaVA modules. Install/activate the official LLaVA repo "
            "or pass --llava-repo-path. Original error: "
            + repr(exc)
        ) from exc
    return {
        "torch": torch,
        "Image": Image,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
        "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
        "SeparatorStyle": SeparatorStyle,
        "conv_templates": conv_templates,
        "get_model_name_from_path": get_model_name_from_path,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "load_pretrained_model": load_pretrained_model,
        "disable_torch_init": disable_torch_init,
    }


def build_official_prompt(
    *,
    question: str,
    model: Any,
    conv_mode: str,
    llava: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    default_image_token = llava["DEFAULT_IMAGE_TOKEN"]
    image_token_se = llava["DEFAULT_IM_START_TOKEN"] + llava["DEFAULT_IMAGE_TOKEN"] + llava["DEFAULT_IM_END_TOKEN"]
    if getattr(model.config, "mm_use_im_start_end", False):
        question_with_image = image_token_se + "\n" + question
    else:
        question_with_image = default_image_token + "\n" + question

    conv = llava["conv_templates"][conv_mode].copy()
    conv.append_message(conv.roles[0], question_with_image)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != llava["SeparatorStyle"].TWO else conv.sep2
    template_info = {
        "conv_mode": conv_mode,
        "roles": list(conv.roles),
        "sep": getattr(conv, "sep", None),
        "sep2": getattr(conv, "sep2", None),
        "sep_style": str(getattr(conv, "sep_style", "")),
        "system": getattr(conv, "system", ""),
        "stop_str": stop_str,
        "mm_use_im_start_end": bool(getattr(model.config, "mm_use_im_start_end", False)),
    }
    return prompt, question_with_image, template_info


def generate_one(
    *,
    sample: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    args: argparse.Namespace,
    llava: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    torch = llava["torch"]
    image = llava["Image"].open(sample["image_path"]).convert("RGB")
    final_question = make_question(str(sample["question"]), str(args.prompt_suffix))
    prompt, question_with_image, template_info = build_official_prompt(
        question=final_question,
        model=model,
        conv_mode=str(args.conv_mode),
        llava=llava,
    )
    input_ids = llava["tokenizer_image_token"](
        prompt,
        tokenizer,
        llava["IMAGE_TOKEN_INDEX"],
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    image_tensor = llava["process_images"]([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [tensor.to(model.device, dtype=torch.float16) for tensor in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            num_beams=int(args.num_beams),
            max_new_tokens=int(args.max_new_tokens),
            use_cache=True,
        )
    # Official LLaVA/Transformers versions differ on whether `generate`
    # returns prompt+completion or only completion. Decode the suffix first,
    # which is what the HF runner does and what POPE parsing expects.
    prompt_len = int(input_ids.shape[1])
    generated_ids = output_ids[0][prompt_len:]
    if generated_ids.numel() == 0:
        generated_ids = output_ids[0]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    raw_full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return raw_output, prompt, {
        "final_question": final_question,
        "question_with_image": question_with_image,
        "raw_full_output": raw_full_output,
        "prompt_token_len": prompt_len,
        "output_token_len": int(output_ids.shape[-1]),
        **template_info,
    }


def raw_output_path(output_dir: Path, dataset: str, setting: str) -> Path:
    dataset_key = "mscoco" if dataset == "MSCOCO" else "gqa"
    return output_dir / "raw" / f"{dataset_key}_{setting}_official_regular.jsonl"


def hf_raw_path(hf_runs_root: Path, dataset: str, setting: str) -> Path:
    return hf_runs_root / "raw" / HF_RAW_NAMES[(dataset, setting)]


def run_official_eval(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    random.seed(int(args.seed))
    llava = import_official_llava(str(args.llava_repo_path))
    llava["disable_torch_init"]()
    model_name = llava["get_model_name_from_path"](str(args.model_path))
    print(f"Official LLaVA model path: {args.model_path}")
    print(f"Official LLaVA model name: {model_name}")
    print(f"Conversation mode: {args.conv_mode}")
    model_path = Path(str(args.model_path))
    if model_path.exists():
        interesting = [
            "config.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        ]
        found = {name: (model_path / name).exists() for name in interesting}
        print(f"Model path exists. Key files: {json.dumps(found, ensure_ascii=False)}")
    else:
        print("Model path does not exist locally; load_pretrained_model will treat it as a model id.")
    try:
        tokenizer, model, image_processor, context_len = llava["load_pretrained_model"](
            str(args.model_path),
            args.model_base,
            model_name,
            device=str(args.device),
        )
    except TypeError:
        tokenizer, model, image_processor, context_len = llava["load_pretrained_model"](
            str(args.model_path),
            args.model_base,
            model_name,
        )
    if args.drop_cache_position:
        orig_forward = model.forward

        def forward_without_cache_position(*forward_args: Any, **forward_kwargs: Any) -> Any:
            forward_kwargs.pop("cache_position", None)
            return orig_forward(*forward_args, **forward_kwargs)

        model.forward = forward_without_cache_position  # type: ignore[method-assign]
        # force_skip_generation_kwargs_validation
        model._validate_model_kwargs = lambda model_kwargs: None
        try:
            type(model)._validate_model_kwargs = lambda self, model_kwargs: None
        except Exception:
            pass
        print("Applied compatibility patch: skipping transformers model_kwargs validation.")
        print("Applied compatibility patch: dropping cache_position before model.forward().")
    model.eval()
    samples_by_group, manifest = load_samples(args)
    output_dir = Path(args.output_dir)
    rows_all = []
    prompt_examples = []

    for (dataset, setting), samples in samples_by_group.items():
        rows = []
        out_path = raw_output_path(output_dir, dataset, setting)
        for index, sample in enumerate(samples, start=1):
            raw_output, full_prompt, prompt_info = generate_one(
                sample=sample,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                args=args,
                llava=llava,
            )
            pred = parse_first_yes_no(raw_output)
            row = {
                "dataset": dataset,
                "setting": setting,
                "method": "OfficialLLaVA-Regular",
                "image_id": sample["image_id"],
                "image_path": sample["image_path"],
                "question": sample["question"],
                "final_question": prompt_info["final_question"],
                "full_prompt": full_prompt if index <= 3 else "",
                "label": sample["label"],
                "raw_output": raw_output,
                "raw_full_output": prompt_info.get("raw_full_output", ""),
                "pred": pred,
                "is_correct": bool(pred == sample["label"]),
                "source_file": sample["source_file"],
                "prompt_token_len": prompt_info.get("prompt_token_len", ""),
                "output_token_len": prompt_info.get("output_token_len", ""),
            }
            rows.append(row)
            rows_all.append(row)
            if index <= 3:
                prompt_examples.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "index": sample["index"],
                        "original_question": sample["question"],
                        "final_question": prompt_info["final_question"],
                        "question_with_image": prompt_info["question_with_image"],
                        "full_prompt": full_prompt,
                        "template_info": {key: value for key, value in prompt_info.items() if key not in {"final_question", "question_with_image"}},
                    }
                )
                print("\n===== Prompt example", dataset, setting, "index", sample["index"], "=====")
                print(full_prompt)
            if int(args.progress_every) > 0 and index % int(args.progress_every) == 0:
                print(f"[official {dataset} {setting}] processed {index}/{len(samples)}")
        write_jsonl_fallback(out_path, rows)
        print(f"Wrote official raw predictions to {out_path}")

    run_config = {
        "model_path": str(args.model_path),
        "model_base": args.model_base,
        "model_name": model_name,
        "context_len": context_len,
        "conv_mode": str(args.conv_mode),
        "limit": int(args.limit),
        "prompt_suffix": str(args.prompt_suffix),
        "decode": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "do_sample": False,
            "num_beams": int(args.num_beams),
            "max_new_tokens": int(args.max_new_tokens),
            "use_cache": True,
        },
        "pope_manifest": manifest,
        "prompt_examples": prompt_examples,
        "official_imports": [
            "from llava.model.builder import load_pretrained_model",
            "from llava.conversation import conv_templates",
            "from llava.mm_utils import tokenizer_image_token, process_images",
        ],
    }
    write_json_fallback(output_dir / "config.json", run_config)
    return rows_all, run_config, prompt_examples


def collect_hf_rows(args: argparse.Namespace) -> dict[tuple[str, str], list[dict[str, Any]]]:
    root = Path(args.hf_runs_root)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for dataset in args.datasets:
        for setting in args.settings:
            path = hf_raw_path(root, dataset, setting)
            if not path.exists():
                groups[(dataset, setting)] = []
                continue
            rows = read_jsonl_fallback(path)[: int(args.limit)]
            groups[(dataset, setting)] = rows
    return groups


def write_summary(
    *,
    args: argparse.Namespace,
    official_rows: list[dict[str, Any]],
    hf_groups: Mapping[tuple[str, str], list[dict[str, Any]]],
    prompt_examples: list[dict[str, Any]],
) -> None:
    output_dir = Path(args.output_dir)
    groups_official: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in official_rows:
        groups_official.setdefault((row["dataset"], row["setting"]), []).append(row)

    rows_summary = []
    for dataset in args.datasets:
        for setting in args.settings:
            official_metrics = compute_metrics(groups_official.get((dataset, setting), []))
            hf_metrics = compute_metrics(hf_groups.get((dataset, setting), []))
            for method, metrics in (("HFRunner-Regular", hf_metrics), ("OfficialLLaVA-Regular", official_metrics)):
                row = {"Dataset": dataset, "Setting": setting, "Method": method}
                row.update(metrics)
                rows_summary.append(row)
            delta_row = {
                "Dataset": dataset,
                "Setting": setting,
                "Method": "Official-minus-HF",
                "N": official_metrics.get("N", 0),
                "Accuracy": official_metrics.get("Accuracy", 0.0) - hf_metrics.get("Accuracy", 0.0),
                "Precision": official_metrics.get("Precision", 0.0) - hf_metrics.get("Precision", 0.0),
                "Recall": official_metrics.get("Recall", 0.0) - hf_metrics.get("Recall", 0.0),
                "F1 Score": official_metrics.get("F1 Score", 0.0) - hf_metrics.get("F1 Score", 0.0),
                "Yes Rate": official_metrics.get("Yes Rate", 0.0) - hf_metrics.get("Yes Rate", 0.0),
                "TP": official_metrics.get("TP", 0) - hf_metrics.get("TP", 0),
                "TN": official_metrics.get("TN", 0) - hf_metrics.get("TN", 0),
                "FP": official_metrics.get("FP", 0) - hf_metrics.get("FP", 0),
                "FN": official_metrics.get("FN", 0) - hf_metrics.get("FN", 0),
                "Invalid": official_metrics.get("Invalid", 0) - hf_metrics.get("Invalid", 0),
            }
            rows_summary.append(delta_row)

    csv_path = output_dir / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Dataset", "Setting", "Method", "N", "Accuracy", "Precision", "Recall", "F1 Score", "Yes Rate", "TP", "TN", "FP", "FN", "Invalid"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows_summary:
            writer.writerow({field: row.get(field, "") for field in fields})

    report_path = output_dir / "SUMMARY.md"
    main_headers = ["Dataset", "Setting", "Method", "N", "Accuracy", "Precision", "Recall", "F1 Score", "Yes Rate"]
    debug_headers = ["Dataset", "Setting", "Method", "TP", "TN", "FP", "FN", "Invalid"]
    lines = [
        "# Official LLaVA POPE Regular Diagnostic",
        "",
        f"- Summary CSV: `{csv_path}`",
        f"- Limit per dataset/setting: `{args.limit}`",
        f"- HF comparison root: `{args.hf_runs_root}`",
        f"- Prompt suffix: `{args.prompt_suffix}`",
        f"- Conversation mode: `{args.conv_mode}`",
        "- Steering/hooks: disabled; Regular baseline only.",
        "- Decode: `temperature=0`, `top_p=1.0`, `do_sample=False`, `num_beams=1`, `max_new_tokens=5` unless overridden.",
        "",
        "## Main Comparison",
        "",
        markdown_table(main_headers, rows_summary),
        "",
        "## Debug Counts",
        "",
        markdown_table(debug_headers, rows_summary),
        "",
        "## Prompt Examples",
        "",
        "```json",
        json.dumps(prompt_examples, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Interpretation Hint",
        "",
        "- If `OfficialLLaVA-Regular` has higher `FP` and higher `Yes Rate` than `HFRunner-Regular`, the previous high baseline likely comes from the HF loader/processor or prompt wrapper being more conservative.",
        "- If `OfficialLLaVA-Regular` is similar to `HFRunner-Regular`, the main gap is more likely POPE annotation version / negative sampling, model checkpoint, or parser rather than conversation template alone.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote official LLaVA POPE summary to {csv_path}")
    print(f"Wrote official LLaVA POPE report to {report_path}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    try:
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite.")
        official_rows, _config, prompt_examples = run_official_eval(args)
        hf_groups = collect_hf_rows(args)
        write_summary(args=args, official_rows=official_rows, hf_groups=hf_groups, prompt_examples=prompt_examples)
    except Exception as exc:
        if os.environ.get("POPE_DIAG_TRACEBACK", "1").strip().lower() not in {"0", "false", "no"}:
            traceback.print_exc()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
