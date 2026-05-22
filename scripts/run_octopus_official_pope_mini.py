"""Run a tiny Octopus-style Official LLaVA POPE alignment check.

This is a diagnostics-only script. It does not import the project's formal
POPE/CatExpert runner, does not enable steering hooks, and does not run any
Octopus method. It only uses the Official LLaVA-style loading/prompt path for
Regular baseline outputs on the first N POPE.MSCOCO adversarial samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_POPE_FILE = "/home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_adversarial.json"
DEFAULT_COCO_IMAGE_ROOT = "/home/huiwei/sy/sy_data/COCO2014/val2014"
DEFAULT_MODEL_PATH = "/home/huiwei/sy/models/llava-v1.5-7b"
DEFAULT_LLAVA_REPO = "/home/huiwei/sy/LLaVA"
DEFAULT_HF_RAW = "data/pope_cat_expert_eval/full_alpha_sweep/raw/mscoco_adversarial_regular.jsonl"
DEFAULT_OUTPUT_DIR = "data/pope_cat_expert_eval/octopus_compare"
OCTOPUS_SUFFIX = "Please answer this question with one word."


@dataclass(frozen=True)
class DecodeConfig:
    name: str
    display: str
    do_sample: bool
    temperature: float
    top_p: float
    num_beams: int
    max_new_tokens: int


DECODE_CONFIGS = (
    DecodeConfig(
        name="default_sampling",
        display="default sampling",
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        num_beams=1,
        max_new_tokens=1024,
    ),
    DecodeConfig(
        name="deterministic",
        display="deterministic",
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        max_new_tokens=5,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--llava-repo-path", default=DEFAULT_LLAVA_REPO)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--pope-file", default=DEFAULT_POPE_FILE)
    parser.add_argument("--coco-image-root", default=DEFAULT_COCO_IMAGE_ROOT)
    parser.add_argument("--hf-raw", default=DEFAULT_HF_RAW)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--prompt-suffix", default=OCTOPUS_SUFFIX)
    parser.add_argument("--decode", choices=["both", "default_sampling", "deterministic"], default="both")
    parser.add_argument(
        "--compat-new-transformers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply small generation API shims needed by older Official LLaVA under newer transformers.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "samples", "questions", "annotations"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list or JSONL file: {path}")
        return [dict(item) for item in payload if isinstance(item, dict)]
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        rows.append(item)
        return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


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


def parse_yes_no(text: Any) -> str:
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
        label = normalize_label(row.get("label", ""))
        pred = str(row.get("pred", "")).strip().lower()
        if pred not in {"yes", "no"}:
            pred = parse_yes_no(first_present(row, ("raw_output", "text", "output", "answer"), ""))
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
        "Acc": metric_div(tp + tn, len(rows)) * 100.0,
        "Precision": precision * 100.0,
        "Recall": recall * 100.0,
        "F1": f1 * 100.0,
        "Yes Rate": metric_div(pred_yes, len(rows)) * 100.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Invalid": invalid,
    }


def resolve_image_path(row: Mapping[str, Any], image_root: Path) -> str:
    candidates: list[str] = []
    for key in ("image_path", "image", "filename", "file_name", "img"):
        value = row.get(key)
        if value not in (None, ""):
            candidates.append(str(value))
    image_id = first_present(row, ("image_id", "id", "coco_id"), "")
    if image_id not in (None, ""):
        image_id_text = str(image_id)
        candidates.append(image_id_text)
        try:
            candidates.append(f"COCO_val2014_{int(image_id_text):012d}.jpg")
        except Exception:
            pass
    for candidate in candidates:
        path = Path(candidate)
        if path.is_absolute() and path.exists():
            return str(path)
        joined = image_root / candidate
        if joined.exists():
            return str(joined)
        joined_name = image_root / path.name
        if joined_name.exists():
            return str(joined_name)
    return ""


def load_pope_samples(path: Path, image_root: Path, limit: int) -> list[dict[str, Any]]:
    raw_rows = read_json_or_jsonl(path)
    samples: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows[:limit]):
        question = str(first_present(row, ("question", "text", "query", "prompt"), "")).strip()
        label = normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target"), ""))
        image_path = resolve_image_path(row, image_root)
        image_id = str(
            first_present(
                row,
                ("image_id", "id", "coco_id"),
                Path(str(first_present(row, ("image", "image_path", "filename", "file_name"), index))).stem,
            )
        )
        sample = {
            "dataset": "MSCOCO",
            "setting": "adversarial",
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
        samples.append(sample)
    if missing:
        preview = [
            {"index": item["index"], "image_id": item["image_id"], "raw_image": item["raw"].get("image")}
            for item in missing[:10]
        ]
        raise FileNotFoundError(f"Missing COCO images: {len(missing)}/{len(samples)}. First missing: {preview}")
    return samples


def import_official_llava(llava_repo_path: str) -> dict[str, Any]:
    repo = Path(llava_repo_path).expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import torch
        from PIL import Image
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
    except Exception as exc:
        raise ImportError(
            "Could not import Official LLaVA modules. Check --llava-repo-path. Original error: "
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
        "tokenizer_image_token": tokenizer_image_token,
        "load_pretrained_model": load_pretrained_model,
        "disable_torch_init": disable_torch_init,
    }


def maybe_apply_new_transformers_compat(model: Any) -> None:
    if getattr(model, "_octopus_mini_compat_applied", False):
        return
    orig_forward = model.forward

    def forward_without_cache_position(*forward_args: Any, **forward_kwargs: Any) -> Any:
        forward_kwargs.pop("cache_position", None)
        return orig_forward(*forward_args, **forward_kwargs)

    model.forward = forward_without_cache_position  # type: ignore[method-assign]
    model._validate_model_kwargs = lambda model_kwargs: None
    try:
        type(model)._validate_model_kwargs = lambda self, model_kwargs: None
    except Exception:
        pass
    setattr(model, "_octopus_mini_compat_applied", True)
    print("Applied compatibility shim: drop cache_position and skip model_kwargs validation.")


def load_model(args: argparse.Namespace, llava: Mapping[str, Any]) -> tuple[Any, Any, Any, int, str]:
    llava["disable_torch_init"]()
    model_name = llava["get_model_name_from_path"](str(args.model_path))
    print(f"model_path: {args.model_path}")
    print(f"model_name: {model_name}")
    print(f"conv_mode: {args.conv_mode}")
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
    if args.compat_new_transformers:
        maybe_apply_new_transformers_compat(model)
    model.eval()
    return tokenizer, model, image_processor, int(context_len), model_name


def make_final_question(question: str, suffix: str) -> str:
    question = str(question).strip()
    suffix = str(suffix).strip()
    if not suffix:
        return question
    if question.lower().endswith(suffix.lower()):
        return question
    return f"{question} {suffix}"


def build_prompt(question: str, model: Any, conv_mode: str, llava: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    image_token = llava["DEFAULT_IMAGE_TOKEN"]
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = llava["DEFAULT_IM_START_TOKEN"] + image_token + llava["DEFAULT_IM_END_TOKEN"]
    question_with_image = image_token + "\n" + question
    conv = llava["conv_templates"][conv_mode].copy()
    conv.append_message(conv.roles[0], question_with_image)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != llava["SeparatorStyle"].TWO else conv.sep2
    return prompt, {
        "question_with_image": question_with_image,
        "system": getattr(conv, "system", ""),
        "roles": list(conv.roles),
        "sep": getattr(conv, "sep", None),
        "sep2": getattr(conv, "sep2", None),
        "sep_style": str(getattr(conv, "sep_style", "")),
        "stop_str": stop_str,
        "mm_use_im_start_end": bool(getattr(model.config, "mm_use_im_start_end", False)),
    }


def decode_suffix(tokenizer: Any, output_ids: Any, prompt_len: int) -> tuple[str, str, int]:
    generated_ids = output_ids[0][prompt_len:]
    if generated_ids.numel() == 0:
        generated_ids = output_ids[0]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    raw_full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return raw_output, raw_full_output, int(output_ids.shape[-1])


def generate_one(
    *,
    sample: Mapping[str, Any],
    decode: DecodeConfig,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    args: argparse.Namespace,
    llava: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    torch = llava["torch"]
    image = llava["Image"].open(sample["image_path"]).convert("RGB")
    final_question = make_final_question(str(sample["question"]), str(args.prompt_suffix))
    prompt, prompt_info = build_prompt(final_question, model, str(args.conv_mode), llava)
    input_ids = llava["tokenizer_image_token"](
        prompt,
        tokenizer,
        llava["IMAGE_TOKEN_INDEX"],
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype=torch.float16)
    prompt_len = int(input_ids.shape[1])

    generate_kwargs = {
        "attention_mask": torch.ones_like(input_ids),
        "images": image_tensor,
        "do_sample": bool(decode.do_sample),
        "temperature": float(decode.temperature),
        "top_p": float(decode.top_p),
        "num_beams": int(decode.num_beams),
        "max_new_tokens": int(decode.max_new_tokens),
        "use_cache": True,
    }
    with torch.inference_mode():
        try:
            output_ids = model.generate(input_ids, **generate_kwargs)
        except Exception as exc:
            if not args.compat_new_transformers:
                raise
            message = repr(exc)
            if not any(token in message for token in ("cache_position", "attention_mask", "model_kwargs", "new_ones")):
                raise
            maybe_apply_new_transformers_compat(model)
            output_ids = model.generate(input_ids, **generate_kwargs)

    raw_output, raw_full_output, output_token_len = decode_suffix(tokenizer, output_ids, prompt_len)
    prompt_info.update(
        {
            "final_question": final_question,
            "raw_full_output": raw_full_output,
            "prompt_token_len": prompt_len,
            "output_token_len": output_token_len,
        }
    )
    return raw_output, prompt, prompt_info


def run_official_decode(
    *,
    decode: DecodeConfig,
    samples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    args: argparse.Namespace,
    llava: Mapping[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    prompt_examples: list[dict[str, Any]] = []
    for one_based_index, sample in enumerate(samples, start=1):
        raw_output, full_prompt, prompt_info = generate_one(
            sample=sample,
            decode=decode,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            args=args,
            llava=llava,
        )
        pred = parse_yes_no(raw_output)
        row = {
            "dataset": "MSCOCO",
            "setting": "adversarial",
            "runner": "OctopusOfficial",
            "decode": decode.display,
            "image_id": sample["image_id"],
            "image_path": sample["image_path"],
            "question": sample["question"],
            "final_question": prompt_info["final_question"],
            "full_prompt": full_prompt if one_based_index <= 3 else "",
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
        if one_based_index <= 3:
            prompt_examples.append(
                {
                    "decode": decode.display,
                    "index": sample["index"],
                    "original_question": sample["question"],
                    "final_question": prompt_info["final_question"],
                    "full_prompt": full_prompt,
                    "template_info": {
                        key: value
                        for key, value in prompt_info.items()
                        if key not in {"final_question", "raw_full_output"}
                    },
                }
            )
            print(f"\n===== Prompt example {decode.display} index {sample['index']} =====")
            print(full_prompt)
        if args.progress_every > 0 and one_based_index % int(args.progress_every) == 0:
            print(f"[OctopusOfficial {decode.display}] processed {one_based_index}/{len(samples)}")
    raw_path = output_dir / "raw" / f"mscoco_adversarial_octopus_{decode.name}.jsonl"
    write_jsonl(raw_path, rows)
    print(f"Wrote raw predictions: {raw_path}")
    return rows, prompt_examples


def collect_hf_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"Warning: HF raw file not found, comparison row will be empty: {path}")
        return []
    rows = read_json_or_jsonl(path)[:limit]
    normalized = []
    for row in rows:
        item = dict(row)
        item["runner"] = "OurHFRunner"
        item["decode"] = "greedy"
        item["pred"] = str(item.get("pred") or parse_yes_no(item.get("raw_output", ""))).lower()
        normalized.append(item)
    return normalized


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return "" if value is None else str(value)


def markdown_table(headers: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def write_report(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    model_name: str,
    context_len: int,
    samples: list[dict[str, Any]],
    hf_rows: list[dict[str, Any]],
    official_rows_by_decode: Mapping[str, list[dict[str, Any]]],
    prompt_examples: list[dict[str, Any]],
) -> None:
    summary_rows: list[dict[str, Any]] = []
    hf_metrics = compute_metrics(hf_rows)
    hf_row = {"Runner": "OurHFRunner", "Decode": "greedy"}
    hf_row.update(hf_metrics)
    summary_rows.append(hf_row)
    for decode in DECODE_CONFIGS:
        metrics = compute_metrics(official_rows_by_decode.get(decode.name, []))
        row = {"Runner": "OctopusOfficial", "Decode": decode.display}
        row.update(metrics)
        summary_rows.append(row)

    csv_path = output_dir / "octopus_mini_run_summary.csv"
    fields = ["Runner", "Decode", "N", "Acc", "Precision", "Recall", "F1", "Yes Rate", "TP", "TN", "FP", "FN", "Invalid"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    raw_preview_rows: list[dict[str, Any]] = []
    for decode in DECODE_CONFIGS:
        for row in official_rows_by_decode.get(decode.name, [])[:20]:
            raw_preview_rows.append(
                {
                    "Decode": decode.display,
                    "Index": len(raw_preview_rows),
                    "Label": row.get("label"),
                    "Pred": row.get("pred"),
                    "Raw Output": repr(row.get("raw_output", "")),
                }
            )

    invalid_notes = []
    for decode in DECODE_CONFIGS:
        invalid = compute_metrics(official_rows_by_decode.get(decode.name, [])).get("Invalid", 0)
        invalid_notes.append(f"- `{decode.display}` invalid count: `{invalid}`")

    headers = ["Runner", "Decode", "N", "Acc", "Precision", "Recall", "F1", "Yes Rate", "TP", "TN", "FP", "FN", "Invalid"]
    preview_headers = ["Decode", "Index", "Label", "Pred", "Raw Output"]
    lines = [
        "# Octopus Official Mini POPE Run Report",
        "",
        "Diagnostics-only Regular baseline; no CatExpert, no steering, no Octopus/VCD/M3ID/AVISC method.",
        "",
        "## Run Config",
        "",
        f"- model_path: `{args.model_path}`",
        f"- model_name: `{model_name}`",
        f"- context_len: `{context_len}`",
        f"- llava_repo_path: `{args.llava_repo_path}`",
        f"- conv_mode: `{args.conv_mode}`",
        f"- pope_file: `{args.pope_file}`",
        f"- coco_image_root: `{args.coco_image_root}`",
        f"- hf_raw: `{args.hf_raw}`",
        f"- limit: `{args.limit}`",
        f"- prompt suffix: `{args.prompt_suffix}`",
        f"- parser: first standalone `yes`/`no`; invalid counted wrong",
        "",
        "## Side-By-Side Metrics",
        "",
        markdown_table(headers, summary_rows),
        "",
        "## Invalid Counts",
        "",
        *invalid_notes,
        "",
        "## Prompt Examples",
        "",
        "```json",
        json.dumps(prompt_examples, indent=2, ensure_ascii=False),
        "```",
        "",
        "## OctopusOfficial Raw Output / Pred Preview",
        "",
        markdown_table(preview_headers, raw_preview_rows),
        "",
        "## First 5 Source Samples",
        "",
        "```json",
        json.dumps(
            [
                {
                    "index": item["index"],
                    "image_id": item["image_id"],
                    "question": item["question"],
                    "label": item["label"],
                    "image_path": item["image_path"],
                }
                for item in samples[:5]
            ],
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
        "## Interpretation Rule",
        "",
        "- If OctopusOfficial has higher `Yes Rate`/`FP` and lower Acc closer to Octopus adversarial 79.77, the gap is mainly runner/template/decode.",
        "- If OctopusOfficial is close to OurHFRunner, the gap is more likely POPE file version or checkpoint.",
        "- If OctopusOfficial has many invalid outputs, the official runner is not comparable yet; fix raw generation before reading metrics.",
    ]
    report_path = output_dir / "OCTOPUS_MINI_RUN_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config_path = output_dir / "octopus_mini_run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "model_name": model_name,
                "context_len": context_len,
                "llava_repo_path": str(args.llava_repo_path),
                "conv_mode": str(args.conv_mode),
                "pope_file": str(args.pope_file),
                "coco_image_root": str(args.coco_image_root),
                "hf_raw": str(args.hf_raw),
                "limit": int(args.limit),
                "prompt_suffix": str(args.prompt_suffix),
                "decode_configs": [decode.__dict__ for decode in DECODE_CONFIGS],
                "summary_csv": str(csv_path),
                "report": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote mini-run summary: {csv_path}")
    print(f"Wrote mini-run report: {report_path}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_path = output_dir / "OCTOPUS_MINI_RUN_REPORT.md"
    if report_path.exists() and not args.overwrite:
        print(f"Error: report already exists: {report_path}. Pass --overwrite.", file=sys.stderr)
        return 1
    try:
        llava = import_official_llava(str(args.llava_repo_path))
        tokenizer, model, image_processor, context_len, model_name = load_model(args, llava)
        samples = load_pope_samples(Path(args.pope_file), Path(args.coco_image_root), int(args.limit))
        hf_rows = collect_hf_rows(Path(args.hf_raw), int(args.limit))
        official_rows_by_decode: dict[str, list[dict[str, Any]]] = {}
        prompt_examples: list[dict[str, Any]] = []
        selected_decodes = [
            d for d in DECODE_CONFIGS
            if str(args.decode) == "both" or d.name == str(args.decode)
        ]
        for decode in DECODE_CONFIGS:
            if decode not in selected_decodes:
                continue
            rows, examples = run_official_decode(
                decode=decode,
                samples=samples,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                args=args,
                llava=llava,
                output_dir=output_dir,
            )
            official_rows_by_decode[decode.name] = rows
            prompt_examples.extend(examples)
        write_report(
            args=args,
            output_dir=output_dir,
            model_name=model_name,
            context_len=context_len,
            samples=samples,
            hf_rows=hf_rows,
            official_rows_by_decode=official_rows_by_decode,
            prompt_examples=prompt_examples,
        )
    except Exception as exc:
        if os.environ.get("POPE_MINI_TRACEBACK", "1").strip().lower() not in {"0", "false", "no"}:
            traceback.print_exc()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
