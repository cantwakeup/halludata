"""Compare baseline vs steered first-token Yes/No logit margins on POPE-style data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import write_json, write_jsonl
from expert_data.steering import (
    ExpertSteeringController,
    build_llava_prefix_prompt,
    normalize_bool,
    parse_csv_items,
)

YES_WORDS = {"yes", "y", "true", "1"}
NO_WORDS = {"no", "n", "false", "0"}
YES_CANDIDATES = ("Yes", " yes", "YES", " yes")
NO_CANDIDATES = ("No", " no", "NO", " no")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for first-token margin debugging."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF LLaVA model ID or local model path.")
    parser.add_argument("--pope-file", required=True, help="POPE JSONL/line-delimited JSON file.")
    parser.add_argument("--image-root", required=True, help="Root directory containing benchmark images.")
    parser.add_argument("--output", default="data/outputs/debug/pope_margin_debug.jsonl")
    parser.add_argument("--limit", type=int, default=100, help="0 means all rows.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--trust-remote-code", default="false")

    parser.add_argument("--steer-vector-path", required=True)
    parser.add_argument("--steer-layers", default="10-20")
    parser.add_argument("--steer-router", choices=["no_filter", "force_cat", "force_attr", "force_rel", "rule"], default="force_cat")
    parser.add_argument("--steer-enabled-experts", default="cat")
    parser.add_argument("--steer-k-heads", type=int, default=64)
    parser.add_argument("--steer-head-select", choices=["norm", "random", "all", "expert_map"], default="norm")
    parser.add_argument("--steer-head-map", default="", help="Head-map JSON for --steer-head-select expert_map.")
    parser.add_argument("--steer-expert-key", default="", help="Vector/head-map expert key for expert_map steering.")
    parser.add_argument("--steer-alpha", type=float, default=4.0)
    parser.add_argument("--steer-prefill", default="true")
    parser.add_argument("--steer-decode", default="false")
    parser.add_argument("--prefill-apply-to", choices=["last_token", "all_tokens"], default="last_token")
    parser.add_argument("--decode-apply-to", choices=["last_token"], default="last_token")
    parser.add_argument("--debug-log-hook-delta", default="false")
    parser.add_argument("--dataset", default="", help="Optional dataset split label stored in output rows.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL, line-delimited `.json`, or JSON-list benchmark rows."""

    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object on line {line_number} of {input_path}")
                rows.append(row)
        return rows
    if isinstance(payload, dict):
        for key in ("data", "samples", "questions", "annotations"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list or line-delimited JSON at {input_path}")
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at index {index} of {input_path}")
        rows.append(row)
    return rows


def first_present(row: Mapping[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    """Return the first non-empty field value under candidate keys."""

    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def normalize_label(value: Any) -> str | None:
    """Normalize common yes/no labels to `yes` or `no`."""

    text = str(value).strip().lower()
    if text in YES_WORDS or text.startswith("yes"):
        return "yes"
    if text in NO_WORDS or text.startswith("no"):
        return "no"
    return None


def normalize_sample(row: Mapping[str, Any], index: int, image_root: Path) -> dict[str, Any]:
    """Normalize one POPE-style row and resolve its image path."""

    question = first_present(row, ("question", "query", "prompt", "text"))
    image_name = first_present(row, ("image_path", "image", "img", "file_name"))
    label = normalize_label(first_present(row, ("answer", "label", "gt_answer", "ground_truth", "target")))
    if not question:
        raise ValueError(f"row {index} is missing a question/text field")
    if not image_name:
        raise ValueError(f"row {index} is missing an image field")
    image_path = Path(str(image_name))
    if not image_path.is_absolute():
        image_path = image_root / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image for row {index} does not exist: {image_path}")
    return {
        "sample_id": str(first_present(row, ("sample_id", "question_id", "id"), index)),
        "image": str(image_name),
        "image_path": str(image_path),
        "question": str(question),
        "label": label,
    }


def resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
    """Resolve a dtype string to a torch dtype."""

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[str(dtype_name).lower()]


def load_llava(model_path: str, device: str, compute_dtype: str, trust_remote_code: bool) -> tuple[Any, Any, Any, Any]:
    """Load torch, PIL, processor, and model for LLaVA first-token scoring."""

    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor

        try:
            from transformers import LlavaForConditionalGeneration
        except ImportError:
            LlavaForConditionalGeneration = None
    except Exception as exc:
        raise RuntimeError("debug_first_token_margin requires torch, transformers, and Pillow.") from exc

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_kwargs = {
        "torch_dtype": resolve_torch_dtype(torch, compute_dtype),
        "trust_remote_code": trust_remote_code,
    }
    model_classes = [candidate for candidate in (LlavaForConditionalGeneration, AutoModelForVision2Seq) if candidate]
    last_error: Exception | None = None
    for model_class in model_classes:
        try:
            model = model_class.from_pretrained(model_path, **model_kwargs)
            break
        except Exception as exc:  # pragma: no cover - depends on cloud model availability.
            last_error = exc
    else:
        raise RuntimeError(f"Failed to load LLaVA model: {model_path}") from last_error
    model.to(device)
    model.eval()
    return torch, Image, processor, model


def inputs_to_device(inputs: Mapping[str, Any], torch: Any, device: str, compute_dtype: Any) -> dict[str, Any]:
    """Move processor outputs to the target device."""

    moved: dict[str, Any] = {}
    del torch
    for key, value in dict(inputs).items():
        if hasattr(value, "to"):
            if key == "pixel_values" and getattr(value, "is_floating_point", lambda: False)():
                moved[key] = value.to(device=device, dtype=compute_dtype)
            else:
                moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def single_token_ids(tokenizer: Any, candidates: Iterable[str]) -> list[int]:
    """Return unique token IDs for candidates that encode as exactly one token."""

    token_ids: list[int] = []
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1 and int(ids[0]) not in token_ids:
            token_ids.append(int(ids[0]))
    if not token_ids:
        raise RuntimeError(f"No single-token IDs found for candidates: {list(candidates)}")
    return token_ids


def first_token_logits(
    sample: Mapping[str, Any],
    *,
    torch: Any,
    Image: Any,
    processor: Any,
    model: Any,
    device: str,
    compute_dtype: Any,
) -> Any:
    """Run one prompt-only forward and return logits for the first generated token."""

    image = Image.open(sample["image_path"]).convert("RGB")
    prompt = build_llava_prefix_prompt(str(sample["question"]))
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = inputs_to_device(inputs, torch, device, compute_dtype)
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
    prompt_len = int(inputs["input_ids"].shape[1])
    return outputs.logits[0, prompt_len - 1, :].detach().float().cpu()


def yes_no_margin(logits: Any, yes_ids: list[int], no_ids: list[int]) -> dict[str, Any]:
    """Compute max yes/no logits and their margin from first-token logits."""

    yes_logit = float(logits[yes_ids].max().item())
    no_logit = float(logits[no_ids].max().item())
    margin = yes_logit - no_logit
    return {
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "margin": margin,
        "prediction": "yes" if margin >= 0.0 else "no",
    }


def summarize(rows: list[dict[str, Any]], controller_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Summarize first-token margin changes and steering hook diagnostics."""

    labeled = [row for row in rows if row.get("label") in {"yes", "no"}]
    flips = [row for row in rows if row["baseline_logit_pred"] != row["steered_logit_pred"]]
    wrong_to_right = [
        row for row in labeled
        if not row["is_baseline_correct"] and row["is_steered_correct"]
    ]
    right_to_wrong = [
        row for row in labeled
        if row["is_baseline_correct"] and not row["is_steered_correct"]
    ]
    label_yes = [row["delta_margin"] for row in rows if row.get("label") == "yes"]
    label_no = [row["delta_margin"] for row in rows if row.get("label") == "no"]
    return {
        "n": len(rows),
        "steering_mode": "fixed_positive",
        "alpha": float(args.steer_alpha),
        "steer_prefill": normalize_bool(args.steer_prefill),
        "steer_decode": normalize_bool(args.steer_decode),
        "avg_delta_margin_all": mean([row["delta_margin"] for row in rows] or [0.0]),
        "avg_delta_margin_label_yes": mean(label_yes or [0.0]),
        "avg_delta_margin_label_no": mean(label_no or [0.0]),
        "baseline_logit_acc": (
            sum(1 for row in labeled if row["is_baseline_correct"]) / len(labeled)
            if labeled else None
        ),
        "steered_logit_acc": (
            sum(1 for row in labeled if row["is_steered_correct"]) / len(labeled)
            if labeled else None
        ),
        "flip_count": len(flips),
        "wrong_to_right": len(wrong_to_right),
        "right_to_wrong": len(right_to_wrong),
        "prefill_hook_call_count": controller_summary.get("prefill_hook_call_count", 0),
        "decode_hook_call_count": controller_summary.get("decode_hook_call_count", 0),
        "prefill_edited_token_count": controller_summary.get("prefill_edited_token_count", 0),
        "decode_edited_token_count": controller_summary.get("decode_edited_token_count", 0),
        "steering_diagnostics": controller_summary,
    }


def main() -> int:
    """Run first-token margin debugging."""

    args = parse_args()
    try:
        output_path = resolve_project_path(args.output)
        image_root = resolve_project_path(args.image_root)
        rows = read_json_rows(resolve_project_path(args.pope_file))
        samples = [normalize_sample(row, index, image_root) for index, row in enumerate(rows)]
        if int(args.limit) > 0:
            samples = samples[: int(args.limit)]
        torch, Image, processor, model = load_llava(
            args.model_path,
            args.device,
            args.compute_dtype,
            normalize_bool(args.trust_remote_code),
        )
        compute_dtype = resolve_torch_dtype(torch, args.compute_dtype)
        tokenizer = getattr(processor, "tokenizer", processor)
        yes_ids = single_token_ids(tokenizer, YES_CANDIDATES)
        no_ids = single_token_ids(tokenizer, NO_CANDIDATES)
        controller = ExpertSteeringController(
            model,
            resolve_project_path(args.steer_vector_path),
            layers=args.steer_layers,
            alpha=float(args.steer_alpha),
            k_heads=int(args.steer_k_heads),
            head_select=str(args.steer_head_select),
            head_map_path=resolve_project_path(args.steer_head_map) if str(args.steer_head_map).strip() else None,
            expert_key=str(args.steer_expert_key).strip() or None,
            router=str(args.steer_router),
            enabled_experts=tuple(parse_csv_items(args.steer_enabled_experts)),
            steer_prefill=normalize_bool(args.steer_prefill),
            steer_decode=normalize_bool(args.steer_decode),
            prefill_apply_to=str(args.prefill_apply_to),
            decode_apply_to=str(args.decode_apply_to),
            debug_log_hook_delta=normalize_bool(args.debug_log_hook_delta),
        )

        output_rows: list[dict[str, Any]] = []
        dataset = str(args.dataset).strip() or Path(args.pope_file).stem.replace("coco_pope_", "")
        for index, sample in enumerate(samples, start=1):
            controller.set_context(str(sample["question"]))
            controller.disable()
            baseline_logits = first_token_logits(
                sample,
                torch=torch,
                Image=Image,
                processor=processor,
                model=model,
                device=args.device,
                compute_dtype=compute_dtype,
            )
            baseline = yes_no_margin(baseline_logits, yes_ids, no_ids)
            controller.set_sign(1.0)
            controller.enable()
            steered_logits = first_token_logits(
                sample,
                torch=torch,
                Image=Image,
                processor=processor,
                model=model,
                device=args.device,
                compute_dtype=compute_dtype,
            )
            controller.disable()
            steered = yes_no_margin(steered_logits, yes_ids, no_ids)
            label = sample.get("label")
            output_rows.append(
                {
                    "dataset": dataset,
                    "image": sample["image"],
                    "question": sample["question"],
                    "label": label,
                    "baseline_yes_logit": baseline["yes_logit"],
                    "baseline_no_logit": baseline["no_logit"],
                    "baseline_margin": baseline["margin"],
                    "baseline_logit_pred": baseline["prediction"],
                    "steered_yes_logit": steered["yes_logit"],
                    "steered_no_logit": steered["no_logit"],
                    "steered_margin": steered["margin"],
                    "steered_logit_pred": steered["prediction"],
                    "delta_margin": steered["margin"] - baseline["margin"],
                    "baseline_correct": baseline["prediction"] == label if label in {"yes", "no"} else None,
                    "steered_correct": steered["prediction"] == label if label in {"yes", "no"} else None,
                    "is_baseline_correct": baseline["prediction"] == label if label in {"yes", "no"} else None,
                    "is_steered_correct": steered["prediction"] == label if label in {"yes", "no"} else None,
                }
            )
            if int(args.progress_every) > 0 and index % int(args.progress_every) == 0:
                print(f"[first-token-margin] processed {index}/{len(samples)} samples")

        write_jsonl(output_path, output_rows)
        summary_path = output_path.with_suffix(".summary.json")
        write_json(summary_path, summarize(output_rows, controller.summary(), args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote first-token margin rows to {output_path}")
    print(f"Wrote first-token margin summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
