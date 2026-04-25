"""Run a one-sample sanity check that steering hooks can move first-token logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from debug_first_token_margin import (  # noqa: E402
    NO_CANDIDATES,
    YES_CANDIDATES,
    first_token_logits,
    inputs_to_device,
    load_llava,
    normalize_sample,
    read_json_rows,
    resolve_project_path,
    resolve_torch_dtype,
    single_token_ids,
    yes_no_margin,
)
from expert_data.activation_cache import write_json  # noqa: E402
from expert_data.steering import (  # noqa: E402
    ExpertSteeringController,
    build_llava_prefix_prompt,
    normalize_bool,
    parse_csv_items,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for hook sanity debugging."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--pope-file", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", default="false")

    parser.add_argument("--steer-vector-path", required=True)
    parser.add_argument("--steer-layers", default="10-20")
    parser.add_argument("--steer-router", choices=["no_filter", "force_cat", "force_attr", "force_rel", "rule"], default="force_cat")
    parser.add_argument("--steer-enabled-experts", default="cat")
    parser.add_argument("--steer-alpha", type=float, default=12.0)
    parser.add_argument("--steer-k-heads", type=int, default=64)
    parser.add_argument("--steer-head-select", choices=["norm", "random", "all"], default="norm")
    parser.add_argument("--steer-prefill", default="true")
    parser.add_argument("--steer-decode", default="false")
    parser.add_argument("--prefill-apply-to", choices=["last_token", "all_tokens"], default="last_token")
    parser.add_argument("--decode-apply-to", choices=["last_token"], default="last_token")
    parser.add_argument("--debug-log-hook-delta", default="true")
    parser.add_argument("--debug-random-vector", default="false")
    return parser.parse_args()


def prompt_length(
    sample: Mapping[str, Any],
    *,
    torch: Any,
    Image: Any,
    processor: Any,
    device: str,
    compute_dtype: Any,
) -> int:
    """Return tokenized prompt length for one image-question sample."""

    image = Image.open(sample["image_path"]).convert("RGB")
    prompt = build_llava_prefix_prompt(str(sample["question"]))
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = inputs_to_device(inputs, torch, device, compute_dtype)
    return int(inputs["input_ids"].shape[1])


def main() -> int:
    """Run the hook sanity CLI."""

    args = parse_args()
    try:
        rows = read_json_rows(resolve_project_path(args.pope_file))
        samples = [
            normalize_sample(row, index, resolve_project_path(args.image_root))
            for index, row in enumerate(rows)
        ]
        if not (0 <= int(args.sample_index) < len(samples)):
            raise ValueError(f"--sample-index {args.sample_index} is out of range for {len(samples)} samples")
        sample = samples[int(args.sample_index)]
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
            router=str(args.steer_router),
            enabled_experts=tuple(parse_csv_items(args.steer_enabled_experts)),
            steer_prefill=normalize_bool(args.steer_prefill),
            steer_decode=normalize_bool(args.steer_decode),
            prefill_apply_to=str(args.prefill_apply_to),
            decode_apply_to=str(args.decode_apply_to),
            debug_log_hook_delta=normalize_bool(args.debug_log_hook_delta),
            debug_random_vector=normalize_bool(args.debug_random_vector),
        )
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
        controller.reset_diagnostics()
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
        baseline_margin = yes_no_margin(baseline_logits, yes_ids, no_ids)
        steered_margin = yes_no_margin(steered_logits, yes_ids, no_ids)
        logit_delta = steered_logits - baseline_logits
        result = {
            "sample_index": int(args.sample_index),
            "image": sample["image"],
            "question": sample["question"],
            "label": sample.get("label"),
            "prompt_length": prompt_length(
                sample,
                torch=torch,
                Image=Image,
                processor=processor,
                device=args.device,
                compute_dtype=compute_dtype,
            ),
            "logits_shape": [int(item) for item in baseline_logits.shape],
            "max_abs_logit_delta": float(logit_delta.abs().max().item()),
            "baseline_margin": baseline_margin["margin"],
            "steered_margin": steered_margin["margin"],
            "delta_margin": steered_margin["margin"] - baseline_margin["margin"],
            "baseline_logit_pred": baseline_margin["prediction"],
            "steered_logit_pred": steered_margin["prediction"],
            "steering_diagnostics": controller.summary(),
        }
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output_path = resolve_project_path(args.output)
        write_json(output_path, result)
        print(f"Wrote hook sanity output to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
