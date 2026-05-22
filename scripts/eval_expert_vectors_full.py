#!/usr/bin/env python3
"""Vector-only expert steering evaluation with the official LLaVA loader.

This evaluator deliberately does not use subtype masks or external head maps.
Each vector selects its own global top-K layer/head pairs by head norm.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from run_pope_official_cat_expert_eval import (
    OfficialLlavaPopeGenerator,
    import_official_llava,
    load_official_model,
    load_pope_samples,
    parse_prediction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import write_json, write_jsonl
from expert_data.steering import ExpertSteeringController


VECTOR_KEYS = ("global", "cat", "attr", "rel")
YES_WORDS = {"yes", "y", "true", "1"}
NO_WORDS = {"no", "n", "false", "0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-type", choices=["pope", "jsonl"], required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--benchmark-family", required=True, choices=["category", "attribute", "relation", "optional"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-vector-file", required=True)
    parser.add_argument("--vectors", default="global,cat,attr,rel")
    parser.add_argument("--alphas", default="0.01,0.05,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--limit", type=int, default=0, help="0 means full benchmark.")
    parser.add_argument("--subtypes", default="", help="Optional comma filter for JSONL subtype rows.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--pope-root", default="")
    parser.add_argument("--datasets", nargs="+", default=["MSCOCO", "GQA"])
    parser.add_argument("--settings", nargs="+", default=["random", "popular", "adversarial"])
    parser.add_argument("--coco-image-root", default="")
    parser.add_argument("--gqa-image-root", default="")
    parser.add_argument("--prompt-suffix", default="Please answer this question in one word.")

    parser.add_argument("--input-jsonl", default="")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--setting-name", default="all")

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--llava-repo-path", "--llava-repo", dest="llava_repo_path", required=True)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compat-new-transformers", action="store_true")
    parser.add_argument("--parser-mode", default="contains_yes_no_octopus_like", choices=["first_yes_no", "contains_yes_no_octopus_like"])
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--layers", default="0-31")
    parser.add_argument("--prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-to", default="last_token", choices=["last_token", "all_tokens"])
    parser.add_argument("--prefill-apply-to", default="last_token", choices=["last_token", "all_tokens"])
    parser.add_argument("--decode-apply-to", default="last_token", choices=["last_token", "all_tokens"])
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).replace(" ", ",").split(",") if item.strip()]


def parse_alphas(value: str) -> list[float]:
    alphas = [float(item) for item in split_csv(value)]
    if not alphas:
        raise ValueError("--alphas cannot be empty")
    return alphas


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value)).strip("_")


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "samples", "questions", "annotations", "items", "rows"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in YES_WORDS or text.startswith("yes"):
        return "yes"
    if text in NO_WORDS or text.startswith("no"):
        return "no"
    raise ValueError(f"Could not normalize yes/no label: {value!r}")


def first_present(row: Mapping[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def resolve_image_path(row: Mapping[str, Any], image_root: Path) -> str:
    candidates: list[str] = []
    for key in ("image_path", "image", "img", "file_name", "filename", "path"):
        value = row.get(key)
        if value not in (None, ""):
            candidates.append(str(value))
    image_id = first_present(row, ("image_id", "id", "coco_id"), "")
    if image_id not in (None, ""):
        image_id_text = str(image_id)
        candidates.extend([image_id_text, f"{image_id_text}.jpg", f"{image_id_text}.png"])
        try:
            candidates.append(f"COCO_val2014_{int(image_id_text):012d}.jpg")
        except Exception:
            pass
    for candidate in candidates:
        path = Path(candidate)
        if path.is_absolute() and path.exists():
            return str(path)
        if str(image_root):
            joined = image_root / candidate
            if joined.exists():
                return str(joined)
    return str(Path(candidates[0]) if candidates else "")


def ensure_prompt(row: Mapping[str, Any]) -> str:
    if row.get("visual_prompt"):
        return str(row["visual_prompt"])
    question = str(first_present(row, ("question", "query", "prompt", "text", "instruction"), "")).strip()
    if not question:
        raise ValueError("JSONL row is missing question/query/prompt/text.")
    if "Please answer" in question or question.startswith("Question:"):
        return question
    return f"Question: {question}\nPlease answer the question based on the image."


def load_jsonl_group(args: argparse.Namespace) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    input_path = resolve(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing JSONL benchmark file: {input_path}")
    image_root = resolve(args.image_root) if str(args.image_root).strip() else Path("")
    subtype_filter = set(split_csv(args.subtypes))
    rows = read_json_or_jsonl(input_path)
    limit = int(args.limit)
    samples: list[dict[str, Any]] = []
    label_counts: dict[str, int] = defaultdict(int)
    subtype_counts: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows):
        subtype = str(row.get("subtype", "") or (row.get("metadata", {}) or {}).get("subtype", ""))
        if subtype_filter and subtype not in subtype_filter:
            continue
        if limit > 0 and len(samples) >= limit:
            break
        label = normalize_label(first_present(row, ("gt_answer", "label", "answer", "ground_truth", "target", "gt"), ""))
        question = str(first_present(row, ("question", "query", "prompt", "text", "instruction"), ""))
        image_path = resolve_image_path(row, image_root)
        if not image_path:
            raise FileNotFoundError(f"Could not resolve image path for row {index}: keys={list(row.keys())}")
        label_counts[label] += 1
        if subtype:
            subtype_counts[subtype] += 1
        samples.append(
            {
                "benchmark_id": args.benchmark_id,
                "benchmark_family": args.benchmark_family,
                "dataset": args.dataset_name or args.benchmark_id,
                "setting": args.setting_name,
                "sample_index": index,
                "sample_id": str(first_present(row, ("sample_id", "question_id", "id"), index)),
                "image_id": str(first_present(row, ("image_id", "id", "coco_id"), "")),
                "image_path": image_path,
                "question": question,
                "prompt": ensure_prompt(row),
                "label": label,
                "subtype": subtype,
                "raw": dict(row),
            }
        )
    manifest = {
        "type": "jsonl",
        "source_file": str(input_path),
        "image_root": str(image_root),
        "num_samples": len(samples),
        "label_counts": dict(label_counts),
        "subtype_counts": dict(subtype_counts),
    }
    return {(args.dataset_name or args.benchmark_id, args.setting_name): samples}, manifest


def load_groups(args: argparse.Namespace) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    if args.benchmark_type == "pope":
        if not str(args.pope_root).strip():
            raise ValueError("--pope-root is required for --benchmark-type pope")
        groups, manifest = load_pope_samples(args)
        for (dataset, setting), samples in groups.items():
            for sample in samples:
                sample["benchmark_id"] = args.benchmark_id
                sample["benchmark_family"] = args.benchmark_family
                sample["sample_id"] = f"{dataset}_{setting}_{sample.get('sample_index', len(sample))}"
        manifest["type"] = "pope"
        return groups, manifest
    return load_jsonl_group(args)


def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = invalid = pred_yes = 0
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
    total = len(rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_rate": pred_yes / total if total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid": invalid,
        "num_samples": total,
    }


def make_controller(args: argparse.Namespace, model: Any, vector: str, alpha: float) -> ExpertSteeringController:
    return ExpertSteeringController(
        model=model,
        vector_path=resolve(args.runtime_vector_file),
        layers=str(args.layers),
        alpha=float(alpha),
        k_heads=int(args.topk),
        head_select="norm",
        router="no_filter",
        enabled_experts=(vector,),
        apply_to=str(args.apply_to),
        steer_prefill=bool(args.prefill),
        steer_decode=bool(args.decode),
        prefill_apply_to=str(args.prefill_apply_to),
        decode_apply_to=str(args.decode_apply_to),
        seed=int(args.seed),
    )


def prediction(
    generator: OfficialLlavaPopeGenerator,
    sample: Mapping[str, Any],
    *,
    vector: str,
    alpha: float | None,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw, prompt_info = generator.generate(sample, mode=mode, sign=1.0 if mode == "steered" else 0.0)
    label = str(sample["label"])
    pred = parse_prediction(raw, label, str(args.parser_mode))
    return {
        "benchmark_id": args.benchmark_id,
        "benchmark_family": args.benchmark_family,
        "dataset": sample.get("dataset", args.benchmark_id),
        "setting": sample.get("setting", "all"),
        "sample_id": sample.get("sample_id", sample.get("sample_index", "")),
        "sample_index": sample.get("sample_index", ""),
        "image_id": sample.get("image_id", ""),
        "image_path": sample.get("image_path", ""),
        "question": sample.get("question", ""),
        "prompt": sample.get("prompt", ""),
        "subtype": sample.get("subtype", ""),
        "vector": vector,
        "alpha": "" if alpha is None else alpha,
        "label": label,
        "raw_output": raw,
        "raw_full_output": prompt_info.get("raw_full_output", ""),
        "pred": pred,
        "is_correct": bool(pred == label),
        "parser_mode": args.parser_mode,
        "prompt_token_len": prompt_info.get("prompt_token_len", ""),
        "output_token_len": prompt_info.get("output_token_len", ""),
    }


def run_predictions(
    *,
    generator: OfficialLlavaPopeGenerator,
    samples: Sequence[Mapping[str, Any]],
    raw_path: Path,
    vector: str,
    alpha: float | None,
    mode: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.skip_existing and raw_path.exists():
        return read_json_or_jsonl(raw_path)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        row = prediction(generator, sample, vector=vector, alpha=alpha, mode=mode, args=args)
        if index > 3:
            row["raw_full_output"] = ""
        rows.append(row)
        if int(args.progress_every) > 0 and index % int(args.progress_every) == 0:
            print(
                f"[{args.benchmark_id} {sample.get('dataset')} {sample.get('setting')} vector={vector} alpha={alpha}] "
                f"processed {index}/{len(samples)}",
                flush=True,
            )
    write_jsonl(raw_path, rows)
    return rows


def alpha_text(alpha: float | None) -> str:
    if alpha is None:
        return "baseline"
    return f"a{alpha:g}".replace("-", "neg").replace(".", "p")


def summarize_with_baseline(rows: list[dict[str, Any]], baseline_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    wrong_to_right = right_to_wrong = changed_pred = 0
    for row in rows:
        base = baseline_by_id.get(str(row["sample_id"]), {})
        base_pred = str(base.get("pred", "invalid"))
        pred = str(row.get("pred", "invalid"))
        label = str(row.get("label", ""))
        if pred == base_pred:
            continue
        changed_pred += 1
        if pred == label and base_pred != label:
            wrong_to_right += 1
        elif base_pred == label and pred != label:
            right_to_wrong += 1
    return {
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "changed_pred": changed_pred,
    }


def changed_cases(rows: list[dict[str, Any]], baseline_by_id: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        base = baseline_by_id.get(str(row["sample_id"]), {})
        base_pred = str(base.get("pred", "invalid"))
        pred = str(row.get("pred", "invalid"))
        label = str(row.get("label", ""))
        if pred == base_pred:
            continue
        if pred == label and base_pred != label:
            change_type = "wrong_to_right"
        elif base_pred == label and pred != label:
            change_type = "right_to_wrong"
        else:
            change_type = "changed_wrong_to_wrong"
        out.append(
            {
                "benchmark": row.get("benchmark_id", ""),
                "benchmark_family": row.get("benchmark_family", ""),
                "dataset": row.get("dataset", ""),
                "setting": row.get("setting", ""),
                "vector": row.get("vector", ""),
                "alpha": row.get("alpha", ""),
                "sample_id": row.get("sample_id", ""),
                "image_path": row.get("image_path", ""),
                "question": row.get("question", ""),
                "gt_answer": label,
                "baseline_answer": base.get("raw_output", ""),
                "baseline_parsed": base_pred,
                "steered_answer": row.get("raw_output", ""),
                "steered_parsed": pred,
                "change_type": change_type,
            }
        )
    return out


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    out_dir = resolve(args.output_dir)
    raw_dir = out_dir / "raw"
    if out_dir.exists() and any(out_dir.iterdir()) and not (args.overwrite or args.skip_existing):
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite or --skip-existing.")
    raw_dir.mkdir(parents=True, exist_ok=True)

    groups, manifest = load_groups(args)
    vectors = [v for v in split_csv(args.vectors) if v in VECTOR_KEYS]
    if not vectors:
        raise ValueError(f"--vectors must include at least one of {VECTOR_KEYS}")
    alphas = parse_alphas(args.alphas)

    llava = import_official_llava(str(args.llava_repo_path))
    tokenizer, model, image_processor, _context_len, model_name = load_official_model(args, llava)
    generator = OfficialLlavaPopeGenerator(
        args=args,
        llava=llava,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        controller=None,
    )

    summary_rows: list[dict[str, Any]] = []
    all_changed: list[dict[str, Any]] = []
    for (dataset, setting), samples in groups.items():
        baseline_name = safe_name(f"{args.benchmark_id}_{dataset}_{setting}_baseline")
        baseline_path = raw_dir / f"{baseline_name}.jsonl"
        baseline_rows = run_predictions(
            generator=generator,
            samples=samples,
            raw_path=baseline_path,
            vector="baseline",
            alpha=None,
            mode="baseline",
            args=args,
        )
        baseline_by_id = {str(row["sample_id"]): row for row in baseline_rows}
        base_metrics = metrics(baseline_rows)
        summary_rows.append(
            {
                "benchmark_id": args.benchmark_id,
                "benchmark_family": args.benchmark_family,
                "dataset": dataset,
                "setting": setting,
                "method": "baseline",
                "vector": "baseline",
                "alpha": "",
                **base_metrics,
                "wrong_to_right": 0,
                "right_to_wrong": 0,
                "changed_pred": 0,
                "raw_path": str(baseline_path),
            }
        )

        for vector in vectors:
            controller = make_controller(args, model, vector, alphas[0])
            generator.controller = controller
            try:
                for alpha in alphas:
                    controller.alpha = float(alpha)
                    run_name = safe_name(f"{args.benchmark_id}_{dataset}_{setting}_{vector}_{alpha_text(alpha)}")
                    raw_path = raw_dir / f"{run_name}.jsonl"
                    rows = run_predictions(
                        generator=generator,
                        samples=samples,
                        raw_path=raw_path,
                        vector=vector,
                        alpha=alpha,
                        mode="steered",
                        args=args,
                    )
                    row_metrics = metrics(rows)
                    changes = summarize_with_baseline(rows, baseline_by_id)
                    summary_rows.append(
                        {
                            "benchmark_id": args.benchmark_id,
                            "benchmark_family": args.benchmark_family,
                            "dataset": dataset,
                            "setting": setting,
                            "method": "steered",
                            "vector": vector,
                            "alpha": alpha,
                            **row_metrics,
                            **changes,
                            "raw_path": str(raw_path),
                        }
                    )
                    all_changed.extend(changed_cases(rows, baseline_by_id))
            finally:
                controller.remove()
                generator.controller = None

    fieldnames = [
        "benchmark_id",
        "benchmark_family",
        "dataset",
        "setting",
        "method",
        "vector",
        "alpha",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "yes_rate",
        "tp",
        "tn",
        "fp",
        "fn",
        "invalid",
        "wrong_to_right",
        "right_to_wrong",
        "changed_pred",
        "num_samples",
        "raw_path",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    write_jsonl(out_dir / "changed_cases_all.jsonl", all_changed)
    write_json(
        out_dir / "config.json",
        {
            "runner": "scripts/eval_expert_vectors_full.py",
            "model_path": str(args.model_path),
            "model_name": model_name,
            "llava_repo_path": str(args.llava_repo_path),
            "conv_mode": str(args.conv_mode),
            "benchmark_type": args.benchmark_type,
            "benchmark_id": args.benchmark_id,
            "benchmark_family": args.benchmark_family,
            "runtime_vector_file": str(resolve(args.runtime_vector_file)),
            "vectors": vectors,
            "alphas": alphas,
            "steering": {
                "head_select": "norm",
                "topk": int(args.topk),
                "layers": str(args.layers),
                "prefill": bool(args.prefill),
                "decode": bool(args.decode),
                "apply_to": str(args.apply_to),
            },
            "decoding": {
                "do_sample": bool(args.do_sample),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "num_beams": int(args.num_beams),
                "max_new_tokens": int(args.max_new_tokens),
                "seed": int(args.seed),
                "parser_mode": str(args.parser_mode),
            },
            "manifest": manifest,
        },
    )
    print(f"Wrote vector-only summary to {out_dir / 'summary.csv'}")
    print(f"Wrote changed cases to {out_dir / 'changed_cases_all.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
