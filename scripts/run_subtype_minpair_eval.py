#!/usr/bin/env python3
"""Run held-out subtype minimal-pair sanity eval with official LLaVA steering."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from expert_data.steering import ExpertSteeringController, normalize_bool  # noqa: E402
from run_pope_official_cat_expert_eval import (  # noqa: E402
    OfficialLlavaPopeGenerator,
    compute_metrics,
    import_official_llava,
    load_official_model,
    parse_prediction,
)


DEFAULT_VECTOR_KEYS = (
    "g_all_clean",
    "g_cat_clean",
    "g_attr_clean",
    "g_rel_clean",
    "d_cat_hard_g1_s05_clean",
    "d_attr_color_g1_s05_clean",
    "d_attr_count_g1_s05_clean",
    "d_rel_spatial_g1_s05_clean",
    "d_rel_contact_g1_s05_clean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--vector-path", required=True)
    parser.add_argument("--vector-keys", default=",".join(DEFAULT_VECTOR_KEYS))
    parser.add_argument("--alphas", default="0.05,0.1,0.25,0.5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--llava-repo-path", "--llava-repo", dest="llava_repo_path", required=True)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--subtypes", default="")
    parser.add_argument("--limit-per-subtype", type=int, default=0)
    parser.add_argument("--layers", default="0-31")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--head-select", default="norm", choices=["norm", "random", "all", "expert_map"])
    parser.add_argument("--prefill", default="true")
    parser.add_argument("--decode", default="true")
    parser.add_argument("--apply-to", default="last_token")
    parser.add_argument("--prefill-apply-to", default="last_token")
    parser.add_argument("--decode-apply-to", default="last_token")
    parser.add_argument("--prompt-suffix", default="Please answer this question with one word.")
    parser.add_argument("--parser-mode", default="contains_yes_no_octopus_like", choices=["first_yes_no", "contains_yes_no_octopus_like"])
    parser.add_argument("--do-sample", default="true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compat-new-transformers", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_csv_items(text: str) -> list[str]:
    return [item.strip() for item in str(text).replace(",", " ").split() if item.strip()]


def parse_alphas(text: str) -> list[float]:
    return [float(item) for item in parse_csv_items(text)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def inspect_vector_keys(path: Path, requested: list[str]) -> tuple[list[str], dict[str, Any]]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    vectors = payload.get("vectors", {})
    available = sorted(str(key) for key in vectors)
    selected = [key for key in requested if key in vectors]
    missing = [key for key in requested if key not in vectors]
    if not selected:
        raise ValueError(f"None of the requested vector keys exist in {path}. Missing={missing[:10]}")
    return selected, {
        "path": str(path),
        "available_count": len(available),
        "selected": selected,
        "missing": missing,
        "layers": list(payload.get("layers", [])),
        "num_heads": int(payload.get("num_heads", 0)),
        "head_dim": int(payload.get("head_dim", 0)),
        "hidden_size": int(payload.get("hidden_size", 0)),
    }


def build_samples(rows: list[Mapping[str, Any]], *, prompt_suffix: str, subtypes: set[str], limit_per_subtype: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subtype = str(row.get("subtype", ""))
        if subtypes and subtype not in subtypes:
            continue
        if int(limit_per_subtype) > 0 and len(grouped[subtype]) >= int(limit_per_subtype):
            continue
        label = str(row.get("gt_answer", row.get("label", ""))).strip().lower()
        if label not in {"yes", "no"}:
            continue
        question = str(row.get("question", "")).strip()
        sample = {
            "id": str(row.get("id", "")),
            "dataset": "subtype_minpair",
            "setting": subtype,
            "subtype": subtype,
            "expert_type": str(row.get("expert_type", "")),
            "image_id": str(row.get("image_id", "")),
            "image_path": str(row.get("image_path", "")),
            "question": question,
            "prompt": f"{question} {prompt_suffix}".strip(),
            "label": label,
            "metadata": row.get("metadata", {}),
        }
        grouped[subtype].append(sample)
    return dict(sorted(grouped.items()))


def prediction_row(
    *,
    sample: Mapping[str, Any],
    method: str,
    vector: str,
    alpha: float | None,
    raw_output: str,
    prompt_info: Mapping[str, Any],
) -> dict[str, Any]:
    pred = parse_prediction(raw_output, str(sample["label"]), str(prompt_info.get("parser_mode", "contains_yes_no_octopus_like")))
    return {
        "id": str(sample.get("id", "")),
        "dataset": "subtype_minpair",
        "setting": str(sample.get("setting", "")),
        "subtype": str(sample.get("subtype", "")),
        "expert_type": str(sample.get("expert_type", "")),
        "method": method,
        "vector": vector,
        "alpha": "" if alpha is None else float(alpha),
        "question": str(sample.get("question", "")),
        "prompt": str(sample.get("prompt", "")),
        "label": str(sample.get("label", "")),
        "pred": pred,
        "raw_output": raw_output,
        "image_id": str(sample.get("image_id", "")),
        "image_path": str(sample.get("image_path", "")),
        "full_prompt": str(prompt_info.get("full_prompt", "")),
        "parser_mode": str(prompt_info.get("parser_mode", "")),
        "output_token_len": prompt_info.get("output_token_len", ""),
    }


def run_subset(
    *,
    generator: OfficialLlavaPopeGenerator,
    samples: list[dict[str, Any]],
    method: str,
    vector: str,
    alpha: float | None,
    output_path: Path,
    progress_every: int,
) -> list[dict[str, Any]]:
    rows = []
    mode = "steered" if method == "steered" else "baseline"
    sign = 1.0 if method == "steered" else 0.0
    for index, sample in enumerate(samples, start=1):
        raw_output, prompt_info = generator.generate(sample, mode=mode, sign=sign)
        row = prediction_row(sample=sample, method=method, vector=vector, alpha=alpha, raw_output=raw_output, prompt_info=prompt_info)
        if index > 3:
            row["full_prompt"] = ""
        rows.append(row)
        if int(progress_every) > 0 and index % int(progress_every) == 0:
            print(f"[{sample['setting']} {method} vector={vector} alpha={alpha}] processed {index}/{len(samples)}")
    write_jsonl(output_path, rows)
    return rows


def output_name(subtype: str, method: str, vector: str, alpha: float | None) -> str:
    safe_vector = str(vector or "baseline").replace(",", "_").replace("/", "_")
    if method == "baseline":
        return f"{subtype}_baseline.jsonl"
    alpha_text = str(alpha).replace("-", "neg").replace(".", "p")
    return f"{subtype}_{safe_vector}_alpha{alpha_text}.jsonl"


def add_changed_metrics(rows: list[Mapping[str, Any]], baseline_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    wrong_to_right = right_to_wrong = changed = 0
    for row in rows:
        base = baseline_by_id.get(str(row.get("id", "")))
        if not base:
            continue
        label = str(row.get("label", "")).lower()
        pred = str(row.get("pred", "")).lower()
        base_pred = str(base.get("pred", "")).lower()
        if pred != base_pred:
            changed += 1
        if base_pred != label and pred == label:
            wrong_to_right += 1
        if base_pred == label and pred != label:
            right_to_wrong += 1
    return {"wrong_to_right": wrong_to_right, "right_to_wrong": right_to_wrong, "changed_pred": changed}


def summarize(raw_dir: Path, summary_csv: Path, report_path: Path) -> None:
    all_rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        all_rows.extend(read_jsonl(path))
    baseline_by_subset: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in all_rows:
        if row.get("method") == "baseline":
            baseline_by_subset[str(row.get("subtype", ""))][str(row.get("id", ""))] = row
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        alpha = "" if row.get("alpha") in (None, "") else str(row.get("alpha"))
        grouped[(str(row.get("subtype")), str(row.get("method")), str(row.get("vector")), alpha)].append(row)
    fields = [
        "eval_subset",
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
    ]
    summary_rows = []
    for (subset, method, vector, alpha), rows in sorted(grouped.items()):
        metrics = compute_metrics(rows)
        changed = add_changed_metrics(rows, baseline_by_subset.get(subset, {}))
        summary_rows.append(
            {
                "eval_subset": subset,
                "method": method,
                "vector": vector,
                "alpha": alpha,
                "accuracy": metrics["Accuracy"] / 100.0,
                "precision": metrics["Precision"] / 100.0,
                "recall": metrics["Recall"] / 100.0,
                "f1": metrics["F1"] / 100.0,
                "yes_rate": metrics["Yes Rate"] / 100.0,
                "tp": metrics["TP"],
                "tn": metrics["TN"],
                "fp": metrics["FP"],
                "fn": metrics["FN"],
                "invalid": metrics["Invalid"],
                **changed,
                "num_samples": metrics["N"],
            }
        )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_report(report_path, summary_rows)


def md_table(headers: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(path: Path, rows: list[Mapping[str, Any]]) -> None:
    best: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("method") != "steered":
            continue
        key = f"{row.get('eval_subset')}::{row.get('vector')}"
        if key not in best or float(row.get("f1", 0.0)) > float(best[key].get("f1", 0.0)):
            best[key] = row
    best_rows = sorted(best.values(), key=lambda row: (str(row.get("eval_subset")), -float(row.get("f1", 0.0))))[:80]
    lines = ["# Subtype Minimal-Pair Held-Out Eval", ""]
    lines.append("## Best Steered Rows By Subset/Vector")
    lines.append(md_table(["eval_subset", "vector", "alpha", "accuracy", "f1", "yes_rate", "wrong_to_right", "right_to_wrong", "changed_pred", "num_samples"], best_rows))
    lines.append("")
    lines.append("## All Rows")
    lines.append(md_table(["eval_subset", "method", "vector", "alpha", "accuracy", "precision", "recall", "f1", "yes_rate", "tp", "tn", "fp", "fn", "wrong_to_right", "right_to_wrong", "changed_pred", "num_samples"], rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        random.seed(int(args.seed))
        output_dir = resolve(args.output_dir)
        raw_dir = output_dir / "raw"
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite.")
        raw_dir.mkdir(parents=True, exist_ok=True)
        vector_keys, vector_info = inspect_vector_keys(resolve(args.vector_path), parse_csv_items(args.vector_keys))
        if vector_info["missing"]:
            print(f"Skipping missing vector keys: {vector_info['missing']}")
        rows = read_jsonl(resolve(args.input_jsonl))
        subtypes = set(parse_csv_items(args.subtypes))
        samples_by_subset = build_samples(rows, prompt_suffix=str(args.prompt_suffix), subtypes=subtypes, limit_per_subtype=int(args.limit_per_subtype))
        if not samples_by_subset:
            raise ValueError("No eval samples selected.")
        llava = import_official_llava(str(args.llava_repo_path))
        llava.torch.manual_seed(int(args.seed))
        if llava.torch.cuda.is_available():
            llava.torch.cuda.manual_seed_all(int(args.seed))
        tokenizer, model, image_processor, context_len, model_name = load_official_model(args, llava)
        generator = OfficialLlavaPopeGenerator(args=args, llava=llava, tokenizer=tokenizer, model=model, image_processor=image_processor, controller=None)

        baseline_rows_by_subset: dict[str, list[dict[str, Any]]] = {}
        for subset, samples in samples_by_subset.items():
            path = raw_dir / output_name(subset, "baseline", "", None)
            if args.skip_existing and path.exists():
                baseline_rows = read_jsonl(path)
            else:
                baseline_rows = run_subset(generator=generator, samples=samples, method="baseline", vector="", alpha=None, output_path=path, progress_every=int(args.progress_every))
            baseline_rows_by_subset[subset] = baseline_rows

        alpha_values = parse_alphas(args.alphas)
        for vector_key in vector_keys:
            controller = ExpertSteeringController(
                model=model,
                vector_path=resolve(args.vector_path),
                layers=str(args.layers),
                alpha=float(alpha_values[0]),
                k_heads=int(args.topk),
                head_select=str(args.head_select),
                router="no_filter",
                enabled_experts=(vector_key,),
                expert_key=vector_key,
                apply_to=str(args.apply_to),
                steer_prefill=normalize_bool(args.prefill),
                steer_decode=normalize_bool(args.decode),
                prefill_apply_to=str(args.prefill_apply_to),
                decode_apply_to=str(args.decode_apply_to),
                seed=int(args.seed),
            )
            generator.controller = controller
            try:
                for alpha in alpha_values:
                    controller.alpha = float(alpha)
                    for subset, samples in samples_by_subset.items():
                        path = raw_dir / output_name(subset, "steered", vector_key, alpha)
                        if args.skip_existing and path.exists():
                            continue
                        run_subset(generator=generator, samples=samples, method="steered", vector=vector_key, alpha=float(alpha), output_path=path, progress_every=int(args.progress_every))
            finally:
                controller.remove()
                generator.controller = None

        config = {
            "runner": "scripts/run_subtype_minpair_eval.py",
            "model_path": str(args.model_path),
            "model_base": args.model_base,
            "model_name": model_name,
            "context_len": context_len,
            "llava_repo_path": str(args.llava_repo_path),
            "conv_mode": str(args.conv_mode),
            "input_jsonl": str(resolve(args.input_jsonl)),
            "vector_info": vector_info,
            "subsets": {key: len(value) for key, value in samples_by_subset.items()},
            "alphas": alpha_values,
            "decode": {
                "do_sample": normalize_bool(args.do_sample),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "num_beams": int(args.num_beams),
                "max_new_tokens": int(args.max_new_tokens),
            },
            "seed": int(args.seed),
            "parser_mode": str(args.parser_mode),
            "steering": {
                "layers": str(args.layers),
                "topk": int(args.topk),
                "head_select": str(args.head_select),
                "prefill": normalize_bool(args.prefill),
                "decode": normalize_bool(args.decode),
                "apply_to": str(args.apply_to),
                "prefill_apply_to": str(args.prefill_apply_to),
                "decode_apply_to": str(args.decode_apply_to),
            },
        }
        write_json(output_dir / "config.json", config)
        summarize(raw_dir, output_dir / "summary.csv", output_dir / "SUMMARY.md")
        print(f"Wrote subtype eval summary to {output_dir / 'SUMMARY.md'}")
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
