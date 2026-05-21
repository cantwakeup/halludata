#!/usr/bin/env python3
"""Evaluate subtype mask steering with separated direction and head mask.

This wrapper keeps the official LLaVA generation path and reuses the existing
ExpertSteeringController `expert_map` mode:

    direction-key -> vector content
    mask-key      -> selected [layer, head] rows

When a mask is supplied, head selection no longer comes from the direction
vector norm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from run_pope_official_cat_expert_eval import (
    OfficialLlavaPopeGenerator,
    import_official_llava,
    load_official_model,
    parse_prediction,
)
from expert_data.steering import ExpertSteeringController


SUBTYPES = [
    "cat_random",
    "cat_popular",
    "cat_hard",
    "attr_color",
    "attr_count",
    "rel_spatial",
    "rel_contact",
]

SUBTYPE_TO_TYPE = {
    "cat_random": "cat",
    "cat_popular": "cat",
    "cat_hard": "cat",
    "attr_color": "attr",
    "attr_count": "attr",
    "rel_spatial": "rel",
    "rel_contact": "rel",
}

MISMATCH_MASKS = {
    "attr_color": [
        "mask_s_attr_count_energy_top64",
        "mask_s_rel_spatial_energy_top64",
        "mask_s_cat_hard_energy_top64",
    ],
    "attr_count": [
        "mask_s_attr_color_energy_top64",
        "mask_s_rel_contact_energy_top64",
        "mask_s_cat_hard_energy_top64",
    ],
    "rel_spatial": [
        "mask_s_rel_contact_energy_top64",
        "mask_s_attr_color_energy_top64",
        "mask_s_cat_hard_energy_top64",
    ],
    "rel_contact": [
        "mask_s_rel_spatial_energy_top64",
        "mask_s_attr_count_energy_top64",
        "mask_s_cat_hard_energy_top64",
    ],
    "cat_hard": [
        "mask_s_attr_color_energy_top64",
        "mask_s_attr_count_energy_top64",
        "mask_s_rel_contact_energy_top64",
    ],
    "cat_random": [
        "mask_s_attr_color_energy_top64",
        "mask_s_rel_spatial_energy_top64",
    ],
    "cat_popular": [
        "mask_s_attr_count_energy_top64",
        "mask_s_rel_contact_energy_top64",
    ],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--vector-file", required=True)
    ap.add_argument("--mask-file", required=True)
    ap.add_argument("--run-specs", help="Optional JSONL run spec file. If omitted, default experiment specs are generated.")
    ap.add_argument("--subtypes", default=",".join(SUBTYPES), help="Comma-separated subtype filter.")
    ap.add_argument("--alphas", default="0.05,0.1,0.25,0.5")
    ap.add_argument("--limit-per-subtype", type=int, default=0, help="0 means use all examples.")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--llava-repo-path", "--llava-repo", dest="llava_repo_path", required=True)
    ap.add_argument("--conv-mode", default="llava_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compat-new-transformers", action="store_true")
    ap.add_argument("--parser-mode", default="contains_yes_no_octopus_like")
    ap.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefill", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--decode", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--apply-to", default="last_token", choices=["last_token", "all_tokens"])
    ap.add_argument("--prefill-apply-to", default=None, choices=["last_token", "all_tokens"])
    ap.add_argument("--decode-apply-to", default=None, choices=["last_token", "all_tokens"])
    ap.add_argument("--progress-every", type=int, default=20)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_alphas(value: str) -> List[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("--alphas cannot be empty")
    return out


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]], append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_pt(path: str | Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_vectors(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if "vectors" in payload and isinstance(payload["vectors"], Mapping):
        return payload["vectors"]
    return {k: v for k, v in payload.items() if torch.is_tensor(v)}


def get_masks(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if "masks" in payload and isinstance(payload["masks"], Mapping):
        return payload["masks"]
    return {k: v for k, v in payload.items() if torch.is_tensor(v) and v.ndim == 2}


def safe_name(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def sample_id(row: Mapping[str, Any], fallback: int) -> str:
    for key in ["id", "sample_id", "question_id"]:
        if row.get(key) is not None:
            return str(row[key])
    return f"row_{fallback}"


def row_subtype(row: Mapping[str, Any]) -> str:
    if row.get("subtype"):
        return str(row["subtype"])
    meta = row.get("metadata")
    if isinstance(meta, Mapping) and meta.get("subtype"):
        return str(meta["subtype"])
    raise KeyError(f"Missing subtype in row keys={list(row.keys())}")


def row_label(row: Mapping[str, Any]) -> str:
    label = str(row.get("gt_answer", row.get("label", ""))).strip().lower()
    if label not in {"yes", "no"}:
        raise ValueError(f"Expected gt_answer yes/no, got {label!r}")
    return label


def row_prompt(row: Mapping[str, Any]) -> str:
    if row.get("visual_prompt"):
        return str(row["visual_prompt"])
    question = str(row.get("question", "")).strip()
    return f"Question: {question}\nPlease answer the question based on the image."


def build_eval_rows(
    rows: Sequence[Mapping[str, Any]],
    subtypes: Sequence[str],
    limit_per_subtype: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)
    subtype_set = set(subtypes)
    for idx, row in enumerate(rows):
        subtype = row_subtype(row)
        if subtype not in subtype_set:
            continue
        if limit_per_subtype and counts[subtype] >= limit_per_subtype:
            continue
        item = dict(row)
        item["_row_index"] = idx
        item["_id"] = sample_id(row, idx)
        item["_subtype"] = subtype
        item["_label"] = row_label(row)
        item["_prompt"] = row_prompt(row)
        selected.append(item)
        counts[subtype] += 1
    return selected


def default_specs(subtypes: Sequence[str]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for subtype in subtypes:
        typ = SUBTYPE_TO_TYPE[subtype]
        g_type = f"g_{typ}_clean"
        mask_type = f"mask_g_{typ}_norm_top64"
        mask_subtype = f"mask_s_{subtype}_energy_top64"
        specs.append(
            {
                "name": f"{subtype}__g_all__g_all_mask",
                "subtype": subtype,
                "direction_key": "g_all_clean",
                "mask_key": "mask_g_all_norm_top64",
                "match_type": "g_all_baseline",
            }
        )
        specs.append(
            {
                "name": f"{subtype}__{g_type}__{mask_type}",
                "subtype": subtype,
                "direction_key": g_type,
                "mask_key": mask_type,
                "match_type": "g_type_baseline",
            }
        )
        specs.append(
            {
                "name": f"{subtype}__{g_type}__{mask_subtype}",
                "subtype": subtype,
                "direction_key": g_type,
                "mask_key": mask_subtype,
                "match_type": "matched_energy",
            }
        )
        specs.append(
            {
                "name": f"{subtype}__g_all_clean__{mask_subtype}",
                "subtype": subtype,
                "direction_key": "g_all_clean",
                "mask_key": mask_subtype,
                "match_type": "matched_energy_g_all",
            }
        )
        for mask_key in MISMATCH_MASKS.get(subtype, []):
            specs.append(
                {
                    "name": f"{subtype}__{g_type}__{mask_key}",
                    "subtype": subtype,
                    "direction_key": g_type,
                    "mask_key": mask_key,
                    "match_type": "mismatched_energy",
                }
            )
        if typ in {"attr", "rel"}:
            specs.append(
                {
                    "name": f"{subtype}__s_direction__{mask_subtype}",
                    "subtype": subtype,
                    "direction_key": f"s_{subtype}_clean",
                    "mask_key": mask_subtype,
                    "match_type": "s_direction_ablation",
                }
            )
            for seed in [0, 1, 2]:
                specs.append(
                    {
                        "name": f"{subtype}__{g_type}__random_seed{seed}",
                        "subtype": subtype,
                        "direction_key": g_type,
                        "mask_key": f"random_mask_top64_seed{seed}",
                        "match_type": "random_mask",
                    }
                )
    return specs


def load_specs(path: Optional[str], subtypes: Sequence[str]) -> List[Dict[str, Any]]:
    if not path:
        return default_specs(subtypes)
    p = Path(path)
    specs: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            data = data.get("runs", [])
        specs = [dict(x) for x in data]
    else:
        specs = [dict(x) for x in load_jsonl(p)]
    subtype_set = set(subtypes)
    return [s for s in specs if str(s.get("subtype")) in subtype_set]


def mask_to_head_rows(mask: torch.Tensor) -> List[List[int]]:
    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape [layers, heads], got {tuple(mask.shape)}")
    rows = []
    for layer, head in mask.to(torch.bool).nonzero(as_tuple=False).tolist():
        rows.append([int(layer), int(head)])
    rows.sort()
    return rows


def prepare_runtime_files(
    *,
    vector_payload: Mapping[str, Any],
    vectors: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    specs: Sequence[Mapping[str, Any]],
    runtime_dir: Path,
) -> Tuple[Path, Path, Dict[str, str]]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    eval_vectors: Dict[str, torch.Tensor] = {}
    head_map: Dict[str, List[List[int]]] = {}
    spec_to_expert_key: Dict[str, str] = {}

    for spec in specs:
        direction_key = str(spec["direction_key"])
        mask_key = str(spec["mask_key"])
        if direction_key not in vectors:
            raise KeyError(f"Direction vector key not found: {direction_key}")
        if mask_key not in masks:
            raise KeyError(f"Mask key not found: {mask_key}")
        expert_key = safe_name(f"{direction_key}__{mask_key}")
        spec_to_expert_key[str(spec["name"])] = expert_key
        eval_vectors[expert_key] = vectors[direction_key].detach().cpu()
        head_rows = mask_to_head_rows(masks[mask_key])
        if not head_rows:
            raise ValueError(f"Mask {mask_key} selected no heads")
        head_map[expert_key] = head_rows

    sample_tensor = next(iter(eval_vectors.values()))
    if sample_tensor.ndim != 3:
        raise ValueError(f"Expected direction vectors [L,H,D], got {tuple(sample_tensor.shape)}")
    layers = list(range(int(sample_tensor.shape[0])))
    num_heads = int(sample_tensor.shape[1])
    head_dim = int(sample_tensor.shape[2])

    payload = {
        "vectors": eval_vectors,
        "layers": vector_payload.get("layers", layers),
        "num_heads": vector_payload.get("num_heads", num_heads),
        "head_dim": vector_payload.get("head_dim", head_dim),
        "hidden_size": vector_payload.get("hidden_size", num_heads * head_dim),
        "config": vector_payload.get("config", {}),
        "components": vector_payload.get("components", {}),
        "stats": vector_payload.get("stats", {}),
        "metadata": {
            "created_by": "scripts/eval_subtype_mask_steering.py",
            "note": "Direction vectors duplicated under run-specific keys; masks are applied via expert_map.",
        },
    }
    vector_path = runtime_dir / "direction_vectors_for_mask_eval.pt"
    head_map_path = runtime_dir / "expert_head_map.json"
    torch.save(payload, vector_path)
    head_map_path.write_text(json.dumps(head_map, indent=2), encoding="utf-8")
    return vector_path, head_map_path, spec_to_expert_key


def make_controller(
    *,
    model: Any,
    vector_file: Path,
    head_map_file: Path,
    expert_key: str,
    alpha: float,
    args: argparse.Namespace,
) -> ExpertSteeringController:
    return ExpertSteeringController(
        model=model,
        vector_path=str(vector_file),
        layers="0-31",
        alpha=alpha,
        k_heads=64,
        head_select="expert_map",
        router="no_filter",
        head_map_path=str(head_map_file),
        expert_key=expert_key,
        enabled_experts=(expert_key,),
        steer_prefill=args.prefill,
        steer_decode=args.decode,
        apply_to=args.apply_to,
        prefill_apply_to=args.prefill_apply_to or args.apply_to,
        decode_apply_to=args.decode_apply_to or args.apply_to,
    )


def generate_one(
    generator: OfficialLlavaPopeGenerator,
    item: Mapping[str, Any],
    steering: Optional[ExpertSteeringController],
    args: argparse.Namespace,
) -> Tuple[str, str]:
    sample = {
        "image_path": item["image_path"],
        "prompt": item["_prompt"],
    }
    previous_controller = generator.controller
    generator.controller = steering
    raw, _prompt_info = generator.generate(
        sample,
        mode="steered" if steering is not None else "baseline",
        sign=1.0 if steering is not None else 0.0,
    )
    generator.controller = previous_controller
    parsed = parse_prediction(raw, str(item["_label"]), str(args.parser_mode))
    return raw, parsed


def summarize_predictions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not (args.overwrite or args.skip_existing):
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite or --skip-existing.")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = out_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    subtypes = split_csv(args.subtypes)
    alphas = parse_alphas(args.alphas)
    all_rows = load_jsonl(args.input_jsonl)
    eval_rows = build_eval_rows(all_rows, subtypes, args.limit_per_subtype)
    rows_by_subtype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        rows_by_subtype[row["_subtype"]].append(row)
    specs = load_specs(args.run_specs, subtypes)

    vector_payload = load_pt(args.vector_file)
    mask_payload = load_pt(args.mask_file)
    vectors = get_vectors(vector_payload)
    masks = get_masks(mask_payload)
    runtime_vector_file, head_map_file, spec_to_expert_key = prepare_runtime_files(
        vector_payload=vector_payload,
        vectors=vectors,
        masks=masks,
        specs=specs,
        runtime_dir=runtime_dir,
    )

    config = {
        "input_jsonl": args.input_jsonl,
        "vector_file": args.vector_file,
        "mask_file": args.mask_file,
        "runtime_vector_file": str(runtime_vector_file),
        "head_map_file": str(head_map_file),
        "subtypes": subtypes,
        "alphas": alphas,
        "limit_per_subtype": args.limit_per_subtype,
        "model_path": args.model_path,
        "llava_repo_path": args.llava_repo_path,
        "conv_mode": args.conv_mode,
        "decode": {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "steering": {
            "topk": 64,
            "head_select": "expert_map",
            "prefill": args.prefill,
            "decode": args.decode,
            "apply_to": args.apply_to,
            "prefill_apply_to": args.prefill_apply_to or args.apply_to,
            "decode_apply_to": args.decode_apply_to or args.apply_to,
        },
        "parser_mode": args.parser_mode,
        "num_specs": len(specs),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "run_specs.resolved.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in specs) + "\n",
        encoding="utf-8",
    )

    llava = import_official_llava(str(args.llava_repo_path))
    llava.torch.manual_seed(int(args.seed))
    if llava.torch.cuda.is_available():
        llava.torch.cuda.manual_seed_all(int(args.seed))
    tokenizer, model, image_processor, _context_len, _model_name = load_official_model(args, llava)
    generator = OfficialLlavaPopeGenerator(
        args=args,
        llava=llava,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        controller=None,
    )

    baseline_by_id: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    changed_rows: List[Dict[str, Any]] = []

    for subtype in subtypes:
        items = rows_by_subtype.get(subtype, [])
        if not items:
            continue
        baseline_raw = raw_dir / f"{safe_name(subtype)}__baseline.jsonl"
        baseline_preds: List[Dict[str, Any]] = []
        if args.skip_existing and baseline_raw.exists():
            baseline_preds = load_jsonl(baseline_raw)
        else:
            for idx, item in enumerate(items, 1):
                raw, pred = generate_one(generator, item, None, args)
                row = {
                    "eval_subset": subtype,
                    "method": "baseline",
                    "id": item["_id"],
                    "image_path": item.get("image_path", ""),
                    "question": item.get("question", ""),
                    "prompt": item["_prompt"],
                    "label": item["_label"],
                    "raw_output": raw,
                    "pred": pred,
                }
                baseline_preds.append(row)
                if args.progress_every and idx % args.progress_every == 0:
                    print(f"[{subtype} baseline] processed {idx}/{len(items)}", flush=True)
            write_jsonl(baseline_raw, baseline_preds)
        for row in baseline_preds:
            baseline_by_id[str(row["id"])] = row
        metrics = summarize_predictions(baseline_preds)
        summary_rows.append(
            {
                "eval_subset": subtype,
                "method": "baseline",
                "direction_key": "",
                "mask_key": "",
                "match_type": "",
                "alpha": "",
                **metrics,
                "wrong_to_right": 0,
                "right_to_wrong": 0,
                "changed_pred": 0,
            }
        )

    for spec in specs:
        subtype = str(spec["subtype"])
        items = rows_by_subtype.get(subtype, [])
        if not items:
            continue
        direction_key = str(spec["direction_key"])
        mask_key = str(spec["mask_key"])
        spec_name = str(spec["name"])
        match_type = str(spec.get("match_type", ""))
        expert_key = spec_to_expert_key[spec_name]
        steering = make_controller(
            model=model,
            vector_file=runtime_vector_file,
            head_map_file=head_map_file,
            expert_key=expert_key,
            alpha=alphas[0],
            args=args,
        )
        for alpha in alphas:
            steering.alpha = float(alpha)
            run_id = safe_name(f"{subtype}__{direction_key}__{mask_key}__a{alpha:g}")
            raw_path = raw_dir / f"{run_id}.jsonl"
            preds: List[Dict[str, Any]] = []
            if args.skip_existing and raw_path.exists():
                preds = load_jsonl(raw_path)
            else:
                for idx, item in enumerate(items, 1):
                    raw, pred = generate_one(generator, item, steering, args)
                    base = baseline_by_id.get(item["_id"], {})
                    base_pred = str(base.get("pred", "invalid"))
                    label = item["_label"]
                    row = {
                        "eval_subset": subtype,
                        "method": "steered",
                        "id": item["_id"],
                        "image_path": item.get("image_path", ""),
                        "question": item.get("question", ""),
                        "prompt": item["_prompt"],
                        "label": label,
                        "raw_output": raw,
                        "pred": pred,
                        "baseline_pred": base_pred,
                        "baseline_raw_output": base.get("raw_output", ""),
                        "direction_key": direction_key,
                        "mask_key": mask_key,
                        "match_type": match_type,
                        "alpha": alpha,
                    }
                    preds.append(row)
                    if base_pred != pred:
                        base_correct = base_pred == label
                        steered_correct = pred == label
                        if steered_correct and not base_correct:
                            change_type = "wrong_to_right"
                        elif base_correct and not steered_correct:
                            change_type = "right_to_wrong"
                        else:
                            change_type = "changed_neutral"
                        changed_rows.append(
                            {
                                "id": item["_id"],
                                "eval_subset": subtype,
                                "image_path": item.get("image_path", ""),
                                "question": item.get("question", ""),
                                "gt_answer": label,
                                "baseline_answer": base.get("raw_output", ""),
                                "baseline_pred": base_pred,
                                "steered_answer": raw,
                                "steered_pred": pred,
                                "direction_key": direction_key,
                                "mask_key": mask_key,
                                "match_type": match_type,
                                "alpha": alpha,
                                "change_type": change_type,
                            }
                        )
                    if args.progress_every and idx % args.progress_every == 0:
                        print(
                            f"[{subtype} direction={direction_key} mask={mask_key} alpha={alpha:g}] "
                            f"processed {idx}/{len(items)}",
                            flush=True,
                        )
                write_jsonl(raw_path, preds)
            metrics = summarize_predictions(preds)
            wrong_to_right = 0
            right_to_wrong = 0
            changed_pred = 0
            for row in preds:
                base = baseline_by_id.get(str(row["id"]), {})
                base_pred = str(base.get("pred", "invalid"))
                pred = str(row["pred"])
                label = str(row["label"])
                if base_pred != pred:
                    changed_pred += 1
                    if pred == label and base_pred != label:
                        wrong_to_right += 1
                    elif base_pred == label and pred != label:
                        right_to_wrong += 1
            summary_rows.append(
                {
                    "eval_subset": subtype,
                    "method": "steered",
                    "direction_key": direction_key,
                    "mask_key": mask_key,
                    "match_type": match_type,
                    "alpha": alpha,
                    **metrics,
                    "wrong_to_right": wrong_to_right,
                    "right_to_wrong": right_to_wrong,
                    "changed_pred": changed_pred,
                }
            )
        steering.remove()

    fieldnames = [
        "eval_subset",
        "method",
        "direction_key",
        "mask_key",
        "match_type",
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
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    write_jsonl(out_dir / "changed_cases.jsonl", changed_rows)
    print(f"Wrote mask steering summary to {summary_path}")
    print(f"Wrote changed cases to {out_dir / 'changed_cases.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
