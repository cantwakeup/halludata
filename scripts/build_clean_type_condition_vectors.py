#!/usr/bin/env python3
"""Build condition-balanced clean type/subtype vectors and head masks.

This is the vector builder for data/clean_type_minpair_v2.

The key difference from the older subtype builder is condition balancing:

1. sample deltas are grouped by metadata["condition_key"];
2. deltas are averaged inside each condition;
3. condition means are averaged with equal weight.

This prevents high-frequency nuisance conditions, object classes, colors, or
predicates from dominating a subtype direction.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ATTRIBUTE_SUBTYPES = [
    "attr_color_clean",
    "attr_count_clean",
    "attr_state_clean",
    "attr_material_clean",
    "attr_shape_clean",
    "attr_action_single_clean",
]
RELATION_SUBTYPES = [
    "rel_left_right_clean",
    "rel_above_below_clean",
    "rel_holding_wearing_clean",
    "rel_sitting_riding_clean",
]
SUBTYPES = ATTRIBUTE_SUBTYPES + RELATION_SUBTYPES
EXPERT_TYPES = ["attr", "rel"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--yesno-direction", default="")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--sample-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--condition-normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--remove-yesno", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yesno-mode", choices=["answer_token", "dataset_pair", "none"], default="answer_token")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("build_clean_type_condition_vectors.py requires torch.") from exc


def torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_rows(torch: Any, x: Any, eps: float = 1e-12) -> Any:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def normalize_vec(torch: Any, x: Any, eps: float = 1e-12) -> Any:
    flat = x.float().reshape(-1)
    return (flat / flat.norm().clamp_min(eps)).reshape(x.shape)


def cosine(torch: Any, a: Any, b: Any) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = af.norm().item() * bf.norm().item()
    if denom <= 0:
        return 0.0
    return float(torch.dot(af, bf).item() / denom)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def subtype_to_type(subtype: str) -> str:
    if subtype.startswith("attr_"):
        return "attr"
    if subtype.startswith("rel_"):
        return "rel"
    return str(subtype).split("_", 1)[0]


def metadata_type(row: Mapping[str, Any]) -> str:
    expert_type = str(row.get("expert_type", "")).strip()
    if expert_type:
        return expert_type
    return subtype_to_type(str(row.get("subtype", "")))


def condition_key(row: Mapping[str, Any]) -> str:
    value = row.get("condition_key")
    if value in (None, ""):
        meta = row.get("metadata", {})
        if isinstance(meta, Mapping):
            value = meta.get("condition_key")
    if value in (None, ""):
        value = [row.get("subtype", ""), row.get("id", "")]
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def group_indices_by_condition(metadata: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> list[tuple[str, list[int]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        grouped[condition_key(metadata[idx])].append(int(idx))
    return sorted(grouped.items(), key=lambda item: item[0])


def flatten_sample_delta(torch: Any, delta: Any, sample_normalize: bool) -> Any:
    x = delta.float().reshape(int(delta.shape[0]), -1)
    if sample_normalize:
        x = normalize_rows(torch, x)
    return x


def condition_balanced_vector(
    torch: Any,
    delta: Any,
    metadata: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    sample_normalize: bool,
    condition_normalize: bool,
) -> tuple[Any, dict[str, Any]]:
    groups = group_indices_by_condition(metadata, indices)
    if not groups:
        raise ValueError("Cannot build condition-balanced vector from an empty group.")
    condition_means = []
    condition_sizes = []
    raw_sample_norms = []
    for _key, group in groups:
        idx = torch.tensor(group, dtype=torch.long)
        samples = delta.index_select(0, idx).float()
        raw_sample_norms.extend(samples.reshape(int(samples.shape[0]), -1).norm(dim=1).tolist())
        x = flatten_sample_delta(torch, samples, sample_normalize)
        mean = x.mean(dim=0)
        if condition_normalize:
            mean = normalize_vec(torch, mean)
        condition_means.append(mean)
        condition_sizes.append(len(group))
    stacked = torch.stack(condition_means, dim=0)
    vector = stacked.mean(dim=0).reshape(delta.shape[1:]).float()
    condition_norms = stacked.norm(dim=1)
    return vector, {
        "num_samples": int(len(indices)),
        "num_conditions": int(len(groups)),
        "condition_size_min": int(min(condition_sizes)),
        "condition_size_max": int(max(condition_sizes)),
        "condition_size_mean": float(sum(condition_sizes) / len(condition_sizes)),
        "sample_norm_mean": float(sum(raw_sample_norms) / len(raw_sample_norms)) if raw_sample_norms else 0.0,
        "condition_mean_norm_mean": float(condition_norms.mean().item()),
        "condition_mean_norm_max": float(condition_norms.max().item()),
        "vector_norm": float(vector.reshape(-1).norm().item()),
        "sample_normalize": bool(sample_normalize),
        "condition_normalize": bool(condition_normalize),
    }


def condition_balanced_energy_score(
    torch: Any,
    delta: Any,
    metadata: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    sample_normalize: bool,
) -> tuple[Any, dict[str, Any]]:
    groups = group_indices_by_condition(metadata, indices)
    if not groups:
        raise ValueError("Cannot build condition-balanced energy score from an empty group.")
    scores = []
    condition_sizes = []
    for _key, group in groups:
        idx = torch.tensor(group, dtype=torch.long)
        samples = delta.index_select(0, idx).float()
        if sample_normalize:
            flat = normalize_rows(torch, samples.reshape(int(samples.shape[0]), -1))
            samples = flat.reshape_as(samples)
        score = torch.sqrt(samples.square().sum(dim=-1).mean(dim=0).clamp_min(0.0))
        scores.append(score.float())
        condition_sizes.append(len(group))
    score = torch.stack(scores, dim=0).mean(dim=0).float()
    return score, {
        "num_samples": int(len(indices)),
        "num_conditions": int(len(groups)),
        "condition_size_min": int(min(condition_sizes)),
        "condition_size_max": int(max(condition_sizes)),
        "score_min": float(score.min().item()),
        "score_mean": float(score.mean().item()),
        "score_max": float(score.max().item()),
        "sample_normalize": bool(sample_normalize),
    }


def remove_projection(torch: Any, vector: Any, basis: Any | None) -> tuple[Any, dict[str, float]]:
    if basis is None:
        return vector.float(), {
            "raw_yesno_cosine": 0.0,
            "clean_yesno_cosine": 0.0,
            "projection_norm_ratio": 0.0,
            "clean_norm_over_raw_norm": 1.0,
        }
    v = vector.float().reshape(-1)
    b = basis.float().reshape(-1)
    b = b / b.norm().clamp_min(1e-12)
    raw_norm = v.norm().clamp_min(1e-12)
    proj = torch.dot(v, b) * b
    clean = v - proj
    clean_norm = clean.norm().clamp_min(1e-12)
    return clean.reshape(vector.shape).float(), {
        "raw_yesno_cosine": float(torch.dot(v, b).item() / raw_norm.item()),
        "clean_yesno_cosine": float(torch.dot(clean, b).item() / clean_norm.item()),
        "projection_norm_ratio": float(proj.norm().item() / raw_norm.item()),
        "clean_norm_over_raw_norm": float(clean_norm.item() / raw_norm.item()),
    }


def topk_mask(torch: Any, score: Any, topk: int) -> tuple[Any, list[dict[str, Any]]]:
    if score.ndim != 2:
        raise ValueError(f"Expected score [layers, heads], got {tuple(score.shape)}")
    flat = score.float().reshape(-1)
    k = min(int(topk), int(flat.numel()))
    values, indices = torch.topk(flat, k=k, largest=True, sorted=True)
    layers, heads = int(score.shape[0]), int(score.shape[1])
    mask = torch.zeros((layers, heads), dtype=torch.bool)
    rows = []
    for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        layer = int(idx // heads)
        head = int(idx % heads)
        mask[layer, head] = True
        rows.append({"rank": rank, "layer": layer, "head": head, "score": float(value)})
    return mask, rows


def random_mask(torch: Any, shape: Sequence[int], topk: int, seed: int) -> tuple[Any, list[dict[str, Any]]]:
    layers, heads = int(shape[0]), int(shape[1])
    generator = torch.Generator().manual_seed(int(seed))
    idxs = torch.randperm(layers * heads, generator=generator)[: min(int(topk), layers * heads)]
    mask = torch.zeros((layers, heads), dtype=torch.bool)
    rows = []
    for rank, idx in enumerate(idxs.tolist(), start=1):
        layer = int(idx // heads)
        head = int(idx % heads)
        mask[layer, head] = True
        rows.append({"rank": rank, "layer": layer, "head": head, "score": float(len(idxs) - rank + 1)})
    return mask, rows


def head_norm_score(vector: Any) -> Any:
    if vector.ndim != 3:
        raise ValueError(f"Expected vector [layers, heads, head_dim], got {tuple(vector.shape)}")
    return vector.float().norm(dim=-1)


def mask_overlap(torch: Any, a: Any, b: Any) -> dict[str, float]:
    a = a.to(torch.bool)
    b = b.to(torch.bool)
    inter = int((a & b).sum().item())
    union = int((a | b).sum().item())
    return {"intersection": inter, "jaccard": float(inter / union) if union else 0.0}


def load_yesno_direction(
    torch: Any,
    path: Path | None,
    g_delta: Any,
    metadata: Sequence[Mapping[str, Any]],
    *,
    sample_normalize: bool,
    condition_normalize: bool,
    mode: str,
) -> tuple[Any | None, dict[str, Any]]:
    if mode == "none":
        return None, {"mode": "none"}
    if path is not None and path.exists():
        payload = torch_load(torch, path)
        direction = payload.get("yesno_direction", payload.get("direction"))
        if direction is None:
            raise ValueError(f"Could not find yesno_direction in {path}")
        return direction.float(), {"mode": "answer_token", "path": str(path)}
    if mode == "answer_token":
        mode = "dataset_pair"
    yes_idx = [idx for idx, row in enumerate(metadata) if str(row.get("gt_answer", row.get("label", ""))).lower() == "yes"]
    no_idx = [idx for idx, row in enumerate(metadata) if str(row.get("gt_answer", row.get("label", ""))).lower() == "no"]
    if not yes_idx or not no_idx:
        return None, {"mode": "dataset_pair", "warning": "Missing yes/no rows; yes/no direction disabled."}
    yes_vec, yes_info = condition_balanced_vector(
        torch,
        g_delta,
        metadata,
        yes_idx,
        sample_normalize=sample_normalize,
        condition_normalize=condition_normalize,
    )
    no_vec, no_info = condition_balanced_vector(
        torch,
        g_delta,
        metadata,
        no_idx,
        sample_normalize=sample_normalize,
        condition_normalize=condition_normalize,
    )
    return (yes_vec - no_vec).float(), {
        "mode": "dataset_pair",
        "warning": "answer_token yes/no direction file was missing; used dataset yes/no split.",
        "yes": yes_info,
        "no": no_info,
    }


def vector_stats(torch: Any, tensor: Any) -> dict[str, Any]:
    t = tensor.float()
    head_norms = t.norm(dim=-1).reshape(-1)
    return {
        "shape": [int(dim) for dim in t.shape],
        "flat_norm": float(t.reshape(-1).norm().item()),
        "head_norm_mean": float(head_norms.mean().item()),
        "head_norm_max": float(head_norms.max().item()),
        "finite": bool(torch.isfinite(t).all().item()),
    }


def md_table(headers: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
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


def top_rows(counter: Mapping[str, int], limit: int = 20) -> list[dict[str, Any]]:
    return [{"item": key, "count": value} for key, value in Counter(counter).most_common(limit)]


def pairwise_overlap_rows(torch: Any, masks: Mapping[str, Any], keys: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for a, b in itertools.combinations(keys, 2):
        if a in masks and b in masks:
            rows.append({"mask_a": a, "mask_b": b, **mask_overlap(torch, masks[a], masks[b])})
    return rows


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    schema: Mapping[str, Any],
    counts: Mapping[str, int],
    condition_counts: Mapping[str, int],
    vectors: Mapping[str, Any],
    masks: Mapping[str, Any],
    top_heads: Mapping[str, Sequence[Mapping[str, Any]]],
    diagnostics: Mapping[str, Any],
    yesno_info: Mapping[str, Any],
    torch: Any,
) -> None:
    lines: list[str] = []
    lines.append("# Clean Type Condition-Balanced Vector Report")
    lines.append("")
    lines.append(f"- Activations: `{args.activations}`")
    lines.append(f"- Output vectors: `{args.output}`")
    lines.append(f"- Output masks: `{args.mask_output}`")
    lines.append(f"- TopK: `{args.topk}`")
    lines.append(f"- sample_normalize: `{args.sample_normalize}`")
    lines.append(f"- condition_normalize: `{args.condition_normalize}`")
    lines.append(f"- remove_yesno: `{args.remove_yesno}`")
    lines.append(f"- yesno mode: `{yesno_info.get('mode')}`")
    if yesno_info.get("warning"):
        lines.append(f"- yesno warning: {yesno_info['warning']}")
    lines.append("")
    lines.append("## Activation Schema")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dict(schema), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Counts")
    lines.append(md_table(["subtype", "samples", "conditions"], [{"subtype": key, "samples": counts.get(key, 0), "conditions": condition_counts.get(key, 0)} for key in SUBTYPES]))
    lines.append("")
    lines.append("## Condition-Balanced Vector Diagnostics")
    vector_diag_rows = []
    for key, info in diagnostics.get("vectors", {}).items():
        vector_diag_rows.append({"vector": key, **info})
    lines.append(md_table(["vector", "num_samples", "num_conditions", "condition_size_min", "condition_size_max", "condition_size_mean", "sample_norm_mean", "condition_mean_norm_mean", "vector_norm"], vector_diag_rows))
    lines.append("")
    lines.append("## Yes/No Projection")
    projection_rows = [{"vector": key, **value} for key, value in diagnostics.get("yesno_projection", {}).items()]
    lines.append(md_table(["vector", "raw_yesno_cosine", "clean_yesno_cosine", "projection_norm_ratio", "clean_norm_over_raw_norm"], projection_rows))
    lines.append("")
    lines.append("## Vector Norms")
    norm_rows = []
    for key in sorted(vectors):
        stats = vector_stats(torch, vectors[key])
        norm_rows.append({"vector": key, **stats})
    lines.append(md_table(["vector", "flat_norm", "head_norm_mean", "head_norm_max", "finite"], norm_rows))
    lines.append("")
    selected = [key for key in ["g_all_clean", "g_attr_clean", "g_rel_clean"] if key in vectors]
    selected += [f"s_{subtype}_clean" for subtype in SUBTYPES if f"s_{subtype}_clean" in vectors]
    if selected:
        lines.append("## Cosine Matrix")
        rows = []
        for key in selected:
            row = {"vector": key}
            for other in selected:
                row[other] = cosine(torch, vectors[key], vectors[other])
            rows.append(row)
        lines.append(md_table(["vector"] + selected, rows))
        lines.append("")
    energy_keys = [f"mask_s_{subtype}_energy_top{args.topk}" for subtype in SUBTYPES if f"mask_s_{subtype}_energy_top{args.topk}" in masks]
    mean_keys = [f"mask_s_{subtype}_mean_top{args.topk}" for subtype in SUBTYPES if f"mask_s_{subtype}_mean_top{args.topk}" in masks]
    g_keys = [key for key in [f"mask_g_all_norm_top{args.topk}", f"mask_g_attr_norm_top{args.topk}", f"mask_g_rel_norm_top{args.topk}"] if key in masks]
    lines.append("## Mask Overlap: S Energy")
    lines.append(md_table(["mask_a", "mask_b", "intersection", "jaccard"], pairwise_overlap_rows(torch, masks, energy_keys)))
    lines.append("")
    lines.append("## Mask Overlap: G Norm")
    lines.append(md_table(["mask_a", "mask_b", "intersection", "jaccard"], pairwise_overlap_rows(torch, masks, g_keys)))
    lines.append("")
    lines.append("## S Mean vs S Energy")
    rows = []
    for subtype in SUBTYPES:
        mean_key = f"mask_s_{subtype}_mean_top{args.topk}"
        energy_key = f"mask_s_{subtype}_energy_top{args.topk}"
        if mean_key in masks and energy_key in masks:
            rows.append({"subtype": subtype, **mask_overlap(torch, masks[mean_key], masks[energy_key])})
    lines.append(md_table(["subtype", "intersection", "jaccard"], rows))
    lines.append("")
    lines.append("## Top Heads")
    for key in sorted(top_heads):
        lines.append(f"### {key}")
        lines.append(md_table(["rank", "layer", "head", "score"], list(top_heads[key])[: min(args.topk, 20)]))
        lines.append("")
    lines.append("## Automatic Interpretation")
    lines.append("")
    max_proj = max((abs(safe_float(row.get("clean_yesno_cosine", 0.0))) for row in diagnostics.get("yesno_projection", {}).values()), default=0.0)
    if max_proj < 1e-3:
        lines.append("- Yes/no projection removal is numerically clean.")
    else:
        lines.append(f"- Some clean vectors retain yes/no cosine up to {max_proj:.4f}; inspect yes/no_direction source.")
    if "rel_holding_wearing_clean" in counts and counts["rel_holding_wearing_clean"] > 0:
        lines.append("- rel_holding_wearing_clean is present; remember the data audit showed wearing is somewhat dominant.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    torch = load_torch()
    output_path = resolve(args.output)
    mask_output = resolve(args.mask_output)
    report_path = resolve(args.report_output)
    for path in [output_path, mask_output, report_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}. Pass --overwrite.")

    payload = torch_load(torch, resolve(args.activations))
    for key in ["z_visual", "z_fact_text", "z_counterfact_text", "metadata"]:
        if key not in payload:
            raise KeyError(f"Activation cache missing key: {key}")
    z_visual = payload["z_visual"]
    z_fact = payload["z_fact_text"]
    z_counter = payload["z_counterfact_text"]
    metadata = [dict(row) for row in payload["metadata"]]
    if z_visual.shape != z_fact.shape or z_visual.shape != z_counter.shape:
        raise ValueError(f"Activation shapes differ: {z_visual.shape}, {z_fact.shape}, {z_counter.shape}")
    if z_visual.ndim != 4:
        raise ValueError(f"Expected [N,L,H,D] activations, got {tuple(z_visual.shape)}")
    if len(metadata) != int(z_visual.shape[0]):
        raise ValueError(f"metadata len {len(metadata)} != tensor N {z_visual.shape[0]}")
    if not all(condition_key(row) for row in metadata):
        raise ValueError("Some metadata rows have empty condition_key.")

    vector_shape = tuple(int(dim) for dim in z_visual.shape[1:])
    layer_count, head_count, head_dim = vector_shape
    if int(args.topk) > layer_count * head_count:
        raise ValueError(f"--topk {args.topk} exceeds total heads {layer_count * head_count}")

    subtype_indices: dict[str, list[int]] = defaultdict(list)
    type_indices: dict[str, list[int]] = defaultdict(list)
    all_indices = list(range(len(metadata)))
    for idx, row in enumerate(metadata):
        subtype = str(row.get("subtype", ""))
        typ = metadata_type(row)
        subtype_indices[subtype].append(idx)
        type_indices[typ].append(idx)

    counts = Counter(str(row.get("subtype", "")) for row in metadata)
    condition_counts = {subtype: len({condition_key(metadata[idx]) for idx in idxs}) for subtype, idxs in subtype_indices.items()}
    diagnostics: dict[str, Any] = {"vectors": {}, "yesno_projection": {}, "energy": {}}
    vectors: dict[str, Any] = {}

    g_delta = z_fact.float() - z_visual.float()
    yesno_path = resolve(args.yesno_direction) if str(args.yesno_direction).strip() else None
    yesno_direction, yesno_info = load_yesno_direction(
        torch,
        yesno_path,
        g_delta,
        metadata,
        sample_normalize=bool(args.sample_normalize),
        condition_normalize=bool(args.condition_normalize),
        mode=str(args.yesno_mode),
    )
    if yesno_direction is not None and tuple(int(dim) for dim in yesno_direction.shape) != vector_shape:
        raise ValueError(f"yesno_direction shape {tuple(yesno_direction.shape)} != vector shape {vector_shape}")

    def store_vector(name: str, delta: Any, indices: Sequence[int]) -> None:
        raw, info = condition_balanced_vector(
            torch,
            delta,
            metadata,
            indices,
            sample_normalize=bool(args.sample_normalize),
            condition_normalize=bool(args.condition_normalize),
        )
        if bool(args.remove_yesno):
            clean, projection = remove_projection(torch, raw, yesno_direction)
        else:
            clean, projection = remove_projection(torch, raw, None)
        vectors[f"{name}_raw"] = raw.float()
        vectors[f"{name}_clean"] = clean.float()
        diagnostics["vectors"][name] = info
        diagnostics["yesno_projection"][name] = projection

    store_vector("g_all", g_delta, all_indices)
    for typ in EXPERT_TYPES:
        if type_indices.get(typ):
            store_vector(f"g_{typ}", g_delta, type_indices[typ])

    del g_delta

    s_delta = z_fact.float() - z_counter.float()
    for subtype in SUBTYPES:
        idxs = subtype_indices.get(subtype, [])
        if idxs:
            store_vector(f"s_{subtype}", s_delta, idxs)

    for name, tensor in vectors.items():
        if tuple(int(dim) for dim in tensor.shape) != vector_shape:
            raise ValueError(f"Vector {name} shape {tuple(tensor.shape)} != {vector_shape}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Vector {name} contains NaN/Inf.")
    if yesno_direction is not None:
        vectors["yesno_direction"] = yesno_direction.float()

    masks: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    top_heads: dict[str, list[dict[str, Any]]] = {}

    for label in ["all", "attr", "rel"]:
        vector_key = f"g_{label}_clean"
        if vector_key not in vectors:
            continue
        score_key = f"score_g_{label}_norm"
        mask_key = f"mask_g_{label}_norm_top{args.topk}"
        score = head_norm_score(vectors[vector_key])
        mask, rows = topk_mask(torch, score, int(args.topk))
        scores[score_key] = score
        masks[mask_key] = mask
        top_heads[mask_key] = rows

    for subtype in SUBTYPES:
        vector_key = f"s_{subtype}_clean"
        if vector_key in vectors:
            score_key = f"score_s_{subtype}_mean"
            mask_key = f"mask_s_{subtype}_mean_top{args.topk}"
            score = head_norm_score(vectors[vector_key])
            mask, rows = topk_mask(torch, score, int(args.topk))
            scores[score_key] = score
            masks[mask_key] = mask
            top_heads[mask_key] = rows
        idxs = subtype_indices.get(subtype, [])
        if idxs:
            score_key = f"score_s_{subtype}_energy"
            mask_key = f"mask_s_{subtype}_energy_top{args.topk}"
            score, info = condition_balanced_energy_score(
                torch,
                s_delta,
                metadata,
                idxs,
                sample_normalize=bool(args.sample_normalize),
            )
            mask, rows = topk_mask(torch, score, int(args.topk))
            scores[score_key] = score
            masks[mask_key] = mask
            top_heads[mask_key] = rows
            diagnostics["energy"][subtype] = info

    for seed in [0, 1, 2]:
        mask_key = f"random_mask_top{args.topk}_seed{seed}"
        mask, rows = random_mask(torch, (layer_count, head_count), int(args.topk), seed)
        masks[mask_key] = mask
        top_heads[mask_key] = rows

    del s_delta

    output_payload = {
        **vectors,
        "vectors": vectors,
        "layers": list(range(layer_count)),
        "num_heads": int(head_count),
        "head_dim": int(head_dim),
        "hidden_size": int(head_count * head_dim),
        "metadata": {
            "created_by": "scripts/build_clean_type_condition_vectors.py",
            "source_activations": str(resolve(args.activations)),
            "sample_normalize": bool(args.sample_normalize),
            "condition_normalize": bool(args.condition_normalize),
            "remove_yesno": bool(args.remove_yesno),
            "condition_balancing": "mean condition delta first, then equal-weight condition mean",
        },
        "diagnostics": diagnostics,
        "counts_by_subtype": dict(sorted(counts.items())),
        "condition_counts_by_subtype": dict(sorted(condition_counts.items())),
    }
    mask_payload = {
        "masks": masks,
        "scores": scores,
        "top_heads": top_heads,
        "metadata": {
            "created_by": "scripts/build_clean_type_condition_vectors.py",
            "source_activations": str(resolve(args.activations)),
            "source_vectors": str(output_path),
            "topk": int(args.topk),
            "method": "condition_balanced_s_energy_and_vector_norm",
        },
        "diagnostics": {"energy": diagnostics["energy"]},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output_path)
    torch.save(mask_payload, mask_output)
    write_json(output_path.with_suffix(".manifest.json"), output_payload["metadata"] | {"shape": list(vector_shape), "mask_output": str(mask_output)})
    write_json(mask_output.with_suffix(".manifest.json"), mask_payload["metadata"] | {"shape": [layer_count, head_count]})
    write_report(
        report_path,
        args=args,
        schema=payload.get("schema", {}),
        counts=dict(sorted(counts.items())),
        condition_counts=condition_counts,
        vectors=vectors,
        masks=masks,
        top_heads=top_heads,
        diagnostics=diagnostics,
        yesno_info=yesno_info,
        torch=torch,
    )
    print(f"Wrote clean condition vectors to {output_path}")
    print(f"Wrote clean condition masks to {mask_output}")
    print(f"Wrote report to {report_path}")
    print(json.dumps({"counts_by_subtype": dict(sorted(counts.items())), "condition_counts_by_subtype": dict(sorted(condition_counts.items()))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
