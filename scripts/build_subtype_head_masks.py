#!/usr/bin/env python3
"""Build type/subtype-specific head masks for shared-direction steering.

The main mask in this experiment is based on sample-level semantic energy:

    s_delta_i = z_fact_text_i - z_counterfact_text_i
    score[l,h] = sqrt(mean_i ||s_delta_i[l,h,:]||_2^2)

This avoids relying on the subtype mean direction, which can cancel out even
when individual examples have strong subtype-sensitive head activity.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch


SUBTYPES = [
    "cat_random",
    "cat_popular",
    "cat_hard",
    "attr_color",
    "attr_count",
    "rel_spatial",
    "rel_contact",
]

G_KEYS = {
    "all": "g_all_clean",
    "cat": "g_cat_clean",
    "attr": "g_attr_clean",
    "rel": "g_rel_clean",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True, help="Subtype train activation .pt file.")
    ap.add_argument("--vectors", required=True, help="Subtype vector .pt file.")
    ap.add_argument("--output", required=True, help="Output subtype head masks .pt path.")
    ap.add_argument("--report-output", required=True, help="Output Markdown report path.")
    ap.add_argument("--topk", type=int, default=64, help="Number of heads selected globally.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def load_pt(path: str | Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_vectors(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if "vectors" in payload and isinstance(payload["vectors"], Mapping):
        return payload["vectors"]
    return {k: v for k, v in payload.items() if torch.is_tensor(v)}


def as_float32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32).cpu()


def finite_check(name: str, t: torch.Tensor) -> None:
    if not torch.isfinite(t).all().item():
        raise ValueError(f"{name} contains NaN/Inf")


def head_norm_score(vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim != 3:
        raise ValueError(f"Expected vector shape [layers, heads, dim], got {tuple(vec.shape)}")
    vec = as_float32(vec)
    finite_check("vector", vec)
    return torch.linalg.vector_norm(vec, dim=-1)


def topk_mask(score: torch.Tensor, topk: int) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    if score.ndim != 2:
        raise ValueError(f"Expected score shape [layers, heads], got {tuple(score.shape)}")
    if topk <= 0:
        raise ValueError("--topk must be positive")
    layers, heads = score.shape
    flat = score.flatten()
    k = min(topk, flat.numel())
    vals, idxs = torch.topk(flat, k=k, largest=True, sorted=True)
    mask = torch.zeros_like(score, dtype=torch.bool)
    rows: List[Dict[str, Any]] = []
    for val, idx in zip(vals.tolist(), idxs.tolist()):
        layer = int(idx // heads)
        head = int(idx % heads)
        mask[layer, head] = True
        rows.append({"layer": layer, "head": head, "score": float(val)})
    return mask, rows


def random_mask_like(shape: Sequence[int], topk: int, seed: int) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    layers, heads = int(shape[0]), int(shape[1])
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(layers * heads, generator=gen)[: min(topk, layers * heads)]
    mask = torch.zeros((layers, heads), dtype=torch.bool)
    rows: List[Dict[str, Any]] = []
    for rank, idx in enumerate(perm.tolist()):
        layer = int(idx // heads)
        head = int(idx % heads)
        mask[layer, head] = True
        rows.append({"layer": layer, "head": head, "score": float(topk - rank)})
    return mask, rows


def layer_matched_random_mask(
    reference_mask: torch.Tensor, seed: int
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    layers, heads = reference_mask.shape
    gen = torch.Generator().manual_seed(seed)
    out = torch.zeros_like(reference_mask, dtype=torch.bool)
    rows: List[Dict[str, Any]] = []
    for layer in range(layers):
        count = int(reference_mask[layer].sum().item())
        if count <= 0:
            continue
        idxs = torch.randperm(heads, generator=gen)[:count]
        for head in idxs.tolist():
            out[layer, int(head)] = True
            rows.append({"layer": layer, "head": int(head), "score": 1.0})
    rows.sort(key=lambda r: (r["layer"], r["head"]))
    return out, rows


def mask_overlap(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a_bool = a.to(torch.bool)
    b_bool = b.to(torch.bool)
    inter = int((a_bool & b_bool).sum().item())
    union = int((a_bool | b_bool).sum().item())
    return {
        "intersection": inter,
        "jaccard": float(inter / union) if union else 0.0,
    }


def score_stats(score: torch.Tensor) -> Dict[str, float]:
    score = score.to(torch.float32)
    return {
        "min": float(score.min().item()),
        "max": float(score.max().item()),
        "mean": float(score.mean().item()),
        "std": float(score.std(unbiased=False).item()),
    }


def md_table(headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def summarize_schema(payload: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            out[key] = {"type": "Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, list):
            out[key] = {"type": "list", "len": len(value)}
        elif isinstance(value, dict):
            out[key] = {"type": "dict", "keys": list(value.keys())[:20], "len": len(value)}
        else:
            out[key] = {"type": type(value).__name__}
    return out


def metadata_subtype(row: Mapping[str, Any]) -> str:
    subtype = row.get("subtype")
    if subtype:
        return str(subtype)
    meta = row.get("metadata")
    if isinstance(meta, Mapping) and meta.get("subtype"):
        return str(meta["subtype"])
    raise KeyError(f"Could not find subtype in metadata row keys={list(row.keys())}")


def compute_energy_scores(
    z_fact: torch.Tensor,
    z_counterfact: torch.Tensor,
    indices: Sequence[int],
    chunk_size: int = 64,
) -> torch.Tensor:
    if not indices:
        raise ValueError("Cannot compute energy score for empty subtype")
    if z_fact.shape != z_counterfact.shape:
        raise ValueError(f"z_fact and z_counterfact shape mismatch: {z_fact.shape} vs {z_counterfact.shape}")
    layers, heads = int(z_fact.shape[1]), int(z_fact.shape[2])
    accum = torch.zeros((layers, heads), dtype=torch.float64)
    count = 0
    idx_tensor = torch.tensor(list(indices), dtype=torch.long)
    for start in range(0, len(idx_tensor), chunk_size):
        idx = idx_tensor[start : start + chunk_size]
        delta = z_fact.index_select(0, idx).to(torch.float32) - z_counterfact.index_select(0, idx).to(torch.float32)
        finite_check("s_delta", delta)
        accum += delta.square().sum(dim=-1).sum(dim=0).to(torch.float64)
        count += int(delta.shape[0])
        del delta
    mean_sq = accum / max(count, 1)
    return torch.sqrt(mean_sq).to(torch.float32)


def build_report(
    *,
    args: argparse.Namespace,
    act_schema: Mapping[str, Any],
    vector_keys: Sequence[str],
    counts: Mapping[str, int],
    masks: Mapping[str, torch.Tensor],
    scores: Mapping[str, torch.Tensor],
    top_heads: Mapping[str, Sequence[Mapping[str, Any]]],
    diagnostics: Mapping[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Subtype Head Mask Report")
    lines.append("")
    lines.append(f"- Activations: `{args.activations}`")
    lines.append(f"- Vectors: `{args.vectors}`")
    lines.append(f"- TopK: `{args.topk}`")
    lines.append(f"- Masks written: `{args.output}`")
    lines.append("")

    lines.append("## Activation Schema")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(act_schema, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Vector Keys")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(list(vector_keys), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Subtype Counts")
    lines.append("")
    lines.append(md_table(["subtype", "count"], [{"subtype": k, "count": v} for k, v in counts.items()]))
    lines.append("")

    stat_rows = []
    for key, score in scores.items():
        row = {"score": key}
        row.update(score_stats(score))
        stat_rows.append(row)
    lines.append("## Score Statistics")
    lines.append("")
    lines.append(md_table(["score", "min", "max", "mean", "std"], stat_rows))
    lines.append("")

    def overlap_rows(keys_a: Sequence[str], keys_b: Sequence[str], same_set: bool = False) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        pairs: Iterable[Tuple[str, str]]
        if same_set:
            pairs = itertools.combinations(keys_a, 2)
        else:
            pairs = itertools.product(keys_a, keys_b)
        for a, b in pairs:
            if a not in masks or b not in masks:
                continue
            ov = mask_overlap(masks[a], masks[b])
            rows.append({"mask_a": a, "mask_b": b, **ov})
        return rows

    g_mask_keys = [k for k in masks if k.startswith("mask_g_") and k.endswith(f"_top{args.topk}")]
    energy_keys = [k for k in masks if "_energy_" in k]
    mean_keys = [k for k in masks if "_mean_" in k]
    random_keys = [k for k in masks if k.startswith("random_mask") or k.startswith("layer_matched_random")]

    lines.append("## G Mask Overlap")
    lines.append("")
    lines.append(md_table(["mask_a", "mask_b", "intersection", "jaccard"], overlap_rows(g_mask_keys, g_mask_keys, True)))
    lines.append("")

    lines.append("## S Energy Mask Overlap")
    lines.append("")
    lines.append(md_table(["mask_a", "mask_b", "intersection", "jaccard"], overlap_rows(energy_keys, energy_keys, True)))
    lines.append("")

    lines.append("## G vs S Energy Overlap")
    lines.append("")
    lines.append(md_table(["mask_a", "mask_b", "intersection", "jaccard"], overlap_rows(g_mask_keys, energy_keys)))
    lines.append("")

    mean_energy_rows = []
    for subtype in SUBTYPES:
        mean_key = f"mask_s_{subtype}_mean_top{args.topk}"
        energy_key = f"mask_s_{subtype}_energy_top{args.topk}"
        if mean_key in masks and energy_key in masks:
            mean_energy_rows.append({"subtype": subtype, **mask_overlap(masks[mean_key], masks[energy_key])})
    lines.append("## S Mean vs S Energy Overlap")
    lines.append("")
    lines.append(md_table(["subtype", "intersection", "jaccard"], mean_energy_rows))
    lines.append("")

    if random_keys:
        lines.append("## Random Mask Overlap With Energy Masks")
        lines.append("")
        lines.append(md_table(["mask_a", "mask_b", "intersection", "jaccard"], overlap_rows(random_keys, energy_keys)))
        lines.append("")

    lines.append("## Top Heads")
    lines.append("")
    for key in sorted(top_heads):
        lines.append(f"### {key}")
        rows = [
            {"rank": idx + 1, "layer": r["layer"], "head": r["head"], "score": r.get("score", "")}
            for idx, r in enumerate(top_heads[key][: args.topk])
        ]
        lines.append(md_table(["rank", "layer", "head", "score"], rows))
        lines.append("")

    lines.append("## Checks")
    lines.append("")
    check_rows = []
    for key, val in diagnostics.get("checks", {}).items():
        check_rows.append({"check": key, "value": val})
    lines.append(md_table(["check", "value"], check_rows))
    lines.append("")

    lines.append("## Automatic Interpretation")
    lines.append("")
    lines.extend(diagnostics.get("interpretation", ["- No automatic interpretation generated."]))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    report_output = Path(args.report_output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Pass --overwrite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    act_payload = load_pt(args.activations)
    vec_payload = load_pt(args.vectors)
    vectors = get_vectors(vec_payload)

    required_act = ["z_fact_text", "z_counterfact_text", "metadata"]
    missing = [k for k in required_act if k not in act_payload]
    if missing:
        raise KeyError(f"Activation file missing required keys: {missing}")
    z_fact = act_payload["z_fact_text"]
    z_counterfact = act_payload["z_counterfact_text"]
    metadata = act_payload["metadata"]
    if not torch.is_tensor(z_fact) or not torch.is_tensor(z_counterfact):
        raise TypeError("z_fact_text and z_counterfact_text must be tensors")
    if len(metadata) != int(z_fact.shape[0]):
        raise ValueError(f"metadata length {len(metadata)} != tensor N {z_fact.shape[0]}")
    if z_fact.ndim != 4:
        raise ValueError(f"Expected activation shape [N,L,H,D], got {tuple(z_fact.shape)}")

    layer_count, head_count = int(z_fact.shape[1]), int(z_fact.shape[2])
    expected_score_shape = (layer_count, head_count)
    if args.topk > layer_count * head_count:
        raise ValueError(f"--topk {args.topk} exceeds total heads {layer_count * head_count}")

    indices_by_subtype: Dict[str, List[int]] = defaultdict(list)
    subtype_counter: Counter[str] = Counter()
    for idx, row in enumerate(metadata):
        subtype = metadata_subtype(row)
        indices_by_subtype[subtype].append(idx)
        subtype_counter[subtype] += 1

    masks: Dict[str, torch.Tensor] = {}
    scores: Dict[str, torch.Tensor] = {}
    top_heads: Dict[str, List[Dict[str, Any]]] = {}

    # 1. Direction-vector norm masks.
    for label, vec_key in G_KEYS.items():
        if vec_key not in vectors:
            continue
        score_key = f"score_g_{label}_norm"
        mask_key = f"mask_g_{label}_norm_top{args.topk}"
        score = head_norm_score(vectors[vec_key])
        if tuple(score.shape) != expected_score_shape:
            raise ValueError(f"{vec_key} score shape {tuple(score.shape)} != activation heads {expected_score_shape}")
        mask, rows = topk_mask(score, args.topk)
        scores[score_key] = score
        masks[mask_key] = mask
        top_heads[mask_key] = rows

    # 2. S-subtype mean-vector norm masks.
    for subtype in SUBTYPES:
        vec_key = f"s_{subtype}_clean"
        if vec_key not in vectors:
            continue
        score_key = f"score_s_{subtype}_mean"
        mask_key = f"mask_s_{subtype}_mean_top{args.topk}"
        score = head_norm_score(vectors[vec_key])
        if tuple(score.shape) != expected_score_shape:
            raise ValueError(f"{vec_key} score shape {tuple(score.shape)} != activation heads {expected_score_shape}")
        mask, rows = topk_mask(score, args.topk)
        scores[score_key] = score
        masks[mask_key] = mask
        top_heads[mask_key] = rows

    # 3. S-subtype sample energy masks.
    for subtype in SUBTYPES:
        idxs = indices_by_subtype.get(subtype, [])
        if not idxs:
            continue
        score_key = f"score_s_{subtype}_energy"
        mask_key = f"mask_s_{subtype}_energy_top{args.topk}"
        score = compute_energy_scores(z_fact, z_counterfact, idxs)
        if tuple(score.shape) != expected_score_shape:
            raise ValueError(f"{subtype} energy score shape {tuple(score.shape)} != {expected_score_shape}")
        mask, rows = topk_mask(score, args.topk)
        scores[score_key] = score
        masks[mask_key] = mask
        top_heads[mask_key] = rows

    # 4. Random controls.
    for seed in [0, 1, 2]:
        mask_key = f"random_mask_top{args.topk}_seed{seed}"
        mask, rows = random_mask_like(expected_score_shape, args.topk, seed)
        masks[mask_key] = mask
        top_heads[mask_key] = rows

    # Optional layer-matched random controls based on each subtype energy mask.
    for subtype in SUBTYPES:
        ref_key = f"mask_s_{subtype}_energy_top{args.topk}"
        if ref_key not in masks:
            continue
        mask_key = f"layer_matched_random_{subtype}_top{args.topk}_seed0"
        mask, rows = layer_matched_random_mask(masks[ref_key], args.seed)
        masks[mask_key] = mask
        top_heads[mask_key] = rows

    checks: Dict[str, Any] = {}
    for subtype in ["rel_spatial", "attr_color", "attr_count", "rel_contact"]:
        mean_key = f"score_s_{subtype}_mean"
        energy_key = f"score_s_{subtype}_energy"
        if mean_key in scores and energy_key in scores:
            checks[f"{subtype}_mean_max"] = f"{scores[mean_key].max().item():.4f}"
            checks[f"{subtype}_energy_max"] = f"{scores[energy_key].max().item():.4f}"

    pair_checks = [
        (
            "attr_color_vs_attr_count_energy",
            f"mask_s_attr_color_energy_top{args.topk}",
            f"mask_s_attr_count_energy_top{args.topk}",
        ),
        (
            "rel_spatial_vs_rel_contact_energy",
            f"mask_s_rel_spatial_energy_top{args.topk}",
            f"mask_s_rel_contact_energy_top{args.topk}",
        ),
        ("g_cat_vs_g_attr_norm", f"mask_g_cat_norm_top{args.topk}", f"mask_g_attr_norm_top{args.topk}"),
        ("g_cat_vs_g_rel_norm", f"mask_g_cat_norm_top{args.topk}", f"mask_g_rel_norm_top{args.topk}"),
        ("g_attr_vs_g_rel_norm", f"mask_g_attr_norm_top{args.topk}", f"mask_g_rel_norm_top{args.topk}"),
    ]
    for label, a, b in pair_checks:
        if a in masks and b in masks:
            ov = mask_overlap(masks[a], masks[b])
            checks[label] = f"intersection={ov['intersection']}, jaccard={ov['jaccard']:.4f}"

    interpretation = []
    for subtype in ["rel_spatial", "attr_color"]:
        mean_key = f"score_s_{subtype}_mean"
        energy_key = f"score_s_{subtype}_energy"
        if mean_key in scores and energy_key in scores:
            mean_max = float(scores[mean_key].max().item())
            energy_max = float(scores[energy_key].max().item())
            if energy_max > 0 and mean_max / energy_max < 0.5:
                interpretation.append(
                    f"- `{subtype}` has much weaker mean score than sample-energy score; energy masks may retain signal that mean directions cancel."
                )
    for label, a, b in [
        ("attribute", f"mask_s_attr_color_energy_top{args.topk}", f"mask_s_attr_count_energy_top{args.topk}"),
        ("relation", f"mask_s_rel_spatial_energy_top{args.topk}", f"mask_s_rel_contact_energy_top{args.topk}"),
    ]:
        if a in masks and b in masks:
            ov = mask_overlap(masks[a], masks[b])
            if ov["jaccard"] > 0.65:
                interpretation.append(
                    f"- {label} energy masks are highly overlapping (Jaccard {ov['jaccard']:.2f}); subtype specificity may be weak."
                )
            else:
                interpretation.append(
                    f"- {label} energy masks are meaningfully separated (Jaccard {ov['jaccard']:.2f}); this is promising for mask steering."
                )
    g_keys = [k for k in masks if k.startswith("mask_g_")]
    if len(g_keys) >= 2:
        g_overlaps = [
            mask_overlap(masks[a], masks[b])["jaccard"]
            for a, b in itertools.combinations(g_keys, 2)
        ]
        if g_overlaps and sum(g_overlaps) / len(g_overlaps) > 0.65:
            interpretation.append("- `g_*` norm masks are highly overlapping, consistent with a shared grounding direction.")
    if not interpretation:
        interpretation.append("- Mask diagnostics generated; inspect overlap tables before deciding whether subtype masks are separable.")

    diagnostics = {
        "checks": checks,
        "interpretation": interpretation,
        "score_stats": {k: score_stats(v) for k, v in scores.items()},
        "mask_counts": {k: int(v.sum().item()) for k, v in masks.items()},
    }

    out_payload = {
        "masks": masks,
        "scores": scores,
        "top_heads": top_heads,
        "metadata": {
            "topk": args.topk,
            "source_activations": str(Path(args.activations).resolve()),
            "source_vectors": str(Path(args.vectors).resolve()),
            "method": "sample_energy_and_vector_norm",
            "created_by": "scripts/build_subtype_head_masks.py",
            "subtype_counts": dict(subtype_counter),
            "activation_shape": list(z_fact.shape),
        },
        "diagnostics": diagnostics,
    }
    torch.save(out_payload, output)

    report = build_report(
        args=args,
        act_schema=summarize_schema(act_payload),
        vector_keys=sorted(vectors.keys()),
        counts=dict(sorted(subtype_counter.items())),
        masks=masks,
        scores=scores,
        top_heads=top_heads,
        diagnostics=diagnostics,
    )
    report_output.write_text(report, encoding="utf-8")
    print(f"Wrote subtype head masks to {output}")
    print(f"Wrote mask report to {report_output}")
    print(json.dumps(out_payload["metadata"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
