#!/usr/bin/env python3
"""Cheap vector-only diagnostics for the next typed-FAS experiment phase.

This script intentionally uses only cached tensors. It does not call APIs,
extract activations, or run LLaVA benchmark generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TYPES = ("cat", "attr", "rel")
PAIRS = (("cat", "attr"), ("cat", "rel"), ("attr", "rel"))
DEFAULT_RUNTIME_VECTORS = (
    "data/after_fas_type_v1_gpt4omini_typed250_text/"
    "official_llava_vectors/runtime_raw_type_vectors.pt"
)
DEFAULT_CONDITION_VECTORS = "data/clean_type_minpair_v2/vectors/condition_vectors.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-vectors", default=DEFAULT_RUNTIME_VECTORS)
    parser.add_argument("--condition-vectors", default=DEFAULT_CONDITION_VECTORS)
    parser.add_argument("--output-root", default="experiments/typed_fas_next/vector_only")
    parser.add_argument("--top-ks", default="16,32,64,128")
    parser.add_argument("--lambdas", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--pca-ks", default="1,2,3,5")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_torch() -> Any:
    import torch

    return torch


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def lambda_tag(value: float) -> str:
    text = f"{value:.4g}".replace("-", "neg").replace(".", "p")
    return text


def load_torch_payload(path: Path) -> Mapping[str, Any]:
    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping payload in {path}")
    return payload


def get_runtime_vectors(payload: Mapping[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    source = payload.get("vectors")
    if not isinstance(source, Mapping):
        raise ValueError("Runtime vector payload is missing payload['vectors']")
    vectors: dict[str, Any] = {}
    for name in TYPES:
        if name not in source:
            raise KeyError(f"Missing runtime vector key: {name}")
        tensor = source[name]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        vectors[name] = tensor.detach().cpu().float()
    shapes = {name: tuple(value.shape) for name, value in vectors.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Runtime vector shapes differ: {shapes}")
    vectors["global"] = torch.stack([vectors[name] for name in TYPES]).mean(dim=0)
    return vectors


def cosine_flat(a: Any, b: Any) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = a_flat.norm() * b_flat.norm()
    if float(denom.item()) <= 0.0:
        return 0.0
    return float((a_flat * b_flat).sum().item() / denom.item())


def headwise_cosine(a: Any, b: Any) -> Any:
    denom = a.norm(dim=-1) * b.norm(dim=-1)
    return (a * b).sum(dim=-1) / denom.clamp_min(1e-12)


def norm_summary(tensor: Any) -> dict[str, float]:
    torch = require_torch()
    head_norm = tensor.float().norm(dim=-1)
    return {
        "flat_norm": float(tensor.float().norm().item()),
        "head_norm_mean": float(head_norm.mean().item()),
        "head_norm_median": float(head_norm.median().item()),
        "head_norm_p90": float(torch.quantile(head_norm.flatten(), 0.90).item()),
        "head_norm_max": float(head_norm.max().item()),
    }


def top_pairs_by_score(score: Any, top_k: int) -> list[tuple[int, int]]:
    flat = score.flatten()
    k = min(max(int(top_k), 0), int(flat.numel()))
    if k <= 0:
        return []
    _values, indices = flat.topk(k)
    num_heads = int(score.shape[1])
    return [(int(index // num_heads), int(index % num_heads)) for index in indices.tolist()]


def mask_stats(masks: Mapping[str, set[tuple[int, int]]]) -> dict[str, Any]:
    pair_rows = {}
    for a, b in PAIRS:
        left = set(masks[a])
        right = set(masks[b])
        union = left | right
        pair_rows[f"{a}_{b}"] = {
            "intersection": len(left & right),
            "union": len(union),
            "jaccard": len(left & right) / len(union) if union else 0.0,
        }
    triple_intersection = set.intersection(*(set(masks[name]) for name in TYPES))
    triple_union = set.union(*(set(masks[name]) for name in TYPES))
    return {
        "pairs": pair_rows,
        "triple_intersection": len(triple_intersection),
        "triple_union": len(triple_union),
        "triple_jaccard": len(triple_intersection) / len(triple_union) if triple_union else 0.0,
    }


def restricted_cosines(vectors: Mapping[str, Any], masks: Mapping[str, set[tuple[int, int]]]) -> dict[str, float]:
    torch = require_torch()
    rows: dict[str, float] = {}
    num_layers = int(next(iter(vectors.values())).shape[0])
    num_heads = int(next(iter(vectors.values())).shape[1])
    for a, b in PAIRS:
        heads = set(masks[a]) | set(masks[b])
        if not heads:
            rows[f"{a}_{b}"] = 0.0
            continue
        mask = torch.zeros((num_layers, num_heads), dtype=torch.bool)
        for layer, head in heads:
            mask[layer, head] = True
        rows[f"{a}_{b}"] = cosine_flat(vectors[a][mask], vectors[b][mask])
    return rows


def mean_other(vectors: Mapping[str, Any], type_name: str) -> Any:
    torch = require_torch()
    others = [vectors[name] for name in TYPES if name != type_name]
    return torch.stack(others).mean(dim=0)


def per_head_normalize(tensor: Any) -> Any:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def build_vector_families(raw: Mapping[str, Any], lambdas: Sequence[float], pca_ks: Sequence[int]) -> dict[str, dict[str, Any]]:
    torch = require_torch()
    families: dict[str, dict[str, Any]] = {"raw": {name: raw[name].clone() for name in TYPES}}
    families["global_residual"] = {}
    global_unit = raw["global"].flatten()
    global_unit = global_unit / global_unit.norm().clamp_min(1e-12)
    for name in TYPES:
        flat = raw[name].flatten()
        projected = (flat * global_unit).sum() * global_unit
        families["global_residual"][name] = (flat - projected).reshape_as(raw[name]).float()

    for lam in lambdas:
        tag = lambda_tag(lam)
        contrast: dict[str, Any] = {}
        headnorm: dict[str, Any] = {}
        for name in TYPES:
            contrast[name] = raw[name] - float(lam) * mean_other(raw, name)
            own = per_head_normalize(raw[name])
            other = torch.stack([per_head_normalize(raw[other]) for other in TYPES if other != name]).mean(dim=0)
            headnorm[name] = own - float(lam) * other
        families[f"contrast_l{tag}"] = contrast
        families[f"headnorm_contrast_l{tag}"] = headnorm

    matrix = torch.stack([raw[name].flatten().float() for name in TYPES], dim=0)
    _u, svals, vh = torch.linalg.svd(matrix, full_matrices=False)
    max_rank = int(vh.shape[0])
    for k in pca_ks:
        actual_k = min(max(int(k), 0), max_rank)
        family: dict[str, Any] = {}
        if actual_k == 0:
            for name in TYPES:
                family[name] = raw[name].clone()
        else:
            basis = vh[:actual_k]
            for name in TYPES:
                flat = raw[name].flatten().float()
                projected = (flat @ basis.T) @ basis
                family[name] = (flat - projected).reshape_as(raw[name]).float()
        family_name = f"pca_remove{int(k)}"
        families[family_name] = family
        families[family_name]["_actual_pca_k"] = actual_k
        families[family_name]["_pca_explained"] = [
            float((value * value / (svals * svals).sum()).item()) for value in svals[:actual_k]
        ]

    for family in families.values():
        tensors = [family[name] for name in TYPES]
        family["global"] = torch.stack(tensors).mean(dim=0)
    return families


def score_maps(vectors: Mapping[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    torch = require_torch()
    norms = {name: vectors[name].norm(dim=-1) for name in TYPES}
    cos = {
        (a, b): headwise_cosine(vectors[a], vectors[b])
        for a, b in PAIRS
    }
    per_type: dict[str, dict[str, Any]] = {}
    all_pairs = [(layer, head) for layer in range(int(vectors["cat"].shape[0])) for head in range(int(vectors["cat"].shape[1]))]
    rng = random.Random(seed)
    shuffled = list(all_pairs)
    rng.shuffle(shuffled)
    random_score = torch.zeros_like(norms["cat"])
    for rank, (layer, head) in enumerate(reversed(shuffled), start=1):
        random_score[layer, head] = float(rank)
    shared = torch.stack([norms[name] for name in TYPES]).mean(dim=0)
    for name in TYPES:
        others = [other for other in TYPES if other != name]
        pair_cos = []
        full_cos = []
        for other in others:
            key = next(pair for pair in PAIRS if name in pair and other in pair)
            pair_cos.append(cos[key])
            full_cos.append(cosine_flat(vectors[name], vectors[other]))
        max_head_cos = torch.stack(pair_cos).max(dim=0).values
        mean_other_norm = torch.stack([norms[other] for other in others]).mean(dim=0)
        contrast_norm = (vectors[name] - torch.stack([vectors[other] for other in others]).mean(dim=0)).norm(dim=-1)
        ratio = norms[name] / mean_other_norm.clamp_min(1e-12)
        ratio = torch.where(norms[name] >= torch.quantile(norms[name].flatten(), 0.5), ratio, torch.zeros_like(ratio))
        baseline_cos = float(sum(full_cos) / len(full_cos))
        margin = torch.clamp(torch.full_like(max_head_cos, baseline_cos) - max_head_cos, min=0.0)
        per_type[name] = {
            "norm": norms[name],
            "specificity_cos": norms[name] * (1.0 - max_head_cos),
            "contrast_norm": contrast_norm,
            "ratio": ratio,
            "hybrid": norms[name] * margin,
            "shared": shared,
            "random": random_score,
        }
    return per_type


def family_diagnostics(name: str, vectors: Mapping[str, Any], top_ks: Sequence[int], seed: int) -> dict[str, Any]:
    scores = score_maps(vectors, seed)
    output: dict[str, Any] = {
        "family": name,
        "norms": {type_name: norm_summary(vectors[type_name]) for type_name in TYPES},
        "pairwise_cosines": {f"{a}_{b}": cosine_flat(vectors[a], vectors[b]) for a, b in PAIRS},
        "cosine_to_family_global": {type_name: cosine_flat(vectors[type_name], vectors["global"]) for type_name in TYPES},
        "topk": {},
    }
    for strategy in ("norm", "specificity_cos", "contrast_norm", "ratio", "hybrid", "shared", "random"):
        output["topk"][strategy] = {}
        for top_k in top_ks:
            masks = {
                type_name: set(top_pairs_by_score(scores[type_name][strategy], top_k))
                for type_name in TYPES
            }
            output["topk"][strategy][str(top_k)] = {
                "overlap": mask_stats(masks),
                "restricted_cosines": restricted_cosines(vectors, masks),
                "heads": {type_name: [[layer, head] for layer, head in sorted(masks[type_name])] for type_name in TYPES},
            }
    return output


def flatten_family_rows(diag: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, payload in diag.items():
        if not isinstance(payload, Mapping):
            continue
        row = {
            "family": family,
            "cat_attr_cos": payload["pairwise_cosines"]["cat_attr"],
            "cat_rel_cos": payload["pairwise_cosines"]["cat_rel"],
            "attr_rel_cos": payload["pairwise_cosines"]["attr_rel"],
            "mean_abs_pairwise_cos": sum(abs(payload["pairwise_cosines"][f"{a}_{b}"]) for a, b in PAIRS) / 3.0,
            "cat_global_cos": payload["cosine_to_family_global"]["cat"],
            "attr_global_cos": payload["cosine_to_family_global"]["attr"],
            "rel_global_cos": payload["cosine_to_family_global"]["rel"],
            "cat_norm": payload["norms"]["cat"]["flat_norm"],
            "attr_norm": payload["norms"]["attr"]["flat_norm"],
            "rel_norm": payload["norms"]["rel"]["flat_norm"],
        }
        rows.append(row)
    rows.sort(key=lambda item: (float(item["mean_abs_pairwise_cos"]), item["family"]))
    return rows


def flatten_mask_rows(diag: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, payload in diag.items():
        for strategy, by_k in payload["topk"].items():
            for top_k, values in by_k.items():
                overlap = values["overlap"]
                restricted = values["restricted_cosines"]
                rows.append({
                    "family": family,
                    "strategy": strategy,
                    "top_k": int(top_k),
                    "triple_intersection": overlap["triple_intersection"],
                    "triple_union": overlap["triple_union"],
                    "triple_jaccard": overlap["triple_jaccard"],
                    "cat_attr_jaccard": overlap["pairs"]["cat_attr"]["jaccard"],
                    "cat_rel_jaccard": overlap["pairs"]["cat_rel"]["jaccard"],
                    "attr_rel_jaccard": overlap["pairs"]["attr_rel"]["jaccard"],
                    "cat_attr_restricted_cos": restricted["cat_attr"],
                    "cat_rel_restricted_cos": restricted["cat_rel"],
                    "attr_rel_restricted_cos": restricted["attr_rel"],
                })
    rows.sort(key=lambda item: (item["top_k"], float(item["triple_jaccard"]), item["family"], item["strategy"]))
    return rows


def condition_vector_summary(path: Path) -> dict[str, Any]:
    payload = load_torch_payload(path)
    source = payload.get("vectors")
    if not isinstance(source, Mapping):
        source = payload
    keys = [
        key for key in source
        if isinstance(key, str)
        and (key.startswith("g_") or key.startswith("s_"))
        and key.endswith("_clean")
    ]
    vectors = {key: source[key].detach().cpu().float() for key in keys}
    coarse = [key for key in ("g_all_clean", "g_attr_clean", "g_rel_clean") if key in vectors]
    subtypes = [key for key in keys if key.startswith("s_")]
    rows = []
    for key in coarse + subtypes:
        best_coarse = ""
        best_coarse_cos = math.nan
        if key not in coarse:
            scored = [(coarse_key, cosine_flat(vectors[key], vectors[coarse_key])) for coarse_key in coarse]
            if scored:
                best_coarse, best_coarse_cos = max(scored, key=lambda item: item[1])
        rows.append({
            "vector": key,
            "kind": "coarse" if key in coarse else "subtype",
            "flat_norm": norm_summary(vectors[key])["flat_norm"],
            "best_coarse": best_coarse,
            "best_coarse_cosine": best_coarse_cos,
        })
    subtype_pairs = []
    for index, left in enumerate(subtypes):
        for right in subtypes[index + 1:]:
            subtype_pairs.append({
                "left": left,
                "right": right,
                "cosine": cosine_flat(vectors[left], vectors[right]),
            })
    return {
        "path": str(path),
        "rows": rows,
        "subtype_pair_mean_abs_cosine": (
            sum(abs(row["cosine"]) for row in subtype_pairs) / len(subtype_pairs)
            if subtype_pairs else None
        ),
        "subtype_pairs": subtype_pairs,
        "counts_by_subtype": payload.get("counts_by_subtype", {}),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def md_table(headers: Sequence[str], rows: Sequence[Mapping[str, Any]], limit: int | None = None) -> str:
    selected = list(rows[:limit]) if limit is not None else list(rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def render_report(payload: Mapping[str, Any]) -> str:
    family_rows = payload["family_rows"]
    mask_rows = payload["mask_rows"]
    raw = next(row for row in family_rows if row["family"] == "raw")
    low_cos = family_rows[:10]
    mask_64 = [row for row in mask_rows if int(row["top_k"]) == 64]
    low_overlap_64 = sorted(mask_64, key=lambda row: (float(row["triple_jaccard"]), float(row["cat_attr_jaccard"]) + float(row["cat_rel_jaccard"]) + float(row["attr_rel_jaccard"])))[:12]
    condition = payload["condition_summary"]
    subtype_mean = condition.get("subtype_pair_mean_abs_cosine")
    lines = [
        "# Vector-Only Typed-FAS Diagnostics",
        "",
        "## Inputs",
        "",
        f"- runtime vectors: `{payload['runtime_vector_path']}`",
        f"- condition vectors: `{payload['condition_vector_path']}`",
        "- computation: cached CPU tensor diagnostics only; no API calls, activation extraction, or benchmark generation.",
        "",
        "## Raw Baseline Recheck",
        "",
        md_table(
            ["family", "cat_attr_cos", "cat_rel_cos", "attr_rel_cos", "mean_abs_pairwise_cos", "cat_norm", "attr_norm", "rel_norm"],
            [raw],
        ),
        "",
        "This matches the prior conclusion: the raw cat/attr/rel directions are highly aligned.",
        "",
        "## Lowest Pairwise-Cosine Vector Variants",
        "",
        md_table(
            ["family", "cat_attr_cos", "cat_rel_cos", "attr_rel_cos", "mean_abs_pairwise_cos", "cat_global_cos", "attr_global_cos", "rel_global_cos", "cat_norm", "attr_norm", "rel_norm"],
            low_cos,
        ),
        "",
        "Lower cosine alone is not enough. PCA removal and head-normalized contrast can make vectors look private while also changing norms and possibly removing the shared effect that made FAS work.",
        "",
        "## Lowest Top-64 Mask Overlap",
        "",
        md_table(
            ["family", "strategy", "top_k", "triple_intersection", "triple_union", "triple_jaccard", "cat_attr_jaccard", "cat_rel_jaccard", "attr_rel_jaccard", "cat_attr_restricted_cos", "cat_rel_restricted_cos", "attr_rel_restricted_cos"],
            low_overlap_64,
        ),
        "",
        "The discriminative strategies can reduce mask overlap, so they are better dev-benchmark candidates than norm-topK. The benchmark claim still needs actual dev runs.",
        "",
        "## Clean Minimal-Pair Context",
        "",
        f"- subtype mean absolute cosine: `{fmt(subtype_mean)}`",
        "- Cached condition vectors are much less aligned than the raw FAS vectors, but previous heldout mask evals showed only limited credible selectivity.",
        "",
        "## Decision",
        "",
        "- Do not scale raw vectors again; the existing large run already falsifies clean expert behavior.",
        "- Best cheap next dev candidates: `contrast_l1` or `global_residual` with `specificity_cos`/`contrast_norm` masks at K=16/32/64, plus negative-alpha checks.",
        "- Anchor cancellation remains unexecuted here because the necessary anchor text activations are not cached; it requires a new text-only activation pass, not an API call.",
        "- Correct-vs-wrong minimal pairs are already cached in `clean_type_minpair_v2`; the useful next step is targeted dev reruns for attr_count/attr_shape and relation bias checks, not another vector-only pass.",
    ]
    return "\n".join(lines) + "\n"


def save_variant_bundle(path: Path, families: Mapping[str, Mapping[str, Any]], source_payload: Mapping[str, Any]) -> None:
    torch = require_torch()
    vectors = {
        f"{family}_{type_name}": values[type_name].detach().cpu().float()
        for family, values in families.items()
        for type_name in TYPES
        if not family.startswith("pca_remove5")
    }
    # Also save concise expert names for the two most useful benchmark candidates.
    if "contrast_l1" in families:
        for type_name in TYPES:
            vectors[f"contrast_{type_name}"] = families["contrast_l1"][type_name].detach().cpu().float()
    if "global_residual" in families:
        for type_name in TYPES:
            vectors[f"residual_{type_name}"] = families["global_residual"][type_name].detach().cpu().float()
    shape = tuple(next(iter(vectors.values())).shape)
    payload = {
        "vectors": vectors,
        "layers": [int(layer) for layer in source_payload.get("layers", list(range(shape[0])))],
        "num_heads": int(shape[1]),
        "head_dim": int(shape[2]),
        "hidden_size": int(shape[1] * shape[2]),
        "metadata": {
            "source": str(resolve(DEFAULT_RUNTIME_VECTORS)),
            "note": "Vector-only transformed variants; benchmark before making expert claims.",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_head_maps(root: Path, diagnostics: Mapping[str, Any]) -> list[dict[str, str]]:
    written: list[dict[str, str]] = []
    for family, payload in diagnostics.items():
        for strategy, by_k in payload["topk"].items():
            for top_k, values in by_k.items():
                compact = {}
                for type_name in TYPES:
                    heads = values["heads"][type_name]
                    compact[type_name] = heads
                    compact[f"{family}_{type_name}"] = heads
                    if family == "contrast_l1":
                        compact[f"contrast_{type_name}"] = heads
                    if family == "global_residual":
                        compact[f"residual_{type_name}"] = heads
                path = root / "head_maps" / f"{family}__{strategy}__top{top_k}.json"
                write_json(path, compact)
                written.append({
                    "family": family,
                    "strategy": strategy,
                    "top_k": str(top_k),
                    "path": str(path),
                })
    return written


def main() -> int:
    args = parse_args()
    runtime_path = resolve(args.runtime_vectors)
    condition_path = resolve(args.condition_vectors)
    output_root = resolve(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_root}. Pass --overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_payload = load_torch_payload(runtime_path)
    raw = get_runtime_vectors(runtime_payload)
    lambdas = parse_floats(args.lambdas)
    top_ks = parse_ints(args.top_ks)
    pca_ks = parse_ints(args.pca_ks)
    families = build_vector_families(raw, lambdas, pca_ks)
    diagnostics = {
        family: family_diagnostics(
            family,
            {type_name: vectors[type_name] for type_name in TYPES + ("global",)},
            top_ks,
            args.random_seed,
        )
        for family, vectors in families.items()
        if all(type_name in vectors for type_name in TYPES)
    }
    family_rows = flatten_family_rows(diagnostics)
    mask_rows = flatten_mask_rows(diagnostics)
    condition_summary = condition_vector_summary(condition_path) if condition_path.exists() else {}
    head_map_rows = save_head_maps(output_root, diagnostics)
    save_variant_bundle(output_root / "vector_only_variants.pt", families, runtime_payload)

    payload = {
        "runtime_vector_path": str(runtime_path),
        "condition_vector_path": str(condition_path),
        "output_root": str(output_root),
        "lambdas": lambdas,
        "top_ks": top_ks,
        "pca_ks_requested": pca_ks,
        "families": diagnostics,
        "family_rows": family_rows,
        "mask_rows": mask_rows,
        "condition_summary": condition_summary,
        "head_map_rows": head_map_rows,
    }
    write_json(output_root / "vector_only_diagnostics.json", payload)
    write_csv(output_root / "vector_family_summary.csv", family_rows)
    write_csv(output_root / "head_mask_summary.csv", mask_rows)
    write_csv(output_root / "head_map_files.csv", head_map_rows)
    if condition_summary:
        write_csv(output_root / "condition_vector_summary.csv", condition_summary["rows"])
        write_csv(output_root / "condition_subtype_pair_cosines.csv", condition_summary["subtype_pairs"])
    (output_root / "VECTOR_ONLY_DIAGNOSTICS.md").write_text(render_report(payload), encoding="utf-8")
    print(f"Wrote vector-only diagnostics to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
