#!/usr/bin/env python3
"""Build subtype-aware grounding and semantic steering vectors.

The builder consumes subtype minimal-pair activations and emits a vector file
compatible with ExpertSteeringController. Denoising is performed on sample-level
delta matrices, not on cat/attr/rel type means.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBTYPES = (
    "cat_random",
    "cat_popular",
    "cat_hard",
    "attr_color",
    "attr_count",
    "rel_spatial",
    "rel_contact",
)
EXPERT_TYPES = ("cat", "attr", "rel")
MIXES = {
    "g_only": (1.0, 0.0),
    "s_only": (0.0, 1.0),
    "g1_s025": (1.0, 0.25),
    "g1_s05": (1.0, 0.5),
    "g1_s1": (1.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--yesno-direction", default="")
    parser.add_argument("--sample-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--svd-k", type=int, default=4)
    parser.add_argument("--denoise-method", choices=["none", "uncentered_svd"], default="uncentered_svd")
    parser.add_argument("--remove-yesno", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yesno-mode", choices=["answer_token", "dataset_pair", "none"], default="answer_token")
    parser.add_argument("--shuffle-subtype-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-heads", default="64")
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
        raise RuntimeError("build_subtype_vectors.py requires torch.") from exc


def torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def flatten_samples(tensor: Any) -> Any:
    return tensor.float().reshape(int(tensor.shape[0]), -1)


def normalize_rows(torch: Any, x: Any, eps: float = 1e-12) -> Any:
    norms = x.norm(dim=1, keepdim=True).clamp_min(eps)
    return x / norms


def normalize_vec(torch: Any, x: Any, eps: float = 1e-12) -> Any:
    norm = x.float().norm().clamp_min(eps)
    return x.float() / norm


def cosine(torch: Any, a: Any, b: Any) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = float(af.norm().item() * bf.norm().item())
    if denom <= 0:
        return 0.0
    return float(torch.dot(af, bf).item() / denom)


def vector_stats(torch: Any, tensor: Any) -> dict[str, Any]:
    t = tensor.float()
    head_norms = t.norm(dim=-1).reshape(-1)
    return {
        "shape": [int(dim) for dim in t.shape],
        "flat_norm": float(t.reshape(-1).norm().item()),
        "head_norm_mean": float(head_norms.mean().item()) if head_norms.numel() else 0.0,
        "head_norm_max": float(head_norms.max().item()) if head_norms.numel() else 0.0,
        "head_norm_min": float(head_norms.min().item()) if head_norms.numel() else 0.0,
        "finite": bool(torch.isfinite(t).all().item()),
    }


def topk_svd_directions(torch: Any, x: Any, k: int) -> tuple[Any, list[float], list[float], str]:
    """Return exact top-k right singular directions using a Gram eigendecomp."""

    if int(k) < 1:
        raise ValueError("--svd-k must be >= 1")
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("Cannot run SVD on an empty sample matrix.")
    x = x.float()
    rank = min(int(k), n)
    gram = x @ x.T
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order][:rank].clamp_min(0.0)
    eigvecs = eigvecs[:, order][:, :rank]
    singular_values = torch.sqrt(eigvals)
    directions = []
    for idx in range(rank):
        sigma = singular_values[idx].clamp_min(1e-12)
        direction = (eigvecs[:, idx].T @ x) / sigma
        directions.append(normalize_vec(torch, direction))
    directions_tensor = torch.stack(directions, dim=0)
    total_energy = float((x * x).sum().item())
    energy = [(float(value.item()) ** 2 / total_energy) if total_energy > 0 else 0.0 for value in singular_values]
    return directions_tensor, [float(value.item()) for value in singular_values], energy, "gram_exact_uncentered_svd"


def denoise_group(
    torch: Any,
    samples: Any,
    *,
    vector_shape: tuple[int, ...],
    sample_normalize: bool,
    denoise_method: str,
    svd_k: int,
) -> tuple[Any, dict[str, Any]]:
    x = flatten_samples(samples)
    raw_norms = x.norm(dim=1)
    if sample_normalize:
        x = normalize_rows(torch, x)
    mu = x.mean(dim=0)
    info: dict[str, Any] = {
        "num_samples": int(x.shape[0]),
        "sample_norm_raw": {
            "mean": float(raw_norms.mean().item()) if raw_norms.numel() else 0.0,
            "min": float(raw_norms.min().item()) if raw_norms.numel() else 0.0,
            "max": float(raw_norms.max().item()) if raw_norms.numel() else 0.0,
        },
        "mean_only_norm": float(mu.norm().item()),
        "method": denoise_method,
    }
    if denoise_method == "none":
        vector = mu
    else:
        directions, singular_values, energy, impl = topk_svd_directions(torch, x, int(svd_k))
        coeffs = directions @ mu
        vector = directions.T @ coeffs
        info.update(
            {
                "svd_impl": impl,
                "svd_k": int(min(svd_k, x.shape[0])),
                "singular_values": singular_values,
                "explained_energy_ratio": energy,
                "projected_norm": float(vector.norm().item()),
            }
        )
    return vector.reshape(vector_shape).float(), info


def remove_projection(torch: Any, vector: Any, basis: Any | None) -> tuple[Any, dict[str, float]]:
    if basis is None:
        return vector.float(), {
            "raw_yesno_cosine": 0.0,
            "clean_yesno_cosine": 0.0,
            "projection_norm_ratio": 0.0,
            "clean_norm_over_raw_norm": 1.0,
        }
    v = vector.float().reshape(-1)
    b = normalize_vec(torch, basis.float().reshape(-1))
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


def rows_for_group(metadata: list[Mapping[str, Any]], key: str, value: str) -> list[int]:
    return [index for index, row in enumerate(metadata) if str(row.get(key, "")) == value]


def load_yesno_direction(torch: Any, path: Path, payload: Mapping[str, Any], yesno_mode: str) -> tuple[Any | None, dict[str, Any]]:
    if yesno_mode == "none":
        return None, {"mode": "none"}
    if path and str(path) != "." and path.exists():
        yesno_payload = torch_load(torch, path)
        direction = yesno_payload.get("yesno_direction", yesno_payload.get("direction"))
        if direction is None:
            raise ValueError(f"Could not find yesno_direction in {path}")
        return direction.float(), {"mode": "answer_token", "path": str(path), "source_schema": yesno_payload.get("schema", {})}
    if yesno_mode == "answer_token":
        # Fall back rather than fail so a partially extracted cache can still be diagnosed.
        yesno_mode = "dataset_pair"
    metadata = list(payload.get("metadata", []))
    z_fact = payload["z_fact_text"].float()
    z_visual = payload["z_visual"].float()
    delta = z_fact - z_visual
    yes_idx = [idx for idx, row in enumerate(metadata) if str(row.get("gt_answer", row.get("label", ""))).lower() == "yes"]
    no_idx = [idx for idx, row in enumerate(metadata) if str(row.get("gt_answer", row.get("label", ""))).lower() == "no"]
    if not yes_idx or not no_idx:
        return None, {"mode": "dataset_pair", "warning": "Could not estimate yes/no direction; missing yes or no rows."}
    direction = delta[yes_idx].float().mean(dim=0) - delta[no_idx].float().mean(dim=0)
    return direction.float(), {
        "mode": "dataset_pair",
        "warning": "answer_token yes/no file was missing; used dataset yes/no split as fallback.",
        "yes_count": len(yes_idx),
        "no_count": len(no_idx),
    }


def md_table(headers: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def cosine_matrix(torch: Any, vectors: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = list(vectors)
    rows = []
    for name in names:
        row: dict[str, Any] = {"vector": name}
        for other in names:
            row[other] = cosine(torch, vectors[name], vectors[other])
        rows.append(row)
    return rows


def top_heads(torch: Any, vector: Any, k: int) -> list[dict[str, Any]]:
    norms = vector.float().norm(dim=-1)
    rows = []
    for layer in range(int(norms.shape[0])):
        for head in range(int(norms.shape[1])):
            rows.append({"layer": layer, "head": head, "norm": float(norms[layer, head].item())})
    rows.sort(key=lambda row: (-row["norm"], row["layer"], row["head"]))
    return rows[:k]


def head_overlap(top_by_vector: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    names = list(top_by_vector)
    rows = []
    for i, left in enumerate(names):
        left_set = {(int(row["layer"]), int(row["head"])) for row in top_by_vector[left]}
        for right in names[i + 1 :]:
            right_set = {(int(row["layer"]), int(row["head"])) for row in top_by_vector[right]}
            inter = len(left_set & right_set)
            union = len(left_set | right_set)
            rows.append({"vector_a": left, "vector_b": right, "intersection": inter, "jaccard": inter / union if union else 0.0})
    return rows


def parse_topk(text: str) -> list[int]:
    return [int(item) for item in str(text).replace(",", " ").split() if item.strip()]


def compose(torch: Any, g: Any, s: Any, mix_g: float, mix_s: float) -> tuple[Any, Any]:
    g_norm = g.float().reshape(-1).norm()
    s_norm = s.float().reshape(-1).norm()
    parts = []
    if abs(float(mix_g)) > 0 and g_norm.item() > 0:
        parts.append(float(mix_g) * normalize_vec(torch, g))
    if abs(float(mix_s)) > 0 and s_norm.item() > 0:
        parts.append(float(mix_s) * normalize_vec(torch, s))
    if not parts:
        combo_unit = torch.zeros_like(g.float())
    else:
        combo_unit = normalize_vec(torch, sum(parts)).reshape(g.shape)
    target_norm = float((g_norm.item() + s_norm.item()) / 2.0) if g_norm.item() > 0 and s_norm.item() > 0 else float(max(g_norm.item(), s_norm.item()))
    return combo_unit.float() * target_norm, combo_unit.float()


def write_report(
    path: Path,
    *,
    metadata: list[Mapping[str, Any]],
    payload_schema: Mapping[str, Any],
    vector_shape: tuple[int, ...],
    counts: Mapping[str, int],
    diagnostics: Mapping[str, Any],
    vectors: Mapping[str, Any],
    yesno_info: Mapping[str, Any],
    topk_values: list[int],
    torch: Any,
) -> None:
    lines: list[str] = []
    lines.append("# Subtype Minimal-Pair Vector Report")
    lines.append("")
    lines.append(f"- Source activations: `{payload_schema.get('source_jsonl', '')}`")
    lines.append(f"- Vector shape: `{list(vector_shape)}`")
    lines.append(f"- Samples: `{len(metadata)}`")
    lines.append(f"- Yes/no direction mode: `{yesno_info.get('mode')}`")
    if yesno_info.get("warning"):
        lines.append(f"- Warning: {yesno_info['warning']}")
    lines.append("")
    lines.append("## Counts")
    lines.append(md_table(["subtype", "count"], [{"subtype": key, "count": value} for key, value in sorted(counts.items())]))
    lines.append("")
    lines.append("## Denoising")
    denoise_rows = []
    for key, info in diagnostics.get("denoise", {}).items():
        denoise_rows.append(
            {
                "vector": key,
                "num_samples": info.get("num_samples", 0),
                "method": info.get("method", ""),
                "svd_k": info.get("svd_k", ""),
                "first_singular": float(info.get("singular_values", [0.0])[0]) if info.get("singular_values") else 0.0,
                "top_energy": float(info.get("explained_energy_ratio", [0.0])[0]) if info.get("explained_energy_ratio") else 0.0,
                "mean_only_norm": float(info.get("mean_only_norm", 0.0)),
                "projected_norm": float(info.get("projected_norm", info.get("mean_only_norm", 0.0))),
            }
        )
    lines.append(md_table(["vector", "num_samples", "method", "svd_k", "first_singular", "top_energy", "mean_only_norm", "projected_norm"], denoise_rows))
    lines.append("")
    lines.append("## Yes/No Projection")
    projection_rows = []
    for key, info in diagnostics.get("yesno_projection", {}).items():
        projection_rows.append({"vector": key, **info})
    lines.append(md_table(["vector", "raw_yesno_cosine", "clean_yesno_cosine", "projection_norm_ratio", "clean_norm_over_raw_norm"], projection_rows))
    lines.append("")
    selected_names = [name for name in ("g_all_clean", "g_cat_clean", "g_attr_clean", "g_rel_clean", "s_cat_hard_clean", "s_attr_color_clean", "s_attr_count_clean", "s_rel_spatial_clean", "s_rel_contact_clean") if name in vectors]
    if selected_names:
        lines.append("## Clean Cosine Matrix")
        matrix = cosine_matrix(torch, {name: vectors[name] for name in selected_names})
        lines.append(md_table(["vector"] + selected_names, matrix))
        lines.append("")
    lines.append("## Vector Norms")
    norm_rows = []
    for name in sorted(vectors):
        stats = vector_stats(torch, vectors[name])
        norm_rows.append({"vector": name, "flat_norm": stats["flat_norm"], "head_norm_mean": stats["head_norm_mean"], "head_norm_max": stats["head_norm_max"], "finite": stats["finite"]})
    lines.append(md_table(["vector", "flat_norm", "head_norm_mean", "head_norm_max", "finite"], norm_rows))
    lines.append("")
    for k in topk_values:
        names = [name for name in selected_names if name in vectors]
        top_by_vector = {name: top_heads(torch, vectors[name], k) for name in names}
        lines.append(f"## Top{k} Head Overlap")
        lines.append(md_table(["vector_a", "vector_b", "intersection", "jaccard"], head_overlap(top_by_vector)))
        lines.append("")
        for name, rows in top_by_vector.items():
            lines.append(f"### {name} Top{k} Heads")
            lines.append(md_table(["layer", "head", "norm"], rows[: min(k, 20)]))
            lines.append("")
    lines.append("## Automatic Interpretation")
    max_clean_yesno = max((abs(float(info.get("clean_yesno_cosine", 0.0))) for info in diagnostics.get("yesno_projection", {}).values()), default=0.0)
    if max_clean_yesno < 1e-3:
        lines.append("- Yes/no projection removal succeeded numerically; clean vectors are nearly orthogonal to the yes/no direction.")
    else:
        lines.append(f"- Some clean vectors still retain yes/no cosine up to {max_clean_yesno:.4f}; inspect the yes/no direction source.")
    if "s_rel_contact_clean" not in vectors:
        lines.append("- rel_contact was not built, likely due insufficient reliable contact relation samples.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    torch = load_torch()
    random.seed(int(args.seed))
    output_path = resolve(args.output)
    report_path = resolve(args.report_output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite.")
    payload = torch_load(torch, resolve(args.activations))
    required = ("z_visual", "z_fact_text", "z_counterfact_text", "metadata")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Activation cache missing keys: {missing}")
    z_visual = payload["z_visual"].float()
    z_fact = payload["z_fact_text"].float()
    z_counter = payload["z_counterfact_text"].float()
    if list(z_visual.shape) != list(z_fact.shape) or list(z_visual.shape) != list(z_counter.shape):
        raise ValueError(f"Activation branch shapes differ: {list(z_visual.shape)}, {list(z_fact.shape)}, {list(z_counter.shape)}")
    if len(z_visual.shape) != 4:
        raise ValueError(f"Expected activation shape [N,L,H,D], got {list(z_visual.shape)}")
    if not torch.isfinite(z_visual).all() or not torch.isfinite(z_fact).all() or not torch.isfinite(z_counter).all():
        raise ValueError("Activation cache contains NaN/Inf.")
    metadata = [dict(row) for row in payload["metadata"]]
    if len(metadata) != int(z_visual.shape[0]):
        raise ValueError(f"Metadata length {len(metadata)} does not match activations N={z_visual.shape[0]}")
    if args.shuffle_subtype_labels:
        subtype_values = [str(row.get("subtype", "")) for row in metadata]
        rng = random.Random(int(args.seed))
        rng.shuffle(subtype_values)
        for row, subtype in zip(metadata, subtype_values):
            row["subtype"] = subtype
        print("Shuffled subtype labels before vector construction.")
    vector_shape = tuple(int(dim) for dim in z_visual.shape[1:])
    g_delta = z_fact - z_visual
    s_delta = z_fact - z_counter
    yesno_direction, yesno_info = load_yesno_direction(torch, resolve(args.yesno_direction) if str(args.yesno_direction).strip() else Path("."), payload, str(args.yesno_mode))
    if yesno_direction is not None and tuple(int(dim) for dim in yesno_direction.shape) != vector_shape:
        raise ValueError(f"yesno_direction shape {list(yesno_direction.shape)} does not match vector shape {list(vector_shape)}")

    raw_vectors: dict[str, Any] = {}
    clean_vectors: dict[str, Any] = {}
    mean_only_vectors: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {"denoise": {}, "yesno_projection": {}, "yesno_info": yesno_info}
    counts = Counter(str(row.get("subtype", "")) for row in metadata)
    type_counts = Counter(str(row.get("expert_type", "")) for row in metadata)

    def build_vector(name: str, samples: Any) -> None:
        vector, info = denoise_group(
            torch,
            samples,
            vector_shape=vector_shape,
            sample_normalize=bool(args.sample_normalize),
            denoise_method=str(args.denoise_method),
            svd_k=int(args.svd_k),
        )
        mean_only, _ = denoise_group(
            torch,
            samples,
            vector_shape=vector_shape,
            sample_normalize=bool(args.sample_normalize),
            denoise_method="none",
            svd_k=1,
        )
        raw_vectors[f"{name}_raw"] = vector
        mean_only_vectors[f"{name}_mean_only"] = mean_only
        if bool(args.remove_yesno):
            clean, projection = remove_projection(torch, vector, yesno_direction)
        else:
            clean, projection = remove_projection(torch, vector, None)
        clean_vectors[f"{name}_clean"] = clean
        diagnostics["denoise"][name] = info
        diagnostics["yesno_projection"][name] = projection

    all_indices = list(range(len(metadata)))
    build_vector("g_all", g_delta[all_indices])
    for expert in EXPERT_TYPES:
        idx = rows_for_group(metadata, "expert_type", expert)
        if idx:
            build_vector(f"g_{expert}", g_delta[idx])
    for subtype in SUBTYPES:
        idx = rows_for_group(metadata, "subtype", subtype)
        if idx:
            build_vector(f"s_{subtype}", s_delta[idx])

    composed: dict[str, Any] = {}
    composed_unit: dict[str, Any] = {}
    for subtype in SUBTYPES:
        expert = subtype.split("_", 1)[0]
        g_key = f"g_{expert}_clean"
        s_key = f"s_{subtype}_clean"
        if g_key not in clean_vectors or s_key not in clean_vectors:
            continue
        for mix_name, (mix_g, mix_s) in MIXES.items():
            vector, unit_vector = compose(torch, clean_vectors[g_key], clean_vectors[s_key], mix_g, mix_s)
            composed[f"d_{subtype}_{mix_name}_clean"] = vector
            composed_unit[f"d_{subtype}_{mix_name}_unit_clean"] = unit_vector

    vectors: dict[str, Any] = {}
    vectors.update(raw_vectors)
    vectors.update(clean_vectors)
    vectors.update(mean_only_vectors)
    vectors.update(composed)
    vectors.update(composed_unit)
    if yesno_direction is not None:
        vectors["yesno_direction"] = yesno_direction.float()
    for name, tensor in vectors.items():
        if tuple(int(dim) for dim in tensor.shape) != vector_shape:
            raise ValueError(f"Vector {name} has shape {list(tensor.shape)}, expected {list(vector_shape)}")
        if not bool(torch.isfinite(tensor.float()).all().item()):
            raise ValueError(f"Vector {name} contains NaN/Inf.")

    output_payload = {
        **{name: tensor for name, tensor in vectors.items() if name in {
            "g_all_raw",
            "g_all_clean",
            "g_cat_raw",
            "g_cat_clean",
            "g_attr_raw",
            "g_attr_clean",
            "g_rel_raw",
            "g_rel_clean",
            "s_cat_random_raw",
            "s_cat_random_clean",
            "s_cat_popular_raw",
            "s_cat_popular_clean",
            "s_cat_hard_raw",
            "s_cat_hard_clean",
            "s_attr_color_raw",
            "s_attr_color_clean",
            "s_attr_count_raw",
            "s_attr_count_clean",
            "s_rel_spatial_raw",
            "s_rel_spatial_clean",
            "s_rel_contact_raw",
            "s_rel_contact_clean",
            "yesno_direction",
        }},
        "composed": composed,
        "vectors": vectors,
        "layers": list(range(vector_shape[0])),
        "num_heads": int(vector_shape[1]),
        "head_dim": int(vector_shape[2]),
        "hidden_size": int(vector_shape[1] * vector_shape[2]),
        "metadata": {
            "created_by": "scripts/build_subtype_vectors.py",
            "source_activations": str(resolve(args.activations)),
            "sample_normalize": bool(args.sample_normalize),
            "denoise_method": str(args.denoise_method),
            "svd_k": int(args.svd_k),
            "remove_yesno": bool(args.remove_yesno),
            "yesno_mode": yesno_info.get("mode"),
            "shuffle_subtype_labels": bool(args.shuffle_subtype_labels),
            "counts_by_subtype": dict(sorted(counts.items())),
            "counts_by_expert_type": dict(sorted(type_counts.items())),
            "vector_shape": list(vector_shape),
        },
        "diagnostics": diagnostics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output_path)
    write_report(
        report_path,
        metadata=metadata,
        payload_schema=payload.get("schema", {}),
        vector_shape=vector_shape,
        counts=counts,
        diagnostics=diagnostics,
        vectors=vectors,
        yesno_info=yesno_info,
        topk_values=parse_topk(args.topk_heads),
        torch=torch,
    )
    print(f"Wrote subtype vectors to {output_path}")
    print(f"Wrote vector report to {report_path}")
    print(f"Vector keys: {', '.join(sorted(vectors)[:20])} ... ({len(vectors)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
