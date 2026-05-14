"""Build shared/global and type-private residual vectors from activation caches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import read_jsonl, sha256_file, write_json
from expert_data.steering import parse_layer_spec


EXPERTS = ("cat", "attr", "rel")
TYPE_FIELDS = ("hallucination_type", "type", "expert", "category", "task_type")
SUBTYPE_TO_TYPE_PREFIXES = {
    "cat": "cat",
    "attr": "attr",
    "cnt": "attr",
    "col": "attr",
    "rel": "rel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True, help="Activation cache .pt path.")
    parser.add_argument("--metadata", default="", help="Optional metadata JSONL. Defaults to sibling train.meta.jsonl/metadata.jsonl.")
    parser.add_argument("--output", required=True, help="Output shared/private vector .pt path.")
    parser.add_argument("--report-output", required=True, help="Output Markdown diagnostic report.")
    parser.add_argument("--compatible-output", default="", help="Optional eval-compatible .pt path.")
    parser.add_argument(
        "--sample-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize each sample delta before type averaging.",
    )
    parser.add_argument("--global-method", choices=["mean_all", "type_mean_svd", "all_svd"], default="type_mean_svd")
    parser.add_argument("--subspace-k", type=int, default=1)
    parser.add_argument("--layers", default="all", help="Layer spec to select from cache layers. Default: all available.")
    parser.add_argument("--topk-heads", default="32,64", help="Comma-separated K values for head-overlap diagnostics.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("build_shared_private_vectors requires torch.") from exc


def load_torch(path: Path) -> dict[str, Any]:
    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return payload


def as_float_tensor(value: Any, name: str = "tensor") -> Any:
    torch = require_torch()
    if value is None:
        raise KeyError(f"Missing required tensor: {name}")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def tensor_info(value: Any) -> dict[str, Any]:
    shape = getattr(value, "shape", None)
    return {
        "type": type(value).__name__,
        "shape": [int(item) for item in shape] if shape is not None else None,
        "dtype": str(getattr(value, "dtype", "")),
    }


def find_first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def sibling_metadata_path(activation_path: Path) -> Path | None:
    candidates = [
        activation_path.with_suffix(".meta.jsonl"),
        activation_path.parent / f"{activation_path.stem}.meta.jsonl",
        activation_path.parent / "train.meta.jsonl",
        activation_path.parent / "metadata.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_metadata(activation_path: Path, explicit_path: str) -> tuple[list[dict[str, Any]], str]:
    if str(explicit_path).strip():
        path = resolve_project_path(explicit_path)
    else:
        path = sibling_metadata_path(activation_path)
    if path is None or not path.exists():
        return [], ""
    return read_jsonl(path), str(path)


def canonical_type(value: Any) -> str:
    text = str(value).strip().lower()
    if text in EXPERTS:
        return text
    for prefix, expert in SUBTYPE_TO_TYPE_PREFIXES.items():
        if text == prefix or text.startswith(prefix + "_") or text.startswith(prefix + "-"):
            return expert
    if "color" in text or "count" in text:
        return "attr"
    if any(token in text for token in ("left", "right", "above", "below", "spatial", "relation")):
        return "rel"
    return text


def type_labels_from_cache(cache: Mapping[str, Any], metadata: list[dict[str, Any]], n_rows: int) -> list[str]:
    for key in ("hallucination_types", "types", "type", "expert_types", "experts"):
        values = cache.get(key)
        if isinstance(values, (list, tuple)) and len(values) == n_rows:
            return [canonical_type(value) for value in values]
    subtypes = cache.get("subtypes")
    if isinstance(subtypes, (list, tuple)) and len(subtypes) == n_rows:
        return [canonical_type(value) for value in subtypes]
    if metadata:
        if len(metadata) != n_rows:
            raise ValueError(f"Metadata rows ({len(metadata)}) do not match activation rows ({n_rows})")
        labels: list[str] = []
        for row in metadata:
            value = find_first(row, TYPE_FIELDS)
            if value is None:
                value = row.get("subtype", "")
            labels.append(canonical_type(value))
        return labels
    raise ValueError("Could not infer cat/attr/rel labels from cache or metadata.")


def select_layers(tensor: Any, cache_layers: list[int], raw_layers: str) -> tuple[Any, list[int], list[int]]:
    torch = require_torch()
    if str(raw_layers).strip().lower() in {"", "all"}:
        selected_layers = list(cache_layers)
    else:
        selected_layers = parse_layer_spec(raw_layers)
    layer_to_index = {int(layer): index for index, layer in enumerate(cache_layers)}
    missing = [layer for layer in selected_layers if int(layer) not in layer_to_index]
    if missing:
        raise ValueError(f"Requested layers missing from activation cache: {missing}; available={cache_layers}")
    indices = [layer_to_index[int(layer)] for layer in selected_layers]
    return tensor.index_select(1, torch.tensor(indices, dtype=torch.long)), selected_layers, indices


def normalize_samples(samples: Any) -> Any:
    flat = samples.float().reshape(samples.shape[0], -1)
    denom = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / denom).reshape_as(samples).float()


def normalize_vector(vector: Any) -> Any:
    denom = vector.float().reshape(-1).norm().clamp_min(1e-12)
    return (vector.float() / denom).float()


def dot_flat(a: Any, b: Any) -> Any:
    return (a.float().reshape(-1) * b.float().reshape(-1)).sum()


def cosine_flat(a: Any, b: Any) -> float:
    denom = float(a.float().reshape(-1).norm().item() * b.float().reshape(-1).norm().item())
    if denom <= 1e-12:
        return 0.0
    return float(dot_flat(a, b).item() / denom)


def flatten_rows(samples: Any) -> Any:
    return samples.float().reshape(samples.shape[0], -1)


def project_onto_subspace(vector: Any, subspace: Any) -> Any:
    torch = require_torch()
    shape = tuple(vector.shape)
    v = vector.float().reshape(-1)
    g = subspace.float()
    if g.ndim == 1:
        denom = torch.dot(g, g).clamp_min(1e-12)
        projected = (torch.dot(v, g) / denom) * g
    else:
        projected = g.T @ (g @ v)
    return projected.reshape(shape).float()


def validate_tensor(name: str, tensor: Any) -> None:
    torch = require_torch()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")


def norm_summary(vector: Any) -> dict[str, float]:
    per_head = vector.float().norm(dim=-1) if vector.ndim >= 3 else vector.float().reshape(-1).abs()
    flat_norm = vector.float().reshape(-1).norm()
    return {
        "mean": float(per_head.mean().item()),
        "max": float(per_head.max().item()),
        "min": float(per_head.min().item()),
        "flat": float(flat_norm.item()),
    }


def top_heads(vector: Any, layers: list[int], top_k: int) -> list[dict[str, Any]]:
    if vector.ndim < 3:
        return []
    norms = vector.float().norm(dim=-1)
    rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        for head in range(int(norms.shape[1])):
            rows.append({"layer": int(layer), "head": int(head), "score": float(norms[layer_index, head].item())})
    rows.sort(key=lambda row: (-float(row["score"]), int(row["layer"]), int(row["head"])))
    return rows[: int(top_k)]


def head_set(rows: list[Mapping[str, Any]]) -> set[tuple[int, int]]:
    return {(int(row["layer"]), int(row["head"])) for row in rows}


def topk_overlap(vectors: Mapping[str, Any], layers: list[int], top_k: int) -> list[dict[str, Any]]:
    sets = {key: head_set(top_heads(vector, layers, top_k)) for key, vector in vectors.items()}
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(sets), 2):
        intersection = len(sets[left] & sets[right])
        union = len(sets[left] | sets[right])
        rows.append(
            {
                "vector_a": left,
                "vector_b": right,
                "intersection": intersection,
                "jaccard": intersection / union if union else 0.0,
            }
        )
    return rows


def cosine_matrix(vectors: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {left: {right: cosine_flat(vectors[left], vectors[right]) for right in vectors} for left in vectors}


def matrix_rows(matrix: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    rows = []
    keys = list(matrix)
    for left in keys:
        row: dict[str, Any] = {"vector": left}
        row.update({right: matrix[left][right] for right in keys})
        rows.append(row)
    return rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def table(headers: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def cache_schema_summary(cache: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in cache.items():
        if isinstance(value, Mapping):
            summary[key] = {"type": type(value).__name__, "keys": list(value.keys())[:20]}
        elif isinstance(value, (list, tuple)):
            summary[key] = {"type": type(value).__name__, "len": len(value)}
        else:
            summary[key] = tensor_info(value)
    return summary


def delta_by_type_from_cache(
    cache: Mapping[str, Any],
    metadata: list[dict[str, Any]],
    layers_arg: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = require_torch()
    working = cache.get("activations", cache)
    if not isinstance(working, Mapping):
        raise ValueError("Activation payload must be a mapping.")

    # Common AFTER-template cache: z_text/z_visual or z_pos/z_neg as [N,L,H,D].
    z_text_value = find_first(working, ("z_text", "z_factual", "z_pos", "positive", "pos"))
    z_visual_value = find_first(working, ("z_visual", "z_counterfactual", "z_neg", "negative", "neg"))
    if z_text_value is not None and z_visual_value is not None:
        z_text = as_float_tensor(z_text_value, "z_text")
        z_visual = as_float_tensor(z_visual_value, "z_visual")
        if tuple(z_text.shape) != tuple(z_visual.shape):
            raise ValueError(f"Activation shapes differ: {list(z_text.shape)} vs {list(z_visual.shape)}")
        if z_text.ndim < 3:
            raise ValueError(f"Expected activations with sample/layer/head dims, got {list(z_text.shape)}")
        cache_layers = [int(layer) for layer in working.get("layers", list(range(int(z_text.shape[1]))))]
        diff = z_text - z_visual
        diff, selected_layers, selected_layer_indices = select_layers(diff, cache_layers, layers_arg)
        labels = type_labels_from_cache(working, metadata, int(diff.shape[0]))
        delta_by_type = {
            expert: diff.index_select(0, torch.tensor([idx for idx, label in enumerate(labels) if label == expert], dtype=torch.long))
            for expert in EXPERTS
        }
        info = {
            "schema": "paired_tensors",
            "z_text": tensor_info(z_text),
            "z_visual": tensor_info(z_visual),
            "cache_layers": cache_layers,
            "selected_layers": selected_layers,
            "selected_layer_indices": selected_layer_indices,
            "type_distribution": {expert: int(delta_by_type[expert].shape[0]) for expert in EXPERTS},
        }
        return delta_by_type, info

    # Mapping by type, with each type containing delta or paired tensors.
    delta_by_type: dict[str, Any] = {}
    for expert in EXPERTS:
        item = working.get(expert)
        if item is None:
            continue
        if isinstance(item, Mapping):
            delta_value = find_first(item, ("delta", "diff", "deltas"))
            if delta_value is None:
                item_text = find_first(item, ("z_text", "z_factual", "z_pos", "positive", "pos"))
                item_visual = find_first(item, ("z_visual", "z_counterfactual", "z_neg", "negative", "neg"))
                if item_text is None or item_visual is None:
                    raise ValueError(f"Type mapping for {expert} has no delta or paired tensor fields.")
                delta = as_float_tensor(item_text, f"{expert}.z_text") - as_float_tensor(item_visual, f"{expert}.z_visual")
            else:
                delta = as_float_tensor(delta_value, f"{expert}.delta")
        else:
            delta = as_float_tensor(item, expert)
        if delta.ndim == 3:
            delta = delta.unsqueeze(0)
        if delta.ndim < 4:
            raise ValueError(f"Expected {expert} deltas as [N,L,H,D], got {list(delta.shape)}")
        cache_layers = [int(layer) for layer in working.get("layers", list(range(int(delta.shape[1]))))]
        delta, selected_layers, selected_layer_indices = select_layers(delta, cache_layers, layers_arg)
        delta_by_type[expert] = delta
    if set(delta_by_type) == set(EXPERTS):
        return delta_by_type, {
            "schema": "dict_by_type",
            "cache_layers": cache_layers,
            "selected_layers": selected_layers,
            "selected_layer_indices": selected_layer_indices,
            "type_distribution": {expert: int(delta_by_type[expert].shape[0]) for expert in EXPERTS},
        }
    raise ValueError("Could not find a supported activation schema for cat/attr/rel deltas.")


def build_vectors(delta_by_type: Mapping[str, Any], global_method: str, subspace_k: int, sample_normalize: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = require_torch()
    if subspace_k < 1:
        raise ValueError("--subspace-k must be >= 1")
    for expert in EXPERTS:
        if int(delta_by_type[expert].shape[0]) == 0:
            raise ValueError(f"No samples for type '{expert}'")
    processed: dict[str, Any] = {}
    for expert, delta in delta_by_type.items():
        validate_tensor(f"delta_{expert}", delta)
        processed[expert] = normalize_samples(delta) if sample_normalize else delta.float()
    raw_vectors = {expert: normalize_vector(processed[expert].mean(dim=0)) for expert in EXPERTS}
    for key, vector in raw_vectors.items():
        validate_tensor(f"{key}_raw", vector)

    all_samples = torch.cat([processed[expert] for expert in EXPERTS], dim=0)
    global_mean_all = normalize_vector(all_samples.mean(dim=0))
    singular_values: dict[str, list[float]] = {}
    shared_subspace = None

    type_matrix = torch.stack([raw_vectors[expert].reshape(-1) for expert in EXPERTS], dim=0)
    _u, s_type, vh_type = torch.linalg.svd(type_matrix, full_matrices=False)
    singular_values["type_mean_svd"] = [float(item) for item in s_type.cpu()]
    global_type_svd = normalize_vector(vh_type[0].reshape_as(global_mean_all))
    type_subspace = vh_type[: min(int(subspace_k), int(vh_type.shape[0]))].float()
    if cosine_flat(global_type_svd, global_mean_all) < 0.0:
        global_type_svd = -global_type_svd
        type_subspace = type_subspace.clone()
        type_subspace[0] = -type_subspace[0]

    if global_method == "mean_all":
        global_vector = global_mean_all
        shared_subspace = global_vector.reshape(1, -1)
    elif global_method == "type_mean_svd":
        global_vector = global_type_svd
        shared_subspace = type_subspace
    elif global_method == "all_svd":
        sample_matrix = flatten_rows(all_samples)
        _u_all, s_all, vh_all = torch.linalg.svd(sample_matrix, full_matrices=False)
        singular_values["all_svd"] = [float(item) for item in s_all[: max(int(subspace_k), 1)].cpu()]
        shared_subspace = vh_all[: int(subspace_k)].float()
        global_vector = normalize_vector(shared_subspace[0].reshape_as(global_mean_all))
        if cosine_flat(global_vector, global_mean_all) < 0.0:
            shared_subspace = shared_subspace.clone()
            shared_subspace[0] = -shared_subspace[0]
            global_vector = -global_vector
    else:
        raise ValueError(f"Unsupported global method: {global_method}")

    residuals: dict[str, Any] = {}
    projection_stats: dict[str, dict[str, float]] = {}
    for expert in EXPERTS:
        raw = raw_vectors[expert]
        projection = project_onto_subspace(raw, shared_subspace)
        residual = raw - projection
        residual_norm = float(residual.reshape(-1).norm().item())
        raw_norm = float(raw.reshape(-1).norm().item())
        projection_norm = float(projection.reshape(-1).norm().item())
        residuals[f"{expert}_res"] = normalize_vector(residual)
        projection_stats[expert] = {
            "raw_norm": raw_norm,
            "proj_norm": projection_norm,
            "proj_norm_over_raw_norm": projection_norm / raw_norm if raw_norm else 0.0,
            "residual_norm": residual_norm,
            "residual_norm_over_raw_norm": residual_norm / raw_norm if raw_norm else 0.0,
            "raw_global_cosine": cosine_flat(raw, global_vector),
            "residual_global_cosine": cosine_flat(residuals[f"{expert}_res"], global_vector),
        }
    vectors = {
        "global_type_svd": global_type_svd.float(),
        "global_mean_all": global_mean_all.float(),
        "global": global_vector.float(),
        "global_all": global_vector.float(),
        "cat_raw": raw_vectors["cat"].float(),
        "attr_raw": raw_vectors["attr"].float(),
        "rel_raw": raw_vectors["rel"].float(),
        "cat_res": residuals["cat_res"].float(),
        "attr_res": residuals["attr_res"].float(),
        "rel_res": residuals["rel_res"].float(),
    }
    vectors["cat"] = vectors["cat_res"]
    vectors["attr"] = vectors["attr_res"]
    vectors["rel"] = vectors["rel_res"]
    for key, vector in vectors.items():
        validate_tensor(key, vector)
    diagnostics = {
        "singular_values": singular_values,
        "projection_stats": projection_stats,
        "raw_cosine_matrix": cosine_matrix(
            {
                "global_type_svd": vectors["global_type_svd"],
                "global_mean_all": vectors["global_mean_all"],
                "cat_raw": vectors["cat_raw"],
                "attr_raw": vectors["attr_raw"],
                "rel_raw": vectors["rel_raw"],
            }
        ),
        "residual_cosine_matrix": cosine_matrix(
            {
                "global": vectors["global"],
                "cat_res": vectors["cat_res"],
                "attr_res": vectors["attr_res"],
                "rel_res": vectors["rel_res"],
            }
        ),
    }
    return vectors, {"diagnostics": diagnostics, "shared_subspace": shared_subspace.float()}


def render_report(stats: Mapping[str, Any]) -> str:
    selected_layers = stats["cache_info"].get("selected_layers", [])
    vector_norm_rows = [
        {"vector": key, **value}
        for key, value in stats.get("vector_norms", {}).items()
    ]
    projection_rows = [
        {"type": expert, **values}
        for expert, values in stats.get("diagnostics", {}).get("projection_stats", {}).items()
    ]
    top_sections: list[str] = []
    for topk, rows in stats.get("topk_overlap", {}).items():
        top_sections.extend(["", f"## Top{topk} Head Overlap", "", table(["vector_a", "vector_b", "intersection", "jaccard"], rows)])

    residual_global_cos = [
        abs(float(values.get("residual_global_cosine", 0.0)))
        for values in stats.get("diagnostics", {}).get("projection_stats", {}).values()
    ]
    residual_cos_matrix = stats.get("diagnostics", {}).get("residual_cosine_matrix", {})
    raw_cos_matrix = stats.get("diagnostics", {}).get("raw_cosine_matrix", {})
    interpretation = []
    if residual_global_cos and max(residual_global_cos) < 1e-3:
        interpretation.append("- Residual/global cosine is near 0, so projection removal succeeded numerically.")
    elif residual_global_cos:
        interpretation.append(f"- Residual/global cosine max is `{max(residual_global_cos):.4f}`; inspect if this is larger than expected.")
    for key in ("rel", "cat", "attr"):
        ratio = stats.get("diagnostics", {}).get("projection_stats", {}).get(key, {}).get("residual_norm_over_raw_norm")
        if ratio is not None and float(ratio) < 0.2:
            interpretation.append(f"- `{key}_res` norm is small relative to raw (`{float(ratio):.4f}`), so much of that signal may be shared/global.")
    if not interpretation:
        interpretation.append("- Residual norms and cosines look numerically well-behaved.")

    lines = [
        "# Shared-Private Vector Report",
        "",
        f"- Activation cache: `{stats['source_activations']}`",
        f"- Metadata: `{stats.get('metadata_path', '')}`",
        f"- Output vector file: `{stats['output']}`",
        f"- Eval-compatible file: `{stats['compatible_output']}`",
        f"- Global method: `{stats['global_method']}`",
        f"- Subspace k: `{stats['subspace_k']}`",
        f"- Sample normalize: `{stats['sample_normalize']}`",
        f"- Delta/vector shape: `{stats['vector_shape']}`",
        f"- Selected layers: `{selected_layers}`",
        f"- Counts: `{stats['counts']}`",
        "",
        "## Activation Cache Schema",
        "",
        "```json",
        json.dumps(stats.get("cache_schema", {}), ensure_ascii=False, indent=2)[:6000],
        "```",
        "",
        "## Singular Values",
        "",
        "```json",
        json.dumps(stats.get("diagnostics", {}).get("singular_values", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Vector Norms",
        "",
        table(["vector", "mean", "max", "min", "flat"], vector_norm_rows),
        "",
        "## Raw Cosine Matrix",
        "",
        table(["vector", *list(raw_cos_matrix.keys())], matrix_rows(raw_cos_matrix)) if raw_cos_matrix else "- missing",
        "",
        "## Residual Cosine Matrix",
        "",
        table(["vector", *list(residual_cos_matrix.keys())], matrix_rows(residual_cos_matrix)) if residual_cos_matrix else "- missing",
        "",
        "## Projection / Residual Ratios",
        "",
        table(
            [
                "type",
                "raw_norm",
                "proj_norm",
                "proj_norm_over_raw_norm",
                "residual_norm",
                "residual_norm_over_raw_norm",
                "raw_global_cosine",
                "residual_global_cosine",
            ],
            projection_rows,
        ),
        *top_sections,
        "",
        "## Automatic Interpretation",
        "",
        *interpretation,
        "",
        "## Eval Notes",
        "",
        "- The eval-compatible file stores residuals as `vectors['cat']`, `vectors['attr']`, and `vectors['rel']`.",
        "- Additional keys include `global_type_svd`, `global_mean_all`, `cat_raw`, `attr_raw`, `rel_raw`, `cat_res`, `attr_res`, `rel_res`.",
        "- Existing `ExpertSteeringController` can load any key if the runner exposes `--steer-enabled-experts`; POPE official runner currently hardcodes `cat`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        torch = require_torch()
        activation_path = resolve_project_path(args.activations)
        output_path = resolve_project_path(args.output)
        report_path = resolve_project_path(args.report_output)
        compatible_path = resolve_project_path(args.compatible_output) if str(args.compatible_output).strip() else output_path.with_name(output_path.stem + "_eval_compatible.pt")
        if (output_path.exists() or compatible_path.exists() or report_path.exists()) and not args.overwrite:
            raise FileExistsError("Output exists. Pass --overwrite to replace shared-private outputs.")
        cache = load_torch(activation_path)
        metadata, metadata_path = load_metadata(activation_path, str(args.metadata))
        delta_by_type, cache_info = delta_by_type_from_cache(cache, metadata, str(args.layers))
        counts = {expert: int(delta_by_type[expert].shape[0]) for expert in EXPERTS}
        vector_shape = list(delta_by_type["cat"].shape[1:])
        for expert in EXPERTS:
            if list(delta_by_type[expert].shape[1:]) != vector_shape:
                raise ValueError(f"Vector shape mismatch for {expert}: {list(delta_by_type[expert].shape[1:])} vs {vector_shape}")
        vectors, extra = build_vectors(delta_by_type, str(args.global_method), int(args.subspace_k), bool(args.sample_normalize))
        shared_subspace = extra["shared_subspace"]
        diagnostics = extra["diagnostics"]
        layers = [int(layer) for layer in cache_info.get("selected_layers", list(range(int(vector_shape[0]))))]
        num_heads = int(vector_shape[1]) if len(vector_shape) >= 2 else 1
        head_dim = int(vector_shape[2]) if len(vector_shape) >= 3 else int(torch.tensor(vector_shape).prod().item())
        hidden_size = num_heads * head_dim
        topk_values = [int(item) for item in str(args.topk_heads).replace(",", " ").split() if item.strip()]
        overlap_vectors = {
            key: vectors[key]
            for key in ("global_type_svd", "cat_raw", "attr_raw", "rel_raw", "cat_res", "attr_res", "rel_res")
        }
        topk_overlap_rows = {
            str(topk): topk_overlap(overlap_vectors, layers, int(topk))
            for topk in topk_values
        }
        vector_norms = {key: norm_summary(value) for key, value in vectors.items()}
        stats = {
            "source_activations": str(activation_path),
            "source_sha256": sha256_file(activation_path) if activation_path.exists() else "",
            "metadata_path": metadata_path,
            "output": str(output_path),
            "compatible_output": str(compatible_path),
            "global_method": str(args.global_method),
            "subspace_k": int(args.subspace_k),
            "sample_normalize": bool(args.sample_normalize),
            "counts": counts,
            "vector_shape": vector_shape,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "cache_schema": cache_schema_summary(cache),
            "cache_info": cache_info,
            "diagnostics": diagnostics,
            "vector_norms": vector_norms,
            "topk_overlap": topk_overlap_rows,
        }
        payload = {
            **{key: vectors[key] for key in ("global_type_svd", "global_mean_all", "cat_raw", "attr_raw", "rel_raw", "cat_res", "attr_res", "rel_res")},
            "global_all": vectors["global_all"],
            "shared_subspace": shared_subspace,
            "vectors": vectors,
            "layers": layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "metadata": {
                "source_activations": str(activation_path),
                "global_method": str(args.global_method),
                "subspace_k": int(args.subspace_k),
                "sample_normalize": bool(args.sample_normalize),
                "counts": counts,
                "vector_shape": vector_shape,
                "created_by": "scripts/build_shared_private_vectors.py",
            },
            "diagnostics": diagnostics,
        }
        compatible_payload = {
            "vectors": vectors,
            "layers": layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "config": {
                "source": "shared_private_vectors",
                "source_activations": str(activation_path),
                "global_method": str(args.global_method),
                "subspace_k": int(args.subspace_k),
                "sample_normalize": bool(args.sample_normalize),
                "default_cat_attr_rel": "residual vectors",
            },
            "components": {
                "shared_subspace": shared_subspace,
                "raw_vectors": {key: vectors[f"{key}_raw"] for key in EXPERTS},
                "residual_vectors": {key: vectors[f"{key}_res"] for key in EXPERTS},
            },
            "stats": stats,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        torch.save(compatible_payload, compatible_path)
        write_json(output_path.with_suffix(".stats.json"), stats)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(stats), encoding="utf-8")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote shared-private vectors to {output_path}")
    print(f"Wrote eval-compatible vectors to {compatible_path}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
