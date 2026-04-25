"""Evaluate expert-vector directions against held-out activation caches without running LLaVA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import load_activation_cache, write_json
from expert_data.steering import parse_layer_spec

VECTOR_EXPERTS = ("cat", "attr", "rel")
SUBTYPE_TO_VECTOR_EXPERT = {
    "cat": "cat",
    "cnt": "attr",
    "col": "attr",
    "rel": "rel",
}
SUBTYPE_TO_ROW_GROUP = {
    "cat": "cat",
    "cnt": "attr_cnt",
    "col": "attr_col",
    "rel": "rel",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for held-out vector direction checks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-path", default="data/outputs/steering/expert_vectors.pt")
    parser.add_argument("--val-cache", required=True, help="Merged validation activation cache directory.")
    parser.add_argument("--test-cache", required=True, help="Merged test activation cache directory.")
    parser.add_argument("--pair-metadata", default="", help="Reserved compatibility arg; cache metadata is used by default.")
    parser.add_argument("--layers", default="10-20")
    parser.add_argument("--head-select", choices=["all", "norm"], default="all")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--output", default="data/outputs/debug/expert_vector_direction_eval.json")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_torch() -> Any:
    """Import torch lazily for offline tensor metrics."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("eval_expert_vector_direction requires a working torch installation.") from exc


def load_vector_payload(path: Path) -> Mapping[str, Any]:
    """Load an expert vector torch file with compatibility for older torch versions."""

    torch = require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def select_vector_layers(payload: Mapping[str, Any], layers: list[int]) -> dict[str, Any]:
    """Return expert vectors restricted to the requested absolute layer IDs."""

    torch = require_torch()
    vector_layers = [int(layer) for layer in payload["layers"]]
    missing = [layer for layer in layers if layer not in vector_layers]
    if missing:
        raise ValueError(f"Requested layer(s) are missing from vector file: {missing}")
    row_indices = torch.tensor([vector_layers.index(layer) for layer in layers], dtype=torch.long)
    return {
        expert: payload["vectors"][expert].detach().cpu().float().index_select(0, row_indices)
        for expert in VECTOR_EXPERTS
    }


def as_float_tensor(value: Any) -> Any:
    """Convert cache arrays to CPU float32 tensors."""

    torch = require_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def cache_tensors(cache_path: Path) -> tuple[Any, Any, list[str]]:
    """Load z_pos, z_neg, and subtype labels from one activation cache."""

    cache = load_activation_cache(cache_path)
    activations = cache["activations"]
    z_pos = as_float_tensor(activations["z_pos"])
    z_neg = as_float_tensor(activations["z_neg"])
    subtypes = [str(item) for item in activations["subtypes"]]
    if tuple(z_pos.shape) != tuple(z_neg.shape):
        raise ValueError(f"z_pos/z_neg shape mismatch in {cache_path}")
    if z_pos.ndim != 4:
        raise ValueError(f"Expected [N,L,H,D] activations, got {list(z_pos.shape)}")
    if len(subtypes) != int(z_pos.shape[0]):
        raise ValueError("subtype count does not match activation rows")
    return z_pos, z_neg, subtypes


def topk_head_indices(vectors: Mapping[str, Any], top_k: int) -> dict[str, list[tuple[int, int]] | None]:
    """Select top-K layer/head pairs per expert by vector norm."""

    if int(top_k) <= 0:
        return {expert: [] for expert in VECTOR_EXPERTS}
    selections: dict[str, list[tuple[int, int]] | None] = {}
    for expert, tensor in vectors.items():
        norms = tensor.norm(dim=-1)
        rows: list[tuple[float, int, int]] = []
        for layer_index in range(int(norms.shape[0])):
            for head in range(int(norms.shape[1])):
                rows.append((float(norms[layer_index, head].item()), layer_index, head))
        rows.sort(key=lambda item: (-item[0], item[1], item[2]))
        selections[expert] = [(layer_index, head) for _score, layer_index, head in rows[: int(top_k)]]
    return selections


def score_delta(delta: Any, vector: Any, heads: list[tuple[int, int]] | None) -> tuple[float, float]:
    """Return dot and cosine alignment for one delta tensor and one expert vector."""

    if heads is not None:
        if not heads:
            return 0.0, 0.0
        torch = require_torch()
        delta_flat = torch.stack([delta[layer_index, head] for layer_index, head in heads], dim=0).flatten()
        vector_flat = torch.stack([vector[layer_index, head] for layer_index, head in heads], dim=0).flatten()
    else:
        delta_flat = delta.flatten()
        vector_flat = vector.flatten()
    dot = float((delta_flat * vector_flat).sum().item())
    denom = float(delta_flat.norm().item() * vector_flat.norm().item())
    cosine = dot / denom if denom > 0.0 else 0.0
    return dot, cosine


def summarize_scores(rows: list[dict[str, float]]) -> dict[str, float | int]:
    """Summarize dot/cosine alignment rows."""

    if not rows:
        return {"n": 0, "dot_positive_rate": 0.0, "mean_dot": 0.0, "mean_cos": 0.0}
    return {
        "n": len(rows),
        "dot_positive_rate": sum(1 for row in rows if row["dot"] > 0.0) / len(rows),
        "mean_dot": mean([row["dot"] for row in rows]),
        "mean_cos": mean([row["cos"] for row in rows]),
    }


def evaluate_split(
    cache_path: Path,
    *,
    layers: list[int],
    vectors: Mapping[str, Any],
    head_select: str,
    top_k: int,
) -> dict[str, Any]:
    """Evaluate vector alignment for one held-out cache split."""

    torch = require_torch()
    z_pos, z_neg, subtypes = cache_tensors(cache_path)
    num_layers = int(z_pos.shape[1])
    if max(layers) >= num_layers:
        raise ValueError(f"Layer index {max(layers)} is out of range for cache with {num_layers} layers")
    layer_tensor = torch.tensor(layers, dtype=torch.long)
    delta = z_pos.index_select(1, layer_tensor) - z_neg.index_select(1, layer_tensor)
    selected_heads = topk_head_indices(vectors, top_k) if head_select == "norm" else {
        expert: None for expert in VECTOR_EXPERTS
    }

    own_rows: dict[str, list[dict[str, float]]] = {group: [] for group in ("cat", "attr_cnt", "attr_col", "rel")}
    cross_rows: dict[str, dict[str, list[dict[str, float]]]] = {
        group: {expert: [] for expert in VECTOR_EXPERTS}
        for group in ("cat", "attr_cnt", "attr_col", "rel")
    }

    for row_index, subtype in enumerate(subtypes):
        if subtype not in SUBTYPE_TO_VECTOR_EXPERT:
            continue
        row_group = SUBTYPE_TO_ROW_GROUP[subtype]
        own_expert = SUBTYPE_TO_VECTOR_EXPERT[subtype]
        row_delta = delta[row_index]
        for expert in VECTOR_EXPERTS:
            dot, cosine = score_delta(row_delta, vectors[expert], selected_heads[expert])
            score_row = {"dot": dot, "cos": cosine}
            cross_rows[row_group][expert].append(score_row)
            if expert == own_expert:
                own_rows[row_group].append(score_row)

    return {
        "cache_path": str(cache_path),
        "head_select": head_select,
        "top_k": int(top_k),
        "by_subtype": {
            group: summarize_scores(rows)
            for group, rows in own_rows.items()
        },
        "cross_expert": {
            group: {
                expert: summarize_scores(rows)
                for expert, rows in expert_rows.items()
            }
            for group, expert_rows in cross_rows.items()
        },
    }


def main() -> int:
    """Run held-out vector direction evaluation."""

    args = parse_args()
    try:
        payload = load_vector_payload(resolve_project_path(args.vector_path))
        layers = parse_layer_spec(args.layers)
        vectors = select_vector_layers(payload, layers)
        result = {
            "config": {
                "vector_path": str(resolve_project_path(args.vector_path)),
                "vector_layers": [int(layer) for layer in payload["layers"]],
                "layers": layers,
                "head_select": str(args.head_select),
                "top_k": int(args.top_k),
                "pair_metadata": str(args.pair_metadata or ""),
            },
            "splits": {
                "val": evaluate_split(
                    resolve_project_path(args.val_cache),
                    layers=layers,
                    vectors=vectors,
                    head_select=str(args.head_select),
                    top_k=int(args.top_k),
                ),
                "test": evaluate_split(
                    resolve_project_path(args.test_cache),
                    layers=layers,
                    vectors=vectors,
                    head_select=str(args.head_select),
                    top_k=int(args.top_k),
                ),
            },
        }
        write_json(resolve_project_path(args.output), result)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote expert-vector direction evaluation to {resolve_project_path(args.output)}")
    print(json.dumps(result["splits"], ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
