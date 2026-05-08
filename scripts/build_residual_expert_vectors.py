"""Remove a shared component from expert vectors and analyze residual experts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-path", required=True, help="Input torch vector payload.")
    parser.add_argument("--experts", default="cat,attr,rel", help="Comma-separated vector keys.")
    parser.add_argument(
        "--output",
        default="data/outputs_after_template_disjoint_v2/steering/after_template_disjoint_v2_residual_vectors.pt",
    )
    parser.add_argument(
        "--stats-output",
        default="data/outputs_after_template_disjoint_v2/steering/after_template_disjoint_v2_residual_vectors.stats.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_csv_items(text: str) -> list[str]:
    """Parse comma-separated non-empty items."""

    return [item.strip() for item in str(text).split(",") if item.strip()]


def require_torch() -> Any:
    """Import torch lazily."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("build_residual_expert_vectors requires torch.") from exc


def load_torch(path: Path) -> dict[str, Any]:
    """Load a torch payload."""

    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    """Write JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def cosine_flat(a: Any, b: Any) -> float:
    """Compute cosine between flattened tensors."""

    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = float(a_flat.norm().item() * b_flat.norm().item())
    if denom == 0.0:
        return 0.0
    return float((a_flat * b_flat).sum().item() / denom)


def norm_summary(vector: Any) -> dict[str, float]:
    """Summarize per-head vector norms."""

    norms = vector.float().norm(dim=-1)
    return {
        "mean": float(norms.mean().item()),
        "max": float(norms.max().item()),
        "min": float(norms.min().item()),
    }


def project_out(vector: Any, shared: Any) -> Any:
    """Remove the flattened projection of vector onto shared."""

    denom = (shared.flatten().float() * shared.flatten().float()).sum().clamp_min(1e-12)
    coeff = (vector.flatten().float() * shared.flatten().float()).sum() / denom
    return vector.float() - coeff * shared.float()


def pairwise_cosines(vectors: dict[str, Any]) -> dict[str, float]:
    """Compute pairwise flattened cosine values."""

    keys = list(vectors)
    result: dict[str, float] = {}
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            result[f"{left}_{right}"] = cosine_flat(vectors[left], vectors[right])
    return result


def main() -> int:
    """Build residual expert vectors."""

    args = parse_args()
    try:
        torch = require_torch()
        vector_path = resolve_project_path(args.vector_path)
        output_path = resolve_project_path(args.output)
        stats_path = resolve_project_path(args.stats_output)
        if (output_path.exists() or stats_path.exists()) and not args.overwrite:
            raise FileExistsError("Output exists. Pass --overwrite to replace residual vector outputs.")

        payload = load_torch(vector_path)
        experts = parse_csv_items(args.experts)
        source_vectors = payload.get("vectors", {})
        if not isinstance(source_vectors, dict):
            raise ValueError("Input payload does not contain a dict `vectors` field")
        missing = [expert for expert in experts if expert not in source_vectors]
        if missing:
            raise KeyError(f"Missing experts in vector payload: {missing}")

        vectors = {expert: source_vectors[expert].detach().cpu().float() for expert in experts}
        shapes = {expert: tuple(vector.shape) for expert, vector in vectors.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Expert vectors must have same shape, got {shapes}")

        shared = torch.stack([vectors[expert] for expert in experts], dim=0).mean(dim=0)
        residual_vectors = {f"{expert}_resid": project_out(vectors[expert], shared) for expert in experts}
        output_vectors = {
            **vectors,
            "shared": shared.float(),
            **residual_vectors,
        }
        base_config = payload.get("config", {})
        if not isinstance(base_config, dict):
            base_config = {}
        base_components = payload.get("components", {})
        if not isinstance(base_components, dict):
            base_components = {}

        output_payload = {
            **payload,
            "vectors": output_vectors,
            "config": {
                **dict(base_config),
                "residual_source_vector_path": str(vector_path),
                "residual_source_experts": experts,
                "residual_shared_definition": "mean(source expert vectors)",
                "residual_direction": "expert - projection(expert, shared)",
            },
            "components": {
                **dict(base_components),
                "shared_vector": shared.float(),
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_payload, output_path)

        residual_source_keys = list(residual_vectors)
        stats = {
            "source": "residual_expert_vectors",
            "input_vector_path": str(vector_path),
            "output": str(output_path),
            "experts": experts,
            "residual_experts": residual_source_keys,
            "layers": [int(layer) for layer in payload.get("layers", [])],
            "shape": list(next(iter(vectors.values())).shape),
            "source_cosines": pairwise_cosines(vectors),
            "residual_cosines": pairwise_cosines(residual_vectors),
            "source_to_shared_cosines": {expert: cosine_flat(vector, shared) for expert, vector in vectors.items()},
            "residual_to_shared_cosines": {
                expert: cosine_flat(vector, shared) for expert, vector in residual_vectors.items()
            },
            "source_norms": {expert: norm_summary(vector) for expert, vector in vectors.items()},
            "shared_norm": norm_summary(shared),
            "residual_norms": {expert: norm_summary(vector) for expert, vector in residual_vectors.items()},
            "notes": [
                "Residual vectors remove one global shared component across the selected experts.",
                "Residual expert keys are suffixed with `_resid`.",
                "Use residual vectors for diagnostics first; benchmark steering may need smaller alpha.",
            ],
        }
        write_json(stats_path, stats)
        print(f"Wrote residual vectors to {output_path}")
        print(f"Wrote residual stats to {stats_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
