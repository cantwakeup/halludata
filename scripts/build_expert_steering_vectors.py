"""Build typed cat/attr/rel steering vectors from an existing activation cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import load_activation_cache, write_json
from expert_data.expert_vectors import (
    build_expert_vectors_from_cache,
    expert_stats_json_path,
    normalize_bool,
    parse_layer_spec,
    save_expert_vectors,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for expert steering-vector construction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cache",
        default="data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/merged",
        help="Merged train activation cache directory or activations.pt file.",
    )
    parser.add_argument(
        "--output-path",
        default="data/outputs/steering/expert_vectors.pt",
        help="Output torch file for typed expert steering vectors.",
    )
    parser.add_argument("--layers", default="10-20", help="Layer spec, e.g. 10-20 or 10,12,14.")
    parser.add_argument("--max-samples-per-type", type=int, default=2000, help="0 means use all rows.")
    parser.add_argument("--normalize", default="false", help="Whether to L2-normalize each layer-head vector.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for optional row subsampling.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing outputs.")

    # Compatibility placeholders for the original full prompt. The current
    # implementation intentionally reuses already extracted LLaVA activations.
    parser.add_argument("--model-path", default="", help="Reserved for future diff-span extraction mode.")
    parser.add_argument("--cat-data", default="", help="Reserved for future raw expert JSONL mode.")
    parser.add_argument("--attr-data", default="", help="Reserved for future raw expert JSONL mode.")
    parser.add_argument("--rel-data", default="", help="Reserved for future raw expert JSONL mode.")
    parser.add_argument("--position-mode", default="activation_cache", help="Recorded for provenance only.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_can_write(path: Path, overwrite: bool) -> None:
    """Validate that a target file can be written."""

    stats_path = expert_stats_json_path(path)
    existing = [candidate for candidate in (path, stats_path) if candidate.exists()]
    if existing and not overwrite:
        names = ", ".join(str(candidate) for candidate in existing)
        raise FileExistsError(f"Output already exists: {names}. Pass --overwrite to replace.")
    path.parent.mkdir(parents=True, exist_ok=True)


def _jsonable_stats(payload: dict[str, Any], args: argparse.Namespace, train_cache_path: Path) -> dict[str, Any]:
    """Build the human-readable stats JSON payload."""

    stats = dict(payload["stats"])
    stats["config"] = {
        **dict(payload["config"]),
        "train_cache": str(train_cache_path),
        "layers": list(payload["layers"]),
        "position_mode": str(args.position_mode),
        "output_path": str(resolve_project_path(args.output_path)),
    }
    stats["shape"] = {
        "num_selected_layers": len(payload["layers"]),
        "num_heads": int(payload["num_heads"]),
        "head_dim": int(payload["head_dim"]),
        "hidden_size": int(payload["hidden_size"]),
    }
    return stats


def print_diagnostics(payload: dict[str, Any]) -> None:
    """Print compact diagnostics for a newly built expert-vector file."""

    stats = payload["stats"]
    print("Expert steering vectors built from factual-minus-counterfactual activation differences.")
    print(f"Layers: {payload['layers']}")
    print(f"Shape per expert: [{len(payload['layers'])}, {payload['num_heads']}, {payload['head_dim']}]")
    for expert, count in stats["sample_counts"].items():
        norms = stats["vector_norms"][expert]
        print(
            f"{expert}: samples={count}, "
            f"mean_norm={norms['mean']:.6f}, max_norm={norms['max']:.6f}, min_norm={norms['min']:.6f}"
        )
    print("Top layer-heads by vector norm:")
    for row in stats["top_heads_by_norm"][:20]:
        print(
            f"  {row['expert']} layer={row['layer']} head={row['head']} norm={float(row['norm']):.6f}"
        )


def main() -> int:
    """Run typed expert steering-vector construction from the CLI."""

    args = parse_args()
    train_cache_path = resolve_project_path(args.train_cache)
    output_path = resolve_project_path(args.output_path)
    try:
        ensure_can_write(output_path, bool(args.overwrite))
        cache_payload = load_activation_cache(train_cache_path)
        activations = cache_payload["activations"]
        num_layers = int(getattr(activations["z_pos"], "shape", [0, 0])[1])
        layers = parse_layer_spec(args.layers, num_layers=num_layers)
        payload = build_expert_vectors_from_cache(
            cache_payload,
            layers=layers,
            max_samples_per_type=int(args.max_samples_per_type),
            normalize=normalize_bool(args.normalize),
            seed=int(args.seed),
        )
        payload["config"]["source_cache"] = str(train_cache_path)
        payload["config"]["position_mode"] = str(args.position_mode)
        save_expert_vectors(output_path, payload)
        stats_path = expert_stats_json_path(output_path)
        write_json(stats_path, _jsonable_stats(payload, args, train_cache_path))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote expert steering vectors to {output_path}")
    print(f"Wrote expert steering stats to {stats_path}")
    print_diagnostics(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
