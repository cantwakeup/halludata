"""Audit real activation prototype signal on train/val/test caches."""

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
from expert_data.real_prototypes import (
    build_subtype_prototypes_from_cache,
    compute_cross_subtype_matrix,
    evaluate_prototypes,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prototype signal evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, help="Merged train activation cache directory.")
    parser.add_argument("--val-cache", required=True, help="Merged validation activation cache directory.")
    parser.add_argument("--test-cache", required=True, help="Merged test activation cache directory.")
    parser.add_argument("--out-dir", required=True, help="Experiment output directory.")
    parser.add_argument("--score-types", nargs="+", default=["axis", "two_proto"], help="Score types to evaluate.")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap repeats for confidence intervals.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for label-shuffle controls.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty out-dir.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Create or validate an experiment output directory."""

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)


def _require_torch() -> Any:
    """Import torch lazily for prototype persistence."""

    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("eval_real_activation_signal.py requires torch to save prototypes.pt.") from exc


def _prototype_summary(prototypes: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON summary for saved prototypes."""

    return {
        "subtypes": list(prototypes["subtypes"]),
        "counts": dict(prototypes["counts"]),
        "warnings": dict(prototypes["warnings"]),
        "shape": {
            "num_subtypes": int(prototypes["axis"].shape[0]),
            "num_layers": int(prototypes["axis"].shape[1]),
            "num_heads": int(prototypes["axis"].shape[2]),
            "head_dim": int(prototypes["axis"].shape[3]),
        },
    }


def _thresholds_from_eval(eval_result: dict[str, Any]) -> dict[str, float]:
    """Extract subtype thresholds from an evaluation result."""

    return {subtype: float(metrics["threshold"]) for subtype, metrics in eval_result["by_subtype"].items()}


def main() -> int:
    """Run prototype signal evaluation."""

    args = parse_args()
    out_dir = resolve_project_path(args.out_dir)
    try:
        ensure_output_dir(out_dir, bool(args.overwrite))
        torch = _require_torch()
        train_cache = load_activation_cache(resolve_project_path(args.train_cache))
        val_cache = load_activation_cache(resolve_project_path(args.val_cache))
        test_cache = load_activation_cache(resolve_project_path(args.test_cache))

        prototypes = build_subtype_prototypes_from_cache(train_cache, seed=int(args.seed))
        shuffled_prototypes = build_subtype_prototypes_from_cache(
            train_cache,
            shuffle_labels=True,
            seed=int(args.seed),
        )
        torch.save(
            {
                "subtypes": prototypes["subtypes"],
                "mu_pos": prototypes["mu_pos"].half(),
                "mu_neg": prototypes["mu_neg"].half(),
                "axis": prototypes["axis"].half(),
                "counts": prototypes["counts"],
                "warnings": prototypes["warnings"],
            },
            out_dir / "prototypes.pt",
        )
        write_json(out_dir / "prototype_summary.json", _prototype_summary(prototypes))

        all_val: dict[str, Any] = {}
        all_test: dict[str, Any] = {}
        shuffle_val: dict[str, Any] = {}
        shuffle_test: dict[str, Any] = {}
        cross_val: dict[str, Any] = {}
        cross_test: dict[str, Any] = {}
        for score_type in args.score_types:
            val_result = evaluate_prototypes(
                prototypes,
                val_cache,
                score_type=str(score_type),
                tune_thresholds=True,
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            )
            test_result = evaluate_prototypes(
                prototypes,
                test_cache,
                score_type=str(score_type),
                thresholds=_thresholds_from_eval(val_result),
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            )
            shuffled_val_result = evaluate_prototypes(
                shuffled_prototypes,
                val_cache,
                score_type=str(score_type),
                tune_thresholds=True,
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            )
            shuffled_test_result = evaluate_prototypes(
                shuffled_prototypes,
                test_cache,
                score_type=str(score_type),
                thresholds=_thresholds_from_eval(shuffled_val_result),
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            )
            all_val[str(score_type)] = val_result
            all_test[str(score_type)] = test_result
            shuffle_val[str(score_type)] = shuffled_val_result
            shuffle_test[str(score_type)] = shuffled_test_result
            cross_val[str(score_type)] = compute_cross_subtype_matrix(prototypes, val_cache, score_type=str(score_type))
            cross_test[str(score_type)] = compute_cross_subtype_matrix(prototypes, test_cache, score_type=str(score_type))

        write_json(out_dir / "all_head_eval_val.json", all_val)
        write_json(out_dir / "all_head_eval_test.json", all_test)
        write_json(out_dir / "label_shuffle_control_val.json", shuffle_val)
        write_json(out_dir / "label_shuffle_control_test.json", shuffle_test)
        write_json(out_dir / "cross_subtype_matrix_val.json", cross_val)
        write_json(out_dir / "cross_subtype_matrix_test.json", cross_test)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote real activation prototype signal audit to {out_dir}")
    for score_type, result in all_test.items():
        print(f"test/{score_type}: macro_pairwise_acc={result['macro_pairwise_acc']:.4f}")
        for subtype, metrics in result["by_subtype"].items():
            print(
                f"  {subtype}: pairwise_acc={metrics['pairwise_acc']:.4f}, "
                f"AUROC={metrics['auroc']}, AP={metrics['average_precision']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
