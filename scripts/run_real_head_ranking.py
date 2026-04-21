"""Rank real LLaVA heads and evaluate Top-K prototype subsets."""

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
from expert_data.real_head_ranking import (
    compute_head_scores,
    evaluate_topk_heads,
    random_topk_baseline,
    topk_overlap_matrix,
)
from expert_data.real_prototypes import evaluate_prototypes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for real head ranking."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, help="Merged train activation cache directory.")
    parser.add_argument("--val-cache", required=True, help="Merged validation activation cache directory.")
    parser.add_argument("--test-cache", required=True, help="Merged test activation cache directory.")
    parser.add_argument("--prototype-path", required=True, help="Prototype .pt path produced by eval_real_activation_signal.py.")
    parser.add_argument("--out-dir", required=True, help="Output directory for head-ranking results.")
    parser.add_argument("--topk", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256], help="Top-K values to sweep.")
    parser.add_argument("--random-repeats", type=int, default=100, help="Random Top-K repeats.")
    parser.add_argument("--score-type", default="two_proto", help="Prototype score type for Top-K evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty out-dir.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Create or validate a head-ranking output directory."""

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)


def _require_torch() -> Any:
    """Import torch lazily for prototype loading."""

    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("run_real_head_ranking.py requires torch.") from exc


def load_prototypes(path: str | Path) -> dict[str, Any]:
    """Load saved subtype prototypes as CPU float tensors."""

    torch = _require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    payload["mu_pos"] = payload["mu_pos"].float()
    payload["mu_neg"] = payload["mu_neg"].float()
    payload["axis"] = payload["axis"].float()
    payload["subtypes"] = [str(item) for item in payload["subtypes"]]
    return payload


def _thresholds_from_eval(eval_result: dict[str, Any]) -> dict[str, float]:
    """Extract subtype thresholds from an evaluation result."""

    return {subtype: float(metrics["threshold"]) for subtype, metrics in eval_result["by_subtype"].items()}


def _best_k_by_subtype(topk_sweep: dict[str, Any], topk_values: list[int]) -> tuple[dict[str, int], dict[str, float]]:
    """Choose the best validation K per subtype using pairwise accuracy."""

    subtypes = sorted(next(iter(topk_sweep.values()))["by_subtype"]) if topk_sweep else []
    best_k: dict[str, int] = {}
    thresholds: dict[str, float] = {}
    for subtype in subtypes:
        best_value = None
        best_metric = -1.0
        for top_k in topk_values:
            result = topk_sweep[str(top_k)]["by_subtype"][subtype]
            metric = float(result["pairwise_acc"])
            if metric > best_metric:
                best_metric = metric
                best_value = int(top_k)
                thresholds[subtype] = float(result["threshold"])
        best_k[subtype] = int(best_value or topk_values[0])
    return best_k, thresholds


def main() -> int:
    """Run real head ranking and Top-K evaluation."""

    args = parse_args()
    out_dir = resolve_project_path(args.out_dir)
    try:
        ensure_output_dir(out_dir, bool(args.overwrite))
        train_cache = load_activation_cache(resolve_project_path(args.train_cache))
        val_cache = load_activation_cache(resolve_project_path(args.val_cache))
        test_cache = load_activation_cache(resolve_project_path(args.test_cache))
        prototypes = load_prototypes(resolve_project_path(args.prototype_path))

        head_scores = compute_head_scores(train_cache, prototypes)
        write_json(out_dir / "head_ranking.json", head_scores)

        all_head_val = evaluate_prototypes(
            prototypes,
            val_cache,
            score_type=args.score_type,
            tune_thresholds=True,
        )
        all_head_test = evaluate_prototypes(
            prototypes,
            test_cache,
            score_type=args.score_type,
            thresholds=_thresholds_from_eval(all_head_val),
        )
        write_json(out_dir / "all_head_val.json", all_head_val)
        write_json(out_dir / "all_head_test.json", all_head_test)

        topk_values = [int(value) for value in args.topk]
        topk_sweep_val: dict[str, Any] = {}
        for top_k in topk_values:
            topk_sweep_val[str(top_k)] = evaluate_topk_heads(
                prototypes,
                val_cache,
                head_scores,
                top_k,
                score_type=args.score_type,
                tune_thresholds=True,
            )
        best_k, thresholds = _best_k_by_subtype(topk_sweep_val, topk_values)
        topk_eval_test = evaluate_topk_heads(
            prototypes,
            test_cache,
            head_scores,
            best_k,
            score_type=args.score_type,
            thresholds=thresholds,
        )
        topk_eval_test["best_k_by_subtype"] = best_k

        random_val = {
            str(top_k): random_topk_baseline(
                prototypes,
                val_cache,
                top_k,
                repeats=int(args.random_repeats),
                seed=int(args.seed) + top_k,
                score_type=args.score_type,
            )
            for top_k in topk_values
        }
        random_test = {
            str(top_k): random_topk_baseline(
                prototypes,
                test_cache,
                top_k,
                repeats=int(args.random_repeats),
                seed=int(args.seed) + 1000 + top_k,
                score_type=args.score_type,
            )
            for top_k in topk_values
        }
        write_json(out_dir / "topk_sweep_val.json", topk_sweep_val)
        write_json(out_dir / "topk_eval_test.json", topk_eval_test)
        write_json(out_dir / "random_topk_baseline_val.json", random_val)
        write_json(out_dir / "random_topk_baseline_test.json", random_test)
        overlap_k = min(max(topk_values), 64)
        write_json(out_dir / "topk_overlap_matrix.json", topk_overlap_matrix(head_scores, overlap_k))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote real head-ranking results to {out_dir}")
    print(f"Best K by subtype: {best_k}")
    for subtype, metrics in topk_eval_test["by_subtype"].items():
        print(
            f"test/topk/{subtype}: pairwise_acc={metrics['pairwise_acc']:.4f}, "
            f"AUROC={metrics['auroc']}, AP={metrics['average_precision']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
