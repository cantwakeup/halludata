"""Run internal teacher-forced steering tests on held-out pair splits."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import read_json, write_json, write_jsonl
from expert_data.io_utils import read_jsonl
from expert_data.steering import (
    LlavaCandidateScorer,
    MockCandidateScorer,
    VALID_SUBTYPES,
    select_top_heads,
    summarize_candidate_scores,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for internal steering evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-pairs", required=True, help="Validation pair JSONL used to select steering config.")
    parser.add_argument("--test-pairs", required=True, help="Held-out test pair JSONL used for final evaluation.")
    parser.add_argument("--out-dir", required=True, help="Output directory for steering results.")
    parser.add_argument("--adapter", choices=["mock", "llava"], default="mock", help="Candidate scorer backend.")
    parser.add_argument("--prototype-path", default="", help="prototypes.pt path from prototype signal audit.")
    parser.add_argument("--head-ranking", default="", help="head_ranking.json path from real head ranking.")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="HF model ID or local model path.")
    parser.add_argument("--image-root", default="", help="COCO image root for LLaVA scoring.")
    parser.add_argument("--instances-json", default="", help="COCO instances JSON for image path resolution.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for LLaVA scoring.")
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--topk", nargs="+", type=int, default=[16, 32, 64], help="Top-K head counts to sweep.")
    parser.add_argument("--alphas", nargs="+", type=float, default=[-1.0, -0.5, 0.5, 1.0], help="Steering strengths.")
    parser.add_argument("--signs", nargs="+", type=float, default=[1.0, -1.0], help="Subtype-level sign choices.")
    parser.add_argument("--max-val-samples-per-subtype", type=int, default=0, help="0 means all validation samples.")
    parser.add_argument("--max-test-samples-per-subtype", type=int, default=0, help="0 means all test samples.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for optional per-subtype limits.")
    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N scored pairs.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty out-dir.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_optional_project_path(raw_path: str | Path | None) -> Path | None:
    """Resolve optional project-relative paths and treat empty strings as absent."""

    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    return resolve_project_path(text)


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Create or validate an output directory."""

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)


def validate_pair_rows(rows: list[dict[str, Any]]) -> None:
    """Validate the minimum fields needed for candidate steering tests."""

    required = ("pair_id", "image_id", "subtype", "question", "response_pos", "response_neg")
    for index, row in enumerate(rows):
        missing = [field for field in required if field not in row or row[field] in {None, ""}]
        if missing:
            raise ValueError(f"row {index} is missing required field(s): {', '.join(sorted(missing))}")
        if str(row["subtype"]) not in VALID_SUBTYPES:
            raise ValueError(f"row {index} has unsupported subtype '{row['subtype']}'")


def group_pairs_by_subtype(rows: list[dict[str, Any]], max_per_subtype: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    """Group pairs by subtype with deterministic optional sampling."""

    grouped: dict[str, list[dict[str, Any]]] = {subtype: [] for subtype in sorted(VALID_SUBTYPES)}
    for row in rows:
        grouped[str(row["subtype"])].append(row)
    if int(max_per_subtype) <= 0:
        return {subtype: rows for subtype, rows in grouped.items() if rows}
    rng = random.Random(int(seed))
    sampled: dict[str, list[dict[str, Any]]] = {}
    for subtype, subtype_rows in grouped.items():
        subtype_rows = list(subtype_rows)
        rng.shuffle(subtype_rows)
        sampled[subtype] = subtype_rows[: int(max_per_subtype)]
    return {subtype: rows for subtype, rows in sampled.items() if rows}


def _require_torch() -> Any:
    """Import torch lazily for real prototype loading."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("LLaVA steering requires a working torch installation to load prototypes.pt.") from exc


def load_prototypes(path: Path | None) -> dict[str, Any] | None:
    """Load saved real prototypes when a path is provided."""

    if path is None:
        return None
    torch = _require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    payload["axis"] = payload["axis"].float()
    payload["subtypes"] = [str(item) for item in payload["subtypes"]]
    return payload


def synthetic_head_ranking(subtypes: list[str]) -> dict[str, list[dict[str, int | float]]]:
    """Create a tiny ranking for mock scoring when no head-ranking file is supplied."""

    return {
        subtype: [{"layer": 0, "head": 0, "score": 1.0, "sep": 1.0, "disp_pos": 0.0, "disp_neg": 0.0}]
        for subtype in subtypes
    }


def subtype_steering_vectors(prototypes: Mapping[str, Any] | None, subtype: str) -> Any | None:
    """Return one subtype's [L,H,D] steering vectors from prototypes."""

    if prototypes is None:
        return None
    subtypes = [str(item) for item in prototypes["subtypes"]]
    if str(subtype) not in subtypes:
        raise KeyError(f"Subtype '{subtype}' is missing from prototypes")
    return prototypes["axis"][subtypes.index(str(subtype))]


def score_rows(
    scorer: Any,
    rows: list[dict[str, Any]],
    *,
    subtype: str,
    alpha: float,
    sign: float,
    top_k: int,
    head_ranking: Mapping[str, list[Mapping[str, Any]]],
    prototypes: Mapping[str, Any] | None,
    progress_label: str,
    progress_every: int,
) -> list[dict[str, Any]]:
    """Score a group of rows under one steering configuration."""

    selected_heads = select_top_heads(head_ranking, subtype, top_k)
    vectors = subtype_steering_vectors(prototypes, subtype)
    scored_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        scores = scorer.score_pair(
            row,
            alpha=float(alpha),
            sign=float(sign),
            selected_heads=selected_heads,
            steering_vectors=vectors,
        )
        scored_rows.append(
            {
                "pair_id": str(row["pair_id"]),
                "image_id": str(row["image_id"]),
                "subtype": str(row["subtype"]),
                "alpha": float(alpha),
                "sign": float(sign),
                "top_k": int(top_k),
                "score_pos": float(scores["score_pos"]),
                "score_neg": float(scores["score_neg"]),
                "margin": float(scores["score_pos"]) - float(scores["score_neg"]),
            }
        )
        if progress_every > 0 and index % int(progress_every) == 0:
            print(f"[steering] {progress_label}: scored {index}/{len(rows)}")
    return scored_rows


def choose_best_config(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best validation steering config by pairwise accuracy and margin."""

    if not candidates:
        raise ValueError("No candidate configs were evaluated")
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["summary"]["pairwise_acc"]),
            -float(row["summary"]["mean_margin"]),
            abs(float(row["alpha"])),
            int(row["top_k"]),
            float(row["sign"]),
        ),
    )[0]


def evaluate_internal_steering(args: argparse.Namespace) -> dict[str, Any]:
    """Run validation tuning and test evaluation for internal steering."""

    out_dir = resolve_project_path(args.out_dir)
    ensure_output_dir(out_dir, bool(args.overwrite))
    val_pairs = read_jsonl(resolve_project_path(args.val_pairs))
    test_pairs = read_jsonl(resolve_project_path(args.test_pairs))
    validate_pair_rows(val_pairs)
    validate_pair_rows(test_pairs)
    val_by_subtype = group_pairs_by_subtype(val_pairs, args.max_val_samples_per_subtype, args.seed)
    test_by_subtype = group_pairs_by_subtype(test_pairs, args.max_test_samples_per_subtype, args.seed)

    prototype_path = resolve_optional_project_path(args.prototype_path)
    head_ranking_path = resolve_optional_project_path(args.head_ranking)
    if args.adapter == "llava" and (prototype_path is None or head_ranking_path is None):
        raise ValueError("--prototype-path and --head-ranking are required when --adapter llava")
    prototypes = load_prototypes(prototype_path)
    if head_ranking_path is not None:
        head_ranking = read_json(head_ranking_path)
    else:
        head_ranking = synthetic_head_ranking(sorted(set(val_by_subtype) | set(test_by_subtype)))

    if args.adapter == "mock":
        scorer = MockCandidateScorer()
    else:
        image_root = resolve_optional_project_path(args.image_root)
        if image_root is None:
            raise ValueError("--image-root is required when --adapter llava")
        scorer = LlavaCandidateScorer(
            model_id=str(args.model_id),
            image_root=image_root,
            instances_json=resolve_optional_project_path(args.instances_json),
            device=str(args.device),
            compute_dtype=str(args.compute_dtype),
        )

    val_tuning: dict[str, Any] = {}
    steering_config: dict[str, Any] = {}
    val_raw_rows: list[dict[str, Any]] = []
    test_raw_rows: list[dict[str, Any]] = []
    test_eval: dict[str, Any] = {}

    for subtype, rows in sorted(val_by_subtype.items()):
        baseline_rows = score_rows(
            scorer,
            rows,
            subtype=subtype,
            alpha=0.0,
            sign=1.0,
            top_k=max(args.topk),
            head_ranking=head_ranking,
            prototypes=prototypes,
            progress_label=f"val/{subtype}/baseline",
            progress_every=int(args.progress_every),
        )
        baseline_summary = summarize_candidate_scores(baseline_rows)
        candidates: list[dict[str, Any]] = []
        for top_k in args.topk:
            for alpha in args.alphas:
                for sign in args.signs:
                    steered_rows = score_rows(
                        scorer,
                        rows,
                        subtype=subtype,
                        alpha=float(alpha),
                        sign=float(sign),
                        top_k=int(top_k),
                        head_ranking=head_ranking,
                        prototypes=prototypes,
                        progress_label=f"val/{subtype}/k{top_k}/a{alpha}/s{sign}",
                        progress_every=int(args.progress_every),
                    )
                    summary = summarize_candidate_scores(steered_rows)
                    candidates.append(
                        {
                            "top_k": int(top_k),
                            "alpha": float(alpha),
                            "sign": float(sign),
                            "summary": summary,
                        }
                    )
                    val_raw_rows.extend(steered_rows)
        best = choose_best_config(candidates)
        val_tuning[subtype] = {
            "baseline": baseline_summary,
            "candidates": candidates,
            "best": best,
        }
        steering_config[subtype] = {
            "top_k": int(best["top_k"]),
            "alpha": float(best["alpha"]),
            "sign": float(best["sign"]),
        }

    for subtype, rows in sorted(test_by_subtype.items()):
        baseline_rows = score_rows(
            scorer,
            rows,
            subtype=subtype,
            alpha=0.0,
            sign=1.0,
            top_k=max(args.topk),
            head_ranking=head_ranking,
            prototypes=prototypes,
            progress_label=f"test/{subtype}/baseline",
            progress_every=int(args.progress_every),
        )
        best_cfg = steering_config[subtype]
        steered_rows = score_rows(
            scorer,
            rows,
            subtype=subtype,
            alpha=float(best_cfg["alpha"]),
            sign=float(best_cfg["sign"]),
            top_k=int(best_cfg["top_k"]),
            head_ranking=head_ranking,
            prototypes=prototypes,
            progress_label=f"test/{subtype}/steered",
            progress_every=int(args.progress_every),
        )
        baseline_summary = summarize_candidate_scores(baseline_rows)
        steered_summary = summarize_candidate_scores(steered_rows)
        test_eval[subtype] = {
            "baseline": baseline_summary,
            "steered": steered_summary,
            "delta_pairwise_acc": steered_summary["pairwise_acc"] - baseline_summary["pairwise_acc"],
            "delta_mean_margin": steered_summary["mean_margin"] - baseline_summary["mean_margin"],
            "config": best_cfg,
        }
        test_raw_rows.extend([{**row, "mode": "baseline"} for row in baseline_rows])
        test_raw_rows.extend([{**row, "mode": "steered"} for row in steered_rows])

    write_json(out_dir / "steering_config.json", steering_config)
    write_json(out_dir / "val_tuning.json", val_tuning)
    write_json(out_dir / "test_eval.json", test_eval)
    write_jsonl(out_dir / "val_raw_scores.jsonl", val_raw_rows)
    write_jsonl(out_dir / "test_raw_scores.jsonl", test_raw_rows)
    return {
        "out_dir": out_dir,
        "steering_config": steering_config,
        "val_tuning": val_tuning,
        "test_eval": test_eval,
    }


def main() -> int:
    """Run internal steering test from CLI."""

    args = parse_args()
    try:
        result = evaluate_internal_steering(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote internal steering test results to {result['out_dir']}")
    for subtype, metrics in result["test_eval"].items():
        print(
            f"test/{subtype}: baseline={metrics['baseline']['pairwise_acc']:.4f}, "
            f"steered={metrics['steered']['pairwise_acc']:.4f}, "
            f"delta={metrics['delta_pairwise_acc']:.4f}, config={metrics['config']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
