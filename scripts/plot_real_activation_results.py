"""Plot real activation signal audit outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for plotting activation results."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, help="Prototype signal experiment directory.")
    parser.add_argument("--out-dir", required=True, help="Plot output directory.")
    parser.add_argument("--score-type", default="two_proto", help="Score type to plot from evaluation JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty out-dir.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Create or validate a plot output directory."""

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite to replace plots.")
    out_dir.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _import_matplotlib() -> Any:
    """Import matplotlib lazily and raise a clear error if it is unavailable."""

    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install it in the analysis environment.") from exc


def plot_score_histograms(experiment_dir: Path, out_dir: Path, score_type: str) -> None:
    """Plot positive/negative score histograms for each subtype."""

    plt = _import_matplotlib()
    payload = read_json(experiment_dir / "all_head_eval_test.json")
    score_payload = payload[str(score_type)]
    for subtype, metrics in score_payload["by_subtype"].items():
        plt.figure()
        plt.hist(metrics["score_pos"], bins=30, alpha=0.6, label="response_pos")
        plt.hist(metrics["score_neg"], bins=30, alpha=0.6, label="response_neg")
        plt.xlabel("prototype score")
        plt.ylabel("count")
        plt.title(f"{subtype} score distribution ({score_type})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"score_hist_{score_type}_{subtype}.png")
        plt.close()


def plot_head_heatmaps(experiment_dir: Path, out_dir: Path) -> None:
    """Plot per-subtype 32x32 head-score heatmaps when head-ranking output exists."""

    plt = _import_matplotlib()
    head_path = experiment_dir / "head_ranking" / "head_ranking.json"
    if not head_path.exists():
        return
    payload = read_json(head_path)
    for subtype, rows in payload.items():
        max_layer = max(int(row["layer"]) for row in rows)
        max_head = max(int(row["head"]) for row in rows)
        heatmap = [[0.0 for _ in range(max_head + 1)] for _ in range(max_layer + 1)]
        for row in rows:
            heatmap[int(row["layer"])][int(row["head"])] = float(row["score"])
        plt.figure()
        plt.imshow(heatmap, aspect="auto")
        plt.colorbar(label="head score")
        plt.xlabel("head")
        plt.ylabel("layer")
        plt.title(f"{subtype} head ranking heatmap")
        plt.tight_layout()
        plt.savefig(out_dir / f"head_heatmap_{subtype}.png")
        plt.close()


def plot_topk_overlap(experiment_dir: Path, out_dir: Path) -> None:
    """Plot Top-K subtype Jaccard overlap if the matrix exists."""

    plt = _import_matplotlib()
    overlap_path = experiment_dir / "head_ranking" / "topk_overlap_matrix.json"
    if not overlap_path.exists():
        return
    payload = read_json(overlap_path)
    subtypes = list(payload["subtypes"])
    matrix = payload["jaccard"]
    plt.figure()
    plt.imshow(matrix, vmin=0.0, vmax=1.0)
    plt.colorbar(label="Jaccard overlap")
    plt.xticks(range(len(subtypes)), subtypes)
    plt.yticks(range(len(subtypes)), subtypes)
    plt.xlabel("subtype")
    plt.ylabel("subtype")
    plt.title(f"Top-{payload['top_k']} head overlap")
    plt.tight_layout()
    plt.savefig(out_dir / "topk_overlap_matrix.png")
    plt.close()


def main() -> int:
    """Run plotting from saved experiment JSON files."""

    args = parse_args()
    experiment_dir = resolve_project_path(args.experiment_dir)
    out_dir = resolve_project_path(args.out_dir)
    try:
        ensure_output_dir(out_dir, bool(args.overwrite))
        plot_score_histograms(experiment_dir, out_dir, str(args.score_type))
        plot_head_heatmaps(experiment_dir, out_dir)
        plot_topk_overlap(experiment_dir, out_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote activation result plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
