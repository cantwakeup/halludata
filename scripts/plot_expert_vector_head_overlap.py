"""Plot expert-vector cosine and Top-K head overlap in one figure."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-path", required=True, help="Torch vector payload with a `vectors` dict.")
    parser.add_argument("--experts", default="cat,attr,rel", help="Comma-separated vector keys to compare.")
    parser.add_argument("--topk", type=int, default=64, help="Top-K heads per expert by vector norm.")
    parser.add_argument(
        "--output-dir",
        default="data/outputs_after_template_disjoint_v2/head_overlap",
        help="Directory for PNG/JSON/Markdown outputs.",
    )
    parser.add_argument("--title", default="Expert Vector Cosine and Top-K Head Overlap")
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
        raise RuntimeError("plot_expert_vector_head_overlap requires torch.") from exc


def require_matplotlib() -> Any:
    """Import matplotlib lazily."""

    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        raise RuntimeError("plot_expert_vector_head_overlap requires matplotlib.") from exc


def load_torch(path: Path) -> dict[str, Any]:
    """Load a vector payload."""

    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return payload


def tensor_for_expert(payload: dict[str, Any], expert: str) -> Any:
    """Return one expert tensor as CPU float."""

    vectors = payload.get("vectors", {})
    if not isinstance(vectors, dict) or expert not in vectors:
        available = sorted(vectors) if isinstance(vectors, dict) else []
        raise KeyError(f"Expert '{expert}' not found. Available keys: {available}")
    tensor = vectors[expert]
    if not hasattr(tensor, "detach"):
        torch = require_torch()
        tensor = torch.tensor(tensor)
    return tensor.detach().cpu().float()


def cosine_flat(a: Any, b: Any) -> float:
    """Compute cosine between two flattened tensors."""

    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = float(a_flat.norm().item() * b_flat.norm().item())
    if denom == 0.0:
        return 0.0
    return float((a_flat * b_flat).sum().item() / denom)


def score_heads(vector: Any, layers: list[int]) -> list[dict[str, float | int]]:
    """Score every layer/head by L2 norm."""

    rows: list[dict[str, float | int]] = []
    norms = vector.float().norm(dim=-1)
    for layer_index, layer in enumerate(layers):
        for head in range(int(norms.shape[1])):
            rows.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "score": float(norms[layer_index, head].item()),
                }
            )
    rows.sort(key=lambda row: (-float(row["score"]), int(row["layer"]), int(row["head"])))
    return rows


def head_set(rows: list[dict[str, float | int]], topk: int) -> set[tuple[int, int]]:
    """Convert Top-K scored heads to a set."""

    return {(int(row["layer"]), int(row["head"])) for row in rows[:topk]}


def matrix_to_dict(experts: list[str], matrix: list[list[float]]) -> dict[str, dict[str, float]]:
    """Convert a square matrix to a nested dict."""

    return {
        left: {right: float(matrix[left_index][right_index]) for right_index, right in enumerate(experts)}
        for left_index, left in enumerate(experts)
    }


def build_overlap(experts: list[str], head_sets: dict[str, set[tuple[int, int]]]) -> tuple[list[list[int]], list[list[float]]]:
    """Build intersection and Jaccard matrices."""

    intersections: list[list[int]] = []
    jaccards: list[list[float]] = []
    for left in experts:
        intersection_row = []
        jaccard_row = []
        for right in experts:
            intersection = len(head_sets[left] & head_sets[right])
            union = len(head_sets[left] | head_sets[right])
            intersection_row.append(intersection)
            jaccard_row.append((intersection / union) if union else 0.0)
        intersections.append(intersection_row)
        jaccards.append(jaccard_row)
    return intersections, jaccards


def shared_head_categories(experts: list[str], head_sets: dict[str, set[tuple[int, int]]]) -> dict[str, list[tuple[int, int]]]:
    """Group heads by ownership pattern."""

    all_heads = sorted(set().union(*head_sets.values()))
    categories: dict[str, list[tuple[int, int]]] = {}
    for head in all_heads:
        owners = tuple(expert for expert in experts if head in head_sets[expert])
        label = "+".join(owners) if len(owners) > 1 else f"{owners[0]} only"
        categories.setdefault(label, []).append(head)
    return categories


def plot_matrix(ax: Any, values: list[list[float]], experts: list[str], title: str, cbar_label: str) -> None:
    """Plot a labeled matrix with numeric cell annotations."""

    image = ax.imshow(values, vmin=-1.0 if "Cosine" in title else 0.0, vmax=1.0)
    ax.set_xticks(range(len(experts)), experts, rotation=35, ha="right")
    ax.set_yticks(range(len(experts)), experts)
    ax.set_title(title)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.55 else "black")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def plot_head_scatter(ax: Any, categories: dict[str, list[tuple[int, int]]], topk: int) -> None:
    """Plot selected heads by layer/head and ownership category."""

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
    ]
    for index, (label, heads) in enumerate(sorted(categories.items())):
        if not heads:
            continue
        ax.scatter(
            [head for _layer, head in heads],
            [layer for layer, _head in heads],
            s=34,
            alpha=0.86,
            label=f"{label} ({len(heads)})",
            color=colors[index % len(colors)],
        )
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"Top-{topk} head ownership")
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))


def plot_layer_distribution(ax: Any, experts: list[str], head_sets: dict[str, set[tuple[int, int]]]) -> None:
    """Plot layer distribution of selected heads."""

    layers = sorted({layer for heads in head_sets.values() for layer, _head in heads})
    width = 0.8 / max(len(experts), 1)
    for expert_index, expert in enumerate(experts):
        counts = Counter(layer for layer, _head in head_sets[expert])
        xs = [index + expert_index * width for index in range(len(layers))]
        ys = [counts.get(layer, 0) for layer in layers]
        ax.bar(xs, ys, width=width, label=expert)
    ax.set_xticks([index + width * (len(experts) - 1) / 2 for index in range(len(layers))], layers, rotation=90)
    ax.set_xlabel("layer")
    ax.set_ylabel("Top-K head count")
    ax.set_title("Layer distribution")
    ax.legend(fontsize=8)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with UTF-8 formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_report(
    output_dir: Path,
    vector_path: Path,
    experts: list[str],
    topk: int,
    cosine_matrix: list[list[float]],
    intersection_matrix: list[list[int]],
    jaccard_matrix: list[list[float]],
    layer_summary: dict[str, dict[str, int]],
) -> None:
    """Write a short Markdown report."""

    def table(headers: list[str], rows: list[list[Any]]) -> str:
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        for row in rows:
            pieces = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
            lines.append("| " + " | ".join(pieces) + " |")
        return "\n".join(lines)

    pair_rows = []
    for i, left in enumerate(experts):
        for j, right in enumerate(experts):
            if i >= j:
                continue
            pair_rows.append([left, right, cosine_matrix[i][j], intersection_matrix[i][j], jaccard_matrix[i][j]])
    layer_rows = [
        [expert, ", ".join(f"{layer}:{count}" for layer, count in summary.items())]
        for expert, summary in layer_summary.items()
    ]
    text = "\n".join(
        [
            "# Expert Vector And Head Overlap Summary",
            "",
            f"- Vector path: `{vector_path}`",
            f"- Experts: `{', '.join(experts)}`",
            f"- Top-K: `{topk}`",
            f"- Figure: `expert_vector_head_overlap_top{topk}.png`",
            "",
            "## Pairwise Summary",
            "",
            table(["expert_a", "expert_b", "vector_cosine", "topk_intersection", "topk_jaccard"], pair_rows),
            "",
            "## Top Layers",
            "",
            table(["expert", "layer:count"], layer_rows),
        ]
    )
    (output_dir / "REPORT.md").write_text(text + "\n", encoding="utf-8")


def main() -> int:
    """Create combined cosine/head-overlap diagnostics."""

    args = parse_args()
    try:
        plt = require_matplotlib()
        vector_path = resolve_project_path(args.vector_path)
        output_dir = resolve_project_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = load_torch(vector_path)
        experts = parse_csv_items(args.experts)
        if len(experts) < 2:
            raise ValueError("--experts must include at least two vector keys")
        layers = [int(layer) for layer in payload.get("layers", [])]

        vectors = {expert: tensor_for_expert(payload, expert) for expert in experts}
        if not layers:
            first = next(iter(vectors.values()))
            layers = list(range(int(first.shape[0])))
        scored = {expert: score_heads(vector, layers) for expert, vector in vectors.items()}
        head_sets = {expert: head_set(rows, args.topk) for expert, rows in scored.items()}

        cosine_matrix = [[cosine_flat(vectors[left], vectors[right]) for right in experts] for left in experts]
        intersection_matrix, jaccard_matrix = build_overlap(experts, head_sets)
        categories = shared_head_categories(experts, head_sets)
        layer_summary = {
            expert: dict(Counter(layer for layer, _head in head_sets[expert]).most_common(8))
            for expert in experts
        }

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(args.title, fontsize=16)
        plot_matrix(axes[0][0], cosine_matrix, experts, "Vector Cosine Similarity", "cosine")
        plot_matrix(axes[0][1], jaccard_matrix, experts, f"Top-{args.topk} Head Jaccard", "Jaccard")
        plot_head_scatter(axes[1][0], categories, args.topk)
        plot_layer_distribution(axes[1][1], experts, head_sets)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        png_path = output_dir / f"expert_vector_head_overlap_top{args.topk}.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)

        summary = {
            "vector_path": str(vector_path),
            "experts": experts,
            "topk": int(args.topk),
            "layers": layers,
            "cosine_matrix": matrix_to_dict(experts, cosine_matrix),
            "topk_intersection_matrix": {
                left: {right: int(intersection_matrix[i][j]) for j, right in enumerate(experts)}
                for i, left in enumerate(experts)
            },
            "topk_jaccard_matrix": matrix_to_dict(experts, jaccard_matrix),
            "head_ownership_counts": {label: len(heads) for label, heads in categories.items()},
            "top_layers": layer_summary,
            "figure": str(png_path),
        }
        write_json(output_dir / f"expert_vector_head_overlap_top{args.topk}.json", summary)
        write_report(output_dir, vector_path, experts, args.topk, cosine_matrix, intersection_matrix, jaccard_matrix, layer_summary)
        print(f"Wrote combined expert overlap figure to {png_path}")
        print(f"Wrote summary JSON to {output_dir / f'expert_vector_head_overlap_top{args.topk}.json'}")
        print(f"Wrote report to {output_dir / 'REPORT.md'}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
