"""Mine typed expert heads from steering vector norms and visualize overlap."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-paths", nargs="+", required=True, help="Named vector paths, e.g. cat_attr=... rel=...")
    parser.add_argument("--experts", required=True, help="Comma-separated vector keys to mine.")
    parser.add_argument("--output-dir", default="data/outputs_after_template_v1/head_analysis")
    parser.add_argument("--topk", default="16,32,64,128", help="Comma-separated Top-K sizes.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_torch() -> Any:
    """Import torch lazily."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("mine_expert_heads requires torch.") from exc


def load_torch(path: Path) -> dict[str, Any]:
    """Load a torch vector payload."""

    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return payload


def parse_vector_paths(items: list[str]) -> dict[str, Path]:
    """Parse key=value vector path specifications."""

    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--vector-paths entries must be key=path, got: {item}")
        key, value = item.split("=", 1)
        result[key.strip()] = resolve_project_path(value.strip())
    return result


def parse_csv_items(text: str) -> list[str]:
    """Parse comma-separated non-empty items."""

    return [piece.strip() for piece in str(text).split(",") if piece.strip()]


def load_vectors(vector_paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Load all vector payloads."""

    return {name: load_torch(path) for name, path in vector_paths.items()}


def find_expert_vector(payloads: dict[str, dict[str, Any]], expert: str) -> tuple[Any, list[int], str]:
    """Find an expert vector in the loaded payloads."""

    for source_name, payload in payloads.items():
        vectors = payload.get("vectors", {})
        if isinstance(vectors, dict) and expert in vectors:
            tensor = vectors[expert].detach().cpu().float()
            layers = [int(layer) for layer in payload.get("layers", list(range(int(tensor.shape[0]))))]
            return tensor, layers, source_name
    raise KeyError(f"Expert vector '{expert}' not found in any vector payload")


def payload_has_expert(payloads: dict[str, dict[str, Any]], expert: str) -> bool:
    """Return whether any loaded vector payload exposes one expert key."""

    for payload in payloads.values():
        vectors = payload.get("vectors", {})
        if isinstance(vectors, dict) and expert in vectors:
            return True
    return False


def score_heads(vector: Any, layers: list[int]) -> list[dict[str, Any]]:
    """Score every layer/head by vector L2 norm."""

    rows: list[dict[str, Any]] = []
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


def head_set(rows: list[list[float | int]], k: int | None = None) -> set[tuple[int, int]]:
    """Convert head-map rows to a set."""

    selected = rows if k is None else rows[:k]
    return {(int(row[0]), int(row[1])) for row in selected}


def write_json(path: Path, payload: Any) -> None:
    """Write JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_overlap(output_dir: Path, head_maps: dict[str, list[list[float | int]]], top_k: int) -> dict[str, Any]:
    """Write overlap JSON/CSV for one Top-K size."""

    experts = list(head_maps)
    sets = {expert: head_set(rows) for expert, rows in head_maps.items()}
    matrix: dict[str, dict[str, dict[str, float | int]]] = {}
    for left in experts:
        matrix[left] = {}
        for right in experts:
            intersection = len(sets[left] & sets[right])
            union = len(sets[left] | sets[right])
            matrix[left][right] = {
                "intersection": intersection,
                "jaccard": (intersection / union) if union else 0.0,
            }
    write_json(output_dir / f"overlap_top{top_k}.json", matrix)
    with (output_dir / f"overlap_top{top_k}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expert_a", "expert_b", "intersection", "jaccard"])
        for left in experts:
            for right in experts:
                values = matrix[left][right]
                writer.writerow([left, right, values["intersection"], values["jaccard"]])
    return matrix


def try_import_matplotlib() -> Any | None:
    """Import matplotlib if available."""

    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_heatmap(plt: Any, output_dir: Path, expert: str, rows: list[dict[str, Any]]) -> None:
    """Plot a layer/head score heatmap."""

    max_layer = max(int(row["layer"]) for row in rows)
    max_head = max(int(row["head"]) for row in rows)
    grid = [[0.0 for _ in range(max_head + 1)] for _ in range(max_layer + 1)]
    for row in rows:
        grid[int(row["layer"])][int(row["head"])] = float(row["score"])
    plt.figure(figsize=(10, 6))
    plt.imshow(grid, aspect="auto", interpolation="nearest")
    plt.colorbar(label="||vector[layer, head]||2")
    plt.xlabel("head")
    plt.ylabel("layer")
    plt.title(f"{expert} head norm heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_{expert}.png", dpi=180)
    plt.close()


def plot_overlap_matrix(plt: Any, output_dir: Path, overlap: dict[str, Any]) -> None:
    """Plot a Jaccard overlap matrix."""

    experts = list(overlap)
    values = [[float(overlap[left][right]["jaccard"]) for right in experts] for left in experts]
    plt.figure(figsize=(8, 7))
    plt.imshow(values, vmin=0.0, vmax=1.0)
    plt.colorbar(label="Jaccard")
    plt.xticks(range(len(experts)), experts, rotation=45, ha="right")
    plt.yticks(range(len(experts)), experts)
    plt.title("Top-K expert head overlap")
    plt.tight_layout()
    plt.savefig(output_dir / "topk_overlap_matrix.png", dpi=180)
    plt.close()


def plot_layer_distribution(plt: Any, output_dir: Path, head_maps: dict[str, list[list[float | int]]]) -> None:
    """Plot layer counts for Top-K heads."""

    experts = list(head_maps)
    layers = sorted({int(row[0]) for rows in head_maps.values() for row in rows})
    width = 0.8 / max(len(experts), 1)
    plt.figure(figsize=(12, 5))
    for expert_index, expert in enumerate(experts):
        counts = Counter(int(row[0]) for row in head_maps[expert])
        xs = [index + expert_index * width for index in range(len(layers))]
        ys = [counts.get(layer, 0) for layer in layers]
        plt.bar(xs, ys, width=width, label=expert)
    plt.xticks([index + width * (len(experts) - 1) / 2 for index in range(len(layers))], layers, rotation=90)
    plt.xlabel("layer")
    plt.ylabel("Top-K head count")
    plt.title("Layer distribution of Top-K expert heads")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "layer_distribution_top64.png", dpi=180)
    plt.close()


def plot_assignment(plt: Any, output_dir: Path, head_maps: dict[str, list[list[float | int]]]) -> None:
    """Plot cat/attr/rel-only vs shared head assignment."""

    groups = {
        "cat": head_set(head_maps.get("cat", [])),
        "attr": head_set(head_maps.get("attr", [])),
        "rel": head_set(head_maps.get("rel_all", head_maps.get("rel", []))),
    }
    all_heads = sorted(set().union(*groups.values())) if groups else []
    colors = {
        "cat-only": "tab:blue",
        "attr-only": "tab:orange",
        "rel-only": "tab:green",
        "shared": "tab:red",
    }
    buckets: dict[str, list[tuple[int, int]]] = {key: [] for key in colors}
    for head in all_heads:
        owners = [name for name, values in groups.items() if head in values]
        if len(owners) > 1:
            buckets["shared"].append(head)
        elif owners:
            buckets[f"{owners[0]}-only"].append(head)
    plt.figure(figsize=(10, 6))
    for label, heads in buckets.items():
        if not heads:
            continue
        plt.scatter([head for _layer, head in heads], [layer for layer, _head in heads], label=label, s=32, c=colors[label])
    plt.xlabel("head")
    plt.ylabel("layer")
    plt.title("Top64 typed expert head assignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "expert_head_assignment_top64.png", dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    experts: list[str],
    source_by_expert: dict[str, str],
    head_maps_by_k: dict[int, dict[str, list[list[float | int]]]],
    overlap_top64: dict[str, Any],
    warnings: list[str],
) -> None:
    """Write a markdown report."""

    top64 = head_maps_by_k.get(64) or head_maps_by_k[max(head_maps_by_k)]
    lines = [
        "# Typed Expert Head Mining Report",
        "",
        "Heads are scored by `||vector[layer, head]||_2` and selected globally across all available layers.",
        "",
        "## Sources",
        "",
    ]
    for expert in experts:
        lines.append(f"- `{expert}`: `{source_by_expert.get(expert, '')}`")
    lines.extend(["", "## Top64 Layer Distribution", ""])
    lines.append("| expert | top layers | num unique layers |")
    lines.append("| --- | --- | ---: |")
    for expert, rows in top64.items():
        counts = Counter(int(row[0]) for row in rows)
        top_layers = ", ".join(f"{layer}:{count}" for layer, count in counts.most_common(6))
        lines.append(f"| {expert} | {top_layers} | {len(counts)} |")
    lines.extend(["", "## Top64 Overlap Jaccard", ""])
    experts64 = list(top64)
    lines.append("| expert | " + " | ".join(experts64) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in experts64) + " |")
    for left in experts64:
        values = [float(overlap_top64[left][right]["jaccard"]) for right in experts64]
        lines.append("| " + left + " | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.extend(
        [
            "",
            "## Visualizations",
            "",
            "- `heatmap_<expert>.png`: norm score by layer/head.",
            "- `topk_overlap_matrix.png`: Top64 Jaccard overlap between experts.",
            "- `layer_distribution_top64.png`: where selected heads concentrate by layer.",
            "- `expert_head_assignment_top64.png`: cat-only / attr-only / rel-only / shared heads.",
            "",
            "## Automatic Reading",
            "",
        ]
    )
    cat_attr = overlap_top64.get("cat", {}).get("attr", {}).get("jaccard")
    cat_rel = overlap_top64.get("cat", {}).get("rel_all", {}).get("jaccard")
    attr_rel = overlap_top64.get("attr", {}).get("rel_all", {}).get("jaccard")
    for label, value in (("cat-attr", cat_attr), ("cat-rel_all", cat_rel), ("attr-rel_all", attr_rel)):
        if value is not None:
            lines.append(f"- `{label}` Top64 Jaccard = {float(value):.4f}.")
    if "rel_left" in overlap_top64 and "rel_right" in overlap_top64["rel_left"]:
        value = float(overlap_top64["rel_left"]["rel_right"]["jaccard"])
        lines.append(f"- `rel_left` / `rel_right` Top64 Jaccard = {value:.4f}; low values suggest direction-specific relation heads.")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Mine expert heads and write analysis artifacts."""

    args = parse_args()
    try:
        output_dir = resolve_project_path(args.output_dir)
        head_map_dir = output_dir / "head_maps"
        head_map_dir.mkdir(parents=True, exist_ok=True)
        vector_paths = parse_vector_paths(args.vector_paths)
        payloads = load_vectors(vector_paths)
        experts = parse_csv_items(args.experts)
        for extra_expert in ("rel_horizontal", "rel_vertical"):
            if extra_expert not in experts and payload_has_expert(payloads, extra_expert):
                experts.append(extra_expert)
        topks = [int(item) for item in parse_csv_items(args.topk)]
        if not topks:
            raise ValueError("--topk must contain at least one integer")

        scored_by_expert: dict[str, list[dict[str, Any]]] = {}
        source_by_expert: dict[str, str] = {}
        warnings: list[str] = []
        for expert in experts:
            vector, layers, source = find_expert_vector(payloads, expert)
            scored_by_expert[expert] = score_heads(vector, layers)
            source_by_expert[expert] = str(vector_paths[source])

        head_maps_by_k: dict[int, dict[str, list[list[float | int]]]] = {}
        for topk in topks:
            head_maps: dict[str, list[list[float | int]]] = {}
            for expert, rows in scored_by_expert.items():
                head_maps[expert] = [
                    [int(row["layer"]), int(row["head"]), float(row["score"])]
                    for row in rows[:topk]
                ]
            head_maps_by_k[topk] = head_maps
            write_json(head_map_dir / f"top{topk}.json", head_maps)

        overlap_top64 = write_overlap(output_dir, head_maps_by_k.get(64) or head_maps_by_k[max(topks)], 64 if 64 in head_maps_by_k else max(topks))
        plt = try_import_matplotlib()
        if plt is None:
            warnings.append("matplotlib is not available; PNG visualizations were skipped")
        else:
            for expert, rows in scored_by_expert.items():
                plot_heatmap(plt, output_dir, expert, rows)
            top64 = head_maps_by_k.get(64) or head_maps_by_k[max(topks)]
            plot_overlap_matrix(plt, output_dir, overlap_top64)
            plot_layer_distribution(plt, output_dir, top64)
            plot_assignment(plt, output_dir, top64)
        write_report(output_dir, experts, source_by_expert, head_maps_by_k, overlap_top64, warnings)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote expert head analysis to {resolve_project_path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
