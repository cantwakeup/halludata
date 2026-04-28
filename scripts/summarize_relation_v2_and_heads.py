"""Summarize relation-v2 data, expert head mining, and steering sweeps."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-pairs-stats", default="data/after_template_rel_v2/pairs/stats.json")
    parser.add_argument("--relation-vector-stats", default="data/outputs_after_template_rel_v2/steering/relation_v2_vectors.stats.json")
    parser.add_argument("--head-analysis-dir", default="data/outputs_after_template_v1/head_analysis")
    parser.add_argument("--mme-position-runs", default="data/outputs_after_template_rel_v2/runs/mme_position")
    parser.add_argument("--sanity-runs", default="data/outputs_after_template_v1/runs/expert_head_sanity")
    parser.add_argument("--output", default="data/outputs_after_template_rel_v2/RELATION_AND_HEAD_ANALYSIS_REPORT.md")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON or return an empty dict if absent."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def nested(payload: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Read nested dict keys."""

    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def fmt(value: Any) -> str:
    """Format markdown values."""

    if value is None:
        return ""
    try:
        if value != "":
            return f"{float(value):.4f}"
    except (TypeError, ValueError):
        pass
    return str(value)


def table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    """Render markdown table rows."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return lines


def alpha_from_name(name: str) -> str:
    """Extract alpha from a run name."""

    match = re.search(r"alpha([-+]?\d+(?:\.\d+)?)$", name)
    return match.group(1) if match else ""


def collect_run_rows(root: Path) -> list[dict[str, Any]]:
    """Collect metrics.json rows recursively."""

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("**/metrics.json")):
        run_name = metrics_path.parent.name
        metrics = read_json(metrics_path)
        fixed = metrics.get("fixed_steering", {})
        if not isinstance(fixed, dict):
            fixed = {}
        config = read_json(metrics_path.parent / "config.json")
        steering = config.get("steering", {}) if isinstance(config.get("steering"), dict) else {}
        baseline_acc = metrics.get("accuracy_baseline", nested(metrics, "baseline", "accuracy"))
        steered_acc = metrics.get("accuracy_steered", nested(metrics, "steered", "accuracy", default=baseline_acc))
        rows.append(
            {
                "run": str(metrics_path.parent.relative_to(root)),
                "run_name": run_name,
                "expert_key": steering.get("expert_key", ""),
                "head_select": steering.get("head_select", ""),
                "alpha": fixed.get("alpha", alpha_from_name(run_name)),
                "accuracy_baseline": baseline_acc,
                "accuracy_steered": steered_acc,
                "delta_accuracy": metrics.get("delta_accuracy", fixed.get("delta_accuracy", "")),
                "f1_steered": metrics.get("f1_yes", nested(metrics, "steered", "f1_yes")),
                "yes_rate_baseline": metrics.get("yes_rate_baseline", nested(metrics, "baseline", "yes_rate")),
                "yes_rate_steered": metrics.get("yes_rate_steered", nested(metrics, "steered", "yes_rate")),
                "wrong_to_right": metrics.get("wrong_to_right", fixed.get("wrong_to_right", "")),
                "right_to_wrong": metrics.get("right_to_wrong", fixed.get("right_to_wrong", "")),
                "avg_delta_margin_label_yes": metrics.get("avg_delta_margin_label_yes", fixed.get("avg_delta_margin_label_yes", "")),
                "avg_delta_margin_label_no": metrics.get("avg_delta_margin_label_no", fixed.get("avg_delta_margin_label_no", "")),
                "changed_pred": metrics.get("changed_pred", fixed.get("changed_pred", "")),
            }
        )
    return sorted(rows, key=lambda row: float(row.get("delta_accuracy") or -999.0), reverse=True)


def best_rows(rows: list[dict[str, Any]], n: int = 12) -> list[dict[str, Any]]:
    """Return best non-baseline rows."""

    return [row for row in rows if "baseline" not in str(row.get("run_name", ""))][:n]


def dict_table(payload: dict[str, Any], key_name: str) -> list[str]:
    """Render a flat dict as a table."""

    lines = [f"| {key_name} | value |", "| --- | ---: |"]
    for key, value in payload.items():
        lines.append(f"| {key} | {fmt(value)} |")
    return lines


def head_overlap_rows(head_analysis_dir: Path) -> list[dict[str, Any]]:
    """Read top64 overlap CSV-ish JSON into compact rows."""

    overlap = read_json(head_analysis_dir / "overlap_top64.json")
    rows: list[dict[str, Any]] = []
    for left, values in overlap.items():
        if not isinstance(values, dict):
            continue
        for right, cell in values.items():
            if left >= right or not isinstance(cell, dict):
                continue
            rows.append(
                {
                    "expert_a": left,
                    "expert_b": right,
                    "intersection": cell.get("intersection", ""),
                    "jaccard": cell.get("jaccard", ""),
                }
            )
    return rows


def build_conclusion(position_rows: list[dict[str, Any]], sanity_rows: list[dict[str, Any]], overlap_rows: list[dict[str, Any]]) -> list[str]:
    """Build automatic conclusions."""

    lines: list[str] = []
    best_position = best_rows(position_rows, 1)
    if best_position:
        delta = float(best_position[0].get("delta_accuracy") or 0.0)
        if delta > 0.0:
            lines.append(f"- Best relation-v2 MME position run improves accuracy by {delta:.4f}; relation template/head changes are helping.")
        else:
            yes_margin = float(best_position[0].get("avg_delta_margin_label_yes") or 0.0)
            no_margin = float(best_position[0].get("avg_delta_margin_label_no") or 0.0)
            if abs(yes_margin) > 0.05 or abs(no_margin) > 0.05:
                lines.append("- Relation-v2 moves margins but does not improve accuracy yet; try gating/adaptive alpha before declaring failure.")
            else:
                lines.append("- Relation-v2 still shows little useful movement; inspect spatial data quality or consider a visual perception vector.")
    if sanity_rows:
        best_sanity = best_rows(sanity_rows, 1)
        if best_sanity:
            lines.append(f"- Best cat/attr expert-head sanity run: `{best_sanity[0]['run']}` delta={fmt(best_sanity[0].get('delta_accuracy'))}.")
    if overlap_rows:
        high = [row for row in overlap_rows if float(row.get("jaccard") or 0.0) > 0.5 and row["expert_a"] != row["expert_b"]]
        if high:
            lines.append("- Some expert head maps overlap strongly; typed-head evidence should be framed cautiously.")
        else:
            lines.append("- Top64 head maps are not highly overlapping overall; this supports a typed-expert-head hypothesis.")
    return lines or ["- Not enough completed artifacts to draw a conclusion yet."]


def write_report(
    output: Path,
    pair_stats: dict[str, Any],
    vector_stats: dict[str, Any],
    head_dir: Path,
    position_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
) -> None:
    """Write the combined relation/head analysis report."""

    overlap_rows = head_overlap_rows(head_dir)
    metric_headers = [
        "run",
        "expert_key",
        "head_select",
        "alpha",
        "accuracy_baseline",
        "accuracy_steered",
        "delta_accuracy",
        "f1_steered",
        "yes_rate_baseline",
        "yes_rate_steered",
        "wrong_to_right",
        "right_to_wrong",
        "avg_delta_margin_label_yes",
        "avg_delta_margin_label_no",
        "changed_pred",
    ]
    lines: list[str] = [
        "# Relation V2 And Typed Head Analysis Report",
        "",
        "## Why Relation V2",
        "",
        "- The previous relation data used open-ended `Where is A relative to B?` templates, while MME position is yes/no.",
        "- Relation v2 uses MME-style yes/no relation questions and trusted factual text with inverse spatial facts.",
        "- It also uses stricter bbox filtering to reduce ambiguous or overlapping object pairs.",
        "",
        "## Relation V2 Data Statistics",
        "",
        f"- Total pairs: {pair_stats.get('total_pairs', 'TBD')}",
        f"- Train/val/test: {pair_stats.get('train_pairs', 'TBD')} / {pair_stats.get('val_pairs', 'TBD')} / {pair_stats.get('test_pairs', 'TBD')}",
        f"- Images: {pair_stats.get('num_images', 'TBD')}",
        "",
        *dict_table(pair_stats.get("true_relation_counts", {}), "true_relation"),
        "",
        *dict_table(pair_stats.get("label_counts", {}), "label"),
        "",
        "### Skipped Reasons",
        "",
        *dict_table(pair_stats.get("skipped", {}), "reason"),
        "",
        "## Relation Vector Statistics",
        "",
        "### Norms",
        "",
        "| vector | mean_norm | max_norm | min_norm |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, norms in vector_stats.get("vector_norms", {}).items():
        if not isinstance(norms, dict):
            continue
        lines.append(f"| {key} | {fmt(norms.get('mean'))} | {fmt(norms.get('max'))} | {fmt(norms.get('min'))} |")
    lines.extend(["", "### Cosines", "", *dict_table(vector_stats.get("cosine_diagnostics", {}), "cosine")])
    lines.extend(
        [
            "",
            "## Expert Head Mining",
            "",
            f"- Head-analysis directory: `{head_dir}`",
            "- Main plots: `heatmap_*.png`, `topk_overlap_matrix.png`, `layer_distribution_top64.png`, `expert_head_assignment_top64.png`.",
            "",
            "### Top64 Overlap",
            "",
            *(table(overlap_rows, ["expert_a", "expert_b", "intersection", "jaccard"]) if overlap_rows else ["No head-overlap artifacts found yet."]),
            "",
            "## MME Position Sweep",
            "",
            *(table(best_rows(position_rows, 20), metric_headers) if position_rows else ["No MME position runs found yet."]),
            "",
            "## Cat/Attr Expert-Head Sanity",
            "",
            *(table(best_rows(sanity_rows, 20), metric_headers) if sanity_rows else ["No cat/attr sanity runs found yet."]),
            "",
            "## Automatic Conclusion",
            "",
            *build_conclusion(position_rows, sanity_rows, overlap_rows),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Generate relation/head analysis report."""

    args = parse_args()
    try:
        pair_stats = read_json(resolve_project_path(args.relation_pairs_stats))
        vector_stats = read_json(resolve_project_path(args.relation_vector_stats))
        head_dir = resolve_project_path(args.head_analysis_dir)
        position_rows = collect_run_rows(resolve_project_path(args.mme_position_runs))
        sanity_rows = collect_run_rows(resolve_project_path(args.sanity_runs))
        write_report(resolve_project_path(args.output), pair_stats, vector_stats, head_dir, position_rows, sanity_rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote relation/head analysis report to {resolve_project_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
