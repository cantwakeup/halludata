"""Summarize AFTER-style v1 data, vectors, and POPE runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for report generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-stats", default="data/after_style_v1/pairs/stats.json")
    parser.add_argument("--vector-stats", default="data/outputs_after_style_v1/steering/after_style_expert_vectors.stats.json")
    parser.add_argument("--runs-root", default="data/outputs_after_style_v1/runs")
    parser.add_argument("--output", default="data/outputs_after_style_v1/REPORT.md")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object if the file is absent."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def fmt(value: Any) -> str:
    """Format values for markdown cells."""

    if value is None:
        return ""
    try:
        if value != "":
            return f"{float(value):.4f}"
    except (TypeError, ValueError):
        pass
    return str(value)


def nested(payload: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Read a nested dictionary field."""

    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def collect_run_rows(runs_root: Path) -> list[dict[str, Any]]:
    """Collect POPE run metrics."""

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("pope_random_after_style*/metrics.json")):
        metrics = read_json(metrics_path)
        run_name = metrics_path.parent.name
        fixed = metrics.get("fixed_steering", {})
        if not isinstance(fixed, dict):
            fixed = {}
        rows.append(
            {
                "run": run_name,
                "alpha": fixed.get("alpha", "" if "baseline" in run_name else run_name.rsplit("alpha", 1)[-1]),
                "accuracy_baseline": metrics.get("accuracy_baseline", nested(metrics, "baseline", "accuracy")),
                "accuracy_steered": metrics.get("accuracy_steered", nested(metrics, "steered", "accuracy", default=nested(metrics, "baseline", "accuracy"))),
                "delta_accuracy": metrics.get("delta_accuracy", fixed.get("delta_accuracy", "")),
                "precision_yes": metrics.get("precision_yes", nested(metrics, "steered", "precision_yes", default=nested(metrics, "baseline", "precision_yes"))),
                "recall_yes": metrics.get("recall_yes", nested(metrics, "steered", "recall_yes", default=nested(metrics, "baseline", "recall_yes"))),
                "f1_yes": metrics.get("f1_yes", nested(metrics, "steered", "f1_yes", default=nested(metrics, "baseline", "f1_yes"))),
                "yes_rate_baseline": metrics.get("yes_rate_baseline", nested(metrics, "baseline", "yes_rate")),
                "yes_rate_steered": metrics.get("yes_rate_steered", nested(metrics, "steered", "yes_rate", default=nested(metrics, "baseline", "yes_rate"))),
                "wrong_to_right": metrics.get("wrong_to_right", fixed.get("wrong_to_right", "")),
                "right_to_wrong": metrics.get("right_to_wrong", fixed.get("right_to_wrong", "")),
                "avg_delta_margin_all": metrics.get("avg_delta_margin_all", fixed.get("avg_delta_margin_all", "")),
                "avg_delta_margin_label_yes": metrics.get("avg_delta_margin_label_yes", fixed.get("avg_delta_margin_label_yes", "")),
                "avg_delta_margin_label_no": metrics.get("avg_delta_margin_label_no", fixed.get("avg_delta_margin_label_no", "")),
                "changed_pred": metrics.get("changed_pred", fixed.get("changed_pred", "")),
                "changed_text": metrics.get("changed_text", fixed.get("changed_text", "")),
                "avg_output_length": metrics.get("avg_output_length", fixed.get("avg_output_length", nested(metrics, "baseline", "average_output_length"))),
            }
        )
    return rows


def table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    """Render rows as a markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return lines


def dict_table(payload: dict[str, Any], title_key: str = "key") -> list[str]:
    """Render a flat dictionary as a two-column table."""

    lines = [f"| {title_key} | value |", "| --- | ---: |"]
    for key, value in payload.items():
        lines.append(f"| {key} | {fmt(value)} |")
    return lines


def build_conclusion(vector_stats: dict[str, Any], run_rows: list[dict[str, Any]]) -> list[str]:
    """Build a short automatic interpretation of the run."""

    lines: list[str] = []
    cosine = nested(vector_stats, "cosine_diagnostics", "cat_present_cat_absent", default=None)
    if cosine is not None:
        try:
            cosine_float = float(cosine)
            if cosine_float < -0.3:
                lines.append("- `cat_present` and `cat_absent` are strongly opposed, so cat steering still appears condition-dependent.")
            elif cosine_float > 0.3:
                lines.append("- `cat_present` and `cat_absent` are positively aligned, which is promising for a shared cat factual-text direction.")
            else:
                lines.append("- `cat_present` and `cat_absent` are weakly aligned; fixed steering may be unstable.")
        except (TypeError, ValueError):
            pass
    steered = [row for row in run_rows if "baseline" not in str(row.get("run", ""))]
    if steered:
        best = max(steered, key=lambda row: float(row.get("delta_accuracy") or 0.0))
        delta = float(best.get("delta_accuracy") or 0.0)
        yes_delta = float(best.get("avg_delta_margin_label_yes") or 0.0)
        no_delta = float(best.get("avg_delta_margin_label_no") or 0.0)
        if delta > 0:
            lines.append(f"- Best POPE delta is positive (`{best['run']}`, delta={delta:.4f}), suggesting AFTER-style factual text helps this cat steering setup.")
        elif abs(yes_delta) > 0.05 or abs(no_delta) > 0.05:
            lines.append("- POPE accuracy does not improve, but first-token margins move; the vector has control power and likely needs conditional/gated steering.")
        else:
            lines.append("- POPE accuracy and margins barely move; inspect extraction position, head/layer choice, or alpha scale.")
        if yes_delta > 0 and no_delta < 0:
            lines.append("- Label-specific margins move in the desired truthfulness-like directions.")
        elif yes_delta > 0 and no_delta > 0:
            lines.append("- Label-specific margins move in the same Yes direction, indicating an existence/Yes bias remains.")
    return lines or ["- Not enough completed runs to draw a conclusion yet."]


def write_report(output: Path, pairs_stats: dict[str, Any], vector_stats: dict[str, Any], run_rows: list[dict[str, Any]]) -> None:
    """Write the AFTER-style v1 markdown report."""

    lines: list[str] = [
        "# AFTER-Style v1 Report",
        "",
        "This experiment is a lightweight AFTER-style factual-text pair pipeline, not a full AFTER QAO reproduction.",
        "All new data lives under `data/after_style_v1/` and `data/outputs_after_style_v1/`; legacy outputs are not overwritten.",
        "",
        "## Data Construction",
        "",
        "- `cat`: present/absent object-existence questions with full factual text answers.",
        "- `attr`: count and optional bbox-color factual/counterfactual text answers.",
        "- `rel`: bbox-derived left/right/above/below relation text answers.",
        "- Difference from the older pair bank: answers are fuller factual text, especially for cat, to reduce pure `Yes`/`No` token-polarity learning.",
        "",
        "## Pair Statistics",
        "",
        f"- Total pairs: {pairs_stats.get('total_pairs', 'TBD')}",
        f"- Selected images: {pairs_stats.get('num_selected_images', 'TBD')}",
        "",
        "### Type Counts",
        "",
        *dict_table(pairs_stats.get("hallucination_type_counts", {}), "hallucination_type"),
        "",
        "### Subtype Counts",
        "",
        *dict_table(pairs_stats.get("subtype_counts", {}), "subtype"),
        "",
        "### Skipped Reasons",
        "",
        *dict_table(pairs_stats.get("skipped_reason_counts", {}), "reason"),
        "",
        "## Vector Statistics",
        "",
        "### Sample Counts By Type",
        "",
        *dict_table(vector_stats.get("sample_counts_by_type", {}), "expert"),
        "",
        "### Cosine Diagnostics",
        "",
        *dict_table(vector_stats.get("cosine_diagnostics", {}), "cosine"),
        "",
        "### Vector Norms",
        "",
        "| expert | mean_norm | max_norm | min_norm |",
        "| --- | ---: | ---: | ---: |",
    ]
    for expert, norms in vector_stats.get("vector_norms", {}).items():
        norms = norms or {}
        lines.append(f"| {expert} | {fmt(norms.get('mean', ''))} | {fmt(norms.get('max', ''))} | {fmt(norms.get('min', ''))} |")
    lines.extend(
        [
            "",
            "## POPE Random Results",
            "",
            *(table(run_rows, [
                "run",
                "alpha",
                "accuracy_baseline",
                "accuracy_steered",
                "delta_accuracy",
                "f1_yes",
                "yes_rate_baseline",
                "yes_rate_steered",
                "wrong_to_right",
                "right_to_wrong",
                "avg_delta_margin_label_yes",
                "avg_delta_margin_label_no",
                "changed_pred",
                "changed_text",
            ]) if run_rows else ["No POPE runs found yet."]),
            "",
            "## Automatic Conclusion",
            "",
            *build_conclusion(vector_stats, run_rows),
            "",
            "## Next-Step Reading",
            "",
            "- If POPE improves, expand to popular/adversarial and attr/rel-specific benchmarks.",
            "- If margins move but accuracy does not, add a non-oracle no-action gate or query-conditioned steering.",
            "- If label=yes and label=no margins move in the same direction, the vector still behaves like an existence polarity direction.",
            "- If margins barely move, revisit hook layer/head selection, activation extraction, or alpha scaling.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Generate the AFTER-style v1 report."""

    args = parse_args()
    try:
        pairs_stats = read_json(resolve_project_path(args.pairs_stats))
        vector_stats = read_json(resolve_project_path(args.vector_stats))
        run_rows = collect_run_rows(resolve_project_path(args.runs_root))
        write_report(resolve_project_path(args.output), pairs_stats, vector_stats, run_rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote AFTER-style report to {resolve_project_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
