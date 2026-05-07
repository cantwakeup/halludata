"""Summarize relation-bucket diagnostic benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default="data/outputs_after_template_disjoint_v2/runs/relation_bucket_diagnostics",
    )
    parser.add_argument(
        "--output",
        default="data/outputs_after_template_disjoint_v2/runs/relation_bucket_diagnostics/REPORT.md",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def fmt(value: Any) -> str:
    """Format table values compactly."""

    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a Markdown table."""

    if not rows:
        return "_No runs found._\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def infer_task(path: Path, runs_root: Path) -> str:
    """Infer task name from run path."""

    try:
        return path.parent.parent.relative_to(runs_root).parts[0]
    except Exception:
        return path.parent.parent.name


def row_from_metrics(path: Path, runs_root: Path) -> dict[str, Any]:
    """Extract one flat summary row from a metrics file."""

    metrics = load_json(path)
    fixed = metrics.get("fixed_steering") if isinstance(metrics.get("fixed_steering"), dict) else {}
    diag = metrics.get("steering_diagnostics") if isinstance(metrics.get("steering_diagnostics"), dict) else {}
    baseline = metrics.get("baseline", {})
    steered = metrics.get("steered", {})
    enabled = diag.get("enabled_experts") or fixed.get("enabled_experts") or []
    expert = ",".join(str(item) for item in enabled) if enabled else ""
    run = path.parent.name
    return {
        "task": infer_task(path, runs_root),
        "run": run,
        "expert": expert,
        "alpha": diag.get("alpha", fixed.get("alpha")),
        "accuracy_baseline": metrics.get("accuracy_baseline", fixed.get("accuracy_baseline", baseline.get("accuracy"))),
        "accuracy_steered": metrics.get("accuracy_steered", fixed.get("accuracy_steered", steered.get("accuracy"))),
        "delta_accuracy": metrics.get("delta_accuracy", fixed.get("delta_accuracy")),
        "yes_rate_baseline": metrics.get("yes_rate_baseline", fixed.get("yes_rate_baseline", baseline.get("yes_rate"))),
        "yes_rate_steered": metrics.get("yes_rate_steered", fixed.get("yes_rate_steered", steered.get("yes_rate"))),
        "wrong_to_right": metrics.get("wrong_to_right", fixed.get("wrong_to_right")),
        "right_to_wrong": metrics.get("right_to_wrong", fixed.get("right_to_wrong")),
        "avg_delta_margin_label_yes": metrics.get(
            "avg_delta_margin_label_yes", fixed.get("avg_delta_margin_label_yes")
        ),
        "avg_delta_margin_label_no": metrics.get("avg_delta_margin_label_no", fixed.get("avg_delta_margin_label_no")),
        "changed_pred": metrics.get("changed_pred", fixed.get("changed_pred")),
    }


def main() -> int:
    """Write the diagnostic summary report."""

    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    output_path = Path(args.output).resolve()
    rows = [row_from_metrics(path, runs_root) for path in sorted(runs_root.glob("*/*/metrics.json"))]
    rows.sort(key=lambda row: (row["task"], str(row.get("expert", "")), -(row.get("delta_accuracy") or -999.0)))

    table_rows = [
        [
            row["task"],
            row["run"],
            row["expert"],
            row["alpha"],
            row["accuracy_baseline"],
            row["accuracy_steered"],
            row["delta_accuracy"],
            row["yes_rate_baseline"],
            row["yes_rate_steered"],
            row["wrong_to_right"],
            row["right_to_wrong"],
            row["avg_delta_margin_label_yes"],
            row["avg_delta_margin_label_no"],
            row["changed_pred"],
        ]
        for row in rows
    ]
    best_rows: list[list[Any]] = []
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task and row.get("delta_accuracy") is not None]
        if not task_rows:
            continue
        best = max(task_rows, key=lambda row: row.get("delta_accuracy") or -999.0)
        best_rows.append(
            [
                task,
                best["run"],
                best["expert"],
                best["alpha"],
                best["accuracy_baseline"],
                best["accuracy_steered"],
                best["delta_accuracy"],
                best["wrong_to_right"],
                best["right_to_wrong"],
            ]
        )

    report = "\n".join(
        [
            "# Relation Bucket Diagnostic Report",
            "",
            "This report summarizes bucket-specific relation steering runs.",
            "",
            "## Best Run Per Task",
            "",
            markdown_table(
                [
                    "task",
                    "run",
                    "expert",
                    "alpha",
                    "accuracy_baseline",
                    "accuracy_steered",
                    "delta_accuracy",
                    "wrong_to_right",
                    "right_to_wrong",
                ],
                best_rows,
            ).rstrip(),
            "",
            "## All Runs",
            "",
            markdown_table(
                [
                    "task",
                    "run",
                    "expert",
                    "alpha",
                    "accuracy_baseline",
                    "accuracy_steered",
                    "delta_accuracy",
                    "yes_rate_baseline",
                    "yes_rate_steered",
                    "wrong_to_right",
                    "right_to_wrong",
                    "avg_delta_margin_label_yes",
                    "avg_delta_margin_label_no",
                    "changed_pred",
                ],
                table_rows,
            ).rstrip(),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote relation bucket diagnostic report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
