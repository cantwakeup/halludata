"""Summarize AFTER-template steering results on prepared AMBER yes/no subsets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORY_ORDER = ("existence", "attribute", "relation")
TASK_LABELS = {
    "existence": "cat / object existence",
    "attribute": "attr / attribute",
    "relation": "rel / relation",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="data/outputs_after_template_disjoint_v1/amber_runs")
    parser.add_argument("--dataset-root", default="data/benchmarks/amber_hallucination")
    parser.add_argument("--output", default="data/outputs_after_template_disjoint_v1/amber_runs/REPORT.md")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning empty dict if absent."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def nested(payload: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Read a nested dictionary field."""

    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def fmt(value: Any) -> str:
    """Format markdown table values."""

    if value is None:
        return ""
    try:
        if value != "":
            return f"{float(value):.4f}"
    except (TypeError, ValueError):
        pass
    return str(value)


def table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    """Render a markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return lines


def alpha_from_run_name(run_name: str) -> str:
    """Extract alpha from run directory name."""

    match = re.search(r"alpha([-+]?\d+(?:\.\d+)?)$", run_name)
    return match.group(1) if match else ""


def collect_rows(runs_root: Path) -> list[dict[str, Any]]:
    """Collect completed AMBER run metrics."""

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("*/*/metrics.json")):
        category = metrics_path.parent.parent.name
        run = metrics_path.parent.name
        metrics = read_json(metrics_path)
        fixed = metrics.get("fixed_steering", {})
        if not isinstance(fixed, dict):
            fixed = {}
        config = read_json(metrics_path.parent / "config.json")
        steering_config = config.get("steering", {}) if isinstance(config.get("steering"), dict) else {}
        enabled_experts = steering_config.get("enabled_experts", [])
        expert = ""
        if isinstance(enabled_experts, list) and enabled_experts:
            expert = str(enabled_experts[0])
        elif "_" in run and run != "baseline":
            expert = run.split("_", 1)[0]
        baseline_acc = metrics.get("accuracy_baseline", nested(metrics, "baseline", "accuracy"))
        steered_acc = metrics.get("accuracy_steered")
        if steered_acc in ("", None):
            steered_acc = nested(metrics, "steered", "accuracy", default=baseline_acc)
        baseline_f1 = fixed.get("f1_baseline", nested(metrics, "baseline", "f1_yes"))
        steered_f1 = fixed.get("f1_steered", nested(metrics, "steered", "f1_yes", default=baseline_f1))
        rows.append(
            {
                "category": category,
                "task": TASK_LABELS.get(category, category),
                "run": run,
                "expert": expert,
                "alpha": fixed.get("alpha", alpha_from_run_name(run)),
                "accuracy_baseline": baseline_acc,
                "accuracy_steered": steered_acc,
                "delta_accuracy": metrics.get("delta_accuracy", fixed.get("delta_accuracy", "")),
                "f1_baseline": baseline_f1,
                "f1_steered": steered_f1,
                "delta_f1": fixed.get("delta_f1", ""),
                "yes_rate_baseline": metrics.get("yes_rate_baseline", nested(metrics, "baseline", "yes_rate")),
                "yes_rate_steered": metrics.get("yes_rate_steered", nested(metrics, "steered", "yes_rate", default=nested(metrics, "baseline", "yes_rate"))),
                "wrong_to_right": metrics.get("wrong_to_right", fixed.get("wrong_to_right", "")),
                "right_to_wrong": metrics.get("right_to_wrong", fixed.get("right_to_wrong", "")),
                "avg_delta_margin_all": metrics.get("avg_delta_margin_all", fixed.get("avg_delta_margin_all", "")),
                "avg_delta_margin_label_yes": metrics.get("avg_delta_margin_label_yes", fixed.get("avg_delta_margin_label_yes", "")),
                "avg_delta_margin_label_no": metrics.get("avg_delta_margin_label_no", fixed.get("avg_delta_margin_label_no", "")),
                "changed_pred": metrics.get("changed_pred", fixed.get("changed_pred", "")),
                "changed_text": metrics.get("changed_text", fixed.get("changed_text", "")),
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[int, int, float]:
        category = str(row.get("category", ""))
        category_rank = DEFAULT_CATEGORY_ORDER.index(category) if category in DEFAULT_CATEGORY_ORDER else 99
        is_baseline = 0 if row.get("run") == "baseline" else 1
        try:
            alpha = float(row.get("alpha") or -1.0)
        except (TypeError, ValueError):
            alpha = -1.0
        return category_rank, is_baseline, alpha

    return sorted(rows, key=sort_key)


def best_rows_by_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return best non-baseline run per category."""

    best: list[dict[str, Any]] = []
    for category in sorted({str(row.get("category", "")) for row in rows}):
        candidates = [row for row in rows if row.get("category") == category and row.get("run") != "baseline"]
        if candidates:
            best.append(max(candidates, key=lambda row: float(row.get("delta_accuracy") or 0.0)))
    return best


def build_conclusion(rows: list[dict[str, Any]]) -> list[str]:
    """Build a compact automatic interpretation."""

    if not rows:
        return ["- No completed AMBER runs found yet."]
    lines = [
        "- This is an AMBER yes/no discriminative subset proxy, not the full AMBER generative CHAIR evaluation.",
        "- Use `existence` to read cat steering, `attribute` to read attr steering, and `relation` to read rel steering.",
    ]
    for row in best_rows_by_category(rows):
        category = str(row.get("category", ""))
        delta = float(row.get("delta_accuracy") or 0.0)
        run = str(row.get("run", ""))
        if delta > 0:
            lines.append(f"- Best `{category}` run is `{run}` with delta_accuracy={delta:.4f}; this is a positive AMBER signal for {TASK_LABELS.get(category, category)}.")
        elif delta < 0:
            lines.append(f"- Best `{category}` run is still negative ({run}, delta_accuracy={delta:.4f}); inspect scale/head choice before claiming mitigation.")
        else:
            lines.append(f"- `{category}` has no accuracy gain yet; inspect margin movement and prediction flips.")
    return lines


def write_report(output: Path, dataset_stats: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the AMBER markdown report."""

    headers = [
        "category",
        "task",
        "run",
        "expert",
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
        "# AFTER-Template AMBER Hallucination Report",
        "",
        "This report evaluates expert steering on prepared AMBER yes/no discriminative hallucination subsets.",
        "It does not evaluate AMBER generative CHAIR metrics.",
        "",
        "## Prepared Dataset",
        "",
        f"- Dataset root: `{dataset_stats.get('out_dir', 'data/benchmarks/amber_hallucination')}`",
        f"- Image root: `{dataset_stats.get('image_root', '')}`",
        f"- Total samples: {dataset_stats.get('total_samples', 'TBD')}",
        f"- Category counts: `{dataset_stats.get('category_counts', {})}`",
        f"- Answer counts: `{dataset_stats.get('answer_counts', {})}`",
        "",
        "## Results",
        "",
        *(table(rows, headers) if rows else ["No run metrics found yet."]),
        "",
        "## Best Run Per Category",
        "",
        *(table(best_rows_by_category(rows), headers[:8] + ["f1_steered", "wrong_to_right", "right_to_wrong"]) if rows else ["No completed steering runs found yet."]),
        "",
        "## Automatic Conclusion",
        "",
        *build_conclusion(rows),
        "",
        "## Reading Guide",
        "",
        "- `existence` is the AMBER object-existence subset.",
        "- `attribute` probes attribute-style yes/no decisions.",
        "- `relation` probes relation-style yes/no decisions.",
        "- If the prepared dataset has zero samples, inspect AMBER field names and rerun `prepare_amber_hallucination.py` with `--input-files`.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Generate the AMBER report."""

    args = parse_args()
    try:
        runs_root = resolve_project_path(args.runs_root)
        dataset_root = resolve_project_path(args.dataset_root)
        stats = read_json(dataset_root / "stats.json")
        rows = collect_rows(runs_root)
        output = resolve_project_path(args.output)
        write_report(output, stats, rows)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Wrote AMBER report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
