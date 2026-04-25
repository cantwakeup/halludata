"""Summarize cat truthfulness vector diagnostics and POPE runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for cat truthfulness result summarization."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/outputs/runs/pope_cat_truth")
    parser.add_argument("--margin-root", default="data/outputs/debug")
    parser.add_argument("--vector-stats", default="data/outputs/steering/cat_truth_vector.stats.json")
    parser.add_argument("--output", default="data/outputs/runs/pope_cat_truth/summary.md")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object, returning an empty dict when absent."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def metric(metrics: dict[str, Any], section: str, key: str, default: Any = "") -> Any:
    """Read a nested metric field."""

    value = metrics.get(section, {})
    return value.get(key, default) if isinstance(value, dict) else default


def collect_run_rows(root: Path) -> list[dict[str, Any]]:
    """Collect baseline and alpha run metrics below the POPE cat truth root."""

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = read_json(metrics_path)
        relative = metrics_path.parent.relative_to(root)
        dataset = relative.parts[0] if relative.parts else "unknown"
        run_name = relative.parts[1] if len(relative.parts) > 1 else "baseline"
        fixed = metrics.get("fixed_steering", {})
        if not isinstance(fixed, dict):
            fixed = {}
        rows.append(
            {
                "dataset": dataset,
                "run": run_name,
                "alpha": fixed.get("alpha", "" if run_name == "baseline" else run_name.replace("alpha", "")),
                "accuracy": fixed.get("accuracy_steered", metric(metrics, "baseline", "accuracy", "")),
                "baseline_accuracy": fixed.get("accuracy_baseline", metric(metrics, "baseline", "accuracy", "")),
                "delta_accuracy": fixed.get("delta_accuracy", metrics.get("delta_accuracy", "")),
                "precision_yes": metric(metrics, "steered", "precision_yes", ""),
                "recall_yes": metric(metrics, "steered", "recall_yes", ""),
                "f1_yes": fixed.get("f1_steered", metric(metrics, "baseline", "f1_yes", "")),
                "yes_rate": fixed.get("yes_rate_steered", metric(metrics, "baseline", "yes_rate", "")),
                "wrong_to_right": fixed.get("wrong_to_right", ""),
                "right_to_wrong": fixed.get("right_to_wrong", ""),
                "avg_delta_margin_label_yes": fixed.get("avg_delta_margin_label_yes", ""),
                "avg_delta_margin_label_no": fixed.get("avg_delta_margin_label_no", ""),
            }
        )
    return rows


def collect_margin_rows(margin_root: Path) -> list[dict[str, Any]]:
    """Collect first-token margin summaries for cat truth vector runs."""

    rows: list[dict[str, Any]] = []
    for path in sorted(margin_root.glob("cat_truth_pope_*margin*.summary.json")):
        payload = read_json(path)
        rows.append(
            {
                "file": path.name,
                "steering_mode": payload.get("steering_mode", "fixed_positive"),
                "alpha": payload.get("alpha", ""),
                "baseline_logit_acc": payload.get("baseline_logit_acc", ""),
                "steered_logit_acc": payload.get("steered_logit_acc", ""),
                "avg_delta_margin_label_yes": payload.get("avg_delta_margin_label_yes", ""),
                "avg_delta_margin_label_no": payload.get("avg_delta_margin_label_no", ""),
                "wrong_to_right": payload.get("wrong_to_right", ""),
                "right_to_wrong": payload.get("right_to_wrong", ""),
            }
        )
    return rows


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


def run_table(rows: list[dict[str, Any]]) -> list[str]:
    """Render POPE generation metrics as markdown."""

    lines = [
        "| dataset | run | alpha | acc | base_acc | delta_acc | f1 | yes_rate | W->R | R->W | delta_yes | delta_no |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["dataset"]), str(item["run"]))):
        lines.append(
            "| {dataset} | {run} | {alpha} | {accuracy} | {baseline_accuracy} | {delta_accuracy} | {f1_yes} | {yes_rate} | "
            "{wrong_to_right} | {right_to_wrong} | {avg_delta_margin_label_yes} | {avg_delta_margin_label_no} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    return lines


def margin_table(rows: list[dict[str, Any]]) -> list[str]:
    """Render first-token margin summaries as markdown."""

    lines = [
        "| file | alpha | base_acc | steer_acc | delta_yes | delta_no | W->R | R->W |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {file} | {alpha} | {baseline_logit_acc} | {steered_logit_acc} | {avg_delta_margin_label_yes} | "
            "{avg_delta_margin_label_no} | {wrong_to_right} | {right_to_wrong} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    return lines


def judge_direction(stats: dict[str, Any], margin_rows: list[dict[str, Any]]) -> list[str]:
    """Generate a short interpretation of whether the vector is truthfulness-like."""

    lines: list[str] = []
    cosine = stats.get("present_absent_cosine")
    if cosine is not None:
        if float(cosine) > 0.3:
            lines.append("- Present-only and absent-only vectors are positively aligned, which supports a possible shared truthfulness direction.")
        elif float(cosine) < -0.3:
            lines.append("- Present-only and absent-only vectors are opposed, suggesting a single static vector is more likely an existence-polarity direction than a truthfulness direction.")
        else:
            lines.append("- Present-only and absent-only vectors are weakly aligned; a single static vector may be unstable.")
    if margin_rows:
        best = margin_rows[0]
        yes_delta = _float(best.get("avg_delta_margin_label_yes"))
        no_delta = _float(best.get("avg_delta_margin_label_no"))
        if yes_delta > 0 and no_delta < 0:
            lines.append("- First-token margins move in the desired directions for both labels: yes increases and no decreases.")
        elif yes_delta > 0 and no_delta > 0:
            lines.append("- First-token margins increase for both yes and no labels, so the vector still behaves like an existence/yes direction.")
        elif abs(yes_delta) < 0.05 and abs(no_delta) < 0.05:
            lines.append("- First-token margins barely move; extraction, head selection, or layer choice may need revision.")
        else:
            lines.append("- First-token margins are mixed; inspect alpha and per-dataset behavior before scaling.")
    return lines or ["- Not enough completed diagnostics to classify the vector yet."]


def _float(value: Any) -> float:
    """Parse a number with a safe fallback."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_summary(output: Path, stats: dict[str, Any], run_rows: list[dict[str, Any]], margin_rows: list[dict[str, Any]]) -> None:
    """Write the cat truthfulness markdown report."""

    lines = [
        "# Cat Truthfulness Steering Summary",
        "",
        "This report evaluates a single fixed-positive `cat_truth_vector`, not oracle signed steering.",
        "The intended intervention is always `activation += alpha * cat_truth_vector`.",
        "",
        "## Vector Diagnostics",
        "",
        f"- Present samples: {stats.get('sample_counts', {}).get('present', 'TBD')}",
        f"- Absent samples: {stats.get('sample_counts', {}).get('absent', 'TBD')}",
        f"- Present/absent cosine: {fmt(stats.get('present_absent_cosine', 'TBD'))}",
        "",
        "## Interpretation",
        "",
        *judge_direction(stats, margin_rows),
        "",
        "## First-Token Margin Summaries",
        "",
        *(margin_table(margin_rows) if margin_rows else ["No margin summaries found yet."]),
        "",
        "## POPE Generation Runs",
        "",
        *(run_table(run_rows) if run_rows else ["No generation runs found yet."]),
        "",
        "## Next-Step Diagnosis",
        "",
        "- If present/absent vectors are opposed, use query-conditioned vectors or retrieval rather than one static vector.",
        "- If margins move correctly but generation accuracy does not improve, try a no-action gate or adaptive alpha.",
        "- If margins barely move, revisit activation extraction position, selected heads, and layer range.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run cat truthfulness summarization."""

    args = parse_args()
    try:
        stats = read_json(resolve_project_path(args.vector_stats))
        run_rows = collect_run_rows(resolve_project_path(args.root))
        margin_rows = collect_margin_rows(resolve_project_path(args.margin_root))
        write_summary(resolve_project_path(args.output), stats, run_rows, margin_rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote cat truthfulness summary to {resolve_project_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
