"""Summarize GQA type-aware diagnostic evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="data/gqa_typeaware_v1/eval_runs")
    parser.add_argument("--output", default="")
    parser.add_argument("--report-output", default="")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_benchmark_rows(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path_text = str(config.get("benchmark_data", ""))
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        candidate = PROJECT_ROOT / path_text
        path = candidate if candidate.exists() else path
    if not path.exists():
        return {}
    rows = read_jsonl(path) if path.suffix.lower() == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        for key in ("data", "samples", "questions", "annotations"):
            if isinstance(rows.get(key), list):
                rows = rows[key]
                break
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return result
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id") or row.get("question_id") or row.get("id") or index)
        result[sample_id] = row
    return result


def safe_accuracy(rows: Iterable[Mapping[str, Any]], pred_key: str) -> float | None:
    labeled = [row for row in rows if row.get("label") in {"yes", "no"}]
    if not labeled:
        return None
    return sum(1 for row in labeled if row.get(pred_key) == row.get("label")) / len(labeled)


def attach_raw(prediction: Mapping[str, Any], raw_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    row = dict(prediction)
    raw = raw_by_id.get(str(row.get("sample_id", "")), {})
    row["raw_type"] = raw.get("type") or raw.get("hallucination_type") or ""
    row["raw_subtype"] = raw.get("subtype") or ""
    return row


def per_group_accuracy(rows: list[dict[str, Any]], pred_key: str, group_key: str) -> dict[str, float | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(row.get(group_key) or "unknown")
        grouped[group].append(row)
    return {group: safe_accuracy(items, pred_key) for group, items in sorted(grouped.items())}


def metrics_for_run(run_dir: Path, baseline_accuracy: float | None) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    config_path = run_dir / "config.json"
    config = read_json(config_path) if config_path.exists() else {}
    raw_by_id = load_benchmark_rows(config)
    predictions = [attach_raw(row, raw_by_id) for row in read_jsonl(run_dir / "predictions.jsonl")]

    steering = metrics.get("steering_diagnostics", {})
    enabled = steering.get("enabled_experts", [])
    vector = ",".join(enabled) if enabled else ""
    alpha = steering.get("alpha")
    eval_subset = run_dir.parent.name
    run_name = run_dir.name
    benchmark_name = str(config.get("benchmark_name") or metrics.get("benchmark_name") or "")
    eval_type = ""
    if raw_by_id:
        types = sorted({str(row.get("type") or row.get("hallucination_type") or "") for row in raw_by_id.values()})
        eval_type = ",".join([item for item in types if item])

    if "fixed_steering" in metrics:
        fixed = metrics["fixed_steering"]
        pred_key = "steered_pred"
        accuracy = fixed.get("accuracy_steered")
        baseline_acc = fixed.get("accuracy_baseline", baseline_accuracy)
        row = {
            "method": "steered",
            "vector": vector or run_name.split("_alpha")[0],
            "alpha": alpha,
            "eval_subset": eval_subset,
            "eval_type": eval_type,
            "run": run_name,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_acc,
            "delta_acc": fixed.get("delta_accuracy"),
            "yes_rate": fixed.get("yes_rate_steered"),
            "no_rate": None if fixed.get("yes_rate_steered") is None else 1.0 - float(fixed.get("yes_rate_steered")),
            "wrong_to_right": fixed.get("wrong_to_right"),
            "right_to_wrong": fixed.get("right_to_wrong"),
            "changed_pred": fixed.get("changed_pred"),
            "num_samples": fixed.get("num_samples"),
        }
    else:
        base = metrics.get("baseline", metrics)
        pred_key = "prediction"
        accuracy = base.get("accuracy")
        row = {
            "method": "baseline",
            "vector": "",
            "alpha": "",
            "eval_subset": eval_subset,
            "eval_type": eval_type,
            "run": run_name,
            "accuracy": accuracy,
            "baseline_accuracy": accuracy,
            "delta_acc": 0.0 if accuracy is not None else None,
            "yes_rate": base.get("yes_rate"),
            "no_rate": None if base.get("yes_rate") is None else 1.0 - float(base.get("yes_rate")),
            "wrong_to_right": 0,
            "right_to_wrong": 0,
            "changed_pred": 0,
            "num_samples": base.get("num_samples"),
        }

    per_type = per_group_accuracy(predictions, pred_key, "raw_type")
    per_subtype = per_group_accuracy(predictions, pred_key, "raw_subtype")
    row["per_type_accuracy"] = json.dumps(per_type, ensure_ascii=False, sort_keys=True)
    row["per_subtype_accuracy"] = json.dumps(per_subtype, ensure_ascii=False, sort_keys=True)
    write_json(run_dir / "gqa_typeaware_metrics.json", {**row, "benchmark_name": benchmark_name})
    return row


def collect_baselines(runs_root: Path) -> dict[str, float | None]:
    baselines: dict[str, float | None] = {}
    for metrics_path in runs_root.glob("*/baseline/metrics.json"):
        metrics = read_json(metrics_path)
        baselines[metrics_path.parent.parent.name] = metrics.get("baseline", metrics).get("accuracy")
    return baselines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "vector",
        "alpha",
        "eval_subset",
        "eval_type",
        "run",
        "accuracy",
        "baseline_accuracy",
        "delta_acc",
        "yes_rate",
        "no_rate",
        "wrong_to_right",
        "right_to_wrong",
        "changed_pred",
        "num_samples",
        "per_type_accuracy",
        "per_subtype_accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "eval_subset",
        "method",
        "vector",
        "alpha",
        "accuracy",
        "baseline_accuracy",
        "delta_acc",
        "yes_rate",
        "wrong_to_right",
        "right_to_wrong",
        "changed_pred",
        "num_samples",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_rows = sorted(
        [row for row in rows if row.get("method") == "steered" and row.get("delta_acc") is not None],
        key=lambda row: float(row.get("delta_acc") or 0.0),
        reverse=True,
    )
    text = [
        "# GQA Type-Aware Diagnostic Eval Summary",
        "",
        f"- Summary CSV: `{csv_path}`",
        f"- Runs summarized: {len(rows)}",
        "",
        "## Best Steered Runs By Delta",
        "",
        markdown_table(best_rows[:12]) if best_rows else "No steered runs found.",
        "",
        "## All Runs",
        "",
        markdown_table(rows),
        "",
        "## Notes",
        "",
        "- `per_type_accuracy` and `per_subtype_accuracy` are also written into each run directory as `gqa_typeaware_metrics.json`.",
        "- Baseline rows have `delta_acc=0`; steered rows compare against the matching subset baseline.",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output = Path(args.output) if args.output else runs_root / "summary.csv"
    report_output = Path(args.report_output) if args.report_output else runs_root / "SUMMARY.md"
    try:
        baselines = collect_baselines(runs_root)
        rows: list[dict[str, Any]] = []
        for metrics_path in sorted(runs_root.glob("*/*/metrics.json")):
            run_dir = metrics_path.parent
            rows.append(metrics_for_run(run_dir, baselines.get(run_dir.parent.name)))
        rows.sort(key=lambda row: (str(row.get("eval_subset")), str(row.get("method")), str(row.get("vector")), str(row.get("alpha"))))
        write_csv(output, rows)
        write_report(report_output, rows, output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote GQA type-aware summary to {output}")
    print(f"Wrote GQA type-aware report to {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
