"""Summarize MME count/color attribute steering sanity runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="data/outputs_attr_sanity_mme/runs")
    parser.add_argument("--output", default="data/outputs_attr_sanity_mme/summary.csv")
    parser.add_argument("--report-output", default="data/outputs_attr_sanity_mme/ATTR_MME_SANITY_REPORT.md")
    parser.add_argument("--data-report", default="data/outputs_attr_sanity_mme/ATTR_MME_DATA_REPORT.md")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            rows.append(payload)
    return rows


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except Exception:
        return default


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def parse_run_name(name: str) -> tuple[str, float | None]:
    if name == "baseline":
        return "", None
    match = re.match(r"(.+)_alpha([0-9pPmM.-]+)$", name)
    if not match:
        return name, None
    vector = match.group(1).replace("__plus__", "+")
    alpha_text = match.group(2).replace("p", ".").replace("P", ".").replace("m", "-").replace("M", "-")
    return vector, fnum(alpha_text, 0.0)


def yesno_metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("label") in {"yes", "no"}]
    tp = sum(1 for row in labeled if row.get("label") == "yes" and row.get(pred_key) == "yes")
    tn = sum(1 for row in labeled if row.get("label") == "no" and row.get(pred_key) == "no")
    fp = sum(1 for row in labeled if row.get("label") == "no" and row.get(pred_key) == "yes")
    fn = sum(1 for row in labeled if row.get("label") == "yes" and row.get(pred_key) != "yes")
    invalid = sum(1 for row in rows if row.get(pred_key) not in {"yes", "no"})
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    yes_rows = [row for row in labeled if row.get("label") == "yes"]
    no_rows = [row for row in labeled if row.get("label") == "no"]
    return {
        "n": len(rows),
        "labeled_n": len(labeled),
        "accuracy": (tp + tn) / len(labeled) if labeled else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_rate": sum(1 for row in rows if row.get(pred_key) == "yes") / len(rows) if rows else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid": invalid,
        "label_yes_accuracy": (
            sum(1 for row in yes_rows if row.get(pred_key) == "yes") / len(yes_rows) if yes_rows else 0.0
        ),
        "label_no_accuracy": (
            sum(1 for row in no_rows if row.get(pred_key) == "no") / len(no_rows) if no_rows else 0.0
        ),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_run(category: str, run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.jsonl"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not predictions_path.exists():
        return None
    config = read_json(config_path) if config_path.exists() else {}
    predictions = read_jsonl(predictions_path)
    vector, alpha = parse_run_name(run_dir.name)
    is_baseline = run_dir.name == "baseline"
    if is_baseline:
        rows = [row for row in predictions if row.get("mode") in {"baseline", ""} or "prediction" in row]
        base_metrics = yesno_metrics(rows, "prediction")
        steered_metrics = dict(base_metrics)
        wrong_to_right = right_to_wrong = changed_pred = yes_to_no = no_to_yes = 0
        delta_margin_all = delta_margin_yes = delta_margin_no = 0.0
    else:
        rows = [row for row in predictions if "baseline_pred" in row and "steered_pred" in row]
        base_metrics = yesno_metrics(rows, "baseline_pred")
        steered_metrics = yesno_metrics(rows, "steered_pred")
        wrong_to_right = sum(
            1
            for row in rows
            if row.get("label") in {"yes", "no"}
            and row.get("baseline_pred") != row.get("label")
            and row.get("steered_pred") == row.get("label")
        )
        right_to_wrong = sum(
            1
            for row in rows
            if row.get("label") in {"yes", "no"}
            and row.get("baseline_pred") == row.get("label")
            and row.get("steered_pred") != row.get("label")
        )
        changed_pred = sum(1 for row in rows if row.get("baseline_pred") != row.get("steered_pred"))
        yes_to_no = sum(1 for row in rows if row.get("baseline_pred") == "yes" and row.get("steered_pred") == "no")
        no_to_yes = sum(1 for row in rows if row.get("baseline_pred") == "no" and row.get("steered_pred") == "yes")
        margins = [fnum(row.get("delta_margin")) for row in rows if row.get("delta_margin") not in (None, "")]
        yes_margins = [fnum(row.get("delta_margin")) for row in rows if row.get("label") == "yes" and row.get("delta_margin") not in (None, "")]
        no_margins = [fnum(row.get("delta_margin")) for row in rows if row.get("label") == "no" and row.get("delta_margin") not in (None, "")]
        delta_margin_all = mean(margins)
        delta_margin_yes = mean(yes_margins)
        delta_margin_no = mean(no_margins)
    steering = config.get("steering", {}) if isinstance(config.get("steering"), dict) else {}
    return {
        "benchmark": f"mme_{category}",
        "category": category,
        "run": run_dir.name,
        "method": "baseline" if is_baseline else "steered",
        "vector": vector,
        "enabled_experts": ",".join(str(item) for item in steering.get("enabled_experts", [])),
        "alpha": "" if alpha is None else alpha,
        "n": steered_metrics["n"],
        "accuracy": steered_metrics["accuracy"],
        "precision": steered_metrics["precision"],
        "recall": steered_metrics["recall"],
        "f1": steered_metrics["f1"],
        "yes_rate": steered_metrics["yes_rate"],
        "tp": steered_metrics["tp"],
        "tn": steered_metrics["tn"],
        "fp": steered_metrics["fp"],
        "fn": steered_metrics["fn"],
        "invalid": steered_metrics["invalid"],
        "label_yes_accuracy": steered_metrics["label_yes_accuracy"],
        "label_no_accuracy": steered_metrics["label_no_accuracy"],
        "baseline_accuracy": base_metrics["accuracy"],
        "baseline_precision": base_metrics["precision"],
        "baseline_recall": base_metrics["recall"],
        "baseline_f1": base_metrics["f1"],
        "baseline_yes_rate": base_metrics["yes_rate"],
        "baseline_tp": base_metrics["tp"],
        "baseline_tn": base_metrics["tn"],
        "baseline_fp": base_metrics["fp"],
        "baseline_fn": base_metrics["fn"],
        "delta_acc": steered_metrics["accuracy"] - base_metrics["accuracy"],
        "delta_f1": steered_metrics["f1"] - base_metrics["f1"],
        "delta_yes_rate": steered_metrics["yes_rate"] - base_metrics["yes_rate"],
        "delta_fp": steered_metrics["fp"] - base_metrics["fp"],
        "delta_fn": steered_metrics["fn"] - base_metrics["fn"],
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "changed_pred": changed_pred,
        "yes_to_no": yes_to_no,
        "no_to_yes": no_to_yes,
        "delta_margin_yes": delta_margin_yes,
        "delta_margin_no": delta_margin_no,
        "delta_margin_all": delta_margin_all,
        "vector_path": str(steering.get("vector_path", "")),
        "layers": str(steering.get("layers", "")),
        "k_heads": str(steering.get("k_heads", "")),
        "head_select": str(steering.get("head_select", "")),
        "run_dir": str(run_dir),
    }


def collect_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        category = category_dir.name
        for run_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            row = summarize_run(category, run_dir)
            if row is not None:
                rows.append(row)
    return rows


def table(headers: list[str], rows: list[dict[str, Any]], digits: int = 4) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, ""), digits) for header in headers) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "benchmark",
        "category",
        "method",
        "vector",
        "enabled_experts",
        "alpha",
        "n",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "yes_rate",
        "tp",
        "tn",
        "fp",
        "fn",
        "invalid",
        "label_yes_accuracy",
        "label_no_accuracy",
        "baseline_accuracy",
        "baseline_f1",
        "baseline_yes_rate",
        "baseline_fp",
        "baseline_fn",
        "delta_acc",
        "delta_f1",
        "delta_yes_rate",
        "delta_fp",
        "delta_fn",
        "wrong_to_right",
        "right_to_wrong",
        "changed_pred",
        "yes_to_no",
        "no_to_yes",
        "delta_margin_yes",
        "delta_margin_no",
        "delta_margin_all",
        "vector_path",
        "layers",
        "k_heads",
        "head_select",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def best_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("method") == "steered":
            by_category[str(row.get("category"))].append(row)
    for category, group in sorted(by_category.items()):
        if not group:
            continue
        best = max(group, key=lambda row: (fnum(row.get(metric)), fnum(row.get("delta_acc")), -fnum(row.get("changed_pred"))))
        out.append(best)
    return out


def diagnose_vector_group(group: list[dict[str, Any]]) -> str:
    if not group:
        return "no runs"
    sorted_group = sorted(group, key=lambda row: fnum(row.get("alpha")))
    best = max(sorted_group, key=lambda row: fnum(row.get("delta_f1")))
    positives = [row for row in sorted_group if fnum(row.get("delta_f1")) > 0.0 or fnum(row.get("delta_acc")) > 0.0]
    first = sorted_group[0]
    last = sorted_group[-1]
    tags: list[str] = []
    if sum(fnum(row.get("wrong_to_right")) for row in sorted_group) > sum(fnum(row.get("right_to_wrong")) for row in sorted_group):
        avg_changed = mean([fnum(row.get("changed_pred")) / max(fnum(row.get("n"), 1), 1) for row in sorted_group])
        if avg_changed <= 0.1:
            tags.append("precise correction")
    if fnum(last.get("yes_rate")) - fnum(first.get("yes_rate")) > 0.05 or fnum(best.get("delta_fp")) > 3:
        tags.append("yes-shift risk")
    if fnum(first.get("yes_rate")) - fnum(last.get("yes_rate")) > 0.05 or fnum(best.get("delta_fn")) > 3:
        tags.append("no-shift risk")
    if len(positives) <= 1 and fnum(best.get("delta_f1")) > 0:
        tags.append("unstable single-point gain")
    if not tags:
        tags.append("weak/no stable effect")
    return ", ".join(tags)


def load_skipped(runs_root: Path) -> list[dict[str, str]]:
    path = runs_root / "SKIPPED.tsv"
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append({key: value for key, value in row.items()})
    return rows


def write_report(path: Path, rows: list[dict[str, Any]], runs_root: Path, output_csv: Path, data_report: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_rows = [row for row in rows if row.get("method") == "baseline"]
    steered_rows = [row for row in rows if row.get("method") == "steered"]
    vector_paths = sorted({str(row.get("vector_path")) for row in steered_rows if str(row.get("vector_path", "")).strip()})
    settings_rows = []
    for row in steered_rows[:1]:
        settings_rows.append(
            {
                "model/runner": "run_steered_benchmark.py HF LLaVA",
                "decode": "greedy do_sample=False; max_new_tokens from runner default unless overridden",
                "layers": row.get("layers", ""),
                "head_select": row.get("head_select", ""),
                "top_heads": row.get("k_heads", ""),
                "apply_to": "prefill+decode last_token",
            }
        )
    by_vec: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in steered_rows:
        by_vec[(str(row.get("category")), str(row.get("vector")))].append(row)
    diag_rows = [
        {"category": category, "vector": vector, "diagnosis": diagnose_vector_group(group)}
        for (category, vector), group in sorted(by_vec.items())
    ]
    skipped_rows = load_skipped(runs_root)
    best_f1 = best_rows(rows, "delta_f1")
    best_acc = best_rows(rows, "delta_acc")
    lines = [
        "# Attribute MME Sanity Report",
        "",
        f"- Runs root: `{runs_root}`",
        f"- Summary CSV: `{output_csv}`",
        f"- Data report: `{data_report}`",
        f"- Runs summarized: `{len(rows)}`",
        "",
        "## Experiment Settings",
        "",
        table(["model/runner", "decode", "layers", "head_select", "top_heads", "apply_to"], settings_rows) if settings_rows else "No steered runs found.",
        "",
        "Vector paths:",
        "",
        *(f"- `{item}`" for item in vector_paths),
        "",
        "Skipped vectors:",
        "",
        table(["category", "vector", "reason"], skipped_rows) if skipped_rows else "No skipped vectors recorded.",
        "",
        "## Baseline",
        "",
        table(["category", "n", "accuracy", "precision", "recall", "f1", "yes_rate", "tp", "tn", "fp", "fn", "label_yes_accuracy", "label_no_accuracy"], baseline_rows),
        "",
        "## Best By F1",
        "",
        table(
            [
                "category",
                "vector",
                "enabled_experts",
                "alpha",
                "baseline_accuracy",
                "accuracy",
                "delta_acc",
                "baseline_f1",
                "f1",
                "delta_f1",
                "precision",
                "recall",
                "delta_fp",
                "delta_fn",
                "delta_yes_rate",
                "wrong_to_right",
                "right_to_wrong",
                "changed_pred",
            ],
            best_f1,
        ),
        "",
        "## Best By Accuracy",
        "",
        table(["category", "vector", "alpha", "baseline_accuracy", "accuracy", "delta_acc", "baseline_f1", "f1", "delta_f1", "delta_fp", "delta_fn", "delta_yes_rate"], best_acc),
        "",
        "## Automatic Diagnostics",
        "",
        table(["category", "vector", "diagnosis"], diag_rows),
        "",
        "## All Runs",
        "",
        table(
            [
                "category",
                "method",
                "vector",
                "enabled_experts",
                "alpha",
                "n",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "yes_rate",
                "tp",
                "tn",
                "fp",
                "fn",
                "delta_acc",
                "delta_f1",
                "delta_yes_rate",
                "wrong_to_right",
                "right_to_wrong",
                "changed_pred",
                "yes_to_no",
                "no_to_yes",
                "delta_margin_yes",
                "delta_margin_no",
            ],
            rows,
        ),
        "",
        "## Reading Guide",
        "",
        "- `FP` means label=no but pred=yes; rising FP with rising yes_rate is a yes-shift risk.",
        "- `FN` means label=yes but pred=no; rising FN with falling yes_rate is a no-shift risk.",
        "- `delta_margin_yes/no` are first-token Yes-vs-No margin changes when available from fixed steering runs.",
        "- Treat single changed predictions on small MME subsets as sanity signals, not final evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_root = resolve_project_path(args.runs_root)
    output = resolve_project_path(args.output)
    report_output = resolve_project_path(args.report_output)
    data_report = resolve_project_path(args.data_report)
    rows = collect_rows(runs_root)
    if not rows:
        raise RuntimeError(f"No completed MME attr sanity runs found under {runs_root}")
    rows = sorted(rows, key=lambda row: (str(row.get("category")), str(row.get("method")), str(row.get("vector")), fnum(row.get("alpha"), -1.0)))
    write_csv(output, rows)
    write_report(report_output, rows, runs_root, output, data_report)
    print(f"Wrote Attribute MME sanity summary to {output}")
    print(f"Wrote Attribute MME sanity report to {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
