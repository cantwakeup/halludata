"""Summarize raw POPE Regular/CatExpert prediction JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="data/pope_cat_expert_eval/full")
    parser.add_argument("--output", default="")
    parser.add_argument("--report-output", default="")
    return parser.parse_args()


def metric_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    tp = tn = fp = fn = invalid = 0
    pred_yes = 0
    for row in rows:
        label = str(row.get("label", "")).lower()
        pred = str(row.get("pred", "invalid")).lower()
        if pred == "yes":
            pred_yes += 1
        if pred not in {"yes", "no"}:
            invalid += 1
        if label == "yes" and pred == "yes":
            tp += 1
        elif label == "no" and pred == "no":
            tn += 1
        elif label == "no" and pred == "yes":
            fp += 1
        elif label == "yes" and pred == "no":
            fn += 1
        elif label == "yes":
            fn += 1
        elif label == "no":
            fp += 1
    total = len(rows)
    precision = metric_div(tp, tp + fp)
    recall = metric_div(tp, tp + fn)
    f1 = metric_div(2 * precision * recall, precision + recall)
    accuracy = metric_div(tp + tn, total)
    yes_rate = metric_div(pred_yes, total)
    return {
        "num_samples": total,
        "accuracy": accuracy * 100.0,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "yes_rate": yes_rate * 100.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid": invalid,
    }


def collect_rows(runs_root: Path) -> list[dict[str, Any]]:
    raw_dir = runs_root / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw prediction directory: {raw_dir}")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for path in sorted(raw_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        for row in rows:
            alpha = "" if row.get("alpha") in (None, "") else str(row.get("alpha"))
            key = (
                str(row.get("dataset", "")),
                str(row.get("setting", "")),
                str(row.get("method", "")),
                alpha,
            )
            grouped.setdefault(key, []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (dataset, setting, method, alpha), rows in sorted(grouped.items()):
        metrics = compute_metrics(rows)
        labels = Counter(str(row.get("label", "")).lower() for row in rows)
        preds = Counter(str(row.get("pred", "")).lower() for row in rows)
        summary_rows.append(
            {
                "Dataset": dataset,
                "Setting": setting,
                "Method": method,
                "Alpha": alpha,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1"],
                "Yes Rate": metrics["yes_rate"],
                "TP": metrics["tp"],
                "TN": metrics["tn"],
                "FP": metrics["fp"],
                "FN": metrics["fn"],
                "Invalid": metrics["invalid"],
                "N": metrics["num_samples"],
                "Label Yes": labels.get("yes", 0),
                "Label No": labels.get("no", 0),
                "Pred Yes": preds.get("yes", 0),
                "Pred No": preds.get("no", 0),
                "Pred Invalid": preds.get("invalid", 0),
            }
        )
    return summary_rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Dataset",
        "Setting",
        "Method",
        "Alpha",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Yes Rate",
        "TP",
        "TN",
        "FP",
        "FN",
        "Invalid",
        "N",
        "Label Yes",
        "Label No",
        "Pred Yes",
        "Pred No",
        "Pred Invalid",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(headers: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def method_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, float]:
    method = str(row.get("Method", ""))
    method_rank = 0 if method == "Regular" else 1
    alpha = row.get("Alpha", "")
    try:
        alpha_value = float(alpha)
    except Exception:
        alpha_value = -1.0
    return (str(row.get("Dataset", "")), str(row.get("Setting", "")), method_rank, alpha_value)


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    main_headers = ["Dataset", "Setting", "Method", "Alpha", "Accuracy", "Precision", "Recall", "F1 Score", "Yes Rate"]
    debug_headers = ["Dataset", "Setting", "Method", "Alpha", "TP", "TN", "FP", "FN", "Invalid", "N"]
    text = [
        "# POPE CatExpert Evaluation Summary",
        "",
        f"- Summary CSV: `{csv_path}`",
        f"- Runs summarized: {len(rows)}",
        "",
        "## Main Table",
        "",
        markdown_table(main_headers, sorted(rows, key=method_sort_key)),
        "",
        "## Debug Counts",
        "",
        markdown_table(debug_headers, sorted(rows, key=method_sort_key)),
        "",
        "## Notes",
        "",
        "- Positive class is `yes`, meaning the queried object exists.",
        "- `FP` is the object-hallucination count: label=no, pred=yes.",
        "- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output = Path(args.output) if str(args.output).strip() else runs_root / "summary.csv"
    report_output = Path(args.report_output) if str(args.report_output).strip() else runs_root / "SUMMARY.md"
    try:
        rows = collect_rows(runs_root)
        if not rows:
            raise RuntimeError(f"No raw prediction rows found under {runs_root / 'raw'}")
        write_csv(output, rows)
        write_report(report_output, rows, output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote POPE CatExpert summary to {output}")
    print(f"Wrote POPE CatExpert report to {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
