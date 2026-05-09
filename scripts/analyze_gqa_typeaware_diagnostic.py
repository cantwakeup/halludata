"""Deep diagnostics for GQA type-aware eval runs.

This script is intentionally read-only with respect to existing run outputs. It
adds one markdown report under the eval_runs directory and does not delete or
rewrite old predictions/metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="data/gqa_typeaware_v1/eval_runs")
    parser.add_argument("--summary", default="data/gqa_typeaware_v1/eval_runs/summary.csv")
    parser.add_argument("--output", default="data/gqa_typeaware_v1/eval_runs/DIAGNOSTIC_REPORT.md")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_benchmark_rows(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path_text = str(config.get("benchmark_data", ""))
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        candidate = PROJECT_ROOT / path_text
        if candidate.exists():
            path = candidate
    if not path.exists():
        return {}
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "samples", "questions", "annotations"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        rows = payload if isinstance(payload, list) else []
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if isinstance(row, dict):
            sample_id = str(row.get("sample_id") or row.get("question_id") or row.get("id") or index)
            result[sample_id] = row
    return result


def normalize_label(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def attach_raw(prediction: Mapping[str, Any], raw_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    row = dict(prediction)
    raw = raw_by_id.get(str(row.get("sample_id", "")), {})
    row["label"] = normalize_label(row.get("label")) or normalize_label(raw.get("answer") or raw.get("label"))
    row["raw_type"] = raw.get("type") or raw.get("hallucination_type") or ""
    row["raw_subtype"] = raw.get("subtype") or ""
    return row


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def classification_metrics(rows: Iterable[Mapping[str, Any]], pred_key: str) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("label") in {"yes", "no"}]
    tp = sum(1 for row in labeled if row.get("label") == "yes" and row.get(pred_key) == "yes")
    tn = sum(1 for row in labeled if row.get("label") == "no" and row.get(pred_key) == "no")
    # Treat non-yes/no or missing predictions as wrong for the true label.
    fp = sum(1 for row in labeled if row.get("label") == "no" and row.get(pred_key) != "no")
    fn = sum(1 for row in labeled if row.get("label") == "yes" and row.get(pred_key) != "yes")
    total = len(labeled)
    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = safe_div(2 * precision * recall, precision + recall)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    yes_rate = safe_div(sum(1 for row in labeled if row.get(pred_key) == "yes"), total)
    label_yes_ratio = safe_div(sum(1 for row in labeled if row.get("label") == "yes"), total)
    return {
        "num_samples": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "yes_rate": yes_rate,
        "label_yes_ratio": label_yes_ratio,
        "label_no_ratio": 1.0 - label_yes_ratio,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def group_metrics(rows: list[dict[str, Any]], pred_key: str, group_key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "unknown")].append(row)
    result: dict[str, dict[str, float | int]] = {}
    for group, items in sorted(grouped.items()):
        metrics = classification_metrics(items, pred_key)
        result[group] = {
            "n": metrics["num_samples"],
            "accuracy": metrics["accuracy"],
            "yes_rate": metrics["yes_rate"],
        }
    return result


def run_prediction_key(metrics: Mapping[str, Any]) -> str:
    return "steered_pred" if "fixed_steering" in metrics else "prediction"


def run_baseline_pred_key(metrics: Mapping[str, Any]) -> str:
    return "baseline_pred" if "fixed_steering" in metrics else "prediction"


def analyze_run(run_dir: Path, subset_baselines: Mapping[str, float]) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    config = read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    gqa_metrics = read_json(run_dir / "gqa_typeaware_metrics.json") if (run_dir / "gqa_typeaware_metrics.json").exists() else {}
    raw_by_id = load_benchmark_rows(config)
    rows = [attach_raw(row, raw_by_id) for row in read_jsonl(run_dir / "predictions.jsonl")]
    pred_key = run_prediction_key(metrics)
    base_pred_key = run_baseline_pred_key(metrics)
    class_metrics = classification_metrics(rows, pred_key)
    baseline_for_subset = subset_baselines.get(run_dir.parent.name)
    steering = metrics.get("steering_diagnostics", {})
    vector = gqa_metrics.get("vector") or ",".join(steering.get("enabled_experts", []))
    method = "steered" if "fixed_steering" in metrics else "baseline"
    alpha = steering.get("alpha", gqa_metrics.get("alpha", ""))
    yes_to_no = sum(1 for row in rows if row.get(base_pred_key) == "yes" and row.get(pred_key) == "no")
    no_to_yes = sum(1 for row in rows if row.get(base_pred_key) == "no" and row.get(pred_key) == "yes")
    label_split = {}
    for label in ("yes", "no"):
        label_rows = [row for row in rows if row.get("label") == label]
        label_split[label] = {
            "wrong_to_right": sum(
                1 for row in label_rows
                if row.get(base_pred_key) != row.get("label") and row.get(pred_key) == row.get("label")
            ),
            "right_to_wrong": sum(
                1 for row in label_rows
                if row.get(base_pred_key) == row.get("label") and row.get(pred_key) != row.get("label")
            ),
        }
    delta_acc = None
    if baseline_for_subset is not None:
        delta_acc = class_metrics["accuracy"] - baseline_for_subset
    yes_rate_delta = class_metrics["yes_rate"] - float(gqa_metrics.get("yes_rate", class_metrics["yes_rate"])) if method == "baseline" else None
    if method == "steered":
        baseline_yes_rate = metrics.get("fixed_steering", {}).get("yes_rate_baseline")
        if baseline_yes_rate is not None:
            yes_rate_delta = class_metrics["yes_rate"] - float(baseline_yes_rate)
    tags: list[str] = []
    if yes_rate_delta is not None and yes_rate_delta > 0.02:
        tags.append("yes-shift")
    if yes_rate_delta is not None and yes_rate_delta < -0.02:
        tags.append("no-shift")
    if delta_acc is not None and delta_acc > 0 and sum(v["wrong_to_right"] for v in label_split.values()) > sum(v["right_to_wrong"] for v in label_split.values()):
        tags.append("precise-correction")
    if not tags:
        tags.append("neutral/mixed")
    return {
        "run_dir": str(run_dir),
        "eval_subset": run_dir.parent.name,
        "run": run_dir.name,
        "method": method,
        "vector": vector,
        "alpha": alpha,
        "delta_acc": delta_acc,
        "yes_to_no": yes_to_no,
        "no_to_yes": no_to_yes,
        "wrong_to_right_label_yes": label_split["yes"]["wrong_to_right"],
        "wrong_to_right_label_no": label_split["no"]["wrong_to_right"],
        "right_to_wrong_label_yes": label_split["yes"]["right_to_wrong"],
        "right_to_wrong_label_no": label_split["no"]["right_to_wrong"],
        "per_subtype": group_metrics(rows, pred_key, "raw_subtype"),
        "tags": ",".join(tags),
        **class_metrics,
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def table(headers: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def subset_label_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = [row for row in run_rows if row["method"] == "baseline"]
    rows = []
    for row in sorted(baselines, key=lambda item: item["eval_subset"]):
        rows.append(
            {
                "eval_subset": row["eval_subset"],
                "n": row["num_samples"],
                "label_yes_ratio": row["label_yes_ratio"],
                "label_no_ratio": row["label_no_ratio"],
                "all_yes_acc": row["label_yes_ratio"],
                "all_no_acc": row["label_no_ratio"],
                "baseline_acc": row["accuracy"],
                "baseline_bal_acc": row["balanced_accuracy"],
                "baseline_mcc": row["mcc"],
            }
        )
    return rows


def right_expert_summary(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {
        "gqa_cat_val": "cat",
        "gqa_attr_val": "attr",
        "gqa_rel_val": "rel",
    }
    rows = []
    for subset, right_vector in expected.items():
        steered = [row for row in run_rows if row["eval_subset"] == subset and row["method"] == "steered"]
        if not steered:
            continue
        by_vector: dict[str, float] = {}
        for vector in sorted({str(row["vector"]) for row in steered}):
            best = max((float(row["delta_acc"] or 0.0) for row in steered if str(row["vector"]) == vector), default=0.0)
            by_vector[vector] = best
        best_vector = max(by_vector.items(), key=lambda item: item[1])[0] if by_vector else ""
        rows.append(
            {
                "eval_subset": subset,
                "right_expert": right_vector,
                "right_best_delta": by_vector.get(right_vector),
                "best_vector": best_vector,
                "best_delta": by_vector.get(best_vector),
                "right_is_best": str(best_vector == right_vector),
            }
        )
    return rows


def subtype_sections(run_rows: list[dict[str, Any]]) -> list[str]:
    sections: list[str] = []
    for row in sorted(run_rows, key=lambda item: (item["eval_subset"], item["run"])):
        if not row["per_subtype"]:
            continue
        rows = [
            {"subtype": subtype, **values}
            for subtype, values in row["per_subtype"].items()
        ]
        sections.extend(
            [
                f"### {row['eval_subset']} / {row['run']}",
                "",
                table(["subtype", "n", "accuracy", "yes_rate"], rows),
                "",
            ]
        )
    return sections


def write_report(output: Path, run_rows: list[dict[str, Any]], summary_rows: list[dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_rows_sorted = sorted(run_rows, key=lambda row: (row["eval_subset"], row["method"], str(row["vector"]), str(row["alpha"])))
    metric_headers = [
        "eval_subset",
        "run",
        "vector",
        "alpha",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "yes_rate",
        "delta_acc",
        "yes_to_no",
        "no_to_yes",
        "wrong_to_right_label_yes",
        "wrong_to_right_label_no",
        "right_to_wrong_label_yes",
        "right_to_wrong_label_no",
        "tags",
    ]
    text = [
        "# GQA Type-Aware Diagnostic Deep Report",
        "",
        f"- Runs summarized: {len(run_rows)}",
        f"- summary.csv rows read: {len(summary_rows)}",
        "",
        "## Label Balance And Trivial Baselines",
        "",
        table(
            [
                "eval_subset",
                "n",
                "label_yes_ratio",
                "label_no_ratio",
                "all_yes_acc",
                "all_no_acc",
                "baseline_acc",
                "baseline_bal_acc",
                "baseline_mcc",
            ],
            subset_label_rows(run_rows),
        ),
        "",
        "## Run-Level Diagnostics",
        "",
        table(metric_headers, run_rows_sorted),
        "",
        "## Right Expert Check",
        "",
        table(["eval_subset", "right_expert", "right_best_delta", "best_vector", "best_delta", "right_is_best"], right_expert_summary(run_rows)),
        "",
        "## Per-Subtype Accuracy And Yes Rate",
        "",
        *subtype_sections(run_rows_sorted),
        "## Automatic Interpretation",
        "",
    ]
    right_rows = right_expert_summary(run_rows)
    if right_rows:
        wins = sum(1 for row in right_rows if row["right_is_best"] == "True")
        text.append(f"- Right expert is best on {wins}/{len(right_rows)} eval subsets.")
    shift_counts = Counter(tag for row in run_rows for tag in str(row["tags"]).split(","))
    text.append(f"- Shift tags: {dict(shift_counts)}")
    output.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    summary_rows = read_summary(Path(args.summary))
    try:
        baseline_acc: dict[str, float] = {}
        for metrics_path in runs_root.glob("*/baseline/gqa_typeaware_metrics.json"):
            metrics = read_json(metrics_path)
            baseline_acc[metrics_path.parent.parent.name] = float(metrics.get("accuracy", 0.0))
        run_rows = []
        for metrics_path in sorted(runs_root.glob("*/*/gqa_typeaware_metrics.json")):
            run_rows.append(analyze_run(metrics_path.parent, baseline_acc))
        if not run_rows:
            raise RuntimeError(f"No gqa_typeaware_metrics.json files found under {runs_root}")
        write_report(Path(args.output), run_rows, summary_rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote GQA diagnostic deep report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
