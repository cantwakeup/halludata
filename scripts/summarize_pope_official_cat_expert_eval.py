"""Summarize official-LLaVA POPE Regular/CatExpert prediction JSONL files."""

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


SUMMARY_FIELDS = [
    "Dataset",
    "Setting",
    "Method",
    "Alpha",
    "N",
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
    "Label Yes",
    "Label No",
    "Pred Yes",
    "Pred No",
    "Pred Invalid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="data/pope_cat_expert_eval/official_llava_cat_expert_alpha_sweep_full")
    parser.add_argument("--output", default="")
    parser.add_argument("--report-output", default="")
    parser.add_argument("--old-summary", default="data/pope_cat_expert_eval/full_alpha_sweep/summary.csv")
    return parser.parse_args()


def div(num: float, den: float) -> float:
    return num / den if den else 0.0


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def compute_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    tp = tn = fp = fn = invalid = pred_yes = 0
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
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    f1 = div(2 * precision * recall, precision + recall)
    return {
        "N": total,
        "Accuracy": div(tp + tn, total) * 100.0,
        "Precision": precision * 100.0,
        "Recall": recall * 100.0,
        "F1 Score": f1 * 100.0,
        "Yes Rate": div(pred_yes, total) * 100.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Invalid": invalid,
    }


def alpha_key(value: Any) -> tuple[int, float]:
    if value in (None, ""):
        return (0, -1.0)
    return (1, to_float(value))


def method_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, tuple[int, float]]:
    method = str(row.get("Method", ""))
    method_rank = 0 if method == "Regular" else 1
    return (str(row.get("Dataset", "")), str(row.get("Setting", "")), method_rank, alpha_key(row.get("Alpha", "")))


def find_raw_files(runs_root: Path) -> list[Path]:
    raw_dir = runs_root / "raw"
    if raw_dir.exists():
        files = sorted(raw_dir.glob("*.jsonl"))
        if files:
            return files
    return sorted(path for path in runs_root.glob("**/raw/*.jsonl") if path.is_file())


def collect_summary_rows(runs_root: Path) -> list[dict[str, Any]]:
    raw_files = find_raw_files(runs_root)
    if not raw_files:
        raise FileNotFoundError(f"No raw prediction JSONL files found under {runs_root}")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for path in raw_files:
        for row in read_jsonl(path):
            alpha = "" if row.get("alpha") in (None, "") else str(row.get("alpha"))
            key = (str(row.get("dataset", "")), str(row.get("setting", "")), str(row.get("method", "")), alpha)
            grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (dataset, setting, method, alpha), group_rows in sorted(grouped.items()):
        metrics = compute_metrics(group_rows)
        labels = Counter(str(row.get("label", "")).lower() for row in group_rows)
        preds = Counter(str(row.get("pred", "")).lower() for row in group_rows)
        rows.append(
            {
                "Dataset": dataset,
                "Setting": setting,
                "Method": method,
                "Alpha": alpha,
                **metrics,
                "Label Yes": labels.get("yes", 0),
                "Label No": labels.get("no", 0),
                "Pred Yes": preds.get("yes", 0),
                "Pred No": preds.get("no", 0),
                "Pred Invalid": preds.get("invalid", 0),
            }
        )
    return rows


def load_old_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = dict(row)
            if "F1" in normalized and "F1 Score" not in normalized:
                normalized["F1 Score"] = normalized.get("F1", "")
            rows.append(normalized)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=method_sort_key):
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def table(headers: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def baseline_by_group(rows: list[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("Method", "")) == "Regular":
            out[(str(row.get("Dataset", "")), str(row.get("Setting", "")))] = row
    return out


def best_cat_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = baseline_by_group(rows)
    cat_rows = [row for row in rows if str(row.get("Method", "")) == "CatExpert"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cat_rows:
        grouped.setdefault((str(row.get("Dataset", "")), str(row.get("Setting", ""))), []).append(row)

    best_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        best = max(group, key=lambda row: (to_float(row.get("F1 Score")), to_float(row.get("Accuracy"))))
        baseline = baselines.get(key, {})
        best_rows.append(
            {
                "Dataset": key[0],
                "Setting": key[1],
                "Best Alpha": best.get("Alpha", ""),
                "Baseline Acc": to_float(baseline.get("Accuracy")),
                "Best Acc": to_float(best.get("Accuracy")),
                "Delta Acc": to_float(best.get("Accuracy")) - to_float(baseline.get("Accuracy")),
                "Baseline F1": to_float(baseline.get("F1 Score")),
                "Best F1": to_float(best.get("F1 Score")),
                "Delta F1": to_float(best.get("F1 Score")) - to_float(baseline.get("F1 Score")),
                "Baseline FP": baseline.get("FP", ""),
                "Best FP": best.get("FP", ""),
                "FP Delta": to_float(best.get("FP")) - to_float(baseline.get("FP")),
                "Baseline Yes Rate": to_float(baseline.get("Yes Rate")),
                "Best Yes Rate": to_float(best.get("Yes Rate")),
            }
        )
    return best_rows


def alpha_zero_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = baseline_by_group(rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("Method", "")) != "CatExpert" or abs(to_float(row.get("Alpha")) - 0.0) > 1e-12:
            continue
        key = (str(row.get("Dataset", "")), str(row.get("Setting", "")))
        baseline = baselines.get(key, {})
        out.append(
            {
                "Dataset": key[0],
                "Setting": key[1],
                "Alpha0 Acc": to_float(row.get("Accuracy")),
                "Regular Acc": to_float(baseline.get("Accuracy")),
                "Acc Diff": to_float(row.get("Accuracy")) - to_float(baseline.get("Accuracy")),
                "Alpha0 F1": to_float(row.get("F1 Score")),
                "Regular F1": to_float(baseline.get("F1 Score")),
                "F1 Diff": to_float(row.get("F1 Score")) - to_float(baseline.get("F1 Score")),
                "Alpha0 Invalid": row.get("Invalid", ""),
                "Regular Invalid": baseline.get("Invalid", ""),
            }
        )
    return out


def old_hf_compare_rows(rows: list[dict[str, Any]], old_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not old_rows:
        return []
    old_baseline = baseline_by_group(old_rows)
    current_baseline = baseline_by_group(rows)
    out: list[dict[str, Any]] = []
    for key, row in sorted(current_baseline.items()):
        old = old_baseline.get(key)
        if not old:
            continue
        out.append(
            {
                "Dataset": key[0],
                "Setting": key[1],
                "Official Acc": to_float(row.get("Accuracy")),
                "HF Acc": to_float(old.get("Accuracy")),
                "Acc Diff": to_float(row.get("Accuracy")) - to_float(old.get("Accuracy")),
                "Official F1": to_float(row.get("F1 Score")),
                "HF F1": to_float(old.get("F1 Score")),
                "F1 Diff": to_float(row.get("F1 Score")) - to_float(old.get("F1 Score")),
                "Official FP": row.get("FP", ""),
                "HF FP": old.get("FP", ""),
                "Official Yes Rate": to_float(row.get("Yes Rate")),
                "HF Yes Rate": to_float(old.get("Yes Rate")),
            }
        )
    return out


def read_configs(runs_root: Path) -> list[dict[str, Any]]:
    configs = []
    for path in sorted(runs_root.glob("**/config.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        configs.append({"path": str(path), "payload": payload})
    return configs


def config_notes(configs: list[dict[str, Any]]) -> list[str]:
    if not configs:
        return ["- No config.json files found under the runs root."]
    first = configs[0]["payload"]
    notes = [
        f"- Config files found: `{len(configs)}`",
        f"- Runner: `{first.get('runner', '')}`",
        f"- Model path: `{first.get('model_path', '')}`",
        f"- LLaVA repo: `{first.get('llava_repo_path', '')}`",
        f"- Conv mode: `{first.get('conv_mode', '')}`",
        f"- Prompt template: `{first.get('prompt_template', '')}`",
        f"- Cat vector source: `{first.get('cat_vector_source', '')}`",
    ]
    decode = first.get("decode", {})
    steering = first.get("steering", {})
    notes.append(f"- Decode: `{json.dumps(decode, ensure_ascii=False)}`")
    notes.append(f"- Steering: `{json.dumps(steering, ensure_ascii=False)}`")
    return notes


def write_report(path: Path, rows: list[dict[str, Any]], old_rows: list[dict[str, Any]], csv_path: Path, runs_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    main_headers = ["Dataset", "Setting", "Method", "Alpha", "N", "Accuracy", "Precision", "Recall", "F1 Score", "Yes Rate"]
    debug_headers = ["Dataset", "Setting", "Method", "Alpha", "TP", "TN", "FP", "FN", "Invalid", "N"]
    best_headers = [
        "Dataset",
        "Setting",
        "Best Alpha",
        "Baseline Acc",
        "Best Acc",
        "Delta Acc",
        "Baseline F1",
        "Best F1",
        "Delta F1",
        "Baseline FP",
        "Best FP",
        "FP Delta",
        "Baseline Yes Rate",
        "Best Yes Rate",
    ]
    alpha0_headers = ["Dataset", "Setting", "Alpha0 Acc", "Regular Acc", "Acc Diff", "Alpha0 F1", "Regular F1", "F1 Diff", "Alpha0 Invalid", "Regular Invalid"]
    old_headers = ["Dataset", "Setting", "Official Acc", "HF Acc", "Acc Diff", "Official F1", "HF F1", "F1 Diff", "Official FP", "HF FP", "Official Yes Rate", "HF Yes Rate"]
    configs = read_configs(runs_root)
    best = best_cat_rows(rows)
    alpha0 = alpha_zero_checks(rows)
    old_compare = old_hf_compare_rows(rows, old_rows)

    text = [
        "# Official LLaVA POPE CatExpert Summary",
        "",
        f"- Summary CSV: `{csv_path}`",
        f"- Runs summarized: {len(rows)}",
        "",
        "## Run Config",
        "",
        *config_notes(configs),
        "",
    ]
    if best:
        text.extend(["## Best CatExpert By F1", "", table(best_headers, best), ""])
    if alpha0:
        text.extend(["## Alpha 0 Check", "", table(alpha0_headers, alpha0), ""])
    if old_compare:
        text.extend(["## Official Regular vs Old HF Regular", "", table(old_headers, old_compare), ""])
    text.extend(
        [
            "## Main Table",
            "",
            table(main_headers, sorted(rows, key=method_sort_key)),
            "",
            "## Debug Counts",
            "",
            table(debug_headers, sorted(rows, key=method_sort_key)),
            "",
            "## Notes",
            "",
            "- Positive class is `yes`, meaning the queried object exists.",
            "- `FP` is object hallucination: label=no, pred=yes.",
            "- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.",
            "- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.",
            "- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.",
        ]
    )
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output = Path(args.output) if str(args.output).strip() else runs_root / "summary.csv"
    report_output = Path(args.report_output) if str(args.report_output).strip() else runs_root / "SUMMARY.md"
    try:
        rows = collect_summary_rows(runs_root)
        if not rows:
            raise RuntimeError(f"No summarized rows found under {runs_root}")
        old_rows = load_old_summary(Path(args.old_summary)) if str(args.old_summary).strip() else []
        write_csv(output, rows)
        write_report(report_output, rows, old_rows, output, runs_root)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote official POPE CatExpert summary to {output}")
    print(f"Wrote official POPE CatExpert report to {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
