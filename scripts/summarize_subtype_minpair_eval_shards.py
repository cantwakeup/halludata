#!/usr/bin/env python3
"""Merge and summarize sharded subtype minimal-pair eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, help="Directory containing shard*/raw/*.jsonl outputs.")
    parser.add_argument("--output-csv", default="", help="Defaults to <eval-root>/summary.csv.")
    parser.add_argument("--output-report", default="", help="Defaults to <eval-root>/SUMMARY.md.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def metric_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = invalid = yes_pred = 0
    for row in rows:
        label = str(row.get("label", "")).strip().lower()
        pred = str(row.get("pred", "")).strip().lower()
        if pred == "yes":
            yes_pred += 1
        if label == "yes" and pred == "yes":
            tp += 1
        elif label == "no" and pred == "no":
            tn += 1
        elif label == "no" and pred == "yes":
            fp += 1
        elif label == "yes" and pred == "no":
            fn += 1
        else:
            invalid += 1
            if label == "yes":
                fn += 1
            elif label == "no":
                fp += 1
    n = len(rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_rate": yes_pred / n if n else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid": invalid,
        "num_samples": n,
    }


def changed_metrics(rows: list[Mapping[str, Any]], baseline_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    wrong_to_right = right_to_wrong = changed_pred = 0
    for row in rows:
        base = baseline_by_id.get(str(row.get("id", "")))
        if not base:
            continue
        label = str(row.get("label", "")).lower()
        pred = str(row.get("pred", "")).lower()
        base_pred = str(base.get("pred", "")).lower()
        if pred != base_pred:
            changed_pred += 1
        if base_pred != label and pred == label:
            wrong_to_right += 1
        if base_pred == label and pred != label:
            right_to_wrong += 1
    return {"wrong_to_right": wrong_to_right, "right_to_wrong": right_to_wrong, "changed_pred": changed_pred}


def match_label(vector: str, subset: str) -> str:
    vector = str(vector)
    subset = str(subset)
    if vector.startswith("g_all"):
        return "global"
    if vector.startswith("g_cat") and subset.startswith("cat_"):
        return "type_matched"
    if vector.startswith("g_attr") and subset.startswith("attr_"):
        return "type_matched"
    if vector.startswith("g_rel") and subset.startswith("rel_"):
        return "type_matched"
    if f"d_{subset}_" in vector or vector.startswith(f"d_{subset}_"):
        return "subtype_matched"
    if subset.startswith("cat_") and "_cat_" in vector:
        return "type_matched"
    if subset.startswith("attr_") and "_attr_" in vector:
        return "type_matched"
    if subset.startswith("rel_") and "_rel_" in vector:
        return "type_matched"
    return "mismatched"


def md_table(headers: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def collect_raw_rows(eval_root: Path) -> list[dict[str, Any]]:
    patterns = ["shards/*/raw/*.jsonl", "raw/*.jsonl"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(eval_root.glob(pattern)))
    dedup: dict[tuple[str, str], Path] = {}
    for path in paths:
        dedup[(path.parent.parent.name, path.name)] = path
    rows: list[dict[str, Any]] = []
    for path in sorted(dedup.values()):
        rows.extend(read_jsonl(path))
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_subset: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("method") == "baseline":
            baseline_by_subset[str(row.get("subtype", row.get("setting", "")))][str(row.get("id", ""))] = row

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subset = str(row.get("subtype", row.get("setting", "")))
        method = str(row.get("method", ""))
        vector = str(row.get("vector", ""))
        alpha = "" if row.get("alpha") in ("", None) else str(row.get("alpha"))
        grouped[(subset, method, vector, alpha)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (subset, method, vector, alpha), group in sorted(grouped.items()):
        metrics = metric_rows(group)
        changed = changed_metrics(group, baseline_by_subset.get(subset, {}))
        summary_rows.append(
            {
                "eval_subset": subset,
                "method": method,
                "vector": vector,
                "match": "" if method == "baseline" else match_label(vector, subset),
                "alpha": alpha,
                **metrics,
                **changed,
            }
        )
    return summary_rows


def best_by_subset_vector(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("method") != "steered":
            continue
        key = (str(row.get("eval_subset")), str(row.get("vector")))
        if key not in best or float(row.get("f1", 0.0)) > float(best[key].get("f1", 0.0)):
            best[key] = row
    return [dict(row) for row in sorted(best.values(), key=lambda row: (str(row.get("eval_subset")), -float(row.get("f1", 0.0))))]


def best_by_subset_match(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("method") != "steered":
            continue
        key = (str(row.get("eval_subset")), str(row.get("match")))
        if key not in best or float(row.get("f1", 0.0)) > float(best[key].get("f1", 0.0)):
            best[key] = row
    return [dict(row) for row in sorted(best.values(), key=lambda row: (str(row.get("eval_subset")), str(row.get("match"))))]


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = [
        "eval_subset",
        "method",
        "vector",
        "match",
        "alpha",
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
        "wrong_to_right",
        "right_to_wrong",
        "changed_pred",
        "num_samples",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[Mapping[str, Any]], raw_count: int) -> None:
    headers = ["eval_subset", "vector", "match", "alpha", "accuracy", "f1", "yes_rate", "wrong_to_right", "right_to_wrong", "changed_pred", "num_samples"]
    lines = ["# Large Subtype Minimal-Pair Held-Out Eval", ""]
    lines.append(f"- Raw prediction rows: `{raw_count}`")
    lines.append("")
    lines.append("## Best By Subset And Match Type")
    lines.append(md_table(headers, best_by_subset_match(rows)))
    lines.append("")
    lines.append("## Best By Subset And Vector")
    lines.append(md_table(headers, best_by_subset_vector(rows)))
    lines.append("")
    lines.append("## All Rows")
    all_headers = ["eval_subset", "method", "vector", "match", "alpha", "accuracy", "precision", "recall", "f1", "yes_rate", "tp", "tn", "fp", "fn", "wrong_to_right", "right_to_wrong", "changed_pred", "num_samples"]
    lines.append(md_table(all_headers, rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    eval_root = resolve(args.eval_root)
    output_csv = resolve(args.output_csv) if str(args.output_csv).strip() else eval_root / "summary.csv"
    output_report = resolve(args.output_report) if str(args.output_report).strip() else eval_root / "SUMMARY.md"
    raw_rows = collect_raw_rows(eval_root)
    if not raw_rows:
        raise FileNotFoundError(f"No raw jsonl rows found under {eval_root}/shards/*/raw or {eval_root}/raw")
    summary_rows = summarize(raw_rows)
    write_csv(output_csv, summary_rows)
    write_report(output_report, summary_rows, raw_count=len(raw_rows))
    print(f"Wrote merged summary CSV to {output_csv}")
    print(f"Wrote merged report to {output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
