"""Summarize COCO/GQA/mixed cat-vector POPE comparison runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SOURCES = ("coco_cat", "gqa_cat", "mixed_cat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default="data/pope_cat_expert_eval/cat_domain_vector_comparison/runs",
        help="Directory containing one subdirectory per source.",
    )
    parser.add_argument("--sources", default=" ".join(DEFAULT_SOURCES))
    parser.add_argument(
        "--output",
        default="data/pope_cat_expert_eval/cat_domain_vector_comparison/summary.csv",
    )
    parser.add_argument(
        "--report-output",
        default="data/pope_cat_expert_eval/cat_domain_vector_comparison/SUMMARY.md",
    )
    return parser.parse_args()


def f(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def method_name(row: Mapping[str, Any]) -> str:
    return str(row.get("Method", row.get("method", ""))).strip()


def dataset_name(row: Mapping[str, Any]) -> str:
    return str(row.get("Dataset", row.get("dataset", ""))).strip()


def setting_name(row: Mapping[str, Any]) -> str:
    return str(row.get("Setting", row.get("setting", ""))).strip()


def alpha_value(row: Mapping[str, Any]) -> str:
    return str(row.get("Alpha", row.get("alpha", ""))).strip()


def metric(row: Mapping[str, Any], name: str) -> float:
    aliases = {
        "F1": ("F1 Score", "F1", "f1"),
        "Accuracy": ("Accuracy", "Acc", "accuracy"),
        "Precision": ("Precision", "precision"),
        "Recall": ("Recall", "recall"),
        "Yes Rate": ("Yes Rate", "yes_rate"),
    }
    for key in aliases.get(name, (name,)):
        if key in row:
            return f(row.get(key))
    return 0.0


def count_metric(row: Mapping[str, Any], name: str) -> int:
    return int(round(f(row.get(name, row.get(name.lower(), 0)))))


def best_cat_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    cat_rows = [row for row in rows if method_name(row).lower() == "catexpert"]
    if not cat_rows:
        return None
    return max(cat_rows, key=lambda row: (metric(row, "F1"), metric(row, "Accuracy"), -f(alpha_value(row), 999.0)))


def regular_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    regular_rows = [row for row in rows if method_name(row).lower() == "regular"]
    return regular_rows[0] if regular_rows else None


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((dataset_name(row), setting_name(row)), []).append(row)
    return grouped


def markdown_table(headers: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    sources = [source for source in str(args.sources).replace(",", " ").split() if source.strip()]
    output_path = Path(args.output)
    report_path = Path(args.report_output)

    summary_rows: list[dict[str, Any]] = []
    all_rows_by_source: dict[str, list[dict[str, str]]] = {}
    for source in sources:
        summary_csv = run_root / source / "summary.csv"
        rows = read_csv(summary_csv)
        all_rows_by_source[source] = rows
        grouped = group_rows(rows)
        for (dataset, setting), group in sorted(grouped.items()):
            base = regular_row(group)
            best = best_cat_row(group)
            if best is None:
                continue
            base_acc = metric(base, "Accuracy") if base else 0.0
            base_f1 = metric(base, "F1") if base else 0.0
            base_fp = count_metric(base, "FP") if base else 0
            base_yes = metric(base, "Yes Rate") if base else 0.0
            summary_rows.append(
                {
                    "Source": source,
                    "Dataset": dataset,
                    "Setting": setting,
                    "Best Alpha": alpha_value(best),
                    "Baseline Acc": base_acc,
                    "Best Acc": metric(best, "Accuracy"),
                    "Delta Acc": metric(best, "Accuracy") - base_acc,
                    "Baseline F1": base_f1,
                    "Best F1": metric(best, "F1"),
                    "Delta F1": metric(best, "F1") - base_f1,
                    "Baseline FP": base_fp,
                    "Best FP": count_metric(best, "FP"),
                    "Delta FP": count_metric(best, "FP") - base_fp,
                    "Baseline Yes Rate": base_yes,
                    "Best Yes Rate": metric(best, "Yes Rate"),
                    "Delta Yes Rate": metric(best, "Yes Rate") - base_yes,
                    "N": count_metric(best, "N"),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Source",
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
        "Delta FP",
        "Baseline Yes Rate",
        "Best Yes Rate",
        "Delta Yes Rate",
        "N",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    by_dataset_rows = sorted(summary_rows, key=lambda row: (str(row["Dataset"]), str(row["Setting"]), str(row["Source"])))
    best_source_rows = []
    for key in sorted({(row["Dataset"], row["Setting"]) for row in summary_rows}):
        candidates = [row for row in summary_rows if (row["Dataset"], row["Setting"]) == key]
        if candidates:
            best_source_rows.append(max(candidates, key=lambda row: (float(row["Best F1"]), float(row["Best Acc"]))))

    missing_sources = [source for source, rows in all_rows_by_source.items() if not rows]
    lines = [
        "# Cat Domain Vector Comparison",
        "",
        f"- Run root: `{run_root}`",
        f"- Combined CSV: `{output_path}`",
        f"- Sources: `{', '.join(sources)}`",
        "",
    ]
    if missing_sources:
        lines.extend(["## Missing Runs", "", "- " + ", ".join(missing_sources), ""])
    lines.extend(
        [
            "## Best Source Per Dataset/Setting",
            "",
            markdown_table(fieldnames, best_source_rows),
            "",
            "## Full Source Comparison",
            "",
            markdown_table(fieldnames, by_dataset_rows),
            "",
            "## Reading Guide",
            "",
            "- If `coco_cat` mostly improves MSCOCO and not GQA, it is domain-specific.",
            "- If `gqa_cat` mostly improves GQA and weakens MSCOCO, it is domain-specific in the other direction.",
            "- If `mixed_cat` is not always best but stays positive/stable on both, it is the safer shared category direction.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote combined summary to {output_path}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
