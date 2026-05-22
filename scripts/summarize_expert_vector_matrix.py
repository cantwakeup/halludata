#!/usr/bin/env python3
"""Summarize vector-only expert benchmark results into a diagonal matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORS = ("baseline", "global", "cat", "attr", "rel")
MAIN_COLUMNS = [
    ("POPE/category avg", "category", "pope"),
    ("AMBER-attribute", "attribute", "amber_attribute"),
    ("GQA/clean-relation", "relation", "relation"),
]
OPTIONAL_COLUMNS = [
    ("AMBER-existence", "category", "amber_existence"),
    ("H-POPE", "attribute", "hpope"),
    ("AMBER-relation", "relation", "amber_relation"),
    ("MME-position", "relation", "mme_position"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/expert_vector_full_eval_v1")
    parser.add_argument("--output", default="")
    parser.add_argument("--report-output", default="")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_summary_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/summary.csv")):
        for row in read_csv(path):
            row["_summary_path"] = str(path)
            rows.append(row)
    return rows


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value)
        if text == "":
            return default
        out = float(text)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def snum(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def matches_column(row: Mapping[str, Any], family: str, token: str) -> bool:
    benchmark_id = str(row.get("benchmark_id", "")).lower()
    benchmark_family = str(row.get("benchmark_family", "")).lower()
    if token == "pope":
        return benchmark_family == "category" and "pope" in benchmark_id
    if token == "relation":
        return benchmark_family == "relation" and ("gqa" in benchmark_id or "clean_relation" in benchmark_id or "relation" in benchmark_id)
    return benchmark_family == family and token in benchmark_id


def weighted_average(values: Iterable[tuple[float, float]]) -> float:
    pairs = list(values)
    denom = sum(weight for _value, weight in pairs)
    return sum(value * weight for value, weight in pairs) / denom if denom else 0.0


def best_for(rows: list[dict[str, Any]], family: str, token: str, vector: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if str(row.get("vector")) == vector and matches_column(row, family, token)]
    if not candidates:
        return None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        alpha = str(row.get("alpha", ""))
        grouped[alpha].append(row)
    best: dict[str, Any] | None = None
    for alpha, group in grouped.items():
        total_n = sum(fnum(row.get("num_samples")) for row in group)
        agg = {
            "vector": vector,
            "alpha": alpha,
            "f1": weighted_average((fnum(row.get("f1")), fnum(row.get("num_samples"))) for row in group),
            "accuracy": weighted_average((fnum(row.get("accuracy")), fnum(row.get("num_samples"))) for row in group),
            "yes_rate": weighted_average((fnum(row.get("yes_rate")), fnum(row.get("num_samples"))) for row in group),
            "wrong_to_right": sum(int(fnum(row.get("wrong_to_right"))) for row in group),
            "right_to_wrong": sum(int(fnum(row.get("right_to_wrong"))) for row in group),
            "changed_pred": sum(int(fnum(row.get("changed_pred"))) for row in group),
            "num_samples": int(total_n),
            "benchmark_ids": ",".join(sorted({str(row.get("benchmark_id")) for row in group})),
            "datasets": ",".join(sorted({str(row.get("dataset")) for row in group})),
            "settings": ",".join(sorted({str(row.get("setting")) for row in group})),
        }
        if best is None or (agg["f1"], agg["accuracy"]) > (best["f1"], best["accuracy"]):
            best = agg
    return best


def cell_text(best: dict[str, Any] | None, baseline: dict[str, Any] | None) -> str:
    if best is None:
        return "unavailable"
    delta = best["f1"] - (baseline["f1"] if baseline else 0.0)
    alpha = best.get("alpha", "")
    alpha_text = "baseline" if str(alpha) == "" else str(alpha)
    flag = " !" if best["yes_rate"] > 0.65 or best["yes_rate"] < 0.35 else ""
    return (
        f"F1={best['f1']:.4f}, Acc={best['accuracy']:.4f}, "
        f"a={alpha_text}, yes={best['yes_rate']:.3f}{flag}, "
        f"dF1={delta:+.4f}, W2R/R2W={best['wrong_to_right']}/{best['right_to_wrong']}"
    )


def table(headers: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def matrix_rows(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any] | None]]:
    cell_map: dict[tuple[str, str], dict[str, Any] | None] = {}
    rendered: list[dict[str, Any]] = []
    baselines = {label: best_for(rows, family, token, "baseline") for label, family, token in columns}
    for vector in VECTORS:
        line = {"vector": vector}
        for label, family, token in columns:
            best = best_for(rows, family, token, vector)
            cell_map[(vector, label)] = best
            line[label] = cell_text(best, baselines[label])
        rendered.append(line)
    return rendered, cell_map


def best_vector_for_column(cell_map: Mapping[tuple[str, str], dict[str, Any] | None], label: str) -> str:
    candidates = []
    for vector in ("global", "cat", "attr", "rel"):
        item = cell_map.get((vector, label))
        if item:
            candidates.append((float(item["f1"]), float(item["accuracy"]), vector))
    if not candidates:
        return "unavailable"
    candidates.sort(reverse=True)
    return candidates[0][2]


def write_best_changed_cases(root: Path, rows: list[dict[str, Any]], cell_map: Mapping[tuple[str, str], dict[str, Any] | None]) -> list[str]:
    changed_root = root / "changed_cases"
    changed_root.mkdir(parents=True, exist_ok=True)
    changed_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("*/changed_cases_all.jsonl")):
        for row in read_jsonl(path):
            changed_by_benchmark[str(row.get("benchmark"))].append(row)
    written: list[str] = []
    for (vector, label), best in cell_map.items():
        if vector == "baseline" or best is None:
            continue
        benchmark_ids = [item for item in str(best.get("benchmark_ids", "")).split(",") if item]
        alpha = str(best.get("alpha", ""))
        out_rows = []
        for benchmark_id in benchmark_ids:
            for row in changed_by_benchmark.get(benchmark_id, []):
                if str(row.get("vector")) == vector and str(row.get("alpha")) == alpha:
                    out_rows.append(row)
        if not out_rows:
            continue
        out_path = changed_root / f"{safe_name(label)}_{vector}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in out_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written.append(str(out_path))
    return written


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value).strip("_")


def interpretation(main_cells: Mapping[tuple[str, str], dict[str, Any] | None]) -> tuple[str, list[str]]:
    winners = {label: best_vector_for_column(main_cells, label) for label, _family, _token in MAIN_COLUMNS}
    lines = [
        f"- POPE/category winner: `{winners['POPE/category avg']}`.",
        f"- AMBER-attribute winner: `{winners['AMBER-attribute']}`.",
        f"- GQA/clean-relation winner: `{winners['GQA/clean-relation']}`.",
    ]
    cat_ok = winners["POPE/category avg"] == "cat"
    attr_ok = winners["AMBER-attribute"] == "attr"
    rel_ok = winners["GQA/clean-relation"] == "rel"
    global_wins = sum(1 for winner in winners.values() if winner == "global")
    diagonal = sum([cat_ok, attr_ok, rel_ok])
    for label, expected in [
        ("POPE/category avg", "cat"),
        ("AMBER-attribute", "attr"),
        ("GQA/clean-relation", "rel"),
    ]:
        item = main_cells.get((expected, label))
        if item and (item["yes_rate"] > 0.65 or item["yes_rate"] < 0.35):
            lines.append(f"- `{expected}` on `{label}` is suspicious due to yes_rate={item['yes_rate']:.3f}.")
    if diagonal == 3:
        return "PASS_DIAGONAL", lines + ["- Clear diagonal advantage: proceed to router/mask/token-level stage only after manual changed-case review."]
    if diagonal in {1, 2}:
        return "PARTIAL", lines + ["- Partial diagonal: keep only established experts and revisit data/vector definitions for the others."]
    if global_wins >= 2:
        return "GLOBAL_ONLY", lines + ["- Global/shared factual direction dominates; separate expert vectors are not justified yet."]
    return "FAIL", lines + ["- Off-diagonal or weak results dominate; do not proceed to router/DPO from these vectors."]


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    output = resolve(args.output) if args.output else root / "EXPERT_MATRIX_REPORT.md"
    report_output = resolve(args.report_output) if args.report_output else root / "REPORT.md"
    rows = collect_summary_rows(root)
    main_matrix, main_cells = matrix_rows(rows, MAIN_COLUMNS)
    optional_matrix, optional_cells = matrix_rows(rows, OPTIONAL_COLUMNS)
    changed_paths = write_best_changed_cases(root, rows, {**main_cells, **optional_cells})
    decision, conclusion_lines = interpretation(main_cells)

    best_rows = []
    for label, family, token in [*MAIN_COLUMNS, *OPTIONAL_COLUMNS]:
        for vector in VECTORS:
            best = best_for(rows, family, token, vector)
            if best:
                best_rows.append({"benchmark": label, **best})

    lines = [
        "# Expert Vector Matrix Report",
        "",
        f"- Summary rows loaded: `{len(rows)}`",
        f"- Decision: `{decision}`",
        "",
        "## Main 3x3 Matrix",
        "",
        table(["vector"] + [label for label, _family, _token in MAIN_COLUMNS], main_matrix),
        "",
        "## Optional Matrix",
        "",
        table(["vector"] + [label for label, _family, _token in OPTIONAL_COLUMNS], optional_matrix),
        "",
        "## Best Rows",
        "",
        table(
            [
                "benchmark",
                "vector",
                "alpha",
                "f1",
                "accuracy",
                "yes_rate",
                "wrong_to_right",
                "right_to_wrong",
                "changed_pred",
                "num_samples",
            ],
            best_rows,
        )
        if best_rows
        else "No completed summary rows found.",
        "",
        "## Automatic Interpretation",
        "",
        *conclusion_lines,
        "",
        "## Changed Cases",
        "",
        *(f"- `{path}`" for path in changed_paths),
        "",
        "## Notes",
        "",
        "- `!` beside yes-rate marks a balanced yes/no benchmark suspicious zone (>0.65 or <0.35).",
        "- POPE/category is averaged across completed POPE dataset/setting groups for the same vector/alpha.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Expert Vector Full Eval V1",
        "",
        "## Goal",
        "",
        "Simplified vector-only full benchmark evaluation. Subtype masks, expert masks, and routing are intentionally disabled.",
        "",
        "## Evaluation Settings",
        "",
        "- Direction: vector itself (`global`, `cat`, `attr`, `rel`).",
        "- Head selection: vector norm top64 over all available 32 layers.",
        "- Hook: official LLaVA decoder self-attention `o_proj` forward pre-hook.",
        "- Apply: prefill=true, decode=true, apply_to=last_token.",
        "- Decoding: do_sample=true, temperature=1.0, top_p=1.0, num_beams=1, max_new_tokens=1024, seed=42.",
        "",
        "## Matrix",
        "",
        f"See `{output}`.",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        *conclusion_lines,
        "",
        "## Changed Cases",
        "",
        *(f"- `{path}`" for path in changed_paths),
    ]
    report_output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote expert matrix report to {output}")
    print(f"Wrote final report to {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
