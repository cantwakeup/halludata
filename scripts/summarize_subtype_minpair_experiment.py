#!/usr/bin/env python3
"""Assemble the subtype minimal-pair experiment report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/subtype_minpair_v1")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_text(path: Path, fallback: str = "_Not available yet._") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else fallback


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def md_table(headers: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = []
        for header in headers:
            value = row.get(header, "")
            try:
                vals.append(f"{float(value):.4f}" if str(value).strip() and header not in {"eval_subset", "method", "vector", "alpha"} else str(value))
            except Exception:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def best_eval_rows(rows: list[Mapping[str, str]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        if row.get("method") != "steered":
            continue
        key = (str(row.get("eval_subset")), str(row.get("vector")))
        if key not in best or float(row.get("f1", 0.0) or 0.0) > float(best[key].get("f1", 0.0) or 0.0):
            best[key] = row
    return [dict(row) for row in sorted(best.values(), key=lambda row: (str(row.get("eval_subset")), -float(row.get("f1", 0.0) or 0.0)))]


def matched_label(vector: str, subset: str) -> str:
    if "g_all" in vector:
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


def matched_matrix(rows: list[Mapping[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in best_eval_rows(rows):
        out.append(
            {
                "eval_subset": row.get("eval_subset", ""),
                "vector": row.get("vector", ""),
                "match": matched_label(str(row.get("vector", "")), str(row.get("eval_subset", ""))),
                "alpha": row.get("alpha", ""),
                "accuracy": row.get("accuracy", ""),
                "f1": row.get("f1", ""),
                "yes_rate": row.get("yes_rate", ""),
                "wrong_to_right": row.get("wrong_to_right", ""),
                "right_to_wrong": row.get("right_to_wrong", ""),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    output = resolve(args.output) if str(args.output).strip() else root / "REPORT.md"
    stats = read_json(root / "minimal_pairs" / "stats.json") or {}
    eval_rows = read_csv(root / "eval" / "heldout_sanity" / "summary.csv")
    shuffle_rows = read_csv(root / "eval" / "shuffle_control" / "summary.csv")
    raw_rows = read_csv(root / "eval" / "raw_clean_ablation" / "summary.csv")
    lines: list[str] = []
    lines.append("# Subtype-Aware Symmetric Minimal-Pair Experiment Report")
    lines.append("")
    lines.append("## 1. Repository/Data Inspection Summary")
    source_status = stats.get("source_status", {})
    lines.append("```json")
    lines.append(json.dumps(source_status, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## 2. Minimal-Pair Dataset Summary")
    lines.append(read_text(root / "minimal_pairs" / "DATA_REPORT.md"))
    lines.append("")
    lines.append("## 3. Activation Extraction Summary")
    manifest = read_json(root / "activations" / "train_activations.manifest.json") or {}
    lines.append("```json")
    lines.append(json.dumps(manifest, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## 4. Vector Construction Summary")
    lines.append(read_text(root / "vectors" / "VECTOR_REPORT.md"))
    lines.append("")
    lines.append("## 5. Evaluation Summary")
    if eval_rows:
        lines.append("### Held-Out Best Rows")
        lines.append(md_table(["eval_subset", "vector", "match", "alpha", "accuracy", "f1", "yes_rate", "wrong_to_right", "right_to_wrong"], matched_matrix(eval_rows)))
    else:
        lines.append("_Held-out eval summary not available yet._")
    lines.append("")
    lines.append("## 6. Controls")
    if shuffle_rows:
        lines.append("### Shuffle Subtype Control")
        lines.append(md_table(["eval_subset", "vector", "match", "alpha", "accuracy", "f1", "yes_rate", "wrong_to_right", "right_to_wrong"], matched_matrix(shuffle_rows)))
    else:
        lines.append("- Shuffle subtype control not available yet.")
    if raw_rows:
        lines.append("### Raw vs Yes/No-Clean Ablation")
        lines.append(md_table(["eval_subset", "vector", "match", "alpha", "accuracy", "f1", "yes_rate", "wrong_to_right", "right_to_wrong"], matched_matrix(raw_rows)))
    else:
        lines.append("- Raw vs clean ablation not available yet.")
    lines.append("")
    lines.append("## 7. Key Conclusion")
    if eval_rows:
        matched_rows = matched_matrix(eval_rows)
        matched = [row for row in matched_rows if row["match"] in {"subtype_matched", "type_matched"}]
        subtype_matched = [row for row in matched_rows if row["match"] == "subtype_matched"]
        mismatched = [row for row in matched_matrix(eval_rows) if row["match"] == "mismatched"]
        matched_f1 = max((float(row["f1"]) for row in matched), default=0.0)
        subtype_f1 = max((float(row["f1"]) for row in subtype_matched), default=0.0)
        mismatched_f1 = max((float(row["f1"]) for row in mismatched), default=0.0)
        if subtype_f1 > mismatched_f1:
            lines.append(f"- Best subtype-matched F1 ({subtype_f1:.4f}) is higher than best mismatched F1 ({mismatched_f1:.4f}); this supports subtype-specific signal.")
        elif matched_f1 > mismatched_f1:
            lines.append(f"- Best type/subtype-matched F1 ({matched_f1:.4f}) is higher than best mismatched F1 ({mismatched_f1:.4f}), but subtype-only F1 ({subtype_f1:.4f}) is not; this supports broad type signal more than subtype signal.")
        else:
            lines.append(f"- Best type/subtype-matched F1 ({matched_f1:.4f}) does not beat best mismatched F1 ({mismatched_f1:.4f}); subtype-specific experts are not established yet.")
    else:
        lines.append("- Eval has not been run yet, so no matched-vs-mismatched conclusion is available.")
    lines.append("")
    lines.append("## 8. Concrete Next Actions")
    lines.append("- If attr_color is the first clear matched winner, expand color data and run a larger GQA/MME-color sanity pass.")
    lines.append("- If attr_count is positive, run an MME-count limit sweep next.")
    lines.append("- If rel_spatial is positive, expand spatial relation data before touching rel_contact.")
    lines.append("- If rel_contact stays weak, inspect relation label quality and counterfact mappings before adding more samples.")
    lines.append("- If all subtype vectors are weak, revisit activation definition and hook timing rather than only tuning alpha/SVD.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote subtype experiment report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
