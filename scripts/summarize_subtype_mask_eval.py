#!/usr/bin/env python3
"""Summarize subtype mask steering held-out evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SUBTYPE_TO_TYPE = {
    "cat_random": "cat",
    "cat_popular": "cat",
    "cat_hard": "cat",
    "attr_color": "attr",
    "attr_count": "attr",
    "rel_spatial": "rel",
    "rel_contact": "rel",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def read_csv(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def f(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def classify(row: Mapping[str, Any]) -> str:
    if row.get("method") == "baseline":
        return "baseline"
    match_type = str(row.get("match_type", ""))
    if match_type:
        return match_type
    mask_key = str(row.get("mask_key", ""))
    subset = str(row.get("eval_subset", ""))
    typ = SUBTYPE_TO_TYPE.get(subset, "")
    if mask_key.startswith("random_mask") or mask_key.startswith("layer_matched_random"):
        return "random_mask"
    if mask_key == "mask_g_all_norm_top64":
        return "g_all_baseline"
    if typ and mask_key == f"mask_g_{typ}_norm_top64":
        return "g_type_baseline"
    if mask_key == f"mask_s_{subset}_energy_top64":
        return "matched_energy"
    if mask_key.startswith("mask_s_") and mask_key.endswith("_energy_top64"):
        return "mismatched_energy"
    return "other"


def best(rows: Sequence[Mapping[str, Any]], metric: str = "f1") -> Dict[str, Any] | None:
    candidates = [dict(r) for r in rows if r.get("method") != "baseline"]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (f(r, metric), f(r, "accuracy")))


def md_table(headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def compact_row(row: Mapping[str, Any] | None, label: str = "") -> Dict[str, Any]:
    if row is None:
        return {"group": label}
    return {
        "group": label or classify(row),
        "eval_subset": row.get("eval_subset", ""),
        "direction_key": row.get("direction_key", ""),
        "mask_key": row.get("mask_key", ""),
        "alpha": row.get("alpha", ""),
        "accuracy": f(row, "accuracy"),
        "f1": f(row, "f1"),
        "yes_rate": f(row, "yes_rate"),
        "wrong_to_right": row.get("wrong_to_right", ""),
        "right_to_wrong": row.get("right_to_wrong", ""),
        "changed_pred": row.get("changed_pred", ""),
        "num_samples": row.get("num_samples", ""),
    }


def suspicious_yes_rate(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    yes_rate = f(row, "yes_rate")
    return yes_rate > 0.65 or yes_rate < 0.35


def main() -> int:
    args = parse_args()
    rows = read_csv(args.summary_csv)
    for row in rows:
        row["group"] = classify(row)
    by_subset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subset = str(row.get("eval_subset", ""))
        if subset:
            by_subset[subset].append(row)

    report_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    conclusions: List[str] = []

    for subset in sorted(by_subset):
        subset_rows = by_subset[subset]
        baseline = next((r for r in subset_rows if r.get("method") == "baseline"), None)
        matched = best([r for r in subset_rows if classify(r) in {"matched_energy", "matched_energy_g_all"}])
        matched_type_only = best([r for r in subset_rows if classify(r) == "matched_energy"])
        mismatched = best([r for r in subset_rows if classify(r) == "mismatched_energy"])
        random_row = best([r for r in subset_rows if classify(r) == "random_mask"])
        g_type = best([r for r in subset_rows if classify(r) == "g_type_baseline"])
        g_all = best([r for r in subset_rows if classify(r) == "g_all_baseline"])
        s_ablation = best([r for r in subset_rows if classify(r) == "s_direction_ablation"])

        matched_adv_mismatch_f1 = f(matched, "f1") - f(mismatched, "f1") if matched and mismatched else 0.0
        matched_adv_random_f1 = f(matched, "f1") - f(random_row, "f1") if matched and random_row else 0.0
        matched_adv_gtype_f1 = f(matched, "f1") - f(g_type, "f1") if matched and g_type else 0.0
        success = bool(
            matched
            and (mismatched is None or matched_adv_mismatch_f1 > 0)
            and (random_row is None or matched_adv_random_f1 > 0)
            and (g_type is None or matched_adv_gtype_f1 >= -0.002)
            and not suspicious_yes_rate(matched)
        )
        report_rows.append(
            {
                "eval_subset": subset,
                "baseline_f1": f(baseline, "f1"),
                "matched_f1": f(matched, "f1"),
                "mismatched_f1": f(mismatched, "f1"),
                "random_f1": f(random_row, "f1"),
                "g_type_f1": f(g_type, "f1"),
                "g_all_f1": f(g_all, "f1"),
                "matched_minus_mismatch_f1": matched_adv_mismatch_f1,
                "matched_minus_random_f1": matched_adv_random_f1,
                "matched_minus_gtype_f1": matched_adv_gtype_f1,
                "matched_yes_rate": f(matched, "yes_rate"),
                "success": "yes" if success else "no",
            }
        )
        for label, row in [
            ("baseline", baseline),
            ("best_matched", matched),
            ("best_matched_g_type_direction", matched_type_only),
            ("best_mismatched", mismatched),
            ("best_random", random_row),
            ("best_g_type_baseline", g_type),
            ("best_g_all_baseline", g_all),
            ("best_s_direction_ablation", s_ablation),
        ]:
            detail_rows.append(compact_row(row, label))

        if success:
            conclusions.append(f"- `{subset}`: matched subtype mask passes the current success criteria.")
        elif matched is None:
            conclusions.append(f"- `{subset}`: no matched mask result was found.")
        elif suspicious_yes_rate(matched):
            conclusions.append(
                f"- `{subset}`: matched mask is suspicious because yes_rate={f(matched, 'yes_rate'):.2f} on a balanced set."
            )
        elif mismatched and f(matched, "f1") <= f(mismatched, "f1"):
            conclusions.append(
                f"- `{subset}`: matched mask does not beat mismatched masks; subtype selectivity is not established."
            )
        elif random_row and f(matched, "f1") <= f(random_row, "f1"):
            conclusions.append(
                f"- `{subset}`: matched mask does not beat random masks; mask localization is not yet convincing."
            )
        elif g_type and f(matched, "f1") < f(g_type, "f1") - 0.002:
            conclusions.append(
                f"- `{subset}`: matched mask trails the g_type norm baseline; the subtype mask is not adding value yet."
            )
        else:
            conclusions.append(f"- `{subset}`: mixed signal; inspect changed cases before promoting this mask.")

    best_rows = []
    for subset in sorted(by_subset):
        row = best(by_subset[subset])
        if row:
            best_rows.append(compact_row(row, "best_any"))

    lines: List[str] = []
    lines.append("# Subtype Mask Steering Eval Report")
    lines.append("")
    lines.append(f"- Summary CSV: `{args.summary_csv}`")
    lines.append("")
    lines.append("## Best Rows By Eval Subset")
    lines.append("")
    lines.append(
        md_table(
            [
                "group",
                "eval_subset",
                "direction_key",
                "mask_key",
                "alpha",
                "accuracy",
                "f1",
                "yes_rate",
                "wrong_to_right",
                "right_to_wrong",
                "changed_pred",
                "num_samples",
            ],
            best_rows,
        )
    )
    lines.append("")

    lines.append("## Matched Advantage")
    lines.append("")
    lines.append(
        md_table(
            [
                "eval_subset",
                "baseline_f1",
                "matched_f1",
                "mismatched_f1",
                "random_f1",
                "g_type_f1",
                "g_all_f1",
                "matched_minus_mismatch_f1",
                "matched_minus_random_f1",
                "matched_minus_gtype_f1",
                "matched_yes_rate",
                "success",
            ],
            report_rows,
        )
    )
    lines.append("")

    lines.append("## Comparison Details")
    lines.append("")
    lines.append(
        md_table(
            [
                "group",
                "eval_subset",
                "direction_key",
                "mask_key",
                "alpha",
                "accuracy",
                "f1",
                "yes_rate",
                "wrong_to_right",
                "right_to_wrong",
                "changed_pred",
                "num_samples",
            ],
            detail_rows,
        )
    )
    lines.append("")

    lines.append("## Yes-Rate Flags")
    lines.append("")
    flag_rows = []
    for row in rows:
        if row.get("method") == "steered" and suspicious_yes_rate(row):
            flag_rows.append(compact_row(row, classify(row)))
    if flag_rows:
        lines.append(
            md_table(
                [
                    "group",
                    "eval_subset",
                    "direction_key",
                    "mask_key",
                    "alpha",
                    "accuracy",
                    "f1",
                    "yes_rate",
                    "wrong_to_right",
                    "right_to_wrong",
                    "changed_pred",
                    "num_samples",
                ],
                flag_rows[:80],
            )
        )
    else:
        lines.append("No steered rows crossed yes_rate < 0.35 or > 0.65.")
    lines.append("")

    lines.append("## Automatic Conclusion")
    lines.append("")
    lines.extend(conclusions)
    lines.append("")
    lines.append("## Success Criteria")
    lines.append("")
    lines.append("- A subtype mask is considered established only if it beats mismatched masks, random masks, and is competitive with the g_type baseline without abnormal yes-rate drift.")
    lines.append("- If matched masks do not beat random masks, this argues against moving to router/DPO yet.")
    lines.append("- If only category masks work, category is ready while attr/rel likely need cleaner data or value-level masks.")
    lines.append("")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote subtype mask eval report to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
