#!/usr/bin/env python3
"""Assemble the final subtype mask steering report from generated artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/subtype_mask_steering_v1")
    ap.add_argument("--output", default="")
    return ap.parse_args()


def read_if_exists(path: Path, max_chars: int | None = None) -> str:
    if not path.exists():
        return f"_Missing artifact: `{path}`_\n"
    text = path.read_text(encoding="utf-8")
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "\n\n_Section truncated in aggregate report; open the source artifact for full details._\n"
    return text


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "REPORT.md"

    inspect = root / "INSPECT.md"
    mask_report = root / "masks" / "MASK_REPORT.md"
    eval_report = root / "eval" / "heldout" / "MASK_EVAL_REPORT.md"
    quality = root / "DATA_QUALITY_NOTES.md"

    lines: List[str] = []
    lines.append("# Type/Subtype-Specific Mask Steering Report")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append("This experiment tests whether expert separation is better expressed as type/subtype-specific intervention masks rather than distinct steering directions.")
    lines.append("")
    lines.append("The intervention is:")
    lines.append("")
    lines.append("```text")
    lines.append("h[l,h] <- h[l,h] + alpha * M_subtype[l,h] * g_type[l,h]")
    lines.append("```")
    lines.append("")
    lines.append("- Direction: `g_all_clean` or `g_type_clean` from subtype minimal-pair vectors.")
    lines.append("- Mask: top64 heads over all 32 layers, selected by subtype sample-level `s_delta = z_fact_text - z_counterfact_text` energy.")
    lines.append("- Loader: official LLaVA only; no old HF runner.")
    lines.append("")

    lines.append(section("Inspection Summary", read_if_exists(inspect, max_chars=12000)))
    lines.append(section("Mask Construction", read_if_exists(mask_report, max_chars=20000)))
    lines.append(section("Held-Out Eval", read_if_exists(eval_report, max_chars=22000)))
    lines.append(section("Data Quality Notes", read_if_exists(quality, max_chars=12000)))

    lines.append("## Final Decision Guide")
    lines.append("")
    lines.append("- If matched subtype masks beat mismatched and random masks without abnormal yes-rate drift, move toward token-level router / DPO routing.")
    lines.append("- If only category masks work, category is ready but attribute/relation data likely needs repair.")
    lines.append("- If attr_count/contact work but color/spatial do not, use finer value/predicate-level masks.")
    lines.append("- If all matched masks fail against random/mismatched, do not build a router yet; revisit data construction or activation definition.")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote final subtype mask report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
