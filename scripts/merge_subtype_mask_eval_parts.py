#!/usr/bin/env python3
"""Merge parallel subtype mask eval part outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts-root", required=True)
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    parts_root = Path(args.parts_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(parts_root.glob("part*/summary.csv"))
    if not summary_paths:
        raise FileNotFoundError(f"No part summary files found under {parts_root}/part*/summary.csv")

    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    for path in summary_paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not fieldnames:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))

    summary_out = output_dir / "summary.csv"
    with summary_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    changed_out = output_dir / "changed_cases.jsonl"
    with changed_out.open("w", encoding="utf-8") as out:
        for path in sorted(parts_root.glob("part*/changed_cases.jsonl")):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)

    print(f"Merged {len(summary_paths)} parts into {summary_out}")
    print(f"Wrote changed cases to {changed_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
