"""Precheck AMBER attribute yes/no data and coarse subtype coverage."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amber-root", default="data/benchmarks/amber_hallucination")
    parser.add_argument("--output", default="data/outputs_attr_sanity_mme/AMBER_ATTRIBUTE_SUBTYPE_PRECHECK.md")
    parser.add_argument("--examples-per-subtype", type=int, default=5)
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "data", "samples", "questions"):
            if isinstance(payload.get(key), list):
                return [row for row in payload[key] if isinstance(row, dict)]
    return []


def find_attribute_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        lowered = str(path).lower()
        if "attr" in lowered or "attribute" in lowered:
            files.append(path)
    if files:
        return files
    return [path for path in sorted(root.glob("*.jsonl")) if "all" not in path.name.lower()]


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return "invalid"


def question_text(row: dict[str, Any]) -> str:
    for key in ("question", "text", "query", "prompt"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def classify_subtype(question: str) -> str:
    lowered = question.lower()
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\bquantity\b|\bseveral\b|\bmany\b", lowered):
        return "count_number"
    if re.search(r"\bcolor\b|\bcolour\b|\bred\b|\bblue\b|\bgreen\b|\byellow\b|\bblack\b|\bwhite\b|\bbrown\b|\bgray\b|\bgrey\b|\borange\b|\bpink\b|\bpurple\b", lowered):
        return "color"
    if re.search(r"\bdoing\b|\bholding\b|\bwearing\b|\briding\b|\bsitting\b|\bstanding\b|\bwalking\b|\brunning\b|\beating\b|\bplaying\b", lowered):
        return "action"
    if re.search(r"\bopen\b|\bclosed\b|\bclean\b|\bdirty\b|\bfull\b|\bempty\b|\bwet\b|\bdry\b|\bold\b|\bnew\b|\bbroken\b", lowered):
        return "state"
    return "other_state"


def table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def fenced_json(payload: Any) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def main() -> int:
    args = parse_args()
    root = resolve_project_path(args.amber_root)
    output = resolve_project_path(args.output)
    files = find_attribute_files(root)
    rows: list[dict[str, Any]] = []
    file_counts: Counter[str] = Counter()
    for path in files:
        loaded = read_rows(path)
        rows.extend(loaded)
        file_counts[str(path)] = len(loaded)
    subtype_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question = question_text(row)
        subtype = str(row.get("subtype") or row.get("attribute_type") or "").strip().lower() or classify_subtype(question)
        label = normalize_label(row.get("label", row.get("answer")))
        subtype_counts[subtype] += 1
        label_counts[subtype][label] += 1
        if len(examples[subtype]) < int(args.examples_per_subtype):
            examples[subtype].append(
                {
                    "question": question,
                    "label": label,
                    "image": row.get("image") or row.get("image_path") or row.get("image_id") or "",
                }
            )
    overview_rows = [
        {
            "subtype": subtype,
            "n": subtype_counts[subtype],
            "yes": label_counts[subtype].get("yes", 0),
            "no": label_counts[subtype].get("no", 0),
            "invalid": label_counts[subtype].get("invalid", 0),
            "aligned_vector": "attr_count" if subtype == "count_number" else ("attr_color" if subtype == "color" else "none/current_attr_only"),
        }
        for subtype in sorted(subtype_counts)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AMBER Attribute Subtype Precheck",
        "",
        f"- AMBER root: `{root}`",
        f"- Attribute-ish files found: `{len(files)}`",
        f"- Total rows loaded: `{len(rows)}`",
        "",
        "## Source Files",
        "",
        table(["file", "rows"], [{"file": path, "rows": count} for path, count in sorted(file_counts.items())]) if file_counts else "No AMBER attribute files found.",
        "",
        "## Subtype Counts",
        "",
        table(["subtype", "n", "yes", "no", "invalid", "aligned_vector"], overview_rows) if overview_rows else "No rows available.",
    ]
    for subtype in sorted(examples):
        lines.extend(["", f"## Examples: {subtype}", "", fenced_json(examples[subtype])])
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `count_number` and `color` are the subtypes most aligned with current MME attr sanity tests.",
            "- `action`, `state`, and `other_state` likely need separate vectors or a broader attr vector before AMBER full runs.",
            "- This script is a data precheck only; it does not run AMBER inference.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote AMBER attribute subtype precheck to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
