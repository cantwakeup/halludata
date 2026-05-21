#!/usr/bin/env python3
"""Write lightweight data-quality diagnostics for subtype mask steering."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


STUFF_OR_PART = {
    "ground",
    "sky",
    "wall",
    "grass",
    "beak",
    "ear",
    "hair",
    "leaf",
    "leaves",
    "seafood",
    "letters",
    "letter",
    "leg",
    "legs",
    "hand",
    "hands",
    "tail",
    "wing",
    "wings",
    "floor",
    "ceiling",
    "road",
    "street",
    "water",
    "snow",
    "sand",
    "post",
    "pole",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def subtype(row: Mapping[str, Any]) -> str:
    if row.get("subtype"):
        return str(row["subtype"])
    meta = row.get("metadata")
    if isinstance(meta, Mapping) and meta.get("subtype"):
        return str(meta["subtype"])
    return ""


def meta_value(row: Mapping[str, Any], *keys: str) -> str:
    meta = row.get("metadata")
    for key in keys:
        if row.get(key):
            return str(row[key])
        if isinstance(meta, Mapping) and meta.get(key):
            return str(meta[key])
    return ""


def md_table(headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def top_rows(counter: Counter[str], n: int = 20) -> List[Dict[str, Any]]:
    return [{"item": key, "count": value} for key, value in counter.most_common(n)]


def object_tokens(row: Mapping[str, Any]) -> List[str]:
    values = [
        meta_value(row, "object", "object_name", "subject", "target_object"),
        meta_value(row, "subject_name", "object_a", "object_b"),
        str(row.get("target_fact", "")),
        str(row.get("question", "")),
    ]
    tokens: List[str] = []
    for value in values:
        tokens.extend(re.findall(r"[a-zA-Z][a-zA-Z_-]+", value.lower()))
    return tokens


def count_word_distribution(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        text = " ".join([str(row.get("question", "")), str(row.get("target_fact", ""))]).lower()
        for word in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]:
            if re.search(rf"\b{word}\b", text):
                counts[word] += 1
    return counts


def predicate(row: Mapping[str, Any]) -> str:
    value = meta_value(row, "predicate", "relation", "rel", "relation_label")
    if value:
        return value.lower()
    text = " ".join([str(row.get("target_fact", "")), str(row.get("question", ""))]).lower()
    for pred in [
        "sitting on",
        "standing on",
        "standing next to",
        "riding",
        "holding",
        "wearing",
        "carrying",
        "eating",
        "touching",
        "lying on",
        "leaning on",
        "left of",
        "right of",
        "above",
        "below",
        "in front of",
        "behind",
    ]:
        if pred in text:
            return pred
    return ""


def grammar_warnings(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    warnings: Counter[str] = Counter()
    patterns = {
        "are_there_one": r"\bare there one\b",
        "watchs": r"\bwatchs\b",
        "womans": r"\bwomans\b",
        "feets": r"\bfeets\b",
        "childs": r"\bchilds\b",
        "mans": r"\bmans\b",
    }
    for row in rows:
        text = " ".join([str(row.get("question", "")), str(row.get("target_fact", "")), str(row.get("target_counterfact", ""))]).lower()
        for label, pattern in patterns.items():
            if re.search(pattern, text):
                warnings[label] += 1
    return warnings


def subtype_rows(rows: Sequence[Mapping[str, Any]], name: str) -> List[Mapping[str, Any]]:
    return [row for row in rows if subtype(row) == name]


def part_stuff_ratio(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = 0
    flagged = 0
    examples: List[str] = []
    for row in rows:
        toks = set(object_tokens(row))
        hit = sorted(toks & STUFF_OR_PART)
        if toks:
            total += 1
        if hit:
            flagged += 1
            if len(examples) < 10:
                examples.append(f"{row.get('id', '')}: {', '.join(hit)}")
    return {
        "total": total,
        "flagged": flagged,
        "ratio": f"{(flagged / total if total else 0.0):.4f}",
        "examples": examples,
    }


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.train_jsonl) + load_jsonl(args.val_jsonl)

    attr_count = subtype_rows(rows, "attr_count")
    attr_color = subtype_rows(rows, "attr_color")
    rel_spatial = subtype_rows(rows, "rel_spatial")
    rel_contact = subtype_rows(rows, "rel_contact")

    color_objects = Counter()
    for row in attr_color:
        obj = meta_value(row, "object", "object_name", "target_object")
        if not obj:
            obj = " ".join(object_tokens(row)[:2])
        color_objects[obj.lower()] += 1

    contact_predicates = Counter(predicate(row) for row in rel_contact)
    spatial_predicates = Counter(predicate(row) for row in rel_spatial)
    contact_predicates.pop("", None)
    spatial_predicates.pop("", None)

    rel_contact_examples = []
    for row in rel_contact:
        text = f"{row.get('target_fact', '')} {row.get('target_counterfact', '')}".lower()
        if "wearing umbrella" in text or "wearing plate" in text or "wearing table" in text:
            rel_contact_examples.append(
                {
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "target_fact": row.get("target_fact", ""),
                    "target_counterfact": row.get("target_counterfact", ""),
                }
            )
            if len(rel_contact_examples) >= 10:
                break

    attr_color_part = part_stuff_ratio(attr_color)
    rel_spatial_part = part_stuff_ratio(rel_spatial)

    lines: List[str] = []
    lines.append("# Subtype Mask Data Quality Notes")
    lines.append("")
    lines.append(f"- Train JSONL: `{args.train_jsonl}`")
    lines.append(f"- Val JSONL: `{args.val_jsonl}`")
    lines.append("")

    lines.append("## Attr Count")
    lines.append("")
    lines.append("### Grammar Warnings")
    lines.append(md_table(["item", "count"], top_rows(grammar_warnings(attr_count), 20)))
    lines.append("")
    lines.append("### Count Word Distribution")
    lines.append(md_table(["item", "count"], top_rows(count_word_distribution(attr_count), 20)))
    lines.append("")

    lines.append("## Attr Color")
    lines.append("")
    lines.append("### Object Distribution")
    lines.append(md_table(["item", "count"], top_rows(color_objects, 20)))
    lines.append("")
    lines.append("### Part/Stuff-Like Object Ratio")
    lines.append(md_table(["total", "flagged", "ratio", "examples"], [attr_color_part]))
    lines.append("")

    lines.append("## Rel Spatial")
    lines.append("")
    lines.append("### Predicate Distribution")
    lines.append(md_table(["item", "count"], top_rows(spatial_predicates, 20)))
    lines.append("")
    lines.append("### Part/Stuff-Like Object Ratio")
    lines.append(md_table(["total", "flagged", "ratio", "examples"], [rel_spatial_part]))
    lines.append("")

    lines.append("## Rel Contact")
    lines.append("")
    lines.append("### Predicate Distribution")
    lines.append(md_table(["item", "count"], top_rows(contact_predicates, 20)))
    lines.append("")
    wearing = contact_predicates.get("wearing", 0)
    total_contact = sum(contact_predicates.values())
    lines.append(f"- Wearing count: `{wearing}` / `{total_contact}` ({wearing / total_contact if total_contact else 0.0:.4f})")
    lines.append("")
    lines.append("### Potentially Unnatural Counterfacts")
    if rel_contact_examples:
        lines.append(md_table(["id", "question", "target_fact", "target_counterfact"], rel_contact_examples))
    else:
        lines.append("No simple `wearing umbrella/plate/table` patterns found.")
    lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote data-quality notes to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
