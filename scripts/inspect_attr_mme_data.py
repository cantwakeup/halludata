"""Inspect prepared MME count/color hallucination subsets for attr sanity eval."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ("existence", "count", "color", "position")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default="data/benchmarks/mme_hallucination")
    parser.add_argument("--image-root", default="", help="Defaults to <bench-root>/images.")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--output", default="data/outputs_attr_sanity_mme/ATTR_MME_DATA_REPORT.md")
    parser.add_argument("--examples-per-category", type=int, default=5)
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object on line {line_number} of {path}")
                rows.append(payload)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    raise ValueError(f"Unsupported JSON shape: {path}")


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return "invalid"


def image_path_for(row: dict[str, Any], image_root: Path) -> Path:
    raw = str(row.get("image_path") or row.get("image") or row.get("image_id") or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return image_root / path


def parser_risks(question: str) -> list[str]:
    risks: list[str] = []
    lowered = question.lower()
    if "yes" in lowered or " no" in f" {lowered}":
        risks.append("question_contains_yes_or_no")
    if len(question.split()) > 40:
        risks.append("long_question")
    if not question.strip().endswith("?"):
        risks.append("not_question_mark_terminated")
    if re.search(r"\bnot\b|\bn't\b|\bwithout\b", lowered):
        risks.append("negation_word")
    return risks


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


def summarize_category(category: str, rows: list[dict[str, Any]], image_root: Path, examples_per_category: int) -> dict[str, Any]:
    labels = Counter(normalize_label(row.get("label", row.get("answer"))) for row in rows)
    image_paths = [image_path_for(row, image_root) for row in rows]
    missing_images = [str(path) for path in image_paths if not path.exists()]
    key_counts: Counter[tuple[str, str]] = Counter()
    duplicate_examples: list[dict[str, Any]] = []
    risk_counts: Counter[str] = Counter()
    risk_examples: list[dict[str, Any]] = []
    for row in rows:
        question = str(row.get("question") or row.get("text") or "").strip()
        key = (str(row.get("image") or row.get("image_path") or row.get("image_id") or ""), question)
        key_counts[key] += 1
        for risk in parser_risks(question):
            risk_counts[risk] += 1
            if len(risk_examples) < examples_per_category:
                risk_examples.append({"risk": risk, "question": question, "label": normalize_label(row.get("label", row.get("answer")))})
    for (image, question), count in key_counts.items():
        if count > 1 and len(duplicate_examples) < examples_per_category:
            duplicate_examples.append({"image": image, "question": question, "count": count})
    examples = []
    for row in rows[:examples_per_category]:
        path = image_path_for(row, image_root)
        examples.append(
            {
                "sample_id": row.get("sample_id", ""),
                "image": row.get("image") or row.get("image_path") or row.get("image_id") or "",
                "image_exists": path.exists(),
                "question": row.get("question") or row.get("text") or "",
                "label": normalize_label(row.get("label", row.get("answer"))),
            }
        )
    return {
        "category": category,
        "n": len(rows),
        "yes": labels.get("yes", 0),
        "no": labels.get("no", 0),
        "invalid": labels.get("invalid", 0),
        "unique_images": len({str(path) for path in image_paths}),
        "missing_images": len(missing_images),
        "missing_image_examples": missing_images[:examples_per_category],
        "duplicate_pairs": sum(1 for count in key_counts.values() if count > 1),
        "duplicate_examples": duplicate_examples,
        "parser_risk_counts": dict(sorted(risk_counts.items())),
        "parser_risk_examples": risk_examples,
        "examples": examples,
    }


def write_report(path: Path, bench_root: Path, image_root: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overview_rows = [
        {
            "category": item["category"],
            "n": item["n"],
            "yes": item["yes"],
            "no": item["no"],
            "invalid": item["invalid"],
            "unique_images": item["unique_images"],
            "missing_images": item["missing_images"],
            "duplicate_pairs": item["duplicate_pairs"],
            "parser_risks": sum(item["parser_risk_counts"].values()),
        }
        for item in summaries
    ]
    lines = [
        "# Attribute MME Data Report",
        "",
        f"- Benchmark root: `{bench_root}`",
        f"- Image root: `{image_root}`",
        "",
        "## Overview",
        "",
        table(["category", "n", "yes", "no", "invalid", "unique_images", "missing_images", "duplicate_pairs", "parser_risks"], overview_rows),
    ]
    for item in summaries:
        lines.extend(
            [
                "",
                f"## {item['category']}",
                "",
                f"- Samples: `{item['n']}`",
                f"- Label counts: yes=`{item['yes']}`, no=`{item['no']}`, invalid=`{item['invalid']}`",
                f"- Missing images: `{item['missing_images']}`",
                f"- Duplicate image/question pairs: `{item['duplicate_pairs']}`",
                f"- Parser risk counts: `{json.dumps(item['parser_risk_counts'], ensure_ascii=False)}`",
                "",
                "Examples:",
                "",
                fenced_json(item["examples"]),
            ]
        )
        if item["missing_image_examples"]:
            lines.extend(["", "Missing image examples:", "", fenced_json(item["missing_image_examples"])])
        if item["duplicate_examples"]:
            lines.extend(["", "Duplicate examples:", "", fenced_json(item["duplicate_examples"])])
        if item["parser_risk_examples"]:
            lines.extend(["", "Parser risk examples:", "", fenced_json(item["parser_risk_examples"])])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    bench_root = resolve_project_path(args.bench_root)
    image_root = resolve_project_path(args.image_root) if str(args.image_root).strip() else bench_root / "images"
    summaries: list[dict[str, Any]] = []
    for category in args.categories:
        path = bench_root / f"{category}.jsonl"
        rows = read_json_rows(path)
        summaries.append(summarize_category(category, rows, image_root, int(args.examples_per_category)))
    output = resolve_project_path(args.output)
    write_report(output, bench_root, image_root, summaries)
    print(f"Wrote Attribute MME data report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
