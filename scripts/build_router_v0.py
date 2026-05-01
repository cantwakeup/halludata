"""Build and optionally evaluate a rule-based query router v0.

This router is intentionally non-learned. It only inspects the query text and
returns multi-label weights when multiple rules match.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_ORDER = (
    "cat",
    "attr_count",
    "attr_color",
    "rel_spatial",
    "rel_horizontal",
    "rel_vertical",
    "rel_contact_placeholder",
    "unknown",
)
PRIMARY_PRIORITY = (
    "cat",
    "attr_count",
    "attr_color",
    "rel_horizontal",
    "rel_vertical",
    "rel_contact_placeholder",
    "rel_spatial",
    "unknown",
)
COUNT_WORDS = {"how many", "number", "total", "one", "two", "three", "four", "five", "six"}
COLOR_WORDS = {"red", "blue", "green", "yellow", "black", "white", "brown", "orange", "color", "colour"}
HORIZONTAL_WORDS = {"left", "right"}
VERTICAL_WORDS = {"above", "below", "top", "bottom"}
CONTACT_PHRASES = {"direct contact", "touch", "touching"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-jsonl", default="", help="Optional JSONL file for evaluation.")
    parser.add_argument("--out", default="data/outputs_after_template_disjoint_v1/router/router_v0_report.md")
    parser.add_argument("--json-out", default="", help="Optional JSON sidecar path.")
    parser.add_argument("--examples", type=int, default=30, help="Number of misroute examples to include.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows."""

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


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def normalize_text(text: Any) -> str:
    """Normalize a query string for keyword rules."""

    raw = str(text or "").lower()
    raw = raw.replace("-", " ")
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def has_word(text: str, word: str) -> bool:
    """Return whether a word or phrase appears with word boundaries."""

    if " " in word:
        return word in text
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def add_score(scores: dict[str, float], key: str, value: float = 1.0) -> None:
    """Add a score to a router label."""

    scores[key] = scores.get(key, 0.0) + value


def route_query(question: str) -> dict[str, float]:
    """Route one query into multi-label normalized weights."""

    text = normalize_text(question)
    scores: dict[str, float] = {}
    count_hit = any(has_word(text, word) for word in COUNT_WORDS)
    color_hit = any(has_word(text, word) for word in COLOR_WORDS)
    horizontal_hit = any(has_word(text, word) for word in HORIZONTAL_WORDS)
    vertical_hit = any(has_word(text, word) for word in VERTICAL_WORDS)
    contact_hit = any(phrase in text for phrase in CONTACT_PHRASES)
    existence_hit = re.search(r"\b(is|are)\s+there\b", text) is not None

    if count_hit:
        add_score(scores, "attr_count")
    if color_hit:
        add_score(scores, "attr_color")
    if horizontal_hit:
        add_score(scores, "rel_horizontal")
        add_score(scores, "rel_spatial")
    if vertical_hit:
        add_score(scores, "rel_vertical")
        add_score(scores, "rel_spatial")
    if contact_hit:
        add_score(scores, "rel_contact_placeholder")
    if existence_hit and not any((count_hit, color_hit, horizontal_hit, vertical_hit, contact_hit)):
        add_score(scores, "cat")
    if not scores:
        scores["unknown"] = 1.0

    total = sum(scores.values()) or 1.0
    return {key: value / total for key, value in sorted(scores.items(), key=lambda item: LABEL_ORDER.index(item[0]))}


def primary_label(weights: Mapping[str, float]) -> str:
    """Return a deterministic primary label from weights."""

    best = max(weights.items(), key=lambda item: (float(item[1]), -PRIMARY_PRIORITY.index(item[0])))
    return best[0]


def positive_labels(weights: Mapping[str, float]) -> set[str]:
    """Return positive labels from a weight map."""

    return {label for label, weight in weights.items() if float(weight) > 0.0}


def relation_axis_from_text(text: str) -> str | None:
    """Infer horizontal/vertical relation axis from a row or question."""

    normalized = normalize_text(text)
    if any(has_word(normalized, word) for word in HORIZONTAL_WORDS):
        return "rel_horizontal"
    if any(has_word(normalized, word) for word in VERTICAL_WORDS):
        return "rel_vertical"
    return None


def expected_labels(row: Mapping[str, Any]) -> set[str]:
    """Map constructed data rows to expected router labels."""

    subtype = str(row.get("subtype") or "").strip()
    hallucination_type = str(row.get("hallucination_type") or "").strip()
    category = str(row.get("category") or "").strip().lower()
    if hallucination_type == "cat" or category == "existence":
        return {"cat"}
    if subtype == "attr_count" or category == "count":
        return {"attr_count"}
    if subtype == "attr_color" or category in {"color", "attribute"}:
        if "color" in normalize_text(row.get("question", "")) or category == "color":
            return {"attr_color"}
        return {"attr_count", "attr_color"} if category == "attribute" else {"attr_color"}
    if hallucination_type == "attr":
        return {"attr_count" if "how many" in normalize_text(row.get("question", "")) else "attr_color"}
    if hallucination_type == "rel" or category in {"position", "relation"}:
        axis = (
            relation_axis_from_text(str(row.get("queried_relation") or ""))
            or relation_axis_from_text(str(row.get("true_relation") or ""))
            or relation_axis_from_text(str(row.get("question") or ""))
        )
        labels = {"rel_spatial"}
        if axis:
            labels.add(axis)
        return labels
    return {"unknown"}


def expected_primary(labels: set[str]) -> str:
    """Choose a deterministic primary expected label."""

    for label in ("cat", "attr_count", "attr_color", "rel_horizontal", "rel_vertical", "rel_contact_placeholder", "rel_spatial", "unknown"):
        if label in labels:
            return label
    return "unknown"


def table(headers: list[str], rows: list[Mapping[str, Any]]) -> list[str]:
    """Render a markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif isinstance(value, dict):
                values.append(" ".join(f"{key}:{val:.2f}" for key, val in value.items()))
            elif isinstance(value, set):
                values.append(",".join(sorted(value)))
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def evaluate(rows: list[dict[str, Any]], example_limit: int) -> dict[str, Any]:
    """Evaluate router rules against constructed rows."""

    confusion: Counter[tuple[str, str]] = Counter()
    predicted_counter: Counter[str] = Counter()
    expected_counter: Counter[str] = Counter()
    strict_hits = 0
    relaxed_hits = 0
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = str(row.get("question") or row.get("query") or row.get("prompt") or "")
        weights = route_query(question)
        predicted = positive_labels(weights)
        expected = expected_labels(row)
        predicted_primary = primary_label(weights)
        expected_primary_label = expected_primary(expected)
        predicted_counter[predicted_primary] += 1
        expected_counter[expected_primary_label] += 1
        confusion[(expected_primary_label, predicted_primary)] += 1
        if predicted == expected:
            strict_hits += 1
        if predicted & expected:
            relaxed_hits += 1
        elif len(examples) < example_limit:
            examples.append(
                {
                    "index": index,
                    "id": row.get("id", row.get("pair_id", "")),
                    "question": question,
                    "expected": ",".join(sorted(expected)),
                    "predicted": ",".join(sorted(predicted)),
                    "weights": weights,
                }
            )
    total = len(rows)
    return {
        "num_rows": total,
        "strict_accuracy": strict_hits / total if total else None,
        "relaxed_accuracy": relaxed_hits / total if total else None,
        "expected_primary_counts": dict(expected_counter),
        "predicted_primary_counts": dict(predicted_counter),
        "confusion": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(confusion.items())
        ],
        "miss_examples": examples,
    }


def render_report(out_path: Path, data_path: Path | None, evaluation: Mapping[str, Any] | None) -> str:
    """Render router report markdown."""

    lines: list[str] = [
        "# Router V0 Rule Report",
        "",
        "This is a rule-based query router skeleton. It does not train a learned router.",
        "",
        "## Supported Labels",
        "",
    ]
    lines.extend(f"- `{label}`" for label in LABEL_ORDER)
    lines.extend(
        [
            "",
            "## Rules",
            "",
            "- `is there` / `are there` without count/color/position/contact keywords -> `cat`",
            "- `how many`, `number`, `total`, or number words one-six -> `attr_count`",
            "- color words or `color` / `colour` -> `attr_color`",
            "- `left` / `right` -> `rel_horizontal` and `rel_spatial`",
            "- `above` / `below` / `top` / `bottom` -> `rel_vertical` and `rel_spatial`",
            "- `direct contact`, `touch`, or `touching` -> `rel_contact_placeholder`",
            "- Multiple matched rules produce multi-label normalized weights.",
            "",
            "## Evaluation",
            "",
        ]
    )
    if evaluation is None:
        lines.append("- No `--data-jsonl` was provided, so only the rule skeleton was written.")
    else:
        lines.extend(
            [
                f"- Data: `{data_path}`",
                f"- Rows: {evaluation.get('num_rows', 0)}",
                f"- Strict accuracy: {evaluation.get('strict_accuracy')}",
                f"- Relaxed hit rate: {evaluation.get('relaxed_accuracy')}",
                "",
                "### Expected Primary Counts",
                "",
            ]
        )
        lines.extend(table(["key", "count"], [{"key": key, "count": value} for key, value in sorted(evaluation.get("expected_primary_counts", {}).items())]))
        lines.extend(["", "### Predicted Primary Counts", ""])
        lines.extend(table(["key", "count"], [{"key": key, "count": value} for key, value in sorted(evaluation.get("predicted_primary_counts", {}).items())]))
        lines.extend(["", "### Primary Confusion", ""])
        lines.extend(table(["expected", "predicted", "count"], list(evaluation.get("confusion", []))))
        lines.extend(["", "### Miss Examples", ""])
        examples = list(evaluation.get("miss_examples", []))
        if examples:
            lines.extend(table(["index", "id", "expected", "predicted", "weights", "question"], examples))
        else:
            lines.append("- No relaxed misses found.")
    lines.extend(["", "## Output", "", f"- Report: `{out_path}`", ""])
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    try:
        out_path = resolve_project_path(args.out)
        data_path = resolve_project_path(args.data_jsonl) if str(args.data_jsonl).strip() else None
        evaluation = evaluate(read_jsonl(data_path), int(args.examples)) if data_path else None
        write_text(out_path, render_report(out_path, data_path, evaluation))
        json_out = resolve_project_path(args.json_out) if str(args.json_out).strip() else out_path.with_suffix(".json")
        write_json(
            json_out,
            {
                "source": "router_v0_rule_based",
                "labels": list(LABEL_ORDER),
                "data_jsonl": str(data_path) if data_path else "",
                "report": str(out_path),
                "evaluation": evaluation,
            },
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote router v0 report to {out_path}")
    print(f"Wrote router v0 JSON to {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
