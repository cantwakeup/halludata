"""Summarize AFTER-template disjoint-v1 sweep and diagnose shift behavior."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = "data/outputs_after_template_disjoint_v1"
DEFAULT_REPORT = "data/outputs_after_template_disjoint_v1/DISJOINT_V1_SWEEP_REPORT.md"
DEFAULT_JSON = "data/outputs_after_template_disjoint_v1/disjoint_v1_sweep_stats.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Root containing disjoint-v1 runs.")
    parser.add_argument("--output", default=DEFAULT_REPORT, help="Markdown report output.")
    parser.add_argument("--json-output", default=DEFAULT_JSON, help="JSON summary output.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows, returning an empty list when missing."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def nested(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Read nested dict values."""

    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
    return value


def as_float(value: Any) -> float | None:
    """Convert a value to float when possible."""

    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def as_int(value: Any) -> int | None:
    """Convert a value to int when possible."""

    number = as_float(value)
    if number is None:
        return None
    return int(number)


def alpha_from_name(name: str) -> float | None:
    """Infer alpha from one run directory name."""

    match = re.search(r"(?:alpha|a)([-+]?\d+(?:\.\d+)?)", name)
    return float(match.group(1)) if match else None


def infer_benchmark(run_path: Path, metrics: Mapping[str, Any]) -> str:
    """Infer benchmark key from metrics and path."""

    for key in ("benchmark_name",):
        value = metrics.get(key) or nested(metrics, "baseline", key)
        if value:
            return str(value)
    parent = run_path.parent.name
    name = run_path.name
    combined = f"{parent}/{name}".lower()
    if "pope" in combined:
        for split in ("random", "popular", "adversarial"):
            if split in combined:
                return f"pope_{split}"
        return "pope"
    if "amber" in combined:
        for category in ("attribute", "existence", "relation"):
            if category in combined:
                return f"amber_{category}"
        return "amber"
    if "mme" in combined:
        for category in ("existence", "count", "color", "position"):
            if category in combined:
                return f"mme_{category}"
        return "mme"
    return parent if parent else name


def infer_expert(run_path: Path, metrics: Mapping[str, Any]) -> str:
    """Infer expert key from diagnostics and run name."""

    diag = metrics.get("steering_diagnostics", {})
    if isinstance(diag, Mapping):
        value = diag.get("expert_key")
        if value:
            return str(value)
        active = diag.get("active_experts")
        if isinstance(active, list) and active:
            return "+".join(str(item) for item in active)
        enabled = diag.get("enabled_experts")
        if isinstance(enabled, list) and enabled:
            return "+".join(str(item) for item in enabled)
    name = run_path.name.lower()
    for expert in ("cat_res", "attr_res", "rel_res", "global_all", "cat", "attr", "rel"):
        if re.search(rf"(^|[_/-]){re.escape(expert)}([_/-]|$)", name):
            return expert
    return ""


def metric_row_from_json(path: Path, root: Path) -> dict[str, Any]:
    """Convert one metrics.json payload to a flat row."""

    metrics = read_json(path)
    fixed = metrics.get("fixed_steering", {})
    if not isinstance(fixed, Mapping):
        fixed = {}
    run_path = path.parent
    alpha = as_float(metrics.get("alpha", fixed.get("alpha", alpha_from_name(run_path.name))))
    row = {
        "source_path": str(path),
        "relative_run": str(run_path.relative_to(root)) if run_path.is_relative_to(root) else str(run_path),
        "run": run_path.name,
        "benchmark": infer_benchmark(run_path, metrics),
        "expert": infer_expert(run_path, metrics),
        "alpha": alpha,
        "accuracy_baseline": as_float(metrics.get("accuracy_baseline", nested(metrics, "baseline", "accuracy"))),
        "accuracy_steered": as_float(metrics.get("accuracy_steered", nested(metrics, "steered", "accuracy"))),
        "delta_accuracy": as_float(metrics.get("delta_accuracy", fixed.get("delta_accuracy"))),
        "f1_steered": as_float(metrics.get("f1_yes", fixed.get("f1_steered", nested(metrics, "steered", "f1_yes")))),
        "f1_baseline": as_float(metrics.get("f1_baseline", fixed.get("f1_baseline", nested(metrics, "baseline", "f1_yes")))),
        "yes_rate_baseline": as_float(metrics.get("yes_rate_baseline", fixed.get("yes_rate_baseline", nested(metrics, "baseline", "yes_rate")))),
        "yes_rate_steered": as_float(metrics.get("yes_rate_steered", fixed.get("yes_rate_steered", nested(metrics, "steered", "yes_rate")))),
        "wrong_to_right": as_int(metrics.get("wrong_to_right", fixed.get("wrong_to_right"))),
        "right_to_wrong": as_int(metrics.get("right_to_wrong", fixed.get("right_to_wrong"))),
        "changed_pred": as_int(metrics.get("changed_pred", fixed.get("changed_pred"))),
        "changed_text": as_int(metrics.get("changed_text", fixed.get("changed_text"))),
        "avg_delta_margin_label_yes": as_float(metrics.get("avg_delta_margin_label_yes", fixed.get("avg_delta_margin_label_yes"))),
        "avg_delta_margin_label_no": as_float(metrics.get("avg_delta_margin_label_no", fixed.get("avg_delta_margin_label_no"))),
        "num_samples": as_int(metrics.get("num_samples", fixed.get("num_samples", nested(metrics, "baseline", "num_samples")))),
    }
    if row["delta_accuracy"] is None and row["accuracy_baseline"] is not None and row["accuracy_steered"] is not None:
        row["delta_accuracy"] = row["accuracy_steered"] - row["accuracy_baseline"]
    return row


def rows_from_csv(path: Path, root: Path) -> list[dict[str, Any]]:
    """Read optional CSV result rows with known metric column names."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            run = str(raw.get("run") or raw.get("setting") or path.stem)
            benchmark = str(raw.get("benchmark") or raw.get("dataset") or path.parent.name)
            row = {
                "source_path": str(path),
                "relative_run": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                "run": run,
                "benchmark": benchmark,
                "expert": str(raw.get("expert") or ""),
                "alpha": as_float(raw.get("alpha")),
                "accuracy_baseline": as_float(raw.get("accuracy_baseline") or raw.get("acc_base")),
                "accuracy_steered": as_float(raw.get("accuracy_steered") or raw.get("acc_steer")),
                "delta_accuracy": as_float(raw.get("delta_accuracy") or raw.get("delta")),
                "f1_steered": as_float(raw.get("f1_steered") or raw.get("f1")),
                "f1_baseline": as_float(raw.get("f1_baseline")),
                "yes_rate_baseline": as_float(raw.get("yes_rate_baseline") or raw.get("yes_base")),
                "yes_rate_steered": as_float(raw.get("yes_rate_steered") or raw.get("yes_steer")),
                "wrong_to_right": as_int(raw.get("wrong_to_right") or raw.get("w2r")),
                "right_to_wrong": as_int(raw.get("right_to_wrong") or raw.get("r2w")),
                "changed_pred": as_int(raw.get("changed_pred") or raw.get("changed")),
                "changed_text": as_int(raw.get("changed_text")),
                "avg_delta_margin_label_yes": as_float(raw.get("avg_delta_margin_label_yes") or raw.get("d_yes")),
                "avg_delta_margin_label_no": as_float(raw.get("avg_delta_margin_label_no") or raw.get("d_no")),
                "num_samples": as_int(raw.get("num_samples") or raw.get("n")),
            }
            if row["delta_accuracy"] is None and row["accuracy_baseline"] is not None and row["accuracy_steered"] is not None:
                row["delta_accuracy"] = row["accuracy_steered"] - row["accuracy_baseline"]
            rows.append(row)
    return rows


def discover_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Find metrics under root, including optional CSV summaries."""

    rows: list[dict[str, Any]] = []
    reports = [str(path) for path in sorted(root.rglob("REPORT*.md"))]
    for path in sorted(root.rglob("metrics.json")):
        try:
            rows.append(metric_row_from_json(path, root))
        except Exception as exc:
            rows.append({"source_path": str(path), "run": path.parent.name, "benchmark": "parse_error", "error": str(exc)})
    for path in sorted(root.rglob("*.csv")):
        if path.name.lower().startswith("overlap_"):
            continue
        try:
            rows.extend(rows_from_csv(path, root))
        except Exception:
            continue
    return rows, reports


def classify_attribute_question(question: Any) -> str:
    """Coarsely classify AMBER attribute questions into attribute subtypes."""

    text = str(question or "").lower()
    if any(word in text for word in ("how many", "number", "total", "count", " one ", " two ", " three ", " four ", " five ")):
        return "count"
    if any(word in text for word in ("color", "colour", "red", "blue", "green", "yellow", "black", "white", "brown", "orange", "gray", "grey")):
        return "color"
    if any(word in text for word in ("sitting", "standing", "running", "lying", "holding", "wearing", "riding", "carrying")):
        return "action_state"
    return "attribute_other"


def safe_accuracy(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    """Compute accuracy for rows with yes/no labels."""

    labeled = [row for row in rows if row.get("label") in {"yes", "no"}]
    if not labeled:
        return None
    return sum(1 for row in labeled if row.get(pred_key) == row.get("label")) / len(labeled)


def safe_yes_rate(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    """Compute yes-rate for one prediction key."""

    if not rows:
        return None
    return sum(1 for row in rows if row.get(pred_key) == "yes") / len(rows)


def attribute_breakdown(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Compute AMBER attribute subtype breakdown from predictions.jsonl files."""

    breakdown: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        benchmark = str(row.get("benchmark") or "").lower()
        if "amber" not in benchmark or "attribute" not in benchmark:
            continue
        predictions_path = Path(str(row.get("source_path", ""))).with_name("predictions.jsonl")
        prediction_rows = read_jsonl(predictions_path)
        if not prediction_rows:
            continue
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prediction in prediction_rows:
            groups[classify_attribute_question(prediction.get("question"))].append(prediction)
        for subtype, group_rows in sorted(groups.items()):
            base_acc = safe_accuracy(group_rows, "baseline_pred")
            steer_acc = safe_accuracy(group_rows, "steered_pred")
            breakdown[str(row.get("run"))].append(
                {
                    "run": row.get("run"),
                    "alpha": row.get("alpha"),
                    "subtype": subtype,
                    "n": len(group_rows),
                    "accuracy_baseline": base_acc,
                    "accuracy_steered": steer_acc,
                    "delta_accuracy": None if base_acc is None or steer_acc is None else steer_acc - base_acc,
                    "yes_rate_baseline": safe_yes_rate(group_rows, "baseline_pred"),
                    "yes_rate_steered": safe_yes_rate(group_rows, "steered_pred"),
                }
            )
    return dict(breakdown)


def monotonic_direction(values: list[float]) -> str:
    """Return increasing/decreasing/mixed/flat for a numeric series."""

    if len(values) < 2:
        return "insufficient"
    eps = 1e-9
    nonincreasing = all(values[i] >= values[i + 1] - eps for i in range(len(values) - 1))
    nondecreasing = all(values[i] <= values[i + 1] + eps for i in range(len(values) - 1))
    if nonincreasing and any(values[i] > values[i + 1] + eps for i in range(len(values) - 1)):
        return "decreasing"
    if nondecreasing and any(values[i] < values[i + 1] - eps for i in range(len(values) - 1)):
        return "increasing"
    if nonincreasing or nondecreasing:
        return "flat"
    return "mixed"


def diagnose_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Diagnose one benchmark group for no/yes-shift and precision."""

    valid = [row for row in rows if row.get("alpha") is not None and row.get("yes_rate_steered") is not None]
    valid.sort(key=lambda row: float(row["alpha"]))
    yes_rates = [float(row["yes_rate_steered"]) for row in valid]
    direction = monotonic_direction(yes_rates)
    changed_values = [row.get("changed_pred") for row in valid if row.get("changed_pred") is not None]
    sample_values = [row.get("num_samples") for row in valid if row.get("num_samples") is not None]
    changed_small = False
    if changed_values:
        changed_ref = min(int(value) for value in changed_values)
        sample_ref = max([int(value) for value in sample_values] or [0])
        changed_small = (changed_ref <= max(5, 0.05 * sample_ref)) if sample_ref else (changed_ref <= 5)

    best = best_row(rows)
    corrected = int(best.get("wrong_to_right") or 0) if best else 0
    broken = int(best.get("right_to_wrong") or 0) if best else 0
    tags: list[str] = []
    if direction == "decreasing":
        tags.append("no-shift risk")
    if direction == "increasing":
        tags.append("yes-shift risk")
    if changed_small and corrected > max(1, 2 * broken):
        tags.append("precise correction")
    if not tags:
        tags.append("mixed or weak shift")
    return {
        "yes_rate_trend": direction,
        "diagnostic_tags": tags,
        "best_run": best,
    }


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the best non-baseline row by delta, falling back to accuracy."""

    candidates = [
        row
        for row in rows
        if row.get("accuracy_steered") is not None and ("baseline" not in str(row.get("run", "")).lower())
    ]
    if not candidates:
        candidates = [row for row in rows if row.get("accuracy_steered") is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row.get("delta_accuracy") if row.get("delta_accuracy") is not None else -999.0),
            float(row.get("accuracy_steered") if row.get("accuracy_steered") is not None else -999.0),
        ),
    )


def fmt(value: Any) -> str:
    """Format markdown table values."""

    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def table(headers: list[str], rows: list[Mapping[str, Any]]) -> list[str]:
    """Render markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return lines


def render_report(
    root: Path,
    rows: list[dict[str, Any]],
    reports: list[str],
    diagnostics: Mapping[str, Any],
    attr_breakdown: Mapping[str, list[dict[str, Any]]],
) -> str:
    """Render markdown report."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("benchmark") or "unknown")].append(row)

    lines: list[str] = [
        "# Disjoint V1 Alpha Sweep Report",
        "",
        "This report summarizes existing AFTER-template image-disjoint v1 runs.",
        "It reads `metrics.json` / CSV result files only and does not modify run outputs.",
        "",
        "## Results First: Best Run By Benchmark",
        "",
    ]
    best_rows: list[dict[str, Any]] = []
    for benchmark in sorted(grouped):
        best = diagnostics.get(benchmark, {}).get("best_run")
        if not best:
            continue
        best_rows.append(
            {
                "benchmark": benchmark,
                "best_run": best.get("run"),
                "expert": best.get("expert"),
                "best_alpha": best.get("alpha"),
                "baseline_acc": best.get("accuracy_baseline"),
                "best_acc": best.get("accuracy_steered"),
                "delta_acc": best.get("delta_accuracy"),
                "f1": best.get("f1_steered"),
                "diagnosis": ", ".join(diagnostics.get(benchmark, {}).get("diagnostic_tags", [])),
            }
        )
    if best_rows:
        lines.extend(table(["benchmark", "best_run", "expert", "best_alpha", "baseline_acc", "best_acc", "delta_acc", "f1", "diagnosis"], best_rows))
    else:
        lines.append("- No completed sweep metrics were found.")

    lines.extend(["", "## Alpha-Level Details", ""])
    detail_headers = [
        "benchmark",
        "run",
        "expert",
        "alpha",
        "accuracy_baseline",
        "accuracy_steered",
        "delta_accuracy",
        "f1_steered",
        "yes_rate_baseline",
        "yes_rate_steered",
        "wrong_to_right",
        "right_to_wrong",
        "changed_pred",
        "avg_delta_margin_label_yes",
        "avg_delta_margin_label_no",
    ]
    detail_rows = sorted(
        [row for row in rows if row.get("accuracy_steered") is not None],
        key=lambda row: (str(row.get("benchmark")), float(row.get("alpha") if row.get("alpha") is not None else -1), str(row.get("run"))),
    )
    if detail_rows:
        lines.extend(table(detail_headers, detail_rows))
    else:
        lines.append("- No metric rows available.")

    lines.extend(["", "## Shift Diagnostics", ""])
    diag_rows = [
        {
            "benchmark": benchmark,
            "yes_rate_trend": data.get("yes_rate_trend"),
            "tags": ", ".join(data.get("diagnostic_tags", [])),
        }
        for benchmark, data in sorted(diagnostics.items())
    ]
    if diag_rows:
        lines.extend(table(["benchmark", "yes_rate_trend", "tags"], diag_rows))
    else:
        lines.append("- No diagnostics available.")

    lines.extend(["", "## Automatic Cat / Attr / Rel Notes", ""])
    for label, needle in (("cat", "cat|pope|existence"), ("attr", "attr|attribute|color|count"), ("rel", "rel|position|relation")):
        matched = [item for item in best_rows if re.search(needle, str(item["benchmark"]).lower()) or re.search(needle, str(item["best_run"]).lower())]
        if not matched:
            lines.append(f"- `{label}`: no matching completed benchmark found.")
            continue
        best = max(matched, key=lambda item: float(item.get("delta_acc") if item.get("delta_acc") is not None else -999.0))
        delta = best.get("delta_acc")
        trend = diagnostics.get(best["benchmark"], {}).get("yes_rate_trend")
        if delta is not None and float(delta) > 0:
            lines.append(f"- `{label}`: best observed delta is positive on `{best['benchmark']}` (`{delta:.4f}`); trend={trend}.")
        else:
            lines.append(f"- `{label}`: best observed delta is not positive on `{best['benchmark']}` (`{fmt(delta)}`); trend={trend}.")

    lines.extend(["", "## AMBER Attribute Breakdown", ""])
    breakdown_rows = [item for items in attr_breakdown.values() for item in items]
    if breakdown_rows:
        lines.extend(table(["run", "alpha", "subtype", "n", "accuracy_baseline", "accuracy_steered", "delta_accuracy", "yes_rate_baseline", "yes_rate_steered"], breakdown_rows))
    else:
        lines.append("- No AMBER attribute `predictions.jsonl` files with fixed-steering rows were found.")

    lines.extend(["", "## Input Discovery", "", f"- Search root: `{root}`"])
    if reports:
        lines.append("- Existing report files discovered:")
        lines.extend(f"  - `{path}`" for path in reports[:50])
        if len(reports) > 50:
            lines.append(f"  - ... {len(reports) - 50} more")
    else:
        lines.append("- No existing REPORT files discovered.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    try:
        root = resolve_project_path(args.root)
        output_path = resolve_project_path(args.output)
        json_path = resolve_project_path(args.json_output)
        rows, reports = discover_rows(root)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("benchmark") or "unknown")].append(row)
        diagnostics = {benchmark: diagnose_group(group_rows) for benchmark, group_rows in grouped.items()}
        attr_breakdown = attribute_breakdown(rows)
        payload = {
            "source": "after_template_disjoint_v1_sweep_summary",
            "root": str(root),
            "num_rows": len(rows),
            "reports_discovered": reports,
            "rows": rows,
            "diagnostics": diagnostics,
            "amber_attribute_breakdown": attr_breakdown,
        }
        write_json(json_path, payload)
        write_text(output_path, render_report(root, rows, reports, diagnostics, attr_breakdown))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote disjoint-v1 sweep report to {output_path}")
    print(f"Wrote disjoint-v1 sweep stats to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
