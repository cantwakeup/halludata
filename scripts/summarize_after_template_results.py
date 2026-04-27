"""Summarize AFTER-template v1 data, activations, vectors, and POPE runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for report generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-stats", default="data/after_template_v1/pairs/stats.json")
    parser.add_argument("--vector-stats", default="data/outputs_after_template_v1/steering/after_template_expert_vectors.stats.json")
    parser.add_argument("--activations-root", default="data/outputs_after_template_v1/activations")
    parser.add_argument("--runs-root", default="data/outputs_after_template_v1/runs")
    parser.add_argument("--output", default="data/outputs_after_template_v1/REPORT.md")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object if absent."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def fmt(value: Any) -> str:
    """Format values for markdown cells."""

    if value is None:
        return ""
    try:
        if value != "":
            return f"{float(value):.4f}"
    except (TypeError, ValueError):
        pass
    return str(value)


def nested(payload: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Read a nested dictionary field."""

    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def dict_table(payload: dict[str, Any], title_key: str = "key") -> list[str]:
    """Render a flat dictionary as a two-column markdown table."""

    lines = [f"| {title_key} | value |", "| --- | ---: |"]
    for key, value in payload.items():
        lines.append(f"| {key} | {fmt(value)} |")
    return lines


def table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    """Render rows as a markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return lines


def collect_activation_rows(root: Path) -> list[dict[str, Any]]:
    """Collect activation manifest rows for train/val/test when available."""

    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        manifest = read_json(root / f"{split}.manifest.json")
        if manifest:
            rows.append(
                {
                    "split": split,
                    "num_pairs": manifest.get("num_pairs", ""),
                    "shape": manifest.get("shape", ""),
                    "position_mode": manifest.get("position_mode", ""),
                    "trusted_input_mode": manifest.get("trusted_input_mode", ""),
                }
            )
    return rows


def collect_run_rows(runs_root: Path) -> list[dict[str, Any]]:
    """Collect POPE run metrics."""

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("pope_random_after_template*/metrics.json")):
        metrics = read_json(metrics_path)
        run_name = metrics_path.parent.name
        fixed = metrics.get("fixed_steering", {})
        if not isinstance(fixed, dict):
            fixed = {}
        rows.append(
            {
                "run": run_name,
                "alpha": fixed.get("alpha", "" if "baseline" in run_name else run_name.rsplit("alpha", 1)[-1]),
                "accuracy_baseline": metrics.get("accuracy_baseline", nested(metrics, "baseline", "accuracy")),
                "accuracy_steered": metrics.get("accuracy_steered", nested(metrics, "steered", "accuracy", default=nested(metrics, "baseline", "accuracy"))),
                "delta_accuracy": metrics.get("delta_accuracy", fixed.get("delta_accuracy", "")),
                "precision_yes": metrics.get("precision_yes", nested(metrics, "steered", "precision_yes", default=nested(metrics, "baseline", "precision_yes"))),
                "recall_yes": metrics.get("recall_yes", nested(metrics, "steered", "recall_yes", default=nested(metrics, "baseline", "recall_yes"))),
                "f1_yes": metrics.get("f1_yes", nested(metrics, "steered", "f1_yes", default=nested(metrics, "baseline", "f1_yes"))),
                "yes_rate_baseline": metrics.get("yes_rate_baseline", nested(metrics, "baseline", "yes_rate")),
                "yes_rate_steered": metrics.get("yes_rate_steered", nested(metrics, "steered", "yes_rate", default=nested(metrics, "baseline", "yes_rate"))),
                "wrong_to_right": metrics.get("wrong_to_right", fixed.get("wrong_to_right", "")),
                "right_to_wrong": metrics.get("right_to_wrong", fixed.get("right_to_wrong", "")),
                "avg_delta_margin_all": metrics.get("avg_delta_margin_all", fixed.get("avg_delta_margin_all", "")),
                "avg_delta_margin_label_yes": metrics.get("avg_delta_margin_label_yes", fixed.get("avg_delta_margin_label_yes", "")),
                "avg_delta_margin_label_no": metrics.get("avg_delta_margin_label_no", fixed.get("avg_delta_margin_label_no", "")),
                "changed_pred": metrics.get("changed_pred", fixed.get("changed_pred", "")),
                "changed_text": metrics.get("changed_text", fixed.get("changed_text", "")),
                "avg_output_length": metrics.get("avg_output_length", fixed.get("avg_output_length", nested(metrics, "baseline", "average_output_length"))),
            }
        )
    return rows


def build_conclusion(vector_stats: dict[str, Any], run_rows: list[dict[str, Any]]) -> list[str]:
    """Build a short automatic interpretation."""

    lines: list[str] = []
    cosine = nested(vector_stats, "cosine_diagnostics", "cat_present_cat_absent", default=None)
    if cosine is not None:
        try:
            cosine_float = float(cosine)
            if cosine_float < -0.3:
                lines.append("- `cat_present` and `cat_absent` are opposed, so even template factual text may still encode conditional object-existence polarity.")
            elif cosine_float > 0.3:
                lines.append("- `cat_present` and `cat_absent` are positively aligned, which supports a shared template factual-text cat direction.")
            else:
                lines.append("- `cat_present` and `cat_absent` are weakly aligned; fixed cat steering may be unstable.")
        except (TypeError, ValueError):
            pass
    steered = [row for row in run_rows if "baseline" not in str(row.get("run", ""))]
    if steered:
        best = max(steered, key=lambda row: float(row.get("delta_accuracy") or 0.0))
        delta = float(best.get("delta_accuracy") or 0.0)
        yes_delta = float(best.get("avg_delta_margin_label_yes") or 0.0)
        no_delta = float(best.get("avg_delta_margin_label_no") or 0.0)
        if delta > 0:
            lines.append(f"- Best POPE delta is positive (`{best['run']}`, delta={delta:.4f}); AFTER-template cat steering helps this run.")
        elif abs(yes_delta) > 0.05 or abs(no_delta) > 0.05:
            lines.append("- Accuracy does not improve, but first-token margins move; the vector has control power and likely needs conditional/gated steering.")
        else:
            lines.append("- Accuracy and margins barely move; inspect activation extraction, hook layer/head choice, or alpha scale.")
        if yes_delta > 0 and no_delta < 0:
            lines.append("- Label=yes and label=no margins move in the desired truthfulness-like directions.")
        elif yes_delta > 0 and no_delta > 0:
            lines.append("- Label=yes and label=no margins move in the same Yes direction, indicating an existence/Yes bias remains.")
    return lines or ["- Not enough completed runs to draw a conclusion yet."]


def write_report(
    output: Path,
    pairs_stats: dict[str, Any],
    vector_stats: dict[str, Any],
    activation_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> None:
    """Write the AFTER-template v1 markdown report."""

    lines: list[str] = [
        "# AFTER-Template v1 Report",
        "",
        "This experiment is a lightweight AFTER-style template factual-text steering pipeline, not a full AFTER QAO reproduction.",
        "The key direction is `mean(z_text - z_visual)`, where `z_visual` comes from image + question and `z_text` comes from template factual text + question.",
        "All new data lives under `data/after_template_v1/` and `data/outputs_after_template_v1/`; legacy outputs are not overwritten.",
        "",
        "## Legacy Archive",
        "",
        "- Registry: `data/legacy_experiments/README.md`",
        "- Manifest: `data/legacy_experiments/manifest.json`",
        "- Status: old files are documented only; no legacy data is moved or overwritten.",
        "",
        "## Data Construction",
        "",
        "- `cat`: present/absent object-existence questions with template factual text, not Yes/No-only answers.",
        "- `attr`: count and optional bbox-color factual template text.",
        "- `rel`: bbox-derived left/right/above/below relation template text.",
        "- Difference from the previous AFTER-style answer-pair pipeline: this version does not compare factual answer vs counterfactual answer. It compares trusted factual text vs untrusted visual query.",
        "",
        "## Pair Statistics",
        "",
        f"- Total pairs: {pairs_stats.get('total_pairs', 'TBD')}",
        f"- Train/val/test pairs: {pairs_stats.get('train_pairs', 'TBD')} / {pairs_stats.get('val_pairs', 'TBD')} / {pairs_stats.get('test_pairs', 'TBD')}",
        f"- Images: {pairs_stats.get('num_images', pairs_stats.get('num_selected_images', 'TBD'))}",
        "",
        "### Type Counts",
        "",
        *dict_table(pairs_stats.get("type_counts", {}), "hallucination_type"),
        "",
        "### Subtype Counts",
        "",
        *dict_table(pairs_stats.get("subtype_counts", {}), "subtype"),
        "",
        "### Skipped Reasons",
        "",
        *dict_table(pairs_stats.get("skipped", {}), "reason"),
        "",
        "## Activation Statistics",
        "",
        *(table(activation_rows, ["split", "num_pairs", "shape", "position_mode", "trusted_input_mode"]) if activation_rows else ["No activation manifests found yet."]),
        "",
        "## Vector Statistics",
        "",
        "### Sample Counts By Type",
        "",
        *dict_table(vector_stats.get("sample_counts_by_type", {}), "expert"),
        "",
        "### Cosine Diagnostics",
        "",
        *dict_table(vector_stats.get("cosine_diagnostics", {}), "cosine"),
        "",
        "### Vector Norms",
        "",
        "| expert | mean_norm | max_norm | min_norm |",
        "| --- | ---: | ---: | ---: |",
    ]
    for expert, norms in vector_stats.get("vector_norms", {}).items():
        norms = norms or {}
        lines.append(f"| {expert} | {fmt(norms.get('mean', ''))} | {fmt(norms.get('max', ''))} | {fmt(norms.get('min', ''))} |")
    lines.extend(
        [
            "",
            "## POPE Random Results",
            "",
            *(table(run_rows, [
                "run",
                "alpha",
                "accuracy_baseline",
                "accuracy_steered",
                "delta_accuracy",
                "f1_yes",
                "yes_rate_baseline",
                "yes_rate_steered",
                "wrong_to_right",
                "right_to_wrong",
                "avg_delta_margin_label_yes",
                "avg_delta_margin_label_no",
                "changed_pred",
                "changed_text",
            ]) if run_rows else ["No POPE runs found yet."]),
            "",
            "## Automatic Conclusion",
            "",
            *build_conclusion(vector_stats, run_rows),
            "",
            "## Next-Step Reading",
            "",
            "- If accuracy improves, expand to POPE popular/adversarial and then typed attr/rel benchmarks.",
            "- If margins move but accuracy does not, add conditional/gated steering rather than treating this as a final mitigation result.",
            "- If label=yes and label=no margins move in the same direction, the cat vector is still closer to an existence/Yes direction.",
            "- If margins barely move, revisit text-only extraction compatibility, head/layer choice, or alpha scaling.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Generate the AFTER-template v1 report."""

    args = parse_args()
    try:
        pairs_stats = read_json(resolve_project_path(args.pairs_stats))
        vector_stats = read_json(resolve_project_path(args.vector_stats))
        activation_rows = collect_activation_rows(resolve_project_path(args.activations_root))
        run_rows = collect_run_rows(resolve_project_path(args.runs_root))
        write_report(resolve_project_path(args.output), pairs_stats, vector_stats, activation_rows, run_rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote AFTER-template report to {resolve_project_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
