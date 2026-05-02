#!/usr/bin/env bash
set -euo pipefail

# Wrong-expert sanity for disjoint-v1 expert-map steering.
#
# It checks whether the right expert beats wrong experts on type-specific
# benchmarks. Default behavior executes jobs. Set DRY_RUN=1 to print commands.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
VECTOR_PATH="${VECTOR_PATH:-data/outputs_after_template_disjoint_v1/steering/after_template_expert_vectors.pt}"
HEAD_MAP="${HEAD_MAP:-data/outputs_after_template_disjoint_v1/head_analysis/head_maps/top64.json}"
RUN_ROOT="${RUN_ROOT:-data/outputs_after_template_disjoint_v1/wrong_expert_sanity}"
GPU="${GPU:-auto}"
DRY_RUN="${DRY_RUN:-0}"

ALPHA_CAT="${ALPHA_CAT:-1.0}"
ALPHA_ATTR="${ALPHA_ATTR:-0.25}"
ALPHA_REL="${ALPHA_REL:-0.1}"

POPE_RANDOM="${POPE_RANDOM:-/home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json}"
POPE_IMAGE_ROOT="${POPE_IMAGE_ROOT:-/home/huiwei/sy/sy_data/COCO2014/val2014}"
POPE_LIMIT="${POPE_LIMIT:-500}"

MME_ROOT="${MME_ROOT:-data/benchmarks/mme_hallucination}"
MME_IMAGE_ROOT="${MME_IMAGE_ROOT:-${MME_ROOT}/images}"
MME_LIMIT="${MME_LIMIT:-0}"

AMBER_ROOT="${AMBER_ROOT:-data/benchmarks/amber_hallucination}"
AMBER_IMAGE_ROOT="${AMBER_IMAGE_ROOT:-/home/huiwei/sy/benchmarks/AMBER}"
AMBER_ATTRIBUTE_LIMIT="${AMBER_ATTRIBUTE_LIMIT:-1000}"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

alpha_for_expert() {
  case "$1" in
    cat) echo "$ALPHA_CAT" ;;
    attr) echo "$ALPHA_ATTR" ;;
    rel) echo "$ALPHA_REL" ;;
    *) echo "1.0" ;;
  esac
}

if [[ ! -f "$VECTOR_PATH" ]]; then
  echo "Missing vector file: $VECTOR_PATH" >&2
  exit 1
fi
if [[ ! -f "$HEAD_MAP" ]]; then
  echo "Missing head map: $HEAD_MAP" >&2
  exit 1
fi

if ! truthy "$DRY_RUN"; then
  source scripts/gpu_sweep_utils.sh
  init_gpu_scheduler
fi

run_or_print() {
  if truthy "$DRY_RUN"; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    run_gpu_job "$@"
  fi
}

run_eval() {
  local data_file="$1"
  local benchmark_name="$2"
  local image_root="$3"
  local limit="$4"
  local out_dir="$5"
  shift 5
  if [[ ! -f "$data_file" ]]; then
    echo "Skipping missing benchmark data: $data_file" >&2
    return 0
  fi
  run_or_print python scripts/run_steered_benchmark.py \
    --benchmark-data "$data_file" \
    --benchmark-name "$benchmark_name" \
    --out-dir "$out_dir" \
    --adapter llava \
    --model-id "$MODEL_PATH" \
    --image-root "$image_root" \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit "$limit" \
    --max-new-tokens 16 \
    --progress-every 20 \
    --overwrite \
    "$@"
}

run_wrong_expert_grid() {
  local key="$1"
  local data_file="$2"
  local benchmark_name="$3"
  local image_root="$4"
  local limit="$5"
  local right_expert="$6"

  run_eval "$data_file" "$benchmark_name" "$image_root" "$limit" "$RUN_ROOT/$key/baseline"
  for expert in cat attr rel; do
    local alpha
    alpha="$(alpha_for_expert "$expert")"
    run_eval "$data_file" "$benchmark_name" "$image_root" "$limit" "$RUN_ROOT/$key/${expert}_top64_alpha${alpha}" \
      --steer-enable \
      --steer-vector-path "$VECTOR_PATH" \
      --steer-alpha "$alpha" \
      --steer-head-select expert_map \
      --steer-head-map "$HEAD_MAP" \
      --steer-expert-key "$expert" \
      --steer-router no_filter \
      --steer-enabled-experts "$expert" \
      --steer-prefill true \
      --steer-decode true \
      --steer-apply-to last_token \
      --prefill-apply-to last_token \
      --decode-apply-to last_token
  done
  printf '%s\t%s\n' "$key" "$right_expert" >> "$RUN_ROOT/.right_experts.tsv"
}

mkdir -p "$RUN_ROOT"
: > "$RUN_ROOT/.right_experts.tsv"

run_wrong_expert_grid "pope_random" "$POPE_RANDOM" "wrong_expert_pope_random" "$POPE_IMAGE_ROOT" "$POPE_LIMIT" "cat"
run_wrong_expert_grid "amber_attribute" "${AMBER_ROOT}/attribute.jsonl" "wrong_expert_amber_attribute" "$AMBER_IMAGE_ROOT" "$AMBER_ATTRIBUTE_LIMIT" "attr"
run_wrong_expert_grid "mme_color" "${MME_ROOT}/color.jsonl" "wrong_expert_mme_color" "$MME_IMAGE_ROOT" "$MME_LIMIT" "attr"
run_wrong_expert_grid "mme_position" "${MME_ROOT}/position.jsonl" "wrong_expert_mme_position" "$MME_IMAGE_ROOT" "$MME_LIMIT" "rel"

if ! truthy "$DRY_RUN"; then
  wait_gpu_jobs
  python - "$RUN_ROOT" <<'PY'
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

run_root = Path(sys.argv[1])
right_experts = {}
right_path = run_root / ".right_experts.tsv"
if right_path.exists():
    for line in right_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, expert = line.split("\t", 1)
        right_experts[key] = expert

def read_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}

def nested(payload, *keys, default=""):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value

def alpha_from_run(name):
    match = re.search(r"alpha([-+]?\d+(?:\.\d+)?)", name)
    return match.group(1) if match else ""

def expert_from_run(name):
    return name.split("_top", 1)[0] if "_top" in name else ""

def fmt(value):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value).replace("|", "\\|")

rows = []
for metrics_path in sorted(run_root.glob("*/*/metrics.json")):
    benchmark = metrics_path.parent.parent.name
    run = metrics_path.parent.name
    metrics = read_json(metrics_path)
    fixed = metrics.get("fixed_steering", {})
    if not isinstance(fixed, dict):
        fixed = {}
    expert = "" if run == "baseline" else expert_from_run(run)
    rows.append({
        "benchmark": benchmark,
        "run": run,
        "expert": expert,
        "right_expert": right_experts.get(benchmark, ""),
        "is_right_expert": expert == right_experts.get(benchmark, ""),
        "alpha": fixed.get("alpha", alpha_from_run(run)),
        "accuracy_baseline": metrics.get("accuracy_baseline", nested(metrics, "baseline", "accuracy")),
        "accuracy_steered": metrics.get("accuracy_steered", nested(metrics, "steered", "accuracy", default=nested(metrics, "baseline", "accuracy"))),
        "delta_accuracy": metrics.get("delta_accuracy", fixed.get("delta_accuracy", "")),
        "f1_steered": metrics.get("f1_yes", fixed.get("f1_steered", nested(metrics, "steered", "f1_yes"))),
        "yes_rate_baseline": metrics.get("yes_rate_baseline", fixed.get("yes_rate_baseline", nested(metrics, "baseline", "yes_rate"))),
        "yes_rate_steered": metrics.get("yes_rate_steered", fixed.get("yes_rate_steered", nested(metrics, "steered", "yes_rate"))),
        "wrong_to_right": metrics.get("wrong_to_right", fixed.get("wrong_to_right", "")),
        "right_to_wrong": metrics.get("right_to_wrong", fixed.get("right_to_wrong", "")),
        "changed_pred": metrics.get("changed_pred", fixed.get("changed_pred", "")),
        "avg_delta_margin_label_yes": metrics.get("avg_delta_margin_label_yes", fixed.get("avg_delta_margin_label_yes", "")),
        "avg_delta_margin_label_no": metrics.get("avg_delta_margin_label_no", fixed.get("avg_delta_margin_label_no", "")),
    })

by_benchmark = defaultdict(list)
for row in rows:
    if row["run"] != "baseline":
        by_benchmark[row["benchmark"]].append(row)

conclusions = []
for benchmark, items in sorted(by_benchmark.items()):
    right = [row for row in items if row["is_right_expert"]]
    wrong = [row for row in items if not row["is_right_expert"]]
    best_right = max(right, key=lambda row: float(row["delta_accuracy"] or -999), default=None)
    best_wrong = max(wrong, key=lambda row: float(row["delta_accuracy"] or -999), default=None)
    if best_right and best_wrong and float(best_right["delta_accuracy"] or -999) > float(best_wrong["delta_accuracy"] or -999):
        note = "right expert wins"
    elif best_wrong and float(best_wrong["delta_accuracy"] or -999) > 0:
        note = "wrong expert also improves: possible global factual grounding effect"
    else:
        note = "no clear right-expert advantage"
    conclusions.append({
        "benchmark": benchmark,
        "right_expert": right_experts.get(benchmark, ""),
        "best_right_delta": "" if best_right is None else best_right["delta_accuracy"],
        "best_wrong": "" if best_wrong is None else best_wrong["run"],
        "best_wrong_delta": "" if best_wrong is None else best_wrong["delta_accuracy"],
        "note": note,
    })

headers = [
    "benchmark",
    "run",
    "expert",
    "right_expert",
    "is_right_expert",
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
lines = [
    "# Wrong-Expert Sanity Report",
    "",
    "## Results First",
    "",
    "| benchmark | right_expert | best_right_delta | best_wrong | best_wrong_delta | note |",
    "| --- | --- | --- | --- | --- | --- |",
]
for row in conclusions:
    lines.append("| " + " | ".join(fmt(row.get(key, "")) for key in ["benchmark", "right_expert", "best_right_delta", "best_wrong", "best_wrong_delta", "note"]) + " |")
lines.extend([
    "",
    "## Conditions",
    "",
    "- Vector: `data/outputs_after_template_disjoint_v1/steering/after_template_expert_vectors.pt`",
    "- Heads: `data/outputs_after_template_disjoint_v1/head_analysis/head_maps/top64.json`",
    "- Expert alphas: cat uses `ALPHA_CAT`, attr uses `ALPHA_ATTR`, rel uses `ALPHA_REL`.",
    "",
    "## All Runs",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join("---" for _ in headers) + " |",
])
for row in rows:
    lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
lines.append("")

report = run_root / "WRONG_EXPERT_SANITY_REPORT.md"
report.write_text("\n".join(lines), encoding="utf-8")
(run_root / "wrong_expert_sanity_stats.json").write_text(json.dumps({"rows": rows, "conclusions": conclusions}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
target = Path("data/outputs_after_template_disjoint_v1/WRONG_EXPERT_SANITY_REPORT.md")
target.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote wrong-expert sanity report to {report}")
print(f"Wrote wrong-expert sanity report to {target}")
PY
else
  echo "[dry-run] No jobs launched. Set DRY_RUN=0 to execute."
fi
