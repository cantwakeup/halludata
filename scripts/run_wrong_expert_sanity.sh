#!/usr/bin/env bash
set -euo pipefail

# Wrong-expert sanity commands for AFTER-template disjoint vectors.
#
# Default behavior executes the runs. Set DRY_RUN=1 to print the commands
# without launching LLaVA.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
VECTOR_PATH="${VECTOR_PATH:-data/outputs_after_template_disjoint_v1/steering/after_template_expert_vectors.pt}"
RUN_ROOT="${RUN_ROOT:-data/outputs_after_template_disjoint_v1/wrong_expert_sanity}"
GPU="${GPU:-auto}"
LIMIT="${LIMIT:-500}"
ALPHAS="${ALPHAS:-1.0}"
DRY_RUN="${DRY_RUN:-0}"
STEER_LAYERS="${STEER_LAYERS:-5-25}"

POPE_ROOT="${POPE_ROOT:-/home/huiwei/sy/benchmarks/POPE/output/coco}"
POPE_IMAGE_ROOT="${POPE_IMAGE_ROOT:-/home/huiwei/sy/sy_data/COCO2014/val2014}"
MME_ROOT="${MME_ROOT:-data/benchmarks/mme_hallucination}"
MME_IMAGE_ROOT="${MME_IMAGE_ROOT:-${MME_ROOT}/images}"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if ! truthy "$DRY_RUN"; then
  source scripts/gpu_sweep_utils.sh
  init_gpu_scheduler
fi

router_for_expert() {
  case "$1" in
    cat) echo "force_cat" ;;
    attr) echo "force_attr" ;;
    rel) echo "force_rel" ;;
    *) echo "no_filter" ;;
  esac
}

run_or_print() {
  if truthy "$DRY_RUN"; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    run_gpu_job "$@"
  fi
}

run_baseline() {
  local data_file="$1"
  local benchmark_name="$2"
  local image_root="$3"
  local out_dir="$4"
  run_or_print python scripts/run_steered_benchmark.py \
    --benchmark-data "$data_file" \
    --benchmark-name "$benchmark_name" \
    --out-dir "$out_dir" \
    --adapter llava \
    --model-id "$MODEL_PATH" \
    --image-root "$image_root" \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit "$LIMIT" \
    --progress-every 20 \
    --overwrite
}

run_steered() {
  local data_file="$1"
  local benchmark_name="$2"
  local image_root="$3"
  local out_dir="$4"
  local expert="$5"
  local alpha="$6"
  local router
  router="$(router_for_expert "$expert")"
  run_or_print python scripts/run_steered_benchmark.py \
    --benchmark-data "$data_file" \
    --benchmark-name "$benchmark_name" \
    --out-dir "$out_dir" \
    --adapter llava \
    --model-id "$MODEL_PATH" \
    --image-root "$image_root" \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit "$LIMIT" \
    --progress-every 20 \
    --steer-enable \
    --steer-vector-path "$VECTOR_PATH" \
    --steer-layers "$STEER_LAYERS" \
    --steer-alpha "$alpha" \
    --steer-k-heads 64 \
    --steer-head-select norm \
    --steer-router "$router" \
    --steer-enabled-experts "$expert" \
    --steer-prefill true \
    --steer-decode true \
    --steer-apply-to last_token \
    --prefill-apply-to last_token \
    --decode-apply-to last_token \
    --overwrite
}

run_benchmark_grid() {
  local key="$1"
  local data_file="$2"
  local benchmark_name="$3"
  local image_root="$4"

  run_baseline "$data_file" "$benchmark_name" "$image_root" "$RUN_ROOT/$key/baseline"
  for expert in cat attr rel; do
    for alpha in $ALPHAS; do
      run_steered "$data_file" "$benchmark_name" "$image_root" "$RUN_ROOT/$key/${expert}_alpha${alpha}" "$expert" "$alpha"
    done
  done
}

for dataset in random popular adversarial; do
  run_benchmark_grid \
    "pope_${dataset}" \
    "${POPE_ROOT}/coco_pope_${dataset}.json" \
    "wrong_expert_pope_${dataset}" \
    "$POPE_IMAGE_ROOT"
done

run_benchmark_grid "mme_color" "${MME_ROOT}/color.jsonl" "wrong_expert_mme_color" "$MME_IMAGE_ROOT"
run_benchmark_grid "mme_position" "${MME_ROOT}/position.jsonl" "wrong_expert_mme_position" "$MME_IMAGE_ROOT"

if ! truthy "$DRY_RUN"; then
  wait_gpu_jobs
  python - "$RUN_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

run_root = Path(sys.argv[1])

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
    match = re.search(r"alpha([-+]?\d+(?:\.\d+)?)$", name)
    return match.group(1) if match else ""

def fmt(value):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)

rows = []
for metrics_path in sorted(run_root.glob("*/*/metrics.json")):
    benchmark = metrics_path.parent.parent.name
    run = metrics_path.parent.name
    metrics = read_json(metrics_path)
    fixed = metrics.get("fixed_steering", {})
    if not isinstance(fixed, dict):
        fixed = {}
    expert = "" if run == "baseline" else run.split("_alpha", 1)[0]
    rows.append({
        "benchmark": benchmark,
        "run": run,
        "expert": expert,
        "alpha": fixed.get("alpha", alpha_from_run(run)),
        "accuracy_baseline": metrics.get("accuracy_baseline", nested(metrics, "baseline", "accuracy")),
        "accuracy_steered": metrics.get("accuracy_steered", nested(metrics, "steered", "accuracy", default=nested(metrics, "baseline", "accuracy"))),
        "delta_accuracy": metrics.get("delta_accuracy", fixed.get("delta_accuracy", "")),
        "corrected": metrics.get("wrong_to_right", fixed.get("wrong_to_right", "")),
        "broken": metrics.get("right_to_wrong", fixed.get("right_to_wrong", "")),
        "changed_pred": metrics.get("changed_pred", fixed.get("changed_pred", "")),
        "avg_delta_margin_label_yes": metrics.get("avg_delta_margin_label_yes", fixed.get("avg_delta_margin_label_yes", "")),
        "avg_delta_margin_label_no": metrics.get("avg_delta_margin_label_no", fixed.get("avg_delta_margin_label_no", "")),
    })

headers = [
    "benchmark",
    "run",
    "expert",
    "alpha",
    "accuracy_baseline",
    "accuracy_steered",
    "delta_accuracy",
    "corrected",
    "broken",
    "changed_pred",
    "avg_delta_margin_label_yes",
    "avg_delta_margin_label_no",
]
lines = [
    "# Wrong-Expert Sanity Report",
    "",
    "This report compares cat/attr/rel expert injection on benchmarks where only one expert should be appropriate.",
    "`corrected` is wrong_to_right and `broken` is right_to_wrong.",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join("---" for _ in headers) + " |",
]
for row in rows:
    lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
lines.append("")
out = run_root / "WRONG_EXPERT_SANITY_REPORT.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote wrong-expert sanity report to {out}")
PY
else
  echo "[dry-run] No jobs launched. Set DRY_RUN=0 to execute."
fi
