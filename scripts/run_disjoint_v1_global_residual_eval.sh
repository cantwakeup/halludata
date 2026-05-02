#!/usr/bin/env bash
set -euo pipefail

# Evaluate global, typed, residual, and global+residual vectors.
#
# Default behavior executes jobs. Set DRY_RUN=1 to print commands.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
VECTOR_PATH="${VECTOR_PATH:-data/outputs_after_template_disjoint_v1/steering/disjoint_v1_global_residual_vectors.pt}"
RUN_ROOT="${RUN_ROOT:-data/outputs_after_template_disjoint_v1/global_residual_eval}"
GPU="${GPU:-auto}"
DRY_RUN="${DRY_RUN:-0}"
ALPHAS="${ALPHAS:-0.1 0.25 0.5 1.0}"
K_HEADS="${K_HEADS:-64}"

POPE_RANDOM="${POPE_RANDOM:-/home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json}"
POPE_IMAGE_ROOT="${POPE_IMAGE_ROOT:-/home/huiwei/sy/sy_data/COCO2014/val2014}"
POPE_LIMIT="${POPE_LIMIT:-500}"

AMBER_ATTRIBUTE="${AMBER_ATTRIBUTE:-data/benchmarks/amber_hallucination/attribute.jsonl}"
AMBER_IMAGE_ROOT="${AMBER_IMAGE_ROOT:-/home/huiwei/sy/benchmarks/AMBER}"
AMBER_ATTRIBUTE_LIMIT="${AMBER_ATTRIBUTE_LIMIT:-1000}"

MME_POSITION="${MME_POSITION:-data/benchmarks/mme_hallucination/position.jsonl}"
MME_IMAGE_ROOT="${MME_IMAGE_ROOT:-data/benchmarks/mme_hallucination/images}"
MME_LIMIT="${MME_LIMIT:-0}"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -f "$VECTOR_PATH" ]]; then
  echo "Missing vector file: $VECTOR_PATH" >&2
  echo "Build it with scripts/build_disjoint_v1_global_and_residual_vectors.py first." >&2
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

safe_name() {
  echo "$1" | tr ',+' '__'
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

run_vector_grid() {
  local family="$1"
  local data_file="$2"
  local benchmark_name="$3"
  local image_root="$4"
  local limit="$5"
  shift 5
  local vector_sets=("$@")

  run_eval "$data_file" "$benchmark_name" "$image_root" "$limit" "$RUN_ROOT/$family/baseline"
  for vector_set in "${vector_sets[@]}"; do
    for alpha in $ALPHAS; do
      local out_key
      out_key="$(safe_name "$vector_set")"
      run_eval "$data_file" "$benchmark_name" "$image_root" "$limit" "$RUN_ROOT/$family/${out_key}_alpha${alpha}" \
        --steer-enable \
        --steer-vector-path "$VECTOR_PATH" \
        --steer-layers 0-31 \
        --steer-alpha "$alpha" \
        --steer-k-heads "$K_HEADS" \
        --steer-head-select norm \
        --steer-router no_filter \
        --steer-enabled-experts "$vector_set" \
        --steer-prefill true \
        --steer-decode true \
        --steer-apply-to last_token \
        --prefill-apply-to last_token \
        --decode-apply-to last_token
    done
  done
}

run_vector_grid \
  "pope_random" \
  "$POPE_RANDOM" \
  "global_residual_pope_random" \
  "$POPE_IMAGE_ROOT" \
  "$POPE_LIMIT" \
  "global_all" "cat" "cat_res" "global_all,cat_res" "global_plus_cat_res"

run_vector_grid \
  "amber_attribute" \
  "$AMBER_ATTRIBUTE" \
  "global_residual_amber_attribute" \
  "$AMBER_IMAGE_ROOT" \
  "$AMBER_ATTRIBUTE_LIMIT" \
  "global_all" "attr" "attr_res" "global_all,attr_res" "global_plus_attr_res"

run_vector_grid \
  "mme_position" \
  "$MME_POSITION" \
  "global_residual_mme_position" \
  "$MME_IMAGE_ROOT" \
  "$MME_LIMIT" \
  "global_all" "rel" "rel_res" "global_all,rel_res" "global_plus_rel_res"

if ! truthy "$DRY_RUN"; then
  wait_gpu_jobs
  python scripts/summarize_disjoint_v1_sweep.py \
    --root "$RUN_ROOT" \
    --output "$RUN_ROOT/GLOBAL_RESIDUAL_EVAL_REPORT.md" \
    --json-output "$RUN_ROOT/global_residual_eval_stats.json"
else
  echo "[dry-run] No jobs launched. Set DRY_RUN=0 to execute."
fi
