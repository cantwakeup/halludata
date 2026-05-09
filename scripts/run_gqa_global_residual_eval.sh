#!/usr/bin/env bash
set -euo pipefail

# Evaluate global/shared and residual GQA expert vectors.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
BASE_VECTOR_PATH="${BASE_VECTOR_PATH:-data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt}"
VECTOR_PATH="${VECTOR_PATH:-data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt}"
ACTIVATION_CACHE="${ACTIVATION_CACHE:-data/gqa_typeaware_v1/activations/train.pt}"
METADATA="${METADATA:-data/gqa_typeaware_v1/activations/train.meta.jsonl}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
EVAL_ROOT="${EVAL_ROOT:-data/gqa_typeaware_v1/${EVAL_SPLIT}_eval}"
RUN_ROOT="${RUN_ROOT:-data/gqa_typeaware_v1/eval_runs_global_residual}"
IMAGE_ROOT="${IMAGE_ROOT:-/}"
ALPHAS="${ALPHAS:-0.1 0.25 0.5 1.0}"
STEER_LAYERS="${STEER_LAYERS:-5-25}"
STEER_K_HEADS="${STEER_K_HEADS:-64}"
LIMIT="${LIMIT:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
OVERWRITE="${OVERWRITE:-false}"
OVERWRITE_VECTORS="${OVERWRITE_VECTORS:-false}"
GPU="${GPU:-auto}"

source scripts/gpu_sweep_utils.sh
init_gpu_scheduler

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

overwrite_arg() {
  if truthy "$OVERWRITE"; then
    echo "--overwrite"
  fi
}

build_vectors() {
  if [[ -s "$VECTOR_PATH" ]] && ! truthy "$OVERWRITE_VECTORS"; then
    echo "[gqa-residual] vector file exists; skip rebuild: $VECTOR_PATH"
    return
  fi
  local maybe_overwrite=""
  if truthy "$OVERWRITE_VECTORS"; then
    maybe_overwrite="--overwrite"
  fi
  "$PYTHON_BIN" scripts/build_gqa_global_residual_vectors.py \
    --activation-cache "$ACTIVATION_CACHE" \
    --metadata "$METADATA" \
    --vector-path "$BASE_VECTOR_PATH" \
    --output "$VECTOR_PATH" \
    --report-output data/gqa_typeaware_v1/steering/GLOBAL_RESIDUAL_REPORT.md \
    $maybe_overwrite
}

run_logged() {
  local out_dir="$1"
  shift
  mkdir -p "$out_dir"
  local command="$*"
  run_gpu_job bash -c "${command} 2>&1 | tee '${out_dir}/log.txt'; status=\${PIPESTATUS[0]}; cp '${out_dir}/config.json' '${out_dir}/run_config.json' 2>/dev/null || true; exit \$status"
}

subset_file() {
  local subset="$1"
  echo "${EVAL_ROOT}/gqa_${subset}_${EVAL_SPLIT}.jsonl"
}

run_baseline() {
  local subset="$1"
  local data_file="$2"
  local out_dir="${RUN_ROOT}/gqa_${subset}_${EVAL_SPLIT}/baseline"
  local maybe_overwrite
  maybe_overwrite="$(overwrite_arg)"
  run_logged "$out_dir" \
    "$PYTHON_BIN" scripts/run_steered_benchmark.py \
      --benchmark-data "$data_file" \
      --benchmark-name "gqa_${subset}_${EVAL_SPLIT}_global_residual" \
      --out-dir "$out_dir" \
      --adapter llava \
      --model-id "$MODEL_PATH" \
      --image-root "$IMAGE_ROOT" \
      --device cuda:0 \
      --compute-dtype bfloat16 \
      --limit "$LIMIT" \
      --progress-every "$PROGRESS_EVERY" \
      $maybe_overwrite
}

run_steered() {
  local subset="$1"
  local data_file="$2"
  local enabled="$3"
  local run_name="$4"
  local alpha="$5"
  local out_dir="${RUN_ROOT}/gqa_${subset}_${EVAL_SPLIT}/${run_name}_alpha${alpha}"
  local maybe_overwrite
  maybe_overwrite="$(overwrite_arg)"
  run_logged "$out_dir" \
    "$PYTHON_BIN" scripts/run_steered_benchmark.py \
      --benchmark-data "$data_file" \
      --benchmark-name "gqa_${subset}_${EVAL_SPLIT}_global_residual" \
      --out-dir "$out_dir" \
      --adapter llava \
      --model-id "$MODEL_PATH" \
      --image-root "$IMAGE_ROOT" \
      --device cuda:0 \
      --compute-dtype bfloat16 \
      --limit "$LIMIT" \
      --progress-every "$PROGRESS_EVERY" \
      --steer-enable \
      --steer-vector-path "$VECTOR_PATH" \
      --steer-layers "$STEER_LAYERS" \
      --steer-alpha "$alpha" \
      --steer-k-heads "$STEER_K_HEADS" \
      --steer-head-select norm \
      --steer-router no_filter \
      --steer-enabled-experts "$enabled" \
      --steer-prefill true \
      --steer-decode true \
      --steer-apply-to last_token \
      --prefill-apply-to last_token \
      --decode-apply-to last_token \
      $maybe_overwrite
}

run_subset() {
  local subset="$1"
  shift
  local data_file
  data_file="$(subset_file "$subset")"
  if [[ ! -s "$data_file" ]]; then
    echo "Missing or empty GQA eval subset: $data_file" >&2
    exit 1
  fi
  run_baseline "$subset" "$data_file"
  while [[ "$#" -gt 0 ]]; do
    local enabled="$1"
    local run_name="$2"
    shift 2
    for alpha in $ALPHAS; do
      run_steered "$subset" "$data_file" "$enabled" "$run_name" "$alpha"
    done
  done
}

build_vectors

run_subset cat \
  "global_all" "global_all" \
  "cat" "cat" \
  "cat_res" "cat_res" \
  "global_all,cat_res" "global_all_plus_cat_res"

run_subset attr \
  "global_all" "global_all" \
  "attr" "attr" \
  "attr_res" "attr_res" \
  "global_all,attr_res" "global_all_plus_attr_res"

run_subset rel \
  "global_all" "global_all" \
  "rel" "rel" \
  "rel_res" "rel_res" \
  "global_all,rel_res" "global_all_plus_rel_res"

wait_gpu_jobs

"$PYTHON_BIN" scripts/summarize_gqa_typeaware_eval.py \
  --runs-root "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.csv" \
  --report-output "$RUN_ROOT/SUMMARY.md"
