#!/usr/bin/env bash
set -euo pipefail

# GQA type-aware diagnostic sweep.
#
# Runs baseline plus cat/attr/rel expert-vector injections on each GQA diagnostic
# subset. The actual model inference remains delegated to run_steered_benchmark.py.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
VECTOR_PATH="${VECTOR_PATH:-data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
EVAL_ROOT="${EVAL_ROOT:-data/gqa_typeaware_v1/${EVAL_SPLIT}_eval}"
RUN_ROOT="${RUN_ROOT:-data/gqa_typeaware_v1/eval_runs}"
IMAGE_ROOT="${IMAGE_ROOT:-/}"
SUBSETS="${SUBSETS:-cat attr rel}"
VECTORS="${VECTORS:-cat attr rel}"
ALPHAS="${ALPHAS:-0.1 0.25 0.5}"
STEER_LAYERS="${STEER_LAYERS:-5-25}"
STEER_K_HEADS="${STEER_K_HEADS:-64}"
LIMIT="${LIMIT:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
OVERWRITE="${OVERWRITE:-false}"
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

run_logged() {
  local out_dir="$1"
  shift
  mkdir -p "$out_dir"
  local command="$*"
  run_gpu_job bash -lc "${command} 2>&1 | tee '${out_dir}/log.txt'; status=\${PIPESTATUS[0]}; cp '${out_dir}/config.json' '${out_dir}/run_config.json' 2>/dev/null || true; exit \$status"
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
    python scripts/run_steered_benchmark.py \
      --benchmark-data "$data_file" \
      --benchmark-name "gqa_${subset}_${EVAL_SPLIT}" \
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
  local vector="$3"
  local alpha="$4"
  local out_dir="${RUN_ROOT}/gqa_${subset}_${EVAL_SPLIT}/${vector}_alpha${alpha}"
  local maybe_overwrite
  maybe_overwrite="$(overwrite_arg)"
  run_logged "$out_dir" \
    python scripts/run_steered_benchmark.py \
      --benchmark-data "$data_file" \
      --benchmark-name "gqa_${subset}_${EVAL_SPLIT}" \
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
      --steer-enabled-experts "$vector" \
      --steer-prefill true \
      --steer-decode true \
      --steer-apply-to last_token \
      --prefill-apply-to last_token \
      --decode-apply-to last_token \
      $maybe_overwrite
}

for subset in $SUBSETS; do
  data_file="$(subset_file "$subset")"
  if [[ ! -s "$data_file" ]]; then
    echo "Missing or empty GQA eval subset: $data_file" >&2
    exit 1
  fi
  run_baseline "$subset" "$data_file"
  for vector in $VECTORS; do
    for alpha in $ALPHAS; do
      run_steered "$subset" "$data_file" "$vector" "$alpha"
    done
  done
done

wait_gpu_jobs

python scripts/summarize_gqa_typeaware_eval.py \
  --runs-root "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.csv" \
  --report-output "$RUN_ROOT/SUMMARY.md"
