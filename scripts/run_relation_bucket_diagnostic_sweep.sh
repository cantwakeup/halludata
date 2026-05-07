#!/usr/bin/env bash
set -euo pipefail

# Relation-bucket diagnostic sweep for AFTER-template disjoint v2 vectors.
#
# This does not re-extract activations. It first rebuilds relation bucket vectors
# from the existing train cache, then evaluates matched buckets on AMBER relation
# and MME position.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
ACTIVATION_CACHE="${ACTIVATION_CACHE:-data/outputs_after_template_disjoint_v2/activations/train.pt}"
METADATA="${METADATA:-data/outputs_after_template_disjoint_v2/activations/train.meta.jsonl}"
VECTOR_PATH="${VECTOR_PATH:-data/outputs_after_template_disjoint_v2/steering/after_template_disjoint_v2_relation_bucket_vectors.pt}"
VECTOR_STATS="${VECTOR_STATS:-data/outputs_after_template_disjoint_v2/steering/after_template_disjoint_v2_relation_bucket_vectors.stats.json}"
VECTOR_REPORT="${VECTOR_REPORT:-data/outputs_after_template_disjoint_v2/steering/RELATION_BUCKET_VECTOR_REPORT.md}"
RUN_ROOT="${RUN_ROOT:-data/outputs_after_template_disjoint_v2/runs/relation_bucket_diagnostics}"
STEER_LAYERS="${STEER_LAYERS:-5-25}"
STEER_K_HEADS="${STEER_K_HEADS:-64}"
ALPHAS="${ALPHAS:--0.75 -0.5 -0.25 -0.1 0.1 0.25 0.5}"

AMBER_RELATION="${AMBER_RELATION:-data/benchmarks/amber_hallucination/relation.jsonl}"
AMBER_IMAGE_ROOT="${AMBER_IMAGE_ROOT:-/home/huiwei/sy/benchmarks/AMBER}"
AMBER_LIMIT="${AMBER_LIMIT:-0}"
AMBER_EXPERTS="${AMBER_EXPERTS:-rel_contact rel_interaction rel_contact_interaction rel_semantic rel}"
RUN_AMBER="${RUN_AMBER:-true}"

MME_POSITION="${MME_POSITION:-data/benchmarks/mme_hallucination/position.jsonl}"
MME_IMAGE_ROOT="${MME_IMAGE_ROOT:-data/benchmarks/mme_hallucination/images}"
MME_LIMIT="${MME_LIMIT:-0}"
MME_EXPERTS="${MME_EXPERTS:-rel_position_2d rel_horizontal rel_vertical rel_depth rel}"
RUN_MME="${RUN_MME:-true}"

GPU="${GPU:-auto}"

source scripts/gpu_sweep_utils.sh
init_gpu_scheduler

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

build_vectors() {
  python scripts/build_after_template_relation_bucket_vectors.py \
    --activation-cache "$ACTIVATION_CACHE" \
    --metadata "$METADATA" \
    --output "$VECTOR_PATH" \
    --stats-output "$VECTOR_STATS" \
    --report-output "$VECTOR_REPORT" \
    --layers "$STEER_LAYERS" \
    --normalize false \
    --overwrite
}

run_baseline() {
  local task="$1"
  local data_file="$2"
  local image_root="$3"
  local limit="$4"
  local benchmark_name="$5"

  run_gpu_job python scripts/run_steered_benchmark.py \
    --benchmark-data "$data_file" \
    --benchmark-name "$benchmark_name" \
    --out-dir "$RUN_ROOT/$task/baseline" \
    --adapter llava \
    --model-id "$MODEL_PATH" \
    --image-root "$image_root" \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit "$limit" \
    --progress-every 20 \
    --overwrite
}

run_steered() {
  local task="$1"
  local data_file="$2"
  local image_root="$3"
  local limit="$4"
  local benchmark_name="$5"
  local expert="$6"
  local alpha="$7"

  run_gpu_job python scripts/run_steered_benchmark.py \
    --benchmark-data "$data_file" \
    --benchmark-name "$benchmark_name" \
    --out-dir "$RUN_ROOT/$task/${expert}_alpha${alpha}" \
    --adapter llava \
    --model-id "$MODEL_PATH" \
    --image-root "$image_root" \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit "$limit" \
    --progress-every 20 \
    --steer-enable \
    --steer-vector-path "$VECTOR_PATH" \
    --steer-layers "$STEER_LAYERS" \
    --steer-alpha "$alpha" \
    --steer-k-heads "$STEER_K_HEADS" \
    --steer-head-select norm \
    --steer-router no_filter \
    --steer-enabled-experts "$expert" \
    --steer-prefill true \
    --steer-decode true \
    --steer-apply-to last_token \
    --prefill-apply-to last_token \
    --decode-apply-to last_token \
    --overwrite
}

run_grid() {
  local task="$1"
  local data_file="$2"
  local image_root="$3"
  local limit="$4"
  local benchmark_name="$5"
  shift 5
  local experts=("$@")

  if [[ ! -s "$data_file" ]]; then
    echo "Missing or empty benchmark file: $data_file" >&2
    return 1
  fi

  run_baseline "$task" "$data_file" "$image_root" "$limit" "$benchmark_name"
  for expert in "${experts[@]}"; do
    for alpha in $ALPHAS; do
      run_steered "$task" "$data_file" "$image_root" "$limit" "$benchmark_name" "$expert" "$alpha"
    done
  done
}

build_vectors

if truthy "$RUN_AMBER"; then
  read -r -a amber_experts <<< "$AMBER_EXPERTS"
  run_grid "amber_relation" "$AMBER_RELATION" "$AMBER_IMAGE_ROOT" "$AMBER_LIMIT" "amber_relation_bucket_diagnostic" "${amber_experts[@]}"
fi

if truthy "$RUN_MME"; then
  read -r -a mme_experts <<< "$MME_EXPERTS"
  run_grid "mme_position" "$MME_POSITION" "$MME_IMAGE_ROOT" "$MME_LIMIT" "mme_position_bucket_diagnostic" "${mme_experts[@]}"
fi

wait_gpu_jobs

python scripts/summarize_relation_bucket_diagnostics.py \
  --runs-root "$RUN_ROOT" \
  --output "$RUN_ROOT/REPORT.md"
