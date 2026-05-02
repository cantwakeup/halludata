#!/usr/bin/env bash
set -euo pipefail

# Targeted benchmark sweep for AFTER-template image-disjoint v1 vectors.
#
# Default behavior executes jobs. Set DRY_RUN=1 to print commands only.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
VECTOR_PATH="${VECTOR_PATH:-data/outputs_after_template_disjoint_v1/steering/after_template_expert_vectors.pt}"
HEAD_MAP_ROOT="${HEAD_MAP_ROOT:-data/outputs_after_template_disjoint_v1/head_analysis/head_maps}"
RUN_ROOT="${RUN_ROOT:-data/outputs_after_template_disjoint_v1/targeted_eval}"
GPU="${GPU:-auto}"
DRY_RUN="${DRY_RUN:-0}"
TOPKS="${TOPKS:-16 32 64 128}"

POPE_ROOT="${POPE_ROOT:-/home/huiwei/sy/benchmarks/POPE/output/coco}"
POPE_IMAGE_ROOT="${POPE_IMAGE_ROOT:-/home/huiwei/sy/sy_data/COCO2014/val2014}"
POPE_LIMIT="${POPE_LIMIT:-500}"

MME_ROOT="${MME_ROOT:-data/benchmarks/mme_hallucination}"
MME_IMAGE_ROOT="${MME_IMAGE_ROOT:-${MME_ROOT}/images}"
MME_LIMIT="${MME_LIMIT:-0}"

AMBER_ROOT="${AMBER_ROOT:-data/benchmarks/amber_hallucination}"
AMBER_IMAGE_ROOT="${AMBER_IMAGE_ROOT:-/home/huiwei/sy/benchmarks/AMBER}"
AMBER_ATTRIBUTE_LIMIT="${AMBER_ATTRIBUTE_LIMIT:-1000}"
AMBER_RELATION_LIMIT="${AMBER_RELATION_LIMIT:-1664}"

CAT_ALPHAS="${CAT_ALPHAS:-0.5 1.0 1.5 2.0}"
ATTR_ALPHAS="${ATTR_ALPHAS:-0.05 0.1 0.25 0.5}"
REL_ALPHAS="${REL_ALPHAS:-0.05 0.1 0.25 0.5 1.0}"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -f "$VECTOR_PATH" ]]; then
  echo "Missing vector file: $VECTOR_PATH" >&2
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

run_steered_grid() {
  local family="$1"
  local data_file="$2"
  local benchmark_name="$3"
  local image_root="$4"
  local limit="$5"
  local expert="$6"
  local alphas="$7"

  run_eval "$data_file" "$benchmark_name" "$image_root" "$limit" "$RUN_ROOT/$family/baseline"

  for topk in $TOPKS; do
    local map_path="$HEAD_MAP_ROOT/top${topk}.json"
    if [[ ! -f "$map_path" ]]; then
      echo "Skipping missing head map: $map_path" >&2
      continue
    fi
    for alpha in $alphas; do
      run_eval "$data_file" "$benchmark_name" "$image_root" "$limit" "$RUN_ROOT/$family/${expert}_top${topk}_alpha${alpha}" \
        --steer-enable \
        --steer-vector-path "$VECTOR_PATH" \
        --steer-alpha "$alpha" \
        --steer-head-select expert_map \
        --steer-head-map "$map_path" \
        --steer-expert-key "$expert" \
        --steer-router no_filter \
        --steer-enabled-experts "$expert" \
        --steer-prefill true \
        --steer-decode true \
        --steer-apply-to last_token \
        --prefill-apply-to last_token \
        --decode-apply-to last_token
    done
  done
}

# cat / object existence
for dataset in random popular adversarial; do
  run_steered_grid \
    "cat/pope_${dataset}" \
    "${POPE_ROOT}/coco_pope_${dataset}.json" \
    "disjoint_v1_pope_${dataset}_cat" \
    "$POPE_IMAGE_ROOT" \
    "$POPE_LIMIT" \
    "cat" \
    "$CAT_ALPHAS"
done

run_steered_grid \
  "cat/mme_existence" \
  "${MME_ROOT}/existence.jsonl" \
  "disjoint_v1_mme_existence_cat" \
  "$MME_IMAGE_ROOT" \
  "$MME_LIMIT" \
  "cat" \
  "$CAT_ALPHAS"

# attr / count + color
run_steered_grid \
  "attr/mme_count" \
  "${MME_ROOT}/count.jsonl" \
  "disjoint_v1_mme_count_attr" \
  "$MME_IMAGE_ROOT" \
  "$MME_LIMIT" \
  "attr" \
  "$ATTR_ALPHAS"

run_steered_grid \
  "attr/mme_color" \
  "${MME_ROOT}/color.jsonl" \
  "disjoint_v1_mme_color_attr" \
  "$MME_IMAGE_ROOT" \
  "$MME_LIMIT" \
  "attr" \
  "$ATTR_ALPHAS"

run_steered_grid \
  "attr/amber_attribute" \
  "${AMBER_ROOT}/attribute.jsonl" \
  "disjoint_v1_amber_attribute_attr" \
  "$AMBER_IMAGE_ROOT" \
  "$AMBER_ATTRIBUTE_LIMIT" \
  "attr" \
  "$ATTR_ALPHAS"

# rel_spatial / position. AMBER relation is a mismatch diagnostic only.
run_steered_grid \
  "rel_spatial/mme_position" \
  "${MME_ROOT}/position.jsonl" \
  "disjoint_v1_mme_position_rel_spatial" \
  "$MME_IMAGE_ROOT" \
  "$MME_LIMIT" \
  "rel" \
  "$REL_ALPHAS"

run_steered_grid \
  "rel_mismatch_diagnostic/amber_relation" \
  "${AMBER_ROOT}/relation.jsonl" \
  "disjoint_v1_amber_relation_rel_mismatch" \
  "$AMBER_IMAGE_ROOT" \
  "$AMBER_RELATION_LIMIT" \
  "rel" \
  "$REL_ALPHAS"

if ! truthy "$DRY_RUN"; then
  wait_gpu_jobs
  python scripts/summarize_disjoint_v1_sweep.py \
    --root "$RUN_ROOT" \
    --output "$RUN_ROOT/TARGETED_EVAL_REPORT.md" \
    --json-output "$RUN_ROOT/targeted_eval_stats.json"
else
  echo "[dry-run] No jobs launched. Set DRY_RUN=0 to execute."
fi
