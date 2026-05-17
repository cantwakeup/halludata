#!/usr/bin/env bash

# Compare COCO-derived, GQA-derived, and mixed cat expert vectors on POPE.
#
# This wrapper keeps the aligned official baseline fixed:
#   prompt suffix: "Please answer this question with one word."
#   decode: do_sample=true, temperature=1.0, top_p=1.0, max_new_tokens=1024
#   parser: contains_yes_no_octopus_like
#
# It does not rebuild activations. It only builds compatible vector bundles from
# existing COCO/GQA expert-vector files, then calls the official POPE sweep.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

ROOT="${ROOT:-data/pope_cat_expert_eval/cat_domain_vector_comparison}"
VECTOR_DIR="${VECTOR_DIR:-$ROOT/vectors}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-$ROOT/runs}"

COCO_VECTOR="${COCO_VECTOR:-data/outputs_after_template_disjoint_v2/steering/after_template_disjoint_v2_expert_vectors.pt}"
GQA_VECTOR="${GQA_VECTOR:-data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt}"
MIX_MODE="${MIX_MODE:-unit_mean}"
MIXED_SCALE="${MIXED_SCALE:-mean_source_norm}"
VECTOR_LAYERS="${VECTOR_LAYERS:-intersection}"

DATASETS="${DATASETS:-MSCOCO GQA}"
SETTINGS="${SETTINGS:-random popular adversarial}"
METHODS="${METHODS:-regular cat}"
ALPHAS="${ALPHAS:-0 0.01 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.75 1.0 1.25 1.5 2.0}"
LIMIT="${LIMIT:-0}"

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-v1.5-7b-official-clean}"
LLAVA_REPO="${LLAVA_REPO:-/home/huiwei/sy/LLaVA-official-clean}"
POPE_ROOT="${POPE_ROOT:-/home/huiwei/sy/benchmarks/POPE}"
COCO_IMAGE_ROOT="${COCO_IMAGE_ROOT:-/home/huiwei/sy/sy_data/COCO2014/val2014}"
GQA_IMAGE_ROOT="${GQA_IMAGE_ROOT:-/home/huiwei/sy/sy_data/GQA/raw/images/images}"

STEER_LAYERS="${STEER_LAYERS:-5-25}"
STEER_TOPK="${STEER_TOPK:-64}"
HEAD_SELECT="${HEAD_SELECT:-norm}"
PREFILL="${PREFILL:-true}"
DECODE="${DECODE:-true}"
PROMPT_SUFFIX="${PROMPT_SUFFIX:-Please answer this question with one word.}"
PARSER_MODE="${PARSER_MODE:-contains_yes_no_octopus_like}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
DO_SAMPLE="${DO_SAMPLE:-true}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
SEED="${SEED:-42}"
GPU_POOL="${GPU_POOL:-0,1,2,3}"
FORCE_PARALLEL="${FORCE_PARALLEL:-true}"
OVERWRITE="${OVERWRITE:-false}"
VECTOR_OVERWRITE="${VECTOR_OVERWRITE:-true}"
CHILD_OVERWRITE="${CHILD_OVERWRITE:-true}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
SKIP_EXISTING_FILES="${SKIP_EXISTING_FILES:-true}"
COMPAT_NEW_TRANSFORMERS="${COMPAT_NEW_TRANSFORMERS:-false}"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

build_overwrite_arg=()
if truthy "$VECTOR_OVERWRITE"; then
  build_overwrite_arg=(--overwrite)
fi

echo "[cat-domain] building compatible vector bundles"
"$PYTHON_BIN" scripts/build_cat_domain_comparison_vectors.py \
  --coco-vector "$COCO_VECTOR" \
  --gqa-vector "$GQA_VECTOR" \
  --output-dir "$VECTOR_DIR" \
  --mix-mode "$MIX_MODE" \
  --mixed-scale "$MIXED_SCALE" \
  --layers "$VECTOR_LAYERS" \
  "${build_overwrite_arg[@]}"

declare -A VECTOR_PATHS=(
  [coco_cat]="$VECTOR_DIR/coco_cat_as_cat.pt"
  [gqa_cat]="$VECTOR_DIR/gqa_cat_as_cat.pt"
  [mixed_cat]="$VECTOR_DIR/mixed_cat_as_cat.pt"
)

for source in coco_cat gqa_cat mixed_cat; do
  echo "[cat-domain] running source=$source"
  RUN_ROOT="$RUN_ROOT_BASE/$source" \
  CAT_VECTOR_PATH="${VECTOR_PATHS[$source]}" \
  CAT_VECTOR_SOURCE="$source" \
  MODEL_PATH="$MODEL_PATH" \
  LLAVA_REPO="$LLAVA_REPO" \
  POPE_ROOT="$POPE_ROOT" \
  COCO_IMAGE_ROOT="$COCO_IMAGE_ROOT" \
  GQA_IMAGE_ROOT="$GQA_IMAGE_ROOT" \
  DATASETS="$DATASETS" \
  SETTINGS="$SETTINGS" \
  METHODS="$METHODS" \
  ALPHAS="$ALPHAS" \
  LIMIT="$LIMIT" \
  STEER_LAYERS="$STEER_LAYERS" \
  STEER_TOPK="$STEER_TOPK" \
  HEAD_SELECT="$HEAD_SELECT" \
  PREFILL="$PREFILL" \
  DECODE="$DECODE" \
  PROMPT_SUFFIX="$PROMPT_SUFFIX" \
  PARSER_MODE="$PARSER_MODE" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  DO_SAMPLE="$DO_SAMPLE" \
  TEMPERATURE="$TEMPERATURE" \
  TOP_P="$TOP_P" \
  NUM_BEAMS="$NUM_BEAMS" \
  SEED="$SEED" \
  GPU_POOL="$GPU_POOL" \
  FORCE_PARALLEL="$FORCE_PARALLEL" \
  OVERWRITE="$CHILD_OVERWRITE" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  SKIP_EXISTING_FILES="$SKIP_EXISTING_FILES" \
  COMPAT_NEW_TRANSFORMERS="$COMPAT_NEW_TRANSFORMERS" \
  bash scripts/run_pope_official_cat_expert_sweep.sh
done

"$PYTHON_BIN" scripts/summarize_cat_domain_vector_comparison.py \
  --run-root "$RUN_ROOT_BASE" \
  --output "$ROOT/summary.csv" \
  --report-output "$ROOT/SUMMARY.md"

echo "[cat-domain] done"
echo "[cat-domain] report: $ROOT/SUMMARY.md"
