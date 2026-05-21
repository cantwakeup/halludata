#!/usr/bin/env bash
set -euo pipefail

# Held-out clean_type_minpair_v2 mask steering eval.

ROOT="${ROOT:-data/clean_type_minpair_v2}"
INPUT_JSONL="${INPUT_JSONL:-${ROOT}/minimal_pairs/val.jsonl}"
VECTOR_FILE="${VECTOR_FILE:-${ROOT}/vectors/condition_vectors.pt}"
MASK_FILE="${MASK_FILE:-${ROOT}/masks/condition_head_masks.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/eval/heldout_mask_limit${LIMIT_PER_SUBTYPE:-100}_seed${SEED:-42}}"
RUN_SPECS="${RUN_SPECS:-${OUTPUT_DIR}/run_specs.clean_v2.jsonl}"
MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-v1.5-7b-official-clean}"
LLAVA_REPO_PATH="${LLAVA_REPO_PATH:-/home/huiwei/sy/LLaVA-official-clean}"
CONV_MODE="${CONV_MODE:-llava_v1}"
GPU_POOL="${GPU_POOL:-0,1,2,3}"
SUBTYPES="${SUBTYPES:-attr_color_clean,attr_count_clean,attr_state_clean,attr_material_clean,attr_shape_clean,attr_action_single_clean,rel_left_right_clean,rel_above_below_clean,rel_holding_wearing_clean,rel_sitting_riding_clean}"
ALPHAS="${ALPHAS:-0.05,0.1,0.25,0.5}"
LIMIT_PER_SUBTYPE="${LIMIT_PER_SUBTYPE:-100}"
SEED="${SEED:-42}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
FORCE_PARALLEL="${FORCE_PARALLEL:-true}"
OVERWRITE="${OVERWRITE:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

mkdir -p "${OUTPUT_DIR}"

echo "[clean-v2-eval] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[clean-v2-eval] GPU_POOL=${GPU_POOL} SUBTYPES=${SUBTYPES}"
echo "[clean-v2-eval] LIMIT_PER_SUBTYPE=${LIMIT_PER_SUBTYPE} ALPHAS=${ALPHAS}"

python scripts/build_clean_type_mask_eval_specs.py \
  --output "${RUN_SPECS}" \
  --subtypes "${SUBTYPES}"

COMMON_ARGS=(
  --input-jsonl "${INPUT_JSONL}"
  --vector-file "${VECTOR_FILE}"
  --mask-file "${MASK_FILE}"
  --run-specs "${RUN_SPECS}"
  --alphas "${ALPHAS}"
  --limit-per-subtype "${LIMIT_PER_SUBTYPE}"
  --model-path "${MODEL_PATH}"
  --llava-repo-path "${LLAVA_REPO_PATH}"
  --conv-mode "${CONV_MODE}"
  --parser-mode contains_yes_no_octopus_like
  --do-sample
  --temperature 1.0
  --top-p 1.0
  --num-beams 1
  --max-new-tokens 1024
  --seed "${SEED}"
  --prefill
  --decode
  --apply-to last_token
  --progress-every "${PROGRESS_EVERY}"
)
if [[ "${OVERWRITE}" == "true" ]]; then
  COMMON_ARGS+=(--overwrite)
fi
if [[ "${SKIP_EXISTING}" == "true" ]]; then
  COMMON_ARGS+=(--skip-existing)
fi

IFS=',' read -r -a GPUS <<< "${GPU_POOL}"
IFS=',' read -r -a SUBTYPE_LIST <<< "${SUBTYPES}"

if [[ "${FORCE_PARALLEL}" == "true" && ${#GPUS[@]} -gt 1 ]]; then
  PARTS_ROOT="${OUTPUT_DIR}/parts"
  mkdir -p "${PARTS_ROOT}"
  SUBTYPE_GROUPS=()
  for ((i=0; i<${#GPUS[@]}; i++)); do
    SUBTYPE_GROUPS[$i]=""
  done
  for ((i=0; i<${#SUBTYPE_LIST[@]}; i++)); do
    idx=$((i % ${#GPUS[@]}))
    if [[ -z "${SUBTYPE_GROUPS[$idx]}" ]]; then
      SUBTYPE_GROUPS[$idx]="${SUBTYPE_LIST[$i]}"
    else
      SUBTYPE_GROUPS[$idx]="${SUBTYPE_GROUPS[$idx]},${SUBTYPE_LIST[$i]}"
    fi
  done
  pids=()
  for ((idx=0; idx<${#GPUS[@]}; idx++)); do
    if [[ -z "${SUBTYPE_GROUPS[$idx]}" ]]; then
      continue
    fi
    part_dir="${PARTS_ROOT}/part${idx}"
    mkdir -p "${part_dir}"
    echo "[clean-v2-eval] launch part${idx} gpu=${GPUS[$idx]} subtypes=${SUBTYPE_GROUPS[$idx]}"
    CUDA_VISIBLE_DEVICES="${GPUS[$idx]}" python scripts/eval_subtype_mask_steering.py \
      "${COMMON_ARGS[@]}" \
      --subtypes "${SUBTYPE_GROUPS[$idx]}" \
      --device cuda:0 \
      --output-dir "${part_dir}" > "${part_dir}/log.txt" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  if [[ "${status}" != "0" ]]; then
    echo "[clean-v2-eval] one or more parts failed; recent errors:" >&2
    grep -RIn "Error:|Traceback|OutOfMemory|CUDA out of memory|CUDA error|Killed|FileExistsError" "${PARTS_ROOT}"/part*/log.txt | tail -100 >&2 || true
    exit 1
  fi
  python scripts/merge_subtype_mask_eval_parts.py \
    --parts-root "${PARTS_ROOT}" \
    --output-dir "${OUTPUT_DIR}"
else
  python scripts/eval_subtype_mask_steering.py \
    "${COMMON_ARGS[@]}" \
    --subtypes "${SUBTYPES}" \
    --device cuda:0 \
    --output-dir "${OUTPUT_DIR}"
fi

python scripts/summarize_subtype_mask_eval.py \
  --summary-csv "${OUTPUT_DIR}/summary.csv" \
  --output "${OUTPUT_DIR}/MASK_EVAL_REPORT.md"

echo "[clean-v2-eval] done"
echo "[clean-v2-eval] summary: ${OUTPUT_DIR}/summary.csv"
echo "[clean-v2-eval] report: ${OUTPUT_DIR}/MASK_EVAL_REPORT.md"
