#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for clean_type_minpair_v2 train activation extraction and
# condition-balanced vector/mask construction. It intentionally does not run
# evaluation.

ROOT="${ROOT:-data/clean_type_minpair_v2}"
INPUT_JSONL="${INPUT_JSONL:-${ROOT}/minimal_pairs/train.jsonl}"
ACT_ROOT="${ACT_ROOT:-${ROOT}/activations}"
VEC_ROOT="${VEC_ROOT:-${ROOT}/vectors}"
MASK_ROOT="${MASK_ROOT:-${ROOT}/masks}"
MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-v1.5-7b-official-clean}"
LLAVA_REPO_PATH="${LLAVA_REPO_PATH:-/home/huiwei/sy/LLaVA-official-clean}"
CONV_MODE="${CONV_MODE:-llava_v1}"
GPU_POOL="${GPU_POOL:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-}"
DTYPE="${DTYPE:-float16}"
TOPK="${TOPK:-64}"
SEED="${SEED:-42}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
OVERWRITE="${OVERWRITE:-false}"
RUN_EXTRACT="${RUN_EXTRACT:-true}"
RUN_MERGE="${RUN_MERGE:-true}"
RUN_BUILD="${RUN_BUILD:-true}"
SAMPLE_NORMALIZE="${SAMPLE_NORMALIZE:-true}"
CONDITION_NORMALIZE="${CONDITION_NORMALIZE:-false}"
REMOVE_YESNO="${REMOVE_YESNO:-true}"

IFS=',' read -r -a GPUS <<< "${GPU_POOL}"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[clean-v2] GPU_POOL is empty" >&2
  exit 1
fi
if [[ -z "${NUM_SHARDS}" ]]; then
  NUM_SHARDS="${#GPUS[@]}"
fi

mkdir -p "${ACT_ROOT}/logs" "${VEC_ROOT}" "${MASK_ROOT}"

overwrite_arg=()
if [[ "${OVERWRITE}" == "true" ]]; then
  overwrite_arg=(--overwrite)
fi
sample_norm_arg=(--sample-normalize)
if [[ "${SAMPLE_NORMALIZE}" != "true" ]]; then
  sample_norm_arg=(--no-sample-normalize)
fi
condition_norm_arg=(--no-condition-normalize)
if [[ "${CONDITION_NORMALIZE}" == "true" ]]; then
  condition_norm_arg=(--condition-normalize)
fi
remove_yesno_arg=(--remove-yesno)
if [[ "${REMOVE_YESNO}" != "true" ]]; then
  remove_yesno_arg=(--no-remove-yesno)
fi

echo "[clean-v2] ROOT=${ROOT}"
echo "[clean-v2] RUN_EXTRACT=${RUN_EXTRACT} RUN_MERGE=${RUN_MERGE} RUN_BUILD=${RUN_BUILD}"
echo "[clean-v2] GPU_POOL=${GPU_POOL} NUM_SHARDS=${NUM_SHARDS} DTYPE=${DTYPE}"

if [[ "${RUN_EXTRACT}" == "true" ]]; then
  pids=()
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
    out="${ACT_ROOT}/train_shard${shard}.pt"
    meta="${ACT_ROOT}/train_shard${shard}.meta.jsonl"
    yesno="${ACT_ROOT}/train_shard${shard}.yesno.pt"
    log="${ACT_ROOT}/logs/train_shard${shard}.log"
    echo "[clean-v2] launch shard=${shard}/${NUM_SHARDS} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/extract_clean_type_minpair_activations.py \
      --input-jsonl "${INPUT_JSONL}" \
      --output "${out}" \
      --metadata-output "${meta}" \
      --yesno-output "${yesno}" \
      --model-path "${MODEL_PATH}" \
      --llava-repo-path "${LLAVA_REPO_PATH}" \
      --conv-mode "${CONV_MODE}" \
      --device cuda:0 \
      --dtype "${DTYPE}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --progress-every "${PROGRESS_EVERY}" \
      --seed "${SEED}" \
      "${overwrite_arg[@]}" > "${log}" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  if [[ "${status}" != "0" ]]; then
    echo "[clean-v2] extraction failed; recent errors:" >&2
    grep -RIn "Error:|Traceback|OutOfMemory|CUDA out of memory|CUDA error|Killed" "${ACT_ROOT}/logs"/train_shard*.log | tail -80 >&2 || true
    exit 1
  fi
fi

activation_files=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  activation_files+=("${ACT_ROOT}/train_shard${shard}.pt")
done

if [[ "${RUN_MERGE}" == "true" ]]; then
  python scripts/merge_subtype_minpair_activations.py \
    --activation-files "${activation_files[@]}" \
    --output "${ACT_ROOT}/train_activations.pt" \
    --metadata-output "${ACT_ROOT}/train_activations.meta.jsonl" \
    --yesno-output "${ACT_ROOT}/train_activations.yesno.pt" \
    "${overwrite_arg[@]}"
fi

if [[ "${RUN_BUILD}" == "true" ]]; then
  python scripts/build_clean_type_condition_vectors.py \
    --activations "${ACT_ROOT}/train_activations.pt" \
    --yesno-direction "${ACT_ROOT}/train_activations.yesno.pt" \
    --output "${VEC_ROOT}/condition_vectors.pt" \
    --mask-output "${MASK_ROOT}/condition_head_masks.pt" \
    --report-output "${VEC_ROOT}/CONDITION_VECTOR_REPORT.md" \
    --topk "${TOPK}" \
    "${sample_norm_arg[@]}" \
    "${condition_norm_arg[@]}" \
    "${remove_yesno_arg[@]}" \
    --seed "${SEED}" \
    "${overwrite_arg[@]}"
fi

echo "[clean-v2] done"
echo "[clean-v2] activations: ${ACT_ROOT}/train_activations.pt"
echo "[clean-v2] vectors: ${VEC_ROOT}/condition_vectors.pt"
echo "[clean-v2] masks: ${MASK_ROOT}/condition_head_masks.pt"
echo "[clean-v2] report: ${VEC_ROOT}/CONDITION_VECTOR_REPORT.md"
