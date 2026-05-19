#!/usr/bin/env bash
set -euo pipefail

# Larger held-out subtype validation. This reuses existing subtype vectors and
# only rebuilds a larger validation JSONL; it does not re-extract activations.

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-v1.5-7b-official-clean}"
LLAVA_REPO_PATH="${LLAVA_REPO_PATH:-/home/huiwei/sy/LLaVA-official-clean}"
CONV_MODE="${CONV_MODE:-llava_v1}"

DATA_DIR="${DATA_DIR:-data/subtype_minpair_v1/minimal_pairs_large500}"
VECTOR_PATH="${VECTOR_PATH:-data/subtype_minpair_v1/vectors/subtype_vectors.pt}"
EVAL_ROOT="${EVAL_ROOT:-data/subtype_minpair_v1/eval/heldout_large500_seed42}"

VAL_PER_SUBTYPE="${VAL_PER_SUBTYPE:-500}"
TRAIN_CAT_PER_SUBTYPE="${TRAIN_CAT_PER_SUBTYPE:-600}"
TRAIN_ATTR_PER_SUBTYPE="${TRAIN_ATTR_PER_SUBTYPE:-500}"
TRAIN_REL_PER_SUBTYPE="${TRAIN_REL_PER_SUBTYPE:-500}"
SEED="${SEED:-42}"

GPU_POOL="${GPU_POOL:-0,1,2,3}"
ALPHAS="${ALPHAS:-0.05,0.1,0.25,0.5}"
VECTOR_KEYS="${VECTOR_KEYS:-g_all_clean,g_cat_clean,g_attr_clean,g_rel_clean,d_cat_random_g1_s05_clean,d_cat_popular_g1_s05_clean,d_cat_hard_g1_s05_clean,d_attr_color_g1_s05_clean,d_attr_count_g1_s05_clean,d_rel_spatial_g1_s05_clean,d_rel_contact_g1_s05_clean}"

PROMPT_SUFFIX="${PROMPT_SUFFIX:-Please answer this question with one word.}"
PARSER_MODE="${PARSER_MODE:-contains_yes_no_octopus_like}"
DO_SAMPLE="${DO_SAMPLE:-true}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
LAYERS="${LAYERS:-0-31}"
TOPK="${TOPK:-64}"
HEAD_SELECT="${HEAD_SELECT:-norm}"
PREFILL="${PREFILL:-true}"
DECODE="${DECODE:-true}"
APPLY_TO="${APPLY_TO:-last_token}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
REBUILD_DATA="${REBUILD_DATA:-false}"
OVERWRITE_SHARDS="${OVERWRITE_SHARDS:-false}"

echo "[large-subtype] data dir: ${DATA_DIR}"
echo "[large-subtype] eval root: ${EVAL_ROOT}"
echo "[large-subtype] vector path: ${VECTOR_PATH}"
echo "[large-subtype] val per subtype target: ${VAL_PER_SUBTYPE}"
echo "[large-subtype] vector keys: ${VECTOR_KEYS}"
echo "[large-subtype] alphas: ${ALPHAS}"

if [[ ! -s "${DATA_DIR}/val.jsonl" || "${REBUILD_DATA}" == "true" ]]; then
  echo "[large-subtype] building larger minimal-pair validation data"
  python scripts/build_subtype_minpair_data.py \
    --output-dir "${DATA_DIR}" \
    --train-cat-per-subtype "${TRAIN_CAT_PER_SUBTYPE}" \
    --train-attr-per-subtype "${TRAIN_ATTR_PER_SUBTYPE}" \
    --train-rel-per-subtype "${TRAIN_REL_PER_SUBTYPE}" \
    --val-cat-per-subtype "${VAL_PER_SUBTYPE}" \
    --val-attr-per-subtype "${VAL_PER_SUBTYPE}" \
    --val-rel-per-subtype "${VAL_PER_SUBTYPE}" \
    --seed "${SEED}" \
    --overwrite
else
  echo "[large-subtype] using existing ${DATA_DIR}/val.jsonl"
fi

mkdir -p "${EVAL_ROOT}/logs" "${EVAL_ROOT}/shards"

IFS=',' read -r -a GPUS <<< "${GPU_POOL}"
SUBTYPE_GROUPS=(
  "cat_random,attr_color"
  "cat_popular,attr_count"
  "cat_hard,rel_spatial"
  "rel_contact"
)

if (( ${#GPUS[@]} < ${#SUBTYPE_GROUPS[@]} )); then
  echo "[large-subtype] need at least ${#SUBTYPE_GROUPS[@]} GPUs in GPU_POOL, got ${GPU_POOL}" >&2
  exit 1
fi

PIDS=()
for idx in "${!SUBTYPE_GROUPS[@]}"; do
  gpu="${GPUS[$idx]}"
  subtypes="${SUBTYPE_GROUPS[$idx]}"
  shard_dir="${EVAL_ROOT}/shards/shard${idx}"
  log_path="${EVAL_ROOT}/logs/shard${idx}.log"

  extra_flags=()
  if [[ "${SKIP_EXISTING}" == "true" ]]; then
    extra_flags+=(--skip-existing)
  fi
  if [[ "${OVERWRITE_SHARDS}" == "true" ]]; then
    extra_flags+=(--overwrite)
  fi

  echo "[large-subtype] launch shard${idx} on GPU ${gpu}: ${subtypes}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/run_subtype_minpair_eval.py \
      --input-jsonl "${DATA_DIR}/val.jsonl" \
      --vector-path "${VECTOR_PATH}" \
      --vector-keys "${VECTOR_KEYS}" \
      --alphas "${ALPHAS}" \
      --output-dir "${shard_dir}" \
      --model-path "${MODEL_PATH}" \
      --llava-repo-path "${LLAVA_REPO_PATH}" \
      --conv-mode "${CONV_MODE}" \
      --device cuda:0 \
      --subtypes "${subtypes}" \
      --limit-per-subtype "${VAL_PER_SUBTYPE}" \
      --layers "${LAYERS}" \
      --topk "${TOPK}" \
      --head-select "${HEAD_SELECT}" \
      --prefill "${PREFILL}" \
      --decode "${DECODE}" \
      --apply-to "${APPLY_TO}" \
      --prefill-apply-to last_token \
      --decode-apply-to last_token \
      --prompt-suffix "${PROMPT_SUFFIX}" \
      --parser-mode "${PARSER_MODE}" \
      --do-sample "${DO_SAMPLE}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --num-beams "${NUM_BEAMS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}" \
      --progress-every "${PROGRESS_EVERY}" \
      "${extra_flags[@]}"
  ) > "${log_path}" 2>&1 &
  PIDS+=("$!")
done

echo "[large-subtype] waiting for ${#PIDS[@]} shards"
for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "[large-subtype] merging shard summaries"
python scripts/summarize_subtype_minpair_eval_shards.py --eval-root "${EVAL_ROOT}"

echo "[large-subtype] done"
echo "[large-subtype] report: ${EVAL_ROOT}/SUMMARY.md"
