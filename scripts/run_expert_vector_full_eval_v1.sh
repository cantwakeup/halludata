#!/usr/bin/env bash
set -euo pipefail

# Vector-only global/cat/attr/rel benchmark matrix.
# No subtype masks, no external masks, no router.

PROJECT_ROOT="${PROJECT_ROOT:-/home/huiwei/sy/halludata}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
ROOT="${ROOT:-data/expert_vector_full_eval_v1}"
MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-v1.5-7b-official-clean}"
LLAVA_REPO_PATH="${LLAVA_REPO_PATH:-/home/huiwei/sy/LLaVA-official-clean}"
CONV_MODE="${CONV_MODE:-llava_v1}"
GPU_POOL="${GPU_POOL:-0,1,2,3}"
PARALLEL="${PARALLEL:-true}"

ALPHAS="${ALPHAS:-0.01,0.05,0.1,0.25,0.5,0.75,1.0}"
VECTORS="${VECTORS:-global,cat,attr,rel}"
LIMIT="${LIMIT:-0}"
POPE_LIMIT="${POPE_LIMIT:-$LIMIT}"
AMBER_LIMIT="${AMBER_LIMIT:-$LIMIT}"
GQA_REL_LIMIT="${GQA_REL_LIMIT:-$LIMIT}"
OPTIONAL_LIMIT="${OPTIONAL_LIMIT:-$LIMIT}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
OVERWRITE="${OVERWRITE:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
COMPAT_NEW_TRANSFORMERS="${COMPAT_NEW_TRANSFORMERS:-false}"

POPE_ROOT="${POPE_ROOT:-/home/huiwei/sy/benchmarks/POPE}"
COCO_IMAGE_ROOT="${COCO_IMAGE_ROOT:-/home/huiwei/sy/sy_data/COCO2014/val2014}"
GQA_IMAGE_ROOT="${GQA_IMAGE_ROOT:-/home/huiwei/sy/sy_data/GQA/raw/images/images}"
POPE_DATASETS="${POPE_DATASETS:-MSCOCO GQA}"
POPE_SETTINGS="${POPE_SETTINGS:-random popular adversarial}"

AMBER_ROOT="${AMBER_ROOT:-/home/huiwei/sy/benchmarks/AMBER}"
AMBER_BENCH_ROOT="${AMBER_BENCH_ROOT:-data/benchmarks/amber_hallucination}"
AMBER_IMAGE_ROOT="${AMBER_IMAGE_ROOT:-$AMBER_ROOT}"
GQA_REL_JSONL="${GQA_REL_JSONL:-data/gqa_typeaware_v1/val_eval/gqa_rel_val.jsonl}"
CLEAN_VAL_JSONL="${CLEAN_VAL_JSONL:-data/clean_type_minpair_v2/minimal_pairs/val.jsonl}"
HPOPE_JSONL="${HPOPE_JSONL:-}"
HPOPE_IMAGE_ROOT="${HPOPE_IMAGE_ROOT:-$COCO_IMAGE_ROOT}"
MME_POSITION_JSONL="${MME_POSITION_JSONL:-data/benchmarks/mme_position/position.jsonl}"
MME_IMAGE_ROOT="${MME_IMAGE_ROOT:-/home/huiwei/sy/benchmarks/MME}"

RUN_POPE="${RUN_POPE:-true}"
RUN_AMBER_ATTRIBUTE="${RUN_AMBER_ATTRIBUTE:-true}"
RUN_GQA_RELATION="${RUN_GQA_RELATION:-true}"
RUN_OPTIONAL="${RUN_OPTIONAL:-true}"

RUNTIME_VECTOR_FILE="${RUNTIME_VECTOR_FILE:-${ROOT}/vectors/expert_vectors_runtime.pt}"
COMMANDS="${ROOT}/COMMANDS.md"

mkdir -p "${ROOT}" "${ROOT}/vectors"
: > "${COMMANDS}"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

common_flags=()
if truthy "${OVERWRITE}"; then
  common_flags+=(--overwrite)
fi
if truthy "${SKIP_EXISTING}"; then
  common_flags+=(--skip-existing)
fi
if truthy "${COMPAT_NEW_TRANSFORMERS}"; then
  common_flags+=(--compat-new-transformers)
fi

echo "# Expert vector full eval commands" >> "${COMMANDS}"
echo "" >> "${COMMANDS}"

echo "[expert-vector] inspect vectors and build runtime bundle"
echo "\`\`\`bash" >> "${COMMANDS}"
echo "${PYTHON_BIN} scripts/inspect_expert_vector_full_eval.py --output-root ${ROOT} --runtime-vector-output ${RUNTIME_VECTOR_FILE} --overwrite" >> "${COMMANDS}"
echo "\`\`\`" >> "${COMMANDS}"
"${PYTHON_BIN}" scripts/inspect_expert_vector_full_eval.py \
  --output-root "${ROOT}" \
  --runtime-vector-output "${RUNTIME_VECTOR_FILE}" \
  --overwrite

IFS=',' read -r -a GPUS <<< "${GPU_POOL}"
job_index=0
pids=()

run_job() {
  local name="$1"
  shift
  local out_dir="$1"
  shift
  mkdir -p "${out_dir}"
  local gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
  job_index=$((job_index + 1))
  echo "[expert-vector] launch ${name} gpu=${gpu}"
  {
    echo ""
    echo "## ${name}"
    echo ""
    echo "\`\`\`bash"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
    printf '%q ' "$@"
    echo ""
    echo "\`\`\`"
  } >> "${COMMANDS}"
  if truthy "${PARALLEL}" && [[ ${#GPUS[@]} -gt 1 ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "$@" > "${out_dir}/log.txt" 2>&1 &
    pids+=("$!")
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "$@" 2>&1 | tee "${out_dir}/log.txt"
  fi
}

eval_common=(
  "${PYTHON_BIN}" scripts/eval_expert_vectors_full.py
  --runtime-vector-file "${RUNTIME_VECTOR_FILE}"
  --vectors "${VECTORS}"
  --alphas "${ALPHAS}"
  --model-path "${MODEL_PATH}"
  --llava-repo-path "${LLAVA_REPO_PATH}"
  --conv-mode "${CONV_MODE}"
  --parser-mode contains_yes_no_octopus_like
  --do-sample
  --temperature 1.0
  --top-p 1.0
  --num-beams 1
  --max-new-tokens 1024
  --seed 42
  --topk 64
  --layers 0-31
  --prefill
  --decode
  --apply-to last_token
  --progress-every "${PROGRESS_EVERY}"
  "${common_flags[@]}"
)

if truthy "${RUN_POPE}" && [[ -d "${POPE_ROOT}" ]]; then
  read -r -a pope_datasets <<< "${POPE_DATASETS}"
  read -r -a pope_settings <<< "${POPE_SETTINGS}"
  run_job "pope_category" "${ROOT}/pope" \
    "${eval_common[@]}" \
    --benchmark-type pope \
    --benchmark-id pope \
    --benchmark-family category \
    --output-dir "${ROOT}/pope" \
    --pope-root "${POPE_ROOT}" \
    --coco-image-root "${COCO_IMAGE_ROOT}" \
    --gqa-image-root "${GQA_IMAGE_ROOT}" \
    --datasets "${pope_datasets[@]}" \
    --settings "${pope_settings[@]}" \
    --limit "${POPE_LIMIT}"
else
  echo "[expert-vector] skip POPE: RUN_POPE=${RUN_POPE}, POPE_ROOT=${POPE_ROOT}"
fi

if truthy "${RUN_AMBER_ATTRIBUTE}"; then
  if [[ ! -s "${AMBER_BENCH_ROOT}/stats.json" && -d "${AMBER_ROOT}" ]]; then
    echo "[expert-vector] prepare AMBER yes/no subsets"
    "${PYTHON_BIN}" scripts/prepare_amber_hallucination.py \
      --amber-root "${AMBER_ROOT}" \
      --image-root "${AMBER_IMAGE_ROOT}" \
      --out-dir "${AMBER_BENCH_ROOT}" \
      --categories existence attribute relation \
      --overwrite
  fi
  if [[ -s "${AMBER_BENCH_ROOT}/attribute.jsonl" ]]; then
    run_job "amber_attribute" "${ROOT}/amber_attribute" \
      "${eval_common[@]}" \
      --benchmark-type jsonl \
      --benchmark-id amber_attribute \
      --benchmark-family attribute \
      --output-dir "${ROOT}/amber_attribute" \
      --input-jsonl "${AMBER_BENCH_ROOT}/attribute.jsonl" \
      --image-root "${AMBER_IMAGE_ROOT}" \
      --dataset-name AMBER \
      --setting-name attribute \
      --limit "${AMBER_LIMIT}"
  else
    echo "[expert-vector] skip AMBER attribute: ${AMBER_BENCH_ROOT}/attribute.jsonl unavailable"
  fi
fi

if truthy "${RUN_GQA_RELATION}"; then
  if [[ -s "${GQA_REL_JSONL}" ]]; then
    run_job "gqa_relation" "${ROOT}/gqa_relation" \
      "${eval_common[@]}" \
      --benchmark-type jsonl \
      --benchmark-id gqa_relation \
      --benchmark-family relation \
      --output-dir "${ROOT}/gqa_relation" \
      --input-jsonl "${GQA_REL_JSONL}" \
      --image-root "/" \
      --dataset-name GQA \
      --setting-name relation \
      --limit "${GQA_REL_LIMIT}"
  elif [[ -s "${CLEAN_VAL_JSONL}" ]]; then
    echo "[expert-vector] GQA relation file missing; using clean relation controlled val"
    run_job "clean_relation_controlled" "${ROOT}/gqa_relation" \
      "${eval_common[@]}" \
      --benchmark-type jsonl \
      --benchmark-id clean_relation_controlled \
      --benchmark-family relation \
      --output-dir "${ROOT}/gqa_relation" \
      --input-jsonl "${CLEAN_VAL_JSONL}" \
      --image-root "/" \
      --dataset-name clean_type_minpair_v2 \
      --setting-name relation_controlled \
      --subtypes rel_left_right_clean,rel_above_below_clean,rel_holding_wearing_clean,rel_sitting_riding_clean \
      --limit "${GQA_REL_LIMIT}"
  else
    echo "[expert-vector] skip relation: no GQA relation or clean relation JSONL found"
  fi
fi

if truthy "${RUN_OPTIONAL}"; then
  if [[ -s "${AMBER_BENCH_ROOT}/existence.jsonl" ]]; then
    run_job "amber_existence" "${ROOT}/amber_existence" \
      "${eval_common[@]}" \
      --benchmark-type jsonl \
      --benchmark-id amber_existence \
      --benchmark-family category \
      --output-dir "${ROOT}/amber_existence" \
      --input-jsonl "${AMBER_BENCH_ROOT}/existence.jsonl" \
      --image-root "${AMBER_IMAGE_ROOT}" \
      --dataset-name AMBER \
      --setting-name existence \
      --limit "${OPTIONAL_LIMIT}"
  fi
  if [[ -s "${AMBER_BENCH_ROOT}/relation.jsonl" ]]; then
    run_job "amber_relation" "${ROOT}/amber_relation" \
      "${eval_common[@]}" \
      --benchmark-type jsonl \
      --benchmark-id amber_relation \
      --benchmark-family relation \
      --output-dir "${ROOT}/amber_relation" \
      --input-jsonl "${AMBER_BENCH_ROOT}/relation.jsonl" \
      --image-root "${AMBER_IMAGE_ROOT}" \
      --dataset-name AMBER \
      --setting-name relation \
      --limit "${OPTIONAL_LIMIT}"
  fi
  if [[ -n "${HPOPE_JSONL}" && -s "${HPOPE_JSONL}" ]]; then
    run_job "hpope" "${ROOT}/hpope" \
      "${eval_common[@]}" \
      --benchmark-type jsonl \
      --benchmark-id hpope \
      --benchmark-family attribute \
      --output-dir "${ROOT}/hpope" \
      --input-jsonl "${HPOPE_JSONL}" \
      --image-root "${HPOPE_IMAGE_ROOT}" \
      --dataset-name H-POPE \
      --setting-name all \
      --limit "${OPTIONAL_LIMIT}"
  fi
  if [[ -s "${MME_POSITION_JSONL}" ]]; then
    run_job "mme_position" "${ROOT}/mme_position" \
      "${eval_common[@]}" \
      --benchmark-type jsonl \
      --benchmark-id mme_position \
      --benchmark-family relation \
      --output-dir "${ROOT}/mme_position" \
      --input-jsonl "${MME_POSITION_JSONL}" \
      --image-root "${MME_IMAGE_ROOT}" \
      --dataset-name MME \
      --setting-name position \
      --limit "${OPTIONAL_LIMIT}"
  fi
fi

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" != "0" ]]; then
  echo "[expert-vector] one or more eval jobs failed; recent errors:" >&2
  grep -RIn "Traceback|Error:|FileNotFoundError|OutOfMemory|CUDA out of memory|CUDA error|Killed" "${ROOT}"/*/log.txt | tail -120 >&2 || true
  exit 1
fi

echo "[expert-vector] summarize matrix"
"${PYTHON_BIN}" scripts/summarize_expert_vector_matrix.py \
  --root "${ROOT}" \
  --output "${ROOT}/EXPERT_MATRIX_REPORT.md" \
  --report-output "${ROOT}/REPORT.md"

echo "[expert-vector] done"
echo "[expert-vector] inspect: ${ROOT}/INSPECT.md"
echo "[expert-vector] matrix: ${ROOT}/EXPERT_MATRIX_REPORT.md"
echo "[expert-vector] report: ${ROOT}/REPORT.md"
