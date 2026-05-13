#!/usr/bin/env bash

# Multi-GPU official-LLaVA POPE Regular/CatExpert sweep.
#
# This wrapper intentionally uses the official LLaVA runner instead of the old
# HF LlavaForConditionalGeneration POPE runner. It can run baseline-only:
#
#   METHODS=regular bash scripts/run_pope_official_cat_expert_sweep.sh
#
# or Regular + CatExpert alpha sweep:
#
#   METHODS="regular cat" CAT_VECTOR_PATH=... bash scripts/run_pope_official_cat_expert_sweep.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source scripts/gpu_sweep_utils.sh

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-v1.5-7b-official-clean}"
MODEL_BASE="${MODEL_BASE:-}"
LLAVA_REPO="${LLAVA_REPO:-/home/huiwei/sy/LLaVA-official-clean}"
CONV_MODE="${CONV_MODE:-llava_v1}"
POPE_ROOT="${POPE_ROOT:-/home/huiwei/sy/benchmarks/POPE}"
COCO_IMAGE_ROOT="${COCO_IMAGE_ROOT:-/home/huiwei/sy/sy_data/COCO2014/val2014}"
GQA_IMAGE_ROOT="${GQA_IMAGE_ROOT:-/home/huiwei/sy/sy_data/GQA/raw/images/images}"

RUN_ROOT="${RUN_ROOT:-data/pope_cat_expert_eval/official_llava_cat_expert_alpha_sweep_full}"
DATASETS="${DATASETS:-MSCOCO GQA}"
SETTINGS="${SETTINGS:-random popular adversarial}"
METHODS="${METHODS:-regular cat}"
ALPHAS="${ALPHAS:-0 0.01 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.75 1.0 1.25 1.5 2.0}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-false}"
SKIP_COMPLETED="${SKIP_COMPLETED:-false}"
SKIP_EXISTING_FILES="${SKIP_EXISTING_FILES:-false}"

CAT_VECTOR_PATH="${CAT_VECTOR_PATH:-}"
CAT_VECTOR_SOURCE="${CAT_VECTOR_SOURCE:-existing_hf_or_unknown}"
STEER_LAYERS="${STEER_LAYERS:-5-25}"
STEER_TOPK="${STEER_TOPK:-64}"
HEAD_SELECT="${HEAD_SELECT:-norm}"
PREFILL="${PREFILL:-true}"
DECODE="${DECODE:-true}"
APPLY_TO="${APPLY_TO:-last_token}"
PREFILL_APPLY_TO="${PREFILL_APPLY_TO:-last_token}"
DECODE_APPLY_TO="${DECODE_APPLY_TO:-last_token}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-5}"
DO_SAMPLE="${DO_SAMPLE:-false}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
PROMPT_SUFFIX="${PROMPT_SUFFIX:-Please answer this question in one word.}"
PARSER_MODE="${PARSER_MODE:-first_yes_no}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
EXPECTED_TORCH="${EXPECTED_TORCH:-2.1.2+cu118}"
OLD_HF_SUMMARY="${OLD_HF_SUMMARY:-data/pope_cat_expert_eval/full_alpha_sweep/summary.csv}"
COMPAT_NEW_TRANSFORMERS="${COMPAT_NEW_TRANSFORMERS:-false}"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

contains_word() {
  local haystack=" $1 "
  local needle=" $2 "
  [[ "$haystack" == *"$needle"* ]]
}

expected_raw_count_per_part() {
  local count=0
  local alpha
  if contains_word "$METHODS" "regular"; then
    count=$((count + 1))
  fi
  if contains_word "$METHODS" "cat"; then
    for alpha in $ALPHAS; do
      count=$((count + 1))
    done
  fi
  echo "$count"
}

shell_quote() {
  printf "%q" "$1"
}

pick_default_cat_vector() {
  local candidate
  for candidate in \
    "data/outputs_after_template_disjoint_v2/steering/after_template_disjoint_v2_expert_vectors.pt" \
    "data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt"; do
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

save_env_snapshot() {
  local env_dir="$RUN_ROOT/env_snapshot"
  mkdir -p "$env_dir"
  if command -v conda >/dev/null 2>&1; then
    conda list > "$env_dir/llava_official_conda_list.txt" || true
  fi
  "$PYTHON_BIN" -m pip freeze > "$env_dir/llava_official_pip_freeze.txt" || true
  EXPECTED_TORCH="$EXPECTED_TORCH" "$PYTHON_BIN" - <<'PY' > "$env_dir/llava_official_version_check.txt"
import os
import sys
import numpy as np
import torch, torchvision
import transformers, tokenizers, sentencepiece, accelerate
import google.protobuf

print("python:", sys.executable)
print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
print("sentencepiece:", sentencepiece.__version__)
print("accelerate:", accelerate.__version__)
print("protobuf:", google.protobuf.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)

expected = os.environ.get("EXPECTED_TORCH", "")
if expected and torch.__version__ != expected:
    raise SystemExit(f"torch version mismatch: expected {expected}, got {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY
}

write_root_config() {
  local git_commit
  git_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  RUN_ROOT="$RUN_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_BASE="$MODEL_BASE" \
  LLAVA_REPO="$LLAVA_REPO" \
  CONV_MODE="$CONV_MODE" \
  POPE_ROOT="$POPE_ROOT" \
  COCO_IMAGE_ROOT="$COCO_IMAGE_ROOT" \
  GQA_IMAGE_ROOT="$GQA_IMAGE_ROOT" \
  DATASETS="$DATASETS" \
  SETTINGS="$SETTINGS" \
  METHODS="$METHODS" \
  ALPHAS="$ALPHAS" \
  LIMIT="$LIMIT" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  SKIP_EXISTING_FILES="$SKIP_EXISTING_FILES" \
  CAT_VECTOR_PATH="$CAT_VECTOR_PATH" \
  CAT_VECTOR_SOURCE="$CAT_VECTOR_SOURCE" \
  STEER_LAYERS="$STEER_LAYERS" \
  STEER_TOPK="$STEER_TOPK" \
  HEAD_SELECT="$HEAD_SELECT" \
  PREFILL="$PREFILL" \
  DECODE="$DECODE" \
  APPLY_TO="$APPLY_TO" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  DO_SAMPLE="$DO_SAMPLE" \
  TEMPERATURE="$TEMPERATURE" \
  TOP_P="$TOP_P" \
  NUM_BEAMS="$NUM_BEAMS" \
  PROMPT_SUFFIX="$PROMPT_SUFFIX" \
  PARSER_MODE="$PARSER_MODE" \
  SEED="$SEED" \
  GIT_COMMIT="$git_commit" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
payload = {key.lower(): value for key, value in os.environ.items() if key in {
    "PYTHON_BIN", "MODEL_PATH", "MODEL_BASE", "LLAVA_REPO", "CONV_MODE",
    "POPE_ROOT", "COCO_IMAGE_ROOT", "GQA_IMAGE_ROOT", "DATASETS", "SETTINGS",
    "METHODS", "ALPHAS", "LIMIT", "SKIP_COMPLETED", "SKIP_EXISTING_FILES", "CAT_VECTOR_PATH", "CAT_VECTOR_SOURCE",
    "STEER_LAYERS", "STEER_TOPK", "HEAD_SELECT", "PREFILL", "DECODE",
    "APPLY_TO", "MAX_NEW_TOKENS", "DO_SAMPLE", "TEMPERATURE", "TOP_P",
    "NUM_BEAMS", "PROMPT_SUFFIX", "PARSER_MODE", "SEED", "GIT_COMMIT",
}}
payload["runner"] = "official_llava_multi_gpu_wrapper"
payload["env_snapshot"] = str(run_root / "env_snapshot")
run_root.mkdir(parents=True, exist_ok=True)
(run_root / "run_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

if contains_word "$METHODS" "cat"; then
  if [[ -z "$CAT_VECTOR_PATH" ]]; then
    CAT_VECTOR_PATH="$(pick_default_cat_vector || true)"
  fi
  if [[ -z "$CAT_VECTOR_PATH" || ! -f "$CAT_VECTOR_PATH" ]]; then
    echo "Error: METHODS includes cat, but CAT_VECTOR_PATH is missing." >&2
    echo "Set CAT_VECTOR_PATH to the expert-vector .pt file containing vectors['cat']." >&2
    exit 1
  fi
fi

mkdir -p "$RUN_ROOT/parts" "$RUN_ROOT/raw"
save_env_snapshot
write_root_config
init_gpu_scheduler

echo "[official-pope] run root: $RUN_ROOT"
echo "[official-pope] python: $PYTHON_BIN"
echo "[official-pope] model: $MODEL_PATH"
echo "[official-pope] llava repo: $LLAVA_REPO"
echo "[official-pope] datasets: $DATASETS"
echo "[official-pope] settings: $SETTINGS"
echo "[official-pope] methods: $METHODS"
echo "[official-pope] alphas: $ALPHAS"
echo "[official-pope] limit: $LIMIT"
echo "[official-pope] prompt suffix: $PROMPT_SUFFIX"
echo "[official-pope] parser mode: $PARSER_MODE"
echo "[official-pope] seed: $SEED"
echo "[official-pope] skip completed parts: $SKIP_COMPLETED"
echo "[official-pope] skip existing raw files: $SKIP_EXISTING_FILES"
if truthy "${FORCE_PARALLEL:-false}"; then
  PARALLEL_GPU_MODE=1
  read -r -a ACTIVE_GPUS <<< "${GPU_POOL//,/ }"
  NEXT_GPU_SLOT=0
  GPU_JOB_PIDS=()
  echo "[official-pope] FORCE_PARALLEL=true; forcing parallel GPU pool: ${ACTIVE_GPUS[*]}"
fi
if contains_word "$METHODS" "cat"; then
  echo "[official-pope] cat vector: $CAT_VECTOR_PATH"
fi

for dataset in $DATASETS; do
  for setting in $SETTINGS; do
    part_dir="$RUN_ROOT/parts/${dataset}_${setting}"
    mkdir -p "$part_dir"
    expected_files="$(expected_raw_count_per_part)"
    existing_files="0"
    if [[ -d "$part_dir/raw" ]]; then
      existing_files="$(find "$part_dir/raw" -maxdepth 1 -name "*.jsonl" -type f | wc -l | tr -d '[:space:]')"
    fi
    if truthy "$SKIP_COMPLETED" && (( existing_files >= expected_files )); then
      echo "[official-pope] skip completed part ${dataset}_${setting}: ${existing_files}/${expected_files} raw files"
      continue
    fi
    run_script="$part_dir/run.sh"
    log_file="$part_dir/log.txt"
    overwrite_arg=""
    skip_existing_arg=""
    compat_arg=""
    model_base_args=()
    cat_args=()
    if truthy "$OVERWRITE"; then
      overwrite_arg="--overwrite"
    fi
    if truthy "$SKIP_EXISTING_FILES"; then
      skip_existing_arg="--skip-existing"
    fi
    if truthy "$COMPAT_NEW_TRANSFORMERS"; then
      compat_arg="--compat-new-transformers"
    fi
    if [[ -n "$MODEL_BASE" ]]; then
      model_base_args=(--model-base "$MODEL_BASE")
    fi
    if contains_word "$METHODS" "cat"; then
      cat_args=(--cat-vector-path "$CAT_VECTOR_PATH" --cat-vector-source "$CAT_VECTOR_SOURCE")
    fi

    {
      echo "#!/usr/bin/env bash"
      echo "set -euo pipefail"
      echo "cd $(shell_quote "$PROJECT_ROOT")"
      echo "exec > >(tee $(shell_quote "$log_file")) 2>&1"
      echo "echo '[official-pope-part] dataset=${dataset} setting=${setting} CUDA_VISIBLE_DEVICES='\"\${CUDA_VISIBLE_DEVICES:-}\""
      printf "%q " "$PYTHON_BIN"
      printf "%q " "scripts/run_pope_official_cat_expert_eval.py"
      printf "%q %q " "--model-path" "$MODEL_PATH"
      for arg in "${model_base_args[@]}"; do printf "%q " "$arg"; done
      printf "%q %q " "--llava-repo-path" "$LLAVA_REPO"
      printf "%q %q " "--conv-mode" "$CONV_MODE"
      printf "%q %q " "--pope-root" "$POPE_ROOT"
      printf "%q %q " "--coco-image-root" "$COCO_IMAGE_ROOT"
      printf "%q %q " "--gqa-image-root" "$GQA_IMAGE_ROOT"
      printf "%q %q " "--datasets" "$dataset"
      printf "%q %q " "--settings" "$setting"
      printf "%q " "--methods"
      for method in $METHODS; do printf "%q " "$method"; done
      if contains_word "$METHODS" "cat"; then
        for arg in "${cat_args[@]}"; do printf "%q " "$arg"; done
        printf "%q %q " "--alphas" "$ALPHAS"
        printf "%q %q " "--layers" "$STEER_LAYERS"
        printf "%q %q " "--topk" "$STEER_TOPK"
        printf "%q %q " "--head-select" "$HEAD_SELECT"
        printf "%q %q " "--prefill" "$PREFILL"
        printf "%q %q " "--decode" "$DECODE"
        printf "%q %q " "--apply-to" "$APPLY_TO"
        printf "%q %q " "--prefill-apply-to" "$PREFILL_APPLY_TO"
        printf "%q %q " "--decode-apply-to" "$DECODE_APPLY_TO"
      fi
      printf "%q %q " "--limit" "$LIMIT"
      printf "%q %q " "--output-dir" "$part_dir"
      printf "%q %q " "--device" "$DEVICE"
      printf "%q %q " "--max-new-tokens" "$MAX_NEW_TOKENS"
      printf "%q %q " "--do-sample" "$DO_SAMPLE"
      printf "%q %q " "--temperature" "$TEMPERATURE"
      printf "%q %q " "--top-p" "$TOP_P"
      printf "%q %q " "--num-beams" "$NUM_BEAMS"
      printf "%q %q " "--prompt-suffix" "$PROMPT_SUFFIX"
      printf "%q %q " "--parser-mode" "$PARSER_MODE"
      printf "%q %q " "--seed" "$SEED"
      printf "%q %q " "--progress-every" "$PROGRESS_EVERY"
      if [[ -n "$overwrite_arg" ]]; then printf "%q " "$overwrite_arg"; fi
      if [[ -n "$skip_existing_arg" ]]; then printf "%q " "$skip_existing_arg"; fi
      if [[ -n "$compat_arg" ]]; then printf "%q " "$compat_arg"; fi
      echo
    } > "$run_script"
    chmod +x "$run_script"
    run_gpu_job bash "$run_script"
  done
done

wait_gpu_jobs

echo "[official-pope] merging raw predictions"
find "$RUN_ROOT/parts" -path "*/raw/*.jsonl" -type f -exec cp -f {} "$RUN_ROOT/raw/" \;

"$PYTHON_BIN" scripts/summarize_pope_official_cat_expert_eval.py \
  --runs-root "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.csv" \
  --report-output "$RUN_ROOT/SUMMARY.md" \
  --old-summary "$OLD_HF_SUMMARY"

echo "[official-pope] done"
echo "[official-pope] summary: $RUN_ROOT/SUMMARY.md"
