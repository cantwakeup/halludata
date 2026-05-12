#!/usr/bin/env bash

# Regular-only POPE baseline alignment ablation.
#
# Runs:
#   1. current POPE + ours_greedy
#   2. current POPE + octopus_like
#   3. Octopus POPE + ours_greedy, if Octopus files exist
#   4. Octopus POPE + octopus_like, if Octopus files exist
#
# No CatExpert, no steering, no Octopus method.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source scripts/gpu_sweep_utils.sh

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-v1.5-7b-official-clean}"
MODEL_BASE="${MODEL_BASE:-}"
LLAVA_REPO="${LLAVA_REPO:-/home/huiwei/sy/LLaVA-official-clean}"
CONV_MODE="${CONV_MODE:-llava_v1}"
OURS_POPE_ROOT="${OURS_POPE_ROOT:-/home/huiwei/sy/benchmarks/POPE}"
OCTOPUS_ROOT="${OCTOPUS_ROOT:-/home/huiwei/sy/Octopus-master}"
COCO_IMAGE_ROOT="${COCO_IMAGE_ROOT:-/home/huiwei/sy/sy_data/COCO2014/val2014}"
GQA_IMAGE_ROOT="${GQA_IMAGE_ROOT:-/home/huiwei/sy/sy_data/GQA/raw/images/images}"
RUN_BASE="${RUN_BASE:-data/pope_cat_expert_eval/alignment_debug}"
DATASETS="${DATASETS:-MSCOCO}"
SETTINGS="${SETTINGS:-random popular adversarial}"
LIMIT="${LIMIT:-300}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-true}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
DEVICE="${DEVICE:-cuda:0}"
EXPECTED_TORCH="${EXPECTED_TORCH:-2.1.2+cu118}"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

shell_quote() {
  printf "%q" "$1"
}

save_env_snapshot() {
  local env_dir="$RUN_BASE/env_snapshot"
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

find_octopus_pope_root() {
  local candidate
  for candidate in \
    "$OCTOPUS_ROOT/data/POPE" \
    "$OCTOPUS_ROOT/POPE" \
    "$OCTOPUS_ROOT/data/pope" \
    "$OCTOPUS_ROOT/datasets/POPE"; do
    if [[ -d "$candidate" ]] && find "$candidate" -iname "*coco*pope*random*.json*" -print -quit | grep -q .; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

run_one() {
  local run_name="$1"
  local pope_root="$2"
  local preset="$3"
  local out_dir="$RUN_BASE/$run_name"
  local log_file="$out_dir/log.txt"
  local prompt_suffix
  local parser_mode
  local do_sample
  local temperature
  local top_p
  local max_new_tokens

  if [[ "$preset" == "ours_greedy" ]]; then
    prompt_suffix="Please answer this question in one word."
    parser_mode="first_yes_no"
    do_sample="false"
    temperature="0.0"
    top_p="1.0"
    max_new_tokens="5"
  elif [[ "$preset" == "octopus_like" ]]; then
    prompt_suffix="Please answer this question with one word."
    parser_mode="contains_yes_no_octopus_like"
    do_sample="true"
    temperature="1.0"
    top_p="1.0"
    max_new_tokens="1024"
  else
    echo "Unknown preset: $preset" >&2
    return 1
  fi

  mkdir -p "$out_dir"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd $(shell_quote "$PROJECT_ROOT")"
    echo "exec > >(tee $(shell_quote "$log_file")) 2>&1"
    printf "%q " "$PYTHON_BIN"
    printf "%q " "scripts/run_pope_official_cat_expert_eval.py"
    printf "%q %q " "--model-path" "$MODEL_PATH"
    if [[ -n "$MODEL_BASE" ]]; then printf "%q %q " "--model-base" "$MODEL_BASE"; fi
    printf "%q %q " "--llava-repo-path" "$LLAVA_REPO"
    printf "%q %q " "--conv-mode" "$CONV_MODE"
    printf "%q %q " "--pope-root" "$pope_root"
    printf "%q %q " "--coco-image-root" "$COCO_IMAGE_ROOT"
    printf "%q %q " "--gqa-image-root" "$GQA_IMAGE_ROOT"
    printf "%q " "--datasets"
    for dataset in $DATASETS; do printf "%q " "$dataset"; done
    printf "%q " "--settings"
    for setting in $SETTINGS; do printf "%q " "$setting"; done
    printf "%q %q " "--methods" "regular"
    printf "%q %q " "--limit" "$LIMIT"
    printf "%q %q " "--output-dir" "$out_dir"
    printf "%q %q " "--device" "$DEVICE"
    printf "%q %q " "--prompt-suffix" "$prompt_suffix"
    printf "%q %q " "--parser-mode" "$parser_mode"
    printf "%q %q " "--do-sample" "$do_sample"
    printf "%q %q " "--temperature" "$temperature"
    printf "%q %q " "--top-p" "$top_p"
    printf "%q %q " "--num-beams" "1"
    printf "%q %q " "--max-new-tokens" "$max_new_tokens"
    printf "%q %q " "--seed" "$SEED"
    printf "%q %q " "--progress-every" "$PROGRESS_EVERY"
    if truthy "$OVERWRITE"; then printf "%q " "--overwrite"; fi
    echo
    printf "%q " "$PYTHON_BIN"
    printf "%q " "scripts/summarize_pope_official_cat_expert_eval.py"
    printf "%q %q " "--runs-root" "$out_dir"
    printf "%q %q " "--output" "$out_dir/summary.csv"
    printf "%q %q " "--report-output" "$out_dir/SUMMARY.md"
    printf "%q %q " "--old-summary" ""
    echo
  } > "$out_dir/run.sh"
  chmod +x "$out_dir/run.sh"
  run_gpu_job bash "$out_dir/run.sh"
}

mkdir -p "$RUN_BASE"
save_env_snapshot
init_gpu_scheduler

echo "[alignment] current POPE root: $OURS_POPE_ROOT"
echo "[alignment] run base: $RUN_BASE"
echo "[alignment] limit: $LIMIT"

run_one "ours_pope_ours_decode" "$OURS_POPE_ROOT" "ours_greedy"
run_one "ours_pope_octopus_decode" "$OURS_POPE_ROOT" "octopus_like"

if octopus_pope_root="$(find_octopus_pope_root)"; then
  echo "[alignment] Octopus POPE root found: $octopus_pope_root"
  run_one "octopus_pope_ours_decode" "$octopus_pope_root" "ours_greedy"
  run_one "octopus_pope_octopus_decode" "$octopus_pope_root" "octopus_like"
else
  echo "[alignment] Octopus POPE files not found under $OCTOPUS_ROOT; skipping Octopus-POPE matrix cells."
fi

wait_gpu_jobs

echo "[alignment] done"
find "$RUN_BASE" -maxdepth 2 -name SUMMARY.md -print
