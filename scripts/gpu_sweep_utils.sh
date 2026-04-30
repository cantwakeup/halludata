#!/usr/bin/env bash

# Shared GPU scheduler for bash sweep scripts.
#
# Behavior:
# - If every GPU in GPU_POOL is idle, launch jobs in batches across the pool.
# - Otherwise run jobs sequentially on GPU, or on the first idle GPU if GPU=auto.

GPU_POOL="${GPU_POOL:-0 1 2 3}"
GPU="${GPU:-auto}"
AUTO_PARALLEL="${AUTO_PARALLEL:-true}"
GPU_IDLE_MAX_MEMORY_MB="${GPU_IDLE_MAX_MEMORY_MB:-1024}"
GPU_IDLE_MAX_UTIL="${GPU_IDLE_MAX_UTIL:-5}"

PARALLEL_GPU_MODE=0
NEXT_GPU_SLOT=0
GPU_JOB_PIDS=()
ACTIVE_GPUS=()

_gpu_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

_gpu_pool_items() {
  echo "${GPU_POOL//,/ }"
}

_gpu_is_idle() {
  local gpu_id="$1"
  local stats
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  stats="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [[ -z "$stats" ]]; then
    return 1
  fi
  local mem util
  IFS=',' read -r mem util <<< "$stats"
  mem="${mem//[[:space:]]/}"
  util="${util//[[:space:]]/}"
  [[ -n "$mem" && -n "$util" ]] || return 1
  (( mem <= GPU_IDLE_MAX_MEMORY_MB && util <= GPU_IDLE_MAX_UTIL ))
}

_gpu_all_pool_idle() {
  local gpu_id
  for gpu_id in $(_gpu_pool_items); do
    _gpu_is_idle "$gpu_id" || return 1
  done
  return 0
}

_gpu_first_idle_or_default() {
  local gpu_id first_gpu=""
  for gpu_id in $(_gpu_pool_items); do
    if [[ -z "$first_gpu" ]]; then
      first_gpu="$gpu_id"
    fi
    if _gpu_is_idle "$gpu_id"; then
      echo "$gpu_id"
      return 0
    fi
  done
  echo "${first_gpu:-0}"
}

init_gpu_scheduler() {
  ACTIVE_GPUS=()
  GPU_JOB_PIDS=()
  NEXT_GPU_SLOT=0
  PARALLEL_GPU_MODE=0

  if _gpu_truthy "$AUTO_PARALLEL" && _gpu_all_pool_idle; then
    read -r -a ACTIVE_GPUS <<< "$(_gpu_pool_items)"
    PARALLEL_GPU_MODE=1
    echo "[gpu-scheduler] all GPUs idle in pool: ${ACTIVE_GPUS[*]}; running sweeps in parallel batches"
  else
    if [[ "$GPU" == "auto" ]]; then
      ACTIVE_GPUS=("$(_gpu_first_idle_or_default)")
    else
      ACTIVE_GPUS=("$GPU")
    fi
    echo "[gpu-scheduler] using single GPU: ${ACTIVE_GPUS[0]}"
  fi
}

wait_gpu_jobs() {
  local status=0
  local pid
  for pid in "${GPU_JOB_PIDS[@]}"; do
    wait "$pid" || status=$?
  done
  GPU_JOB_PIDS=()
  return "$status"
}

run_gpu_job() {
  if [[ "${#ACTIVE_GPUS[@]}" -eq 0 ]]; then
    init_gpu_scheduler
  fi
  local gpu_id="${ACTIVE_GPUS[$NEXT_GPU_SLOT]}"
  if [[ "$PARALLEL_GPU_MODE" -eq 1 ]]; then
    echo "[gpu-scheduler] launch GPU ${gpu_id}: $*"
    ( CUDA_VISIBLE_DEVICES="$gpu_id" "$@" ) &
    GPU_JOB_PIDS+=("$!")
    NEXT_GPU_SLOT=$(( (NEXT_GPU_SLOT + 1) % ${#ACTIVE_GPUS[@]} ))
    if (( ${#GPU_JOB_PIDS[@]} >= ${#ACTIVE_GPUS[@]} )); then
      wait_gpu_jobs
    fi
  else
    echo "[gpu-scheduler] run GPU ${gpu_id}: $*"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$@"
  fi
}
