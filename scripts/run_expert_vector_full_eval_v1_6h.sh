#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for the time-boxed expert-vector benchmark matrix.
# It intentionally disables optional benchmarks and runs only the main
# POPE / AMBER-attribute / relation matrix with a compact alpha grid.

PROFILE="${PROFILE:-6h}"

case "${PROFILE}" in
  6h)
    ROOT="${ROOT:-data/expert_vector_full_eval_v1_6h}"
    POPE_LIMIT="${POPE_LIMIT:-300}"
    AMBER_LIMIT="${AMBER_LIMIT:-300}"
    GQA_REL_LIMIT="${GQA_REL_LIMIT:-300}"
    ;;
  safe|9pm|9pm_safe)
    ROOT="${ROOT:-data/expert_vector_full_eval_v1_9pm_safe}"
    POPE_LIMIT="${POPE_LIMIT:-200}"
    AMBER_LIMIT="${AMBER_LIMIT:-250}"
    GQA_REL_LIMIT="${GQA_REL_LIMIT:-250}"
    ;;
  *)
    echo "Unknown PROFILE='${PROFILE}'. Use PROFILE=6h or PROFILE=safe." >&2
    exit 2
    ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/huiwei/sy/halludata}"
GPU_POOL="${GPU_POOL:-0,1,2,3}"
PARALLEL="${PARALLEL:-true}"
RUN_OPTIONAL="${RUN_OPTIONAL:-false}"
ALPHAS="${ALPHAS:-0.05,0.1,0.25,0.5}"
VECTORS="${VECTORS:-global,cat,attr,rel}"
OVERWRITE="${OVERWRITE:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

read -r -a alpha_arr <<< "${ALPHAS//,/ }"
read -r -a vector_arr <<< "${VECTORS//,/ }"
read -r -a pope_dataset_arr <<< "${POPE_DATASETS:-MSCOCO GQA}"
read -r -a pope_setting_arr <<< "${POPE_SETTINGS:-random popular adversarial}"

runs_per_group=$((1 + ${#alpha_arr[@]} * ${#vector_arr[@]}))
pope_generations=$((POPE_LIMIT * runs_per_group * ${#pope_dataset_arr[@]} * ${#pope_setting_arr[@]}))
amber_generations=$((AMBER_LIMIT * runs_per_group))
gqa_generations=$((GQA_REL_LIMIT * runs_per_group))
total_generations=$((pope_generations + amber_generations + gqa_generations))

case "${ROOT}" in
  /*) MONITOR_ROOT="${ROOT}" ;;
  *) MONITOR_ROOT="${PROJECT_ROOT}/${ROOT}" ;;
esac

echo "[expert-vector-6h] PROFILE=${PROFILE}"
echo "[expert-vector-6h] ROOT=${ROOT}"
echo "[expert-vector-6h] GPU_POOL=${GPU_POOL} PARALLEL=${PARALLEL}"
echo "[expert-vector-6h] RUN_OPTIONAL=${RUN_OPTIONAL}"
echo "[expert-vector-6h] ALPHAS=${ALPHAS} VECTORS=${VECTORS}"
echo "[expert-vector-6h] POPE_LIMIT=${POPE_LIMIT} AMBER_LIMIT=${AMBER_LIMIT} GQA_REL_LIMIT=${GQA_REL_LIMIT}"
echo "[expert-vector-6h] estimated main yes/no generations=${total_generations} (POPE=${pope_generations}, AMBER-attr=${amber_generations}, relation=${gqa_generations})"
echo "[expert-vector-6h] monitor with: ROOT=${MONITOR_ROOT} bash scripts/monitor_expert_vector_full_eval_v1.sh"

export PROJECT_ROOT
export ROOT
export GPU_POOL
export PARALLEL
export RUN_OPTIONAL
export ALPHAS
export VECTORS
export POPE_LIMIT
export AMBER_LIMIT
export GQA_REL_LIMIT
export OVERWRITE
export SKIP_EXISTING

exec bash scripts/run_expert_vector_full_eval_v1.sh
