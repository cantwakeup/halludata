#!/usr/bin/env bash
set -euo pipefail

# Fixed-sign cat truthfulness steering sweep for POPE COCO.
# This intentionally does not use oracle labels or signed routing.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/huiwei/sy/sy_data/COCO2014/val2014}"
POPE_ROOT="${POPE_ROOT:-/home/huiwei/sy/benchmarks/POPE/output/coco}"
VECTOR_PATH="${VECTOR_PATH:-data/outputs/steering/cat_truth_vector.pt}"
LIMIT="${LIMIT:-500}"
GPU="${GPU:-auto}"

source scripts/gpu_sweep_utils.sh
init_gpu_scheduler

for dataset in random popular adversarial; do
  run_gpu_job python scripts/run_steered_benchmark.py \
    --benchmark-data "${POPE_ROOT}/coco_pope_${dataset}.json" \
    --benchmark-name "pope_${dataset}" \
    --out-dir "data/outputs/runs/pope_cat_truth/${dataset}/baseline" \
    --adapter llava \
    --model-id "${MODEL_PATH}" \
    --image-root "${IMAGE_ROOT}" \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit "${LIMIT}" \
    --progress-every 50 \
    --overwrite

  for alpha in 1 2 4 8; do
    run_gpu_job python scripts/run_steered_benchmark.py \
      --benchmark-data "${POPE_ROOT}/coco_pope_${dataset}.json" \
      --benchmark-name "pope_${dataset}" \
      --out-dir "data/outputs/runs/pope_cat_truth/${dataset}/alpha${alpha}" \
      --adapter llava \
      --model-id "${MODEL_PATH}" \
      --image-root "${IMAGE_ROOT}" \
      --device cuda:0 \
      --compute-dtype bfloat16 \
      --limit "${LIMIT}" \
      --progress-every 50 \
      --steer-enable \
      --steer-vector-path "${VECTOR_PATH}" \
      --steer-layers 10-20 \
      --steer-alpha "${alpha}" \
      --steer-k-heads 64 \
      --steer-head-select norm \
      --steer-router force_cat \
      --steer-enabled-experts cat \
      --steer-prefill true \
      --steer-decode true \
      --steer-apply-to last_token \
      --prefill-apply-to last_token \
      --decode-apply-to last_token \
      --overwrite
  done
done

wait_gpu_jobs
