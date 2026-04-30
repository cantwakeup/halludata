#!/usr/bin/env bash
set -euo pipefail

# Fixed-positive AFTER-template cat steering sweep for POPE COCO random.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/huiwei/sy/sy_data/COCO2014/val2014}"
POPE_FILE="${POPE_FILE:-/home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json}"
VECTOR_PATH="${VECTOR_PATH:-data/outputs_after_template_v1/steering/after_template_expert_vectors.pt}"
LIMIT="${LIMIT:-500}"
GPU="${GPU:-auto}"
ALPHAS="${ALPHAS:-1 2 4 8}"
RUN_ROOT="${RUN_ROOT:-data/outputs_after_template_v1/runs}"

source scripts/gpu_sweep_utils.sh
init_gpu_scheduler

run_gpu_job python scripts/run_steered_benchmark.py \
  --benchmark-data "$POPE_FILE" \
  --benchmark-name pope_random_after_template \
  --out-dir "$RUN_ROOT/pope_random_after_template_baseline" \
  --adapter llava \
  --model-id "$MODEL_PATH" \
  --image-root "$IMAGE_ROOT" \
  --device cuda:0 \
  --compute-dtype bfloat16 \
  --limit "$LIMIT" \
  --progress-every 20 \
  --overwrite

for alpha in $ALPHAS; do
  run_gpu_job python scripts/run_steered_benchmark.py \
    --benchmark-data "$POPE_FILE" \
    --benchmark-name pope_random_after_template \
    --out-dir "$RUN_ROOT/pope_random_after_template_cat_alpha${alpha}" \
    --adapter llava \
    --model-id "$MODEL_PATH" \
    --image-root "$IMAGE_ROOT" \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit "$LIMIT" \
    --progress-every 20 \
    --steer-enable \
    --steer-vector-path "$VECTOR_PATH" \
    --steer-layers 10-20 \
    --steer-alpha "$alpha" \
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

wait_gpu_jobs

python scripts/summarize_after_template_results.py \
  --pairs-stats data/after_template_v1/pairs/stats.json \
  --vector-stats data/outputs_after_template_v1/steering/after_template_expert_vectors.stats.json \
  --activations-root data/outputs_after_template_v1/activations \
  --runs-root "$RUN_ROOT" \
  --output data/outputs_after_template_v1/REPORT.md
