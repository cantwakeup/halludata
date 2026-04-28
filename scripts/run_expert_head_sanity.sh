#!/usr/bin/env bash
set -euo pipefail

# Sanity checks that expert-map Top64 keeps prior cat/attr gains alive.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
POPE_FILE="${POPE_FILE:-/home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json}"
POPE_IMAGE_ROOT="${POPE_IMAGE_ROOT:-/home/huiwei/sy/sy_data/COCO2014/val2014}"
MME_ROOT="${MME_ROOT:-data/benchmarks/mme_hallucination}"
MME_IMAGE_ROOT="${MME_IMAGE_ROOT:-${MME_ROOT}/images}"
VECTOR_PATH="${VECTOR_PATH:-data/outputs_after_template_v1/steering/after_template_expert_vectors.pt}"
HEAD_MAP="${HEAD_MAP:-data/outputs_after_template_v1/head_analysis/head_maps/top64.json}"
RUN_ROOT="${RUN_ROOT:-data/outputs_after_template_v1/runs/expert_head_sanity}"
GPU="${GPU:-0}"

for alpha in 1.0 1.5 2.0; do
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_steered_benchmark.py \
    --benchmark-data "$POPE_FILE" \
    --benchmark-name pope_random_after_template_expert_heads \
    --out-dir "$RUN_ROOT/pope_random_cat_top64_alpha${alpha}" \
    --adapter llava \
    --model-id "$MODEL_PATH" \
    --image-root "$POPE_IMAGE_ROOT" \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit 500 \
    --progress-every 20 \
    --steer-enable \
    --steer-vector-path "$VECTOR_PATH" \
    --steer-alpha "$alpha" \
    --steer-head-select expert_map \
    --steer-head-map "$HEAD_MAP" \
    --steer-expert-key cat \
    --steer-router no_filter \
    --steer-enabled-experts cat \
    --steer-prefill true \
    --steer-decode true \
    --overwrite
done

for alpha in 0.5 1.0 1.5; do
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_steered_benchmark.py \
    --benchmark-data "$MME_ROOT/color.jsonl" \
    --benchmark-name mme_color_after_template_expert_heads \
    --out-dir "$RUN_ROOT/mme_color_attr_top64_alpha${alpha}" \
    --adapter llava \
    --model-id "$MODEL_PATH" \
    --image-root "$MME_IMAGE_ROOT" \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit 0 \
    --progress-every 20 \
    --steer-enable \
    --steer-vector-path "$VECTOR_PATH" \
    --steer-alpha "$alpha" \
    --steer-head-select expert_map \
    --steer-head-map "$HEAD_MAP" \
    --steer-expert-key attr \
    --steer-router no_filter \
    --steer-enabled-experts attr \
    --steer-prefill true \
    --steer-decode true \
    --overwrite
done

python scripts/summarize_relation_v2_and_heads.py \
  --relation-pairs-stats data/after_template_rel_v2/pairs/stats.json \
  --relation-vector-stats data/outputs_after_template_rel_v2/steering/relation_v2_vectors.stats.json \
  --head-analysis-dir data/outputs_after_template_v1/head_analysis \
  --mme-position-runs data/outputs_after_template_rel_v2/runs/mme_position \
  --sanity-runs "$RUN_ROOT" \
  --output data/outputs_after_template_rel_v2/RELATION_AND_HEAD_ANALYSIS_REPORT.md
