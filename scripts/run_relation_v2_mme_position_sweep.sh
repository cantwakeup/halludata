#!/usr/bin/env bash
set -euo pipefail

# Relation-v2 MME position sweep with norm-selected and expert-map heads.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
BENCH_ROOT="${BENCH_ROOT:-data/benchmarks/mme_hallucination}"
IMAGE_ROOT="${IMAGE_ROOT:-${BENCH_ROOT}/images}"
POSITION_FILE="${POSITION_FILE:-${BENCH_ROOT}/position.jsonl}"
VECTOR_PATH="${VECTOR_PATH:-data/outputs_after_template_rel_v2/steering/relation_v2_vectors.pt}"
HEAD_MAP_ROOT="${HEAD_MAP_ROOT:-data/outputs_after_template_v1/head_analysis/head_maps}"
RUN_ROOT="${RUN_ROOT:-data/outputs_after_template_rel_v2/runs/mme_position}"
GPU="${GPU:-0}"
ALPHAS="${ALPHAS:-0.05 0.1 0.25 0.5 1.0}"

if [[ ! -f "$POSITION_FILE" ]]; then
  echo "Missing MME position JSONL: $POSITION_FILE" >&2
  echo "Run scripts/prepare_mme_from_parquet.py first." >&2
  exit 1
fi

run_eval() {
  local out_dir="$1"
  shift
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_steered_benchmark.py \
    --benchmark-data "$POSITION_FILE" \
    --benchmark-name mme_position_relation_v2 \
    --out-dir "$out_dir" \
    --adapter llava \
    --model-id "$MODEL_PATH" \
    --image-root "$IMAGE_ROOT" \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit 0 \
    --progress-every 20 \
    "$@" \
    --overwrite
}

run_eval "$RUN_ROOT/baseline"

for alpha in $ALPHAS; do
  run_eval "$RUN_ROOT/rel_all_norm64_mid_alpha${alpha}" \
    --steer-enable \
    --steer-vector-path "$VECTOR_PATH" \
    --steer-layers 10-20 \
    --steer-alpha "$alpha" \
    --steer-k-heads 64 \
    --steer-head-select norm \
    --steer-router force_rel \
    --steer-enabled-experts rel \
    --steer-prefill true \
    --steer-decode true
done

for k in 16 32 64 128; do
  map_path="$HEAD_MAP_ROOT/top${k}.json"
  for alpha in $ALPHAS; do
    run_eval "$RUN_ROOT/rel_all_expertmap_top${k}_alpha${alpha}" \
      --steer-enable \
      --steer-vector-path "$VECTOR_PATH" \
      --steer-alpha "$alpha" \
      --steer-head-select expert_map \
      --steer-head-map "$map_path" \
      --steer-expert-key rel_all \
      --steer-router no_filter \
      --steer-enabled-experts rel_all \
      --steer-prefill true \
      --steer-decode true
  done
done

for expert in rel_horizontal rel_vertical; do
  for k in 16 32 64; do
    map_path="$HEAD_MAP_ROOT/top${k}.json"
    for alpha in $ALPHAS; do
      run_eval "$RUN_ROOT/${expert}_expertmap_top${k}_alpha${alpha}" \
        --steer-enable \
        --steer-vector-path "$VECTOR_PATH" \
        --steer-alpha "$alpha" \
        --steer-head-select expert_map \
        --steer-head-map "$map_path" \
        --steer-expert-key "$expert" \
        --steer-router no_filter \
        --steer-enabled-experts "$expert" \
        --steer-prefill true \
        --steer-decode true
    done
  done
done

for expert in rel_left rel_right rel_above rel_below; do
  for k in 16 32 64; do
    map_path="$HEAD_MAP_ROOT/top${k}.json"
    for alpha in $ALPHAS; do
      run_eval "$RUN_ROOT/${expert}_expertmap_top${k}_alpha${alpha}" \
        --steer-enable \
        --steer-vector-path "$VECTOR_PATH" \
        --steer-alpha "$alpha" \
        --steer-head-select expert_map \
        --steer-head-map "$map_path" \
        --steer-expert-key "$expert" \
        --steer-router no_filter \
        --steer-enabled-experts "$expert" \
        --steer-prefill true \
        --steer-decode true
    done
  done
done

python scripts/summarize_relation_v2_and_heads.py \
  --relation-pairs-stats data/after_template_rel_v2/pairs/stats.json \
  --relation-vector-stats data/outputs_after_template_rel_v2/steering/relation_v2_vectors.stats.json \
  --head-analysis-dir data/outputs_after_template_v1/head_analysis \
  --mme-position-runs "$RUN_ROOT" \
  --sanity-runs data/outputs_after_template_v1/runs/expert_head_sanity \
  --output data/outputs_after_template_rel_v2/RELATION_AND_HEAD_ANALYSIS_REPORT.md
