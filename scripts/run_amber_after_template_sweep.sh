#!/usr/bin/env bash
set -euo pipefail

# AFTER-template expert steering sweep on prepared AMBER yes/no subsets.
#
# This script evaluates the discriminative yes/no side only. It does not run
# AMBER generative CHAIR evaluation.

MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
AMBER_ROOT="${AMBER_ROOT:-/home/huiwei/sy/benchmarks/AMBER}"
BENCH_ROOT="${BENCH_ROOT:-data/benchmarks/amber_hallucination}"
IMAGE_ROOT="${IMAGE_ROOT:-$AMBER_ROOT}"
VECTOR_PATH="${VECTOR_PATH:-data/outputs_after_template_disjoint_v1/steering/after_template_expert_vectors.pt}"
RUN_ROOT="${RUN_ROOT:-data/outputs_after_template_disjoint_v1/amber_runs}"
GPU="${GPU:-0}"
LIMIT="${LIMIT:-0}"
ALPHAS="${ALPHAS:-0.25 0.5 0.75 1 1.5 2}"
CATEGORIES="${CATEGORIES:-existence attribute relation}"

expert_for_category() {
  case "$1" in
    existence) echo "cat" ;;
    attribute) echo "attr" ;;
    relation) echo "rel" ;;
    *) echo "cat" ;;
  esac
}

router_for_expert() {
  case "$1" in
    cat) echo "force_cat" ;;
    attr) echo "force_attr" ;;
    rel) echo "force_rel" ;;
    *) echo "no_filter" ;;
  esac
}

if [[ ! -f "$BENCH_ROOT/stats.json" ]]; then
  python scripts/prepare_amber_hallucination.py \
    --amber-root "$AMBER_ROOT" \
    --image-root "$IMAGE_ROOT" \
    --out-dir "$BENCH_ROOT" \
    --categories $CATEGORIES \
    --overwrite
fi

if [[ ! -f "$VECTOR_PATH" ]]; then
  echo "Missing vector file: $VECTOR_PATH" >&2
  echo "Build disjoint after-template vectors first or set VECTOR_PATH=..." >&2
  exit 1
fi

for category in $CATEGORIES; do
  data_file="${BENCH_ROOT}/${category}.jsonl"
  if [[ ! -f "$data_file" ]]; then
    echo "Missing prepared AMBER file: $data_file" >&2
    exit 1
  fi
  if [[ ! -s "$data_file" ]]; then
    echo "Skipping empty AMBER category file: $data_file" >&2
    continue
  fi

  expert="$(expert_for_category "$category")"
  router="$(router_for_expert "$expert")"
  benchmark_name="amber_${category}_after_template"

  CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_steered_benchmark.py \
    --benchmark-data "$data_file" \
    --benchmark-name "$benchmark_name" \
    --out-dir "$RUN_ROOT/$category/baseline" \
    --adapter llava \
    --model-id "$MODEL_PATH" \
    --image-root "$IMAGE_ROOT" \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit "$LIMIT" \
    --progress-every 20 \
    --overwrite

  for alpha in $ALPHAS; do
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_steered_benchmark.py \
      --benchmark-data "$data_file" \
      --benchmark-name "$benchmark_name" \
      --out-dir "$RUN_ROOT/$category/${expert}_alpha${alpha}" \
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
      --steer-router "$router" \
      --steer-enabled-experts "$expert" \
      --steer-prefill true \
      --steer-decode true \
      --steer-apply-to last_token \
      --prefill-apply-to last_token \
      --decode-apply-to last_token \
      --overwrite
  done
done

python scripts/summarize_amber_after_template_results.py \
  --runs-root "$RUN_ROOT" \
  --dataset-root "$BENCH_ROOT" \
  --output "$RUN_ROOT/REPORT.md"
