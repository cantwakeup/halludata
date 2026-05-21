#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/huiwei/sy/halludata}
cd "$PROJECT_ROOT"

ROOT=${ROOT:-data/subtype_mask_steering_v1}
MINPAIR_ROOT=${MINPAIR_ROOT:-data/subtype_minpair_v1}
TRAIN_JSONL=${TRAIN_JSONL:-$MINPAIR_ROOT/minimal_pairs/train.jsonl}
VAL_JSONL=${VAL_JSONL:-$MINPAIR_ROOT/minimal_pairs/val.jsonl}
ACTIVATIONS=${ACTIVATIONS:-$MINPAIR_ROOT/activations/train_activations.pt}
VECTORS=${VECTORS:-$MINPAIR_ROOT/vectors/subtype_vectors.pt}
MASKS=${MASKS:-$ROOT/masks/subtype_head_masks.pt}
MASK_REPORT=${MASK_REPORT:-$ROOT/masks/MASK_REPORT.md}
INSPECT=${INSPECT:-$ROOT/INSPECT.md}
QUALITY_NOTES=${QUALITY_NOTES:-$ROOT/DATA_QUALITY_NOTES.md}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/eval/heldout}

MODEL_PATH=${MODEL_PATH:-/home/huiwei/sy/models/llava-v1.5-7b-official-clean}
LLAVA_REPO_PATH=${LLAVA_REPO_PATH:-/home/huiwei/sy/LLaVA-official-clean}
CONV_MODE=${CONV_MODE:-llava_v1}
GPU_POOL=${GPU_POOL:-}
FORCE_PARALLEL=${FORCE_PARALLEL:-false}
RUN_EVAL=${RUN_EVAL:-true}
OVERWRITE=${OVERWRITE:-false}
SKIP_EXISTING=${SKIP_EXISTING:-true}
LIMIT_PER_SUBTYPE=${LIMIT_PER_SUBTYPE:-0}
ALPHAS=${ALPHAS:-0.05,0.1,0.25,0.5}
PROGRESS_EVERY=${PROGRESS_EVERY:-20}
RUN_EVAL_NORM=$(printf '%s' "$RUN_EVAL" | tr '[:upper:]' '[:lower:]')

mkdir -p "$ROOT" "$ROOT/masks" "$EVAL_ROOT"

echo "[subtype-mask] config: RUN_EVAL=$RUN_EVAL_NORM FORCE_PARALLEL=$FORCE_PARALLEL GPU_POOL=$GPU_POOL LIMIT_PER_SUBTYPE=$LIMIT_PER_SUBTYPE ALPHAS=$ALPHAS"

echo "[subtype-mask] inspect inputs"
python scripts/inspect_subtype_mask_inputs.py \
  --train-jsonl "$TRAIN_JSONL" \
  --val-jsonl "$VAL_JSONL" \
  --activations "$ACTIVATIONS" \
  --vectors "$VECTORS" \
  --output "$INSPECT"

echo "[subtype-mask] build masks"
MASK_ARGS=(--activations "$ACTIVATIONS" --vectors "$VECTORS" --output "$MASKS" --report-output "$MASK_REPORT" --topk 64)
if [[ "$OVERWRITE" == "true" ]]; then
  MASK_ARGS+=(--overwrite)
fi
python scripts/build_subtype_head_masks.py "${MASK_ARGS[@]}"

echo "[subtype-mask] data quality notes"
python scripts/write_subtype_mask_data_quality_notes.py \
  --train-jsonl "$TRAIN_JSONL" \
  --val-jsonl "$VAL_JSONL" \
  --output "$QUALITY_NOTES"

case "$RUN_EVAL_NORM" in
true|1|yes|on)
  echo "[subtype-mask] eval gate accepted"
  COMMON_EVAL_ARGS=(
    --input-jsonl "$VAL_JSONL"
    --vector-file "$VECTORS"
    --mask-file "$MASKS"
    --model-path "$MODEL_PATH"
    --llava-repo-path "$LLAVA_REPO_PATH"
    --conv-mode "$CONV_MODE"
    --alphas "$ALPHAS"
    --limit-per-subtype "$LIMIT_PER_SUBTYPE"
    --parser-mode contains_yes_no_octopus_like
    --do-sample
    --temperature 1.0
    --top-p 1.0
    --num-beams 1
    --max-new-tokens 1024
    --seed 42
    --prefill
    --decode
    --apply-to last_token
    --progress-every "$PROGRESS_EVERY"
  )
  if [[ "$SKIP_EXISTING" == "true" ]]; then
    COMMON_EVAL_ARGS+=(--skip-existing)
  fi
  if [[ "$OVERWRITE" == "true" ]]; then
    COMMON_EVAL_ARGS+=(--overwrite)
  fi

  if [[ "$FORCE_PARALLEL" == "true" && -n "$GPU_POOL" ]]; then
    IFS=',' read -r -a GPUS <<< "$GPU_POOL"
    GROUPS=(
      "cat_random,attr_color"
      "cat_popular,attr_count"
      "cat_hard,rel_spatial"
      "rel_contact"
    )
    PART_ROOT="$EVAL_ROOT/parts"
    mkdir -p "$PART_ROOT"
    echo "[subtype-mask] parallel eval on GPU pool: $GPU_POOL"
    pids=()
    for idx in "${!GROUPS[@]}"; do
      gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      part_dir="$PART_ROOT/part$idx"
      mkdir -p "$part_dir"
      echo "[subtype-mask] launch part$idx GPU=$gpu subtypes=${GROUPS[$idx]}"
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_subtype_mask_steering.py \
        "${COMMON_EVAL_ARGS[@]}" \
        --subtypes "${GROUPS[$idx]}" \
        --device cuda:0 \
        --output-dir "$part_dir" \
        > "$part_dir/log.txt" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    python scripts/merge_subtype_mask_eval_parts.py \
      --parts-root "$PART_ROOT" \
      --output-dir "$EVAL_ROOT"
  else
    echo "[subtype-mask] sequential eval"
    python scripts/eval_subtype_mask_steering.py \
      "${COMMON_EVAL_ARGS[@]}" \
      --subtypes "cat_random,cat_popular,cat_hard,attr_color,attr_count,rel_spatial,rel_contact" \
      --device cuda \
      --output-dir "$EVAL_ROOT"
  fi

  echo "[subtype-mask] summarize held-out eval"
  python scripts/summarize_subtype_mask_eval.py \
    --summary-csv "$EVAL_ROOT/summary.csv" \
    --output "$EVAL_ROOT/MASK_EVAL_REPORT.md"
  ;;
*)
  echo "[subtype-mask] RUN_EVAL=$RUN_EVAL_NORM; skipping held-out eval"
  ;;
esac

echo "[subtype-mask] assemble final report"
python scripts/assemble_subtype_mask_report.py \
  --root "$ROOT" \
  --output "$ROOT/REPORT.md"

echo "[subtype-mask] done"
echo "[subtype-mask] report: $ROOT/REPORT.md"
