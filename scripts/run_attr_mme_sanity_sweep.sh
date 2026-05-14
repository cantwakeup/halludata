#!/usr/bin/env bash
set -euo pipefail

# Attribute steering sanity sweep on prepared MME count/color yes/no subsets.
# This wrapper intentionally avoids routers and large benchmark grids.

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_PATH="${MODEL_PATH:-/home/huiwei/sy/models/llava-1.5-7b-hf}"
BENCH_ROOT="${BENCH_ROOT:-data/benchmarks/mme_hallucination}"
IMAGE_ROOT="${IMAGE_ROOT:-${BENCH_ROOT}/images}"
RUN_ROOT="${RUN_ROOT:-data/outputs_attr_sanity_mme/runs}"
DATA_REPORT="${DATA_REPORT:-data/outputs_attr_sanity_mme/ATTR_MME_DATA_REPORT.md}"
SUMMARY_CSV="${SUMMARY_CSV:-data/outputs_attr_sanity_mme/summary.csv}"
SUMMARY_REPORT="${SUMMARY_REPORT:-data/outputs_attr_sanity_mme/ATTR_MME_SANITY_REPORT.md}"

GQA_VECTOR_PATH="${GQA_VECTOR_PATH:-data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt}"
DISJOINT_V2_VECTOR_PATH="${DISJOINT_V2_VECTOR_PATH:-}"
ATTR_COUNT_VECTOR_PATH="${ATTR_COUNT_VECTOR_PATH:-}"
ATTR_COLOR_VECTOR_PATH="${ATTR_COLOR_VECTOR_PATH:-}"

GPU="${GPU:-auto}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-false}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
STEER_LAYERS="${STEER_LAYERS:-5-25}"
STEER_K_HEADS="${STEER_K_HEADS:-64}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-bfloat16}"
RUN_CATEGORIES="${RUN_CATEGORIES:-count color}"

GLOBAL_ALPHAS="${GLOBAL_ALPHAS:-0.05 0.1 0.25 0.5 1.0}"
ATTR_ALPHAS="${ATTR_ALPHAS:-0.025 0.05 0.1 0.25 0.5 0.75 1.0}"
GLOBAL_ATTR_RES_ALPHAS="${GLOBAL_ATTR_RES_ALPHAS:-0.05 0.1 0.25 0.5}"

source scripts/gpu_sweep_utils.sh
init_gpu_scheduler

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

overwrite_arg() {
  if truthy "$OVERWRITE"; then
    echo "--overwrite"
  fi
}

alpha_token() {
  local raw="$1"
  raw="${raw//./p}"
  raw="${raw//- /m}"
  echo "$raw"
}

safe_name() {
  local raw="$1"
  raw="${raw//,/__plus__}"
  raw="${raw//\//_}"
  raw="${raw// /_}"
  echo "$raw"
}

run_logged() {
  local out_dir="$1"
  shift
  mkdir -p "$out_dir"
  local command="$*"
  run_gpu_job bash -c "${command} 2>&1 | tee '${out_dir}/log.txt'; status=\${PIPESTATUS[0]}; cp '${out_dir}/config.json' '${out_dir}/run_config.json' 2>/dev/null || true; exit \$status"
}

check_vector_key() {
  local vector_path="$1"
  local enabled="$2"
  "$PYTHON_BIN" - "$vector_path" "$enabled" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
keys = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
if not path.exists():
    print(f"missing_path:{path}")
    raise SystemExit(1)
try:
    import torch
    payload = torch.load(path, map_location="cpu")
except Exception as exc:
    print(f"load_error:{type(exc).__name__}:{exc}")
    raise SystemExit(1)
vectors = payload.get("vectors", payload) if isinstance(payload, dict) else {}
missing = [key for key in keys if key not in vectors]
if missing:
    print("missing_keys:" + ",".join(missing))
    raise SystemExit(1)
print("ok")
PY
}

discover_disjoint_v2_path() {
  if [[ -n "$DISJOINT_V2_VECTOR_PATH" ]]; then
    echo "$DISJOINT_V2_VECTOR_PATH"
    return
  fi
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
candidates = []
for root in (Path("data/outputs_after_template_disjoint_v2"), Path("data/after_template_disjoint_v2")):
    if root.exists():
        candidates.extend(sorted(root.rglob("*.pt")))
preferred = [
    path for path in candidates
    if "vector" in path.name.lower() or "steering" in str(path).lower()
]
print(str((preferred or candidates or [""])[0]))
PY
}

write_skip() {
  local category="$1"
  local vector_name="$2"
  local reason="$3"
  local skip_file="${RUN_ROOT}/SKIPPED.tsv"
  mkdir -p "$RUN_ROOT"
  if [[ ! -f "$skip_file" ]]; then
    printf "category\tvector\treason\n" > "$skip_file"
  fi
  printf "%s\t%s\t%s\n" "$category" "$vector_name" "$reason" >> "$skip_file"
  echo "[attr-mme] skipped category=${category} vector=${vector_name}: ${reason}"
}

run_baseline() {
  local category="$1"
  local data_file="${BENCH_ROOT}/${category}.jsonl"
  local out_dir="${RUN_ROOT}/${category}/baseline"
  local maybe_overwrite
  maybe_overwrite="$(overwrite_arg)"
  run_logged "$out_dir" \
    "$PYTHON_BIN" scripts/run_steered_benchmark.py \
      --benchmark-data "$data_file" \
      --benchmark-name "mme_${category}_attr_sanity" \
      --out-dir "$out_dir" \
      --adapter llava \
      --model-id "$MODEL_PATH" \
      --image-root "$IMAGE_ROOT" \
      --device cuda:0 \
      --compute-dtype "$COMPUTE_DTYPE" \
      --limit "$LIMIT" \
      --progress-every "$PROGRESS_EVERY" \
      $maybe_overwrite
}

run_vector_alpha() {
  local category="$1"
  local vector_name="$2"
  local vector_path="$3"
  local enabled="$4"
  local alpha="$5"
  local data_file="${BENCH_ROOT}/${category}.jsonl"
  local out_dir="${RUN_ROOT}/${category}/$(safe_name "$vector_name")_alpha$(alpha_token "$alpha")"
  local maybe_overwrite
  maybe_overwrite="$(overwrite_arg)"
  run_logged "$out_dir" \
    "$PYTHON_BIN" scripts/run_steered_benchmark.py \
      --benchmark-data "$data_file" \
      --benchmark-name "mme_${category}_attr_sanity" \
      --out-dir "$out_dir" \
      --adapter llava \
      --model-id "$MODEL_PATH" \
      --image-root "$IMAGE_ROOT" \
      --device cuda:0 \
      --compute-dtype "$COMPUTE_DTYPE" \
      --limit "$LIMIT" \
      --progress-every "$PROGRESS_EVERY" \
      --steer-enable \
      --steer-vector-path "$vector_path" \
      --steer-layers "$STEER_LAYERS" \
      --steer-alpha "$alpha" \
      --steer-k-heads "$STEER_K_HEADS" \
      --steer-head-select norm \
      --steer-router no_filter \
      --steer-enabled-experts "$enabled" \
      --steer-prefill true \
      --steer-decode true \
      --steer-apply-to last_token \
      --prefill-apply-to last_token \
      --decode-apply-to last_token \
      $maybe_overwrite
}

run_vector_grid() {
  local category="$1"
  local vector_name="$2"
  local vector_path="$3"
  local enabled="$4"
  local alphas="$5"
  local check_output
  if ! check_output="$(check_vector_key "$vector_path" "$enabled" 2>&1)"; then
    write_skip "$category" "$vector_name" "$check_output"
    return
  fi
  echo "[attr-mme] vector ready category=${category} vector=${vector_name} enabled=${enabled} path=${vector_path}"
  for alpha in $alphas; do
    run_vector_alpha "$category" "$vector_name" "$vector_path" "$enabled" "$alpha"
  done
}

mkdir -p "$RUN_ROOT"
rm -f "${RUN_ROOT}/SKIPPED.tsv"

"$PYTHON_BIN" scripts/inspect_attr_mme_data.py \
  --bench-root "$BENCH_ROOT" \
  --image-root "$IMAGE_ROOT" \
  --categories existence count color position \
  --output "$DATA_REPORT"

disjoint_v2_path="$(discover_disjoint_v2_path)"

for category in $RUN_CATEGORIES; do
  data_file="${BENCH_ROOT}/${category}.jsonl"
  if [[ ! -s "$data_file" ]]; then
    echo "Missing prepared MME file: $data_file" >&2
    echo "Prepare it first with scripts/prepare_mme_from_parquet.py." >&2
    exit 1
  fi

  run_baseline "$category"

  run_vector_grid "$category" "global_all" "$GQA_VECTOR_PATH" "global_all" "$GLOBAL_ALPHAS"
  run_vector_grid "$category" "attr" "$GQA_VECTOR_PATH" "attr" "$ATTR_ALPHAS"
  run_vector_grid "$category" "attr_res" "$GQA_VECTOR_PATH" "attr_res" "$ATTR_ALPHAS"
  run_vector_grid "$category" "global_all_plus_attr_res" "$GQA_VECTOR_PATH" "global_all,attr_res" "$GLOBAL_ATTR_RES_ALPHAS"
  write_skip "$category" "global_all_plus_attr_res_separate_alpha_grid" "current ExpertSteeringController exposes one shared --steer-alpha for all enabled vectors"

  if [[ -n "$disjoint_v2_path" ]]; then
    run_vector_grid "$category" "disjoint_v2_attr" "$disjoint_v2_path" "attr" "$ATTR_ALPHAS"
  else
    write_skip "$category" "disjoint_v2_attr" "no disjoint-v2 vector .pt discovered"
  fi

  if [[ "$category" == "count" ]]; then
    if [[ -n "$ATTR_COUNT_VECTOR_PATH" ]]; then
      run_vector_grid "$category" "attr_count" "$ATTR_COUNT_VECTOR_PATH" "attr_count" "$ATTR_ALPHAS"
    else
      write_skip "$category" "attr_count" "ATTR_COUNT_VECTOR_PATH not set"
    fi
  fi
  if [[ "$category" == "color" ]]; then
    if [[ -n "$ATTR_COLOR_VECTOR_PATH" ]]; then
      run_vector_grid "$category" "attr_color" "$ATTR_COLOR_VECTOR_PATH" "attr_color" "$ATTR_ALPHAS"
    else
      write_skip "$category" "attr_color" "ATTR_COLOR_VECTOR_PATH not set"
    fi
  fi
done

wait_gpu_jobs

"$PYTHON_BIN" scripts/summarize_attr_mme_sanity.py \
  --runs-root "$RUN_ROOT" \
  --output "$SUMMARY_CSV" \
  --report-output "$SUMMARY_REPORT" \
  --data-report "$DATA_REPORT"

echo "[attr-mme] summary: $SUMMARY_REPORT"
