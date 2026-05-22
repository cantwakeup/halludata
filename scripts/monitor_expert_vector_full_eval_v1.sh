#!/usr/bin/env bash
set -u

# Lightweight monitor for expert-vector eval runs.
# Safe to run repeatedly while jobs are active.

PROJECT_ROOT="${PROJECT_ROOT:-/home/huiwei/sy/halludata}"
ROOT="${ROOT:-data/expert_vector_full_eval_v1_6h}"
TAIL_N="${TAIL_N:-30}"

case "${ROOT}" in
  /*) ABS_ROOT="${ROOT}" ;;
  *) ABS_ROOT="${PROJECT_ROOT}/${ROOT}" ;;
esac

echo "== root =="
echo "${ABS_ROOT}"

echo ""
echo "== live processes =="
pgrep -af "eval_expert_vectors_full|run_expert_vector_full_eval" || true

echo ""
echo "== gpu =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader || nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

echo ""
echo "== latest progress =="
for name in pope amber_attribute gqa_relation amber_existence amber_relation hpope mme_position; do
  log="${ABS_ROOT}/${name}/log.txt"
  if [[ -s "${log}" ]]; then
    echo "---- ${log}"
    grep -a "processed" "${log}" | tail -n "${TAIL_N}" || true
    grep -a "Wrote .*summary" "${log}" | tail -1 || true
  fi
done

echo ""
echo "== completed group-run markers =="
for name in pope amber_attribute gqa_relation amber_existence amber_relation hpope mme_position; do
  log="${ABS_ROOT}/${name}/log.txt"
  if [[ -s "${log}" ]]; then
    completed=$(grep -a "processed" "${log}" | sed -n 's/.*processed \([0-9][0-9]*\)\/\([0-9][0-9]*\).*/\1 \2/p' | awk '$1 == $2 { c++ } END { print c + 0 }')
    echo "${name}: ${completed}"
  fi
done

echo ""
echo "== summaries =="
find "${ABS_ROOT}" -name summary.csv -print 2>/dev/null || true

echo ""
echo "== reports =="
ls -lh "${ABS_ROOT}/EXPERT_MATRIX_REPORT.md" "${ABS_ROOT}/REPORT.md" 2>/dev/null || true

echo ""
echo "== raw output files =="
find "${ABS_ROOT}" -path "*/raw/*.jsonl" -print 2>/dev/null | wc -l || true
