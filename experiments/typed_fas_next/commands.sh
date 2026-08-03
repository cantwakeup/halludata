#!/usr/bin/env bash
set -euo pipefail

# Repo/context audit commands run in this pass.
find . -maxdepth 2 -type d | sort | sed -n '1,220p'
find . -maxdepth 2 -type f \( -iname 'README*' -o -iname '*REPORT*.md' -o -iname '*.md' \) | sort | sed -n '1,220p'
find scripts -maxdepth 1 -type f | sort | sed -n '1,240p'
sed -n '1,240p' data/after_fas_type_v1/REPORT.md
sed -n '1,260p' data/after_fas_type_v1/DATA_REPORT.md
sed -n '1,260p' data/after_fas_type_v1_gpt4omini_typed250_text/REPORT.md
sed -n '1,260p' data/after_fas_type_v1_gpt4omini_typed250_text/official_llava_vectors/raw_type_vectors.REPORT.md
sed -n '1,300p' data/after_fas_type_v1_gpt4omini_typed250_text/bench_sanity_raw_injection_v2_large/LARGE_RUN_REPORT.md
sed -n '1,260p' data/clean_type_minpair_v2/REPORT.md
sed -n '1,260p' data/clean_type_minpair_v2/vectors/CONDITION_VECTOR_REPORT.md
sed -n '1,240p' data/clean_type_minpair_v2/eval/heldout_mask_limit100_seed42/MASK_EVAL_REPORT.md

# Vector schema inspection.
python - <<'PY'
import torch
from pathlib import Path
for p in [
    'data/after_fas_type_v1_gpt4omini_typed250_text/official_llava_vectors/raw_type_vectors.pt',
    'data/after_fas_type_v1_gpt4omini_typed250_text/official_llava_vectors/runtime_raw_type_vectors.pt',
    'data/clean_type_minpair_v2/vectors/condition_vectors.pt',
]:
    obj = torch.load(Path(p), map_location='cpu', weights_only=False)
    print(p, list(obj.keys()) if isinstance(obj, dict) else type(obj))
    if isinstance(obj, dict) and isinstance(obj.get('vectors'), dict):
        print('vector keys:', list(obj['vectors'].keys())[:30])
PY

# Existing raw large selectivity reconstruction.
python - <<'PY'
import csv, pathlib
root = pathlib.Path('data/after_fas_type_v1_gpt4omini_typed250_text/bench_sanity_raw_injection_v2_large')
bench = [
    ('pope_mscoco_random_1000','cat'),
    ('pope_mscoco_popular_1000','cat'),
    ('pope_mscoco_adversarial_1000','cat'),
    ('amber_attribute_1000','attr'),
    ('gqa_relation_full802','rel'),
]
for dirname, matched in bench:
    rows = list(csv.DictReader(open(root / dirname / 'summary.csv')))
    base = next(row for row in rows if row['method'] == 'baseline')
    baseline_acc = float(base['accuracy'])
    by_vector = {}
    for vector in ['cat', 'attr', 'rel', 'global']:
        candidates = [row for row in rows if row.get('method') == 'steered' and row.get('vector') == vector]
        by_vector[vector] = max(candidates, key=lambda row: (float(row['accuracy']), float(row.get('f1') or 0)))
    matched_delta = float(by_vector[matched]['accuracy']) - baseline_acc
    mismatch_delta = max(float(by_vector[v]['accuracy']) - baseline_acc for v in by_vector if v not in [matched, 'global'])
    global_delta = float(by_vector['global']['accuracy']) - baseline_acc
    print(dirname, matched_delta - max(mismatch_delta, global_delta))
PY

# New vector-only experiment run.
python experiments/typed_fas_next/run_vector_only_diagnostics.py --overwrite

# GPU availability check that led to skipping real dev benchmark.
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# Candidate dev command template, not run in this pass.
# python scripts/run_steered_benchmark.py \
#   --benchmark-data <dev_json_or_jsonl> \
#   --benchmark-name <pope_dev_or_amber_dev_or_gqa_dev> \
#   --out-dir results/typed_fas_next/dev/<run_name> \
#   --adapter llava \
#   --model-id /home/huiwei/sy/models/llava-v1.5-7b \
#   --image-root <image_root> \
#   --instances-json <instances_json_if_needed> \
#   --device cuda:0 \
#   --limit 100 \
#   --steer-enable \
#   --steer-vector-path experiments/typed_fas_next/vector_only/vector_only_variants.pt \
#   --steer-head-select expert_map \
#   --steer-head-map experiments/typed_fas_next/vector_only/head_maps/contrast_l0p75__specificity_cos__top32.json \
#   --steer-expert-key contrast_l0p75_cat \
#   --steer-alpha 0.1 \
#   --steer-prefill true \
#   --steer-decode true \
#   --overwrite
