# GQA Type-Aware Diagnostic

This diagnostic adds a GQA-based data source for checking whether the three expert vectors are type-specific:

- `cat_vector` should help most on object/category existence questions.
- `attr_vector` should help most on attribute questions such as color, state, material, size, and count.
- `rel_vector` should help most on object-object relation questions.

The goal is not to replace AMBER/MME. GQA scene graphs are used as a controlled diagnostic source because they provide object, attribute, and relation annotations in one format.

## Why Split Train And Val

`train_vector` is only for activation extraction and expert-vector construction. `val_eval` is only for benchmark evaluation. The builder checks image ids against the other split stats when available and raises an error if any image overlaps.

This keeps the experiment from asking whether the model memorized or overfit specific images used to build the steering directions.

## Build Data

Build the expert-vector data from GQA train:

```bash
python scripts/build_gqa_typeaware_data.py \
  --gqa-root /home/huiwei/sy/sy_data/GQA \
  --split train \
  --out-root data/gqa_typeaware_v1 \
  --max-cat 3000 \
  --max-attr 3000 \
  --max-rel 4000 \
  --seed 42
```

Build the diagnostic evaluation data from GQA val:

```bash
python scripts/build_gqa_typeaware_data.py \
  --gqa-root /home/huiwei/sy/sy_data/GQA \
  --split val \
  --out-root data/gqa_typeaware_v1 \
  --max-cat 1000 \
  --max-attr 1000 \
  --max-rel 1000 \
  --seed 42
```

Outputs:

- `data/gqa_typeaware_v1/train_vector/cat.jsonl`
- `data/gqa_typeaware_v1/train_vector/attr.jsonl`
- `data/gqa_typeaware_v1/train_vector/rel.jsonl`
- `data/gqa_typeaware_v1/train_vector/all.jsonl`
- `data/gqa_typeaware_v1/val_eval/gqa_cat_val.jsonl`
- `data/gqa_typeaware_v1/val_eval/gqa_attr_val.jsonl`
- `data/gqa_typeaware_v1/val_eval/gqa_rel_val.jsonl`
- `data/gqa_typeaware_v1/val_eval/gqa_all_val.jsonl`

Each row contains the GQA diagnostic fields plus the existing AFTER-template fields required by `scripts/extract_after_template_activations.py`.

## Build Expert Vectors

Extract activations with the existing AFTER-template extraction pipeline:

```bash
python scripts/extract_after_template_activations.py \
  --model-path /home/huiwei/sy/models/llava-1.5-7b-hf \
  --pair-file data/gqa_typeaware_v1/train_vector/all.jsonl \
  --image-root / \
  --output data/gqa_typeaware_v1/activations/train.pt \
  --metadata-output data/gqa_typeaware_v1/activations/train.meta.jsonl \
  --layers all \
  --position-mode last_token \
  --trusted-input-mode text_only \
  --batch-size 1 \
  --device cuda:0 \
  --overwrite
```

Build the three expert vectors with the existing vector builder:

```bash
python scripts/build_after_template_vectors.py \
  --activation-cache data/gqa_typeaware_v1/activations/train.pt \
  --metadata data/gqa_typeaware_v1/activations/train.meta.jsonl \
  --output data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt \
  --stats-output data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.stats.json \
  --layers 5-25 \
  --normalize false \
  --overwrite
```

The direction remains the repository's current convention:

```text
mean(z_text - z_visual)
```

where `z_visual` is image plus visual question prompt, and `z_text` is trusted factual text prompt.

## Run Eval

Run baseline plus cat/attr/rel vector injections on all three diagnostic subsets:

```bash
MODEL_PATH=/home/huiwei/sy/models/llava-1.5-7b-hf \
VECTOR_PATH=data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt \
GPU=0 \
OVERWRITE=true \
bash scripts/run_gqa_typeaware_eval.sh
```

Useful quick smoke setting:

```bash
LIMIT=50 ALPHAS="0.1" GPU=0 OVERWRITE=true bash scripts/run_gqa_typeaware_eval.sh
```

Outputs:

- `data/gqa_typeaware_v1/eval_runs/*/*/predictions.jsonl`
- `data/gqa_typeaware_v1/eval_runs/*/*/metrics.json`
- `data/gqa_typeaware_v1/eval_runs/*/*/run_config.json`
- `data/gqa_typeaware_v1/eval_runs/*/*/log.txt`
- `data/gqa_typeaware_v1/eval_runs/summary.csv`
- `data/gqa_typeaware_v1/eval_runs/SUMMARY.md`

## Read Summary

`summary.csv` compares every vector on every subset:

- `eval_subset`: `gqa_cat_val`, `gqa_attr_val`, or `gqa_rel_val`.
- `vector`: the injected vector, empty for baseline.
- `accuracy`: steered accuracy for steered runs, baseline accuracy for baseline rows.
- `baseline_accuracy`: matching subset baseline.
- `delta_acc`: accuracy change against the matching baseline.
- `wrong_to_right` / `right_to_wrong`: number of predictions corrected or broken by steering.
- `per_subtype_accuracy`: JSON object with subtype-level accuracy, especially useful for relation buckets.

The expected positive pattern is diagonal: cat vector helps cat-val most, attr vector helps attr-val most, and rel vector helps rel-val most. Strong off-diagonal drops are evidence of cross-type interference.

## Current Limits

- GQA is not a dedicated hallucination benchmark.
- Scene-graph relations are incomplete, so a generated negative relation is not guaranteed to be visually impossible.
- Attribute labels depend on GQA scene-graph annotation quality.
- Relation negatives are diagnostic controls, not official AMBER/MME labels.
- This flow does not implement a learned router. It only creates data and runs expert-vector cross-evaluation. An oracle router can later map cat questions to `cat`, attr questions to `attr`, and rel questions to `rel`.
