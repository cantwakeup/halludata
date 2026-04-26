# Legacy Experiment Registry

This directory records where the pre-AFTER-style halludata artifacts live. It is a manifest-only archive: no old data, activation cache, vector, or benchmark result is moved, deleted, or overwritten.

## Original COCO Fact-Counterfact Pair Bank

- Fact index: `data/outputs/fact_index_v0.jsonl`
- Atomic facts: `data/outputs/atomic_facts_v0.jsonl`
- Unbalanced pairs: `data/outputs/pairs_unbalanced_v0.jsonl`
- Balanced pairs: `data/outputs/pairs_balanced_v0.jsonl`
- Pair stats: `data/outputs/pair_stats_v0.json`
- Category negative bank: `data/outputs/cat_neg_bank_v2.json`
- Shell bank: `data/outputs/shell_bank_v1.json`

## Original Image-Level Split

- Split root: `data/outputs/splits/v0_mini_seed42/`
- Train pairs: `data/outputs/splits/v0_mini_seed42/pairs_train.jsonl`
- Val pairs: `data/outputs/splits/v0_mini_seed42/pairs_val.jsonl`
- Test pairs: `data/outputs/splits/v0_mini_seed42/pairs_test.jsonl`
- Assignments: `data/outputs/splits/v0_mini_seed42/split_assignments.jsonl`
- Stats: `data/outputs/splits/v0_mini_seed42/split_stats.json`
- Manifest: `data/outputs/splits/v0_mini_seed42/dataset_manifest.json`

## Original LLaVA Activation Caches

- Train cache: `data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/merged`
  - Shape: `[658, 32, 32, 128]`
- Val cache: `data/outputs/activations/llava_v15_7b/v0_mini_seed42/val/merged`
  - Shape: `[391, 32, 32, 128]`
- Test cache: `data/outputs/activations/llava_v15_7b/v0_mini_seed42/test/merged`
  - Shape: `[386, 32, 32, 128]`

## Original Expert Vectors

- Vector: `data/outputs/steering/expert_vectors.pt`
- Stats: `data/outputs/steering/expert_vectors.stats.json`
- Construction: `cat = cat`, `attr = cnt + col`, `rel = rel`
- Interpretation: useful for steering diagnostics, but the cat vector behaved mostly like an object-existence / Yes direction.

## Balanced Cat Truthfulness Experiment

- Pair bank root: `data/outputs/pair_banks/`
- Train pairs: `data/outputs/pair_banks/cat_truthfulness_train.jsonl`
- Val pairs: `data/outputs/pair_banks/cat_truthfulness_val.jsonl`
- Test pairs: `data/outputs/pair_banks/cat_truthfulness_test.jsonl`
- Activations:
  - `data/outputs/activations/cat_truthfulness_train.pt`
  - `data/outputs/activations/cat_truthfulness_val.pt`
  - `data/outputs/activations/cat_truthfulness_test.pt`
- Vector: `data/outputs/steering/cat_truth_vector.pt`
- Stats: `data/outputs/steering/cat_truth_vector.stats.json`
- Key result: `present_absent_cosine = -0.661042502152643`, so present and absent factual directions were strongly anti-aligned.

## POPE Steering Smoke And Margin Debug

- Old POPE smoke runs: `data/outputs/runs/`
- Cat truth POPE/debug outputs:
  - `data/outputs/debug/cat_truth_pope_random500_margin_alpha4.jsonl`
  - `data/outputs/debug/cat_truth_pope_random500_margin_alpha4.summary.json`
- Key result: fixed positive `cat_truth_vector` increased `yes_logit - no_logit` for both label=yes and label=no examples, so it remained closer to an existence/Yes direction than a label-agnostic truthfulness direction.

## Current Status

These legacy artifacts are preserved for comparison and ablation only. The AFTER-style v1 experiment writes only to:

- Pair bank: `data/after_style_v1/`
- New outputs: `data/outputs_after_style_v1/`

