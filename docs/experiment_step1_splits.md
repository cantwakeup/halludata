# Experiment Step 1: Image-Level Pair Splits

This step freezes a reproducible train/val/test split for the pair bank before any real LVLM activation extraction. It is experiment preparation, not a paper result by itself.

## Why Image-Level Splits

The pair bank can contain multiple fact-counterfact pairs from the same COCO image. If pairs from one image are split independently, the same visual scene can appear in train and test at the same time. That would let downstream prototype, detection, reranking, or steering experiments benefit from image-specific leakage instead of learning subtype-level hallucination signals.

An image-level split assigns all pairs with the same `image_id` to exactly one split. This keeps the evaluation closer to the intended setting: train on some images, tune on different images, and report on held-out images.

## Why Not Pair-Level Random Splits

Pair-level random splits are simpler but unsafe here. One image may produce a category pair, a count pair, a color pair, and several relation pairs. If those rows are randomly separated, future activation experiments could see the same image in both prototype construction and evaluation.

The split tool therefore groups by `image_id` first, then assigns image groups to train/val/test with deterministic greedy balancing.

## How The Split Is Used Later

Use the split files as the stable input for real LVLM experiments:

- Extract real LVLM activations separately for `pairs_train.jsonl`, `pairs_val.jsonl`, and `pairs_test.jsonl`.
- Build subtype prototypes only on train activations.
- Tune thresholds, selected heads, or reranking hyperparameters on val.
- Report hallucination detection, reranking, or steering results on test.

The current mock prototype/head-ranking scaffold can read split files, but scientific claims should wait until `MockActivationAdapter` is replaced by a real LVLM activation adapter.

## Command

```bash
python scripts/make_pair_splits.py \
  --pairs data/outputs/pairs_balanced_v0.jsonl \
  --out-dir data/outputs/splits/v0_mini_seed42 \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --seed 42 \
  --overwrite
```

## Outputs

The command writes:

- `pairs_train.jsonl`: train pairs, with original pair rows preserved.
- `pairs_val.jsonl`: validation pairs, with original pair rows preserved.
- `pairs_test.jsonl`: test pairs, with original pair rows preserved.
- `split_assignments.jsonl`: one row per image with its assigned split and subtype counts.
- `split_stats.json`: split sizes, subtype counts, template counts, and leakage checks.
- `dataset_manifest.json`: reproducibility metadata including input SHA-256 and split file names.

## What To Check

The most important sanity checks are in `split_stats.json`:

- `image_overlap_train_val` should be `0`.
- `image_overlap_train_test` should be `0`.
- `image_overlap_val_test` should be `0`.
- `num_duplicate_pair_ids` should be `0`.
- `num_missing_pair_ids` should be `0`.

The subtype distribution will not be perfect because the split is constrained by image groups, but it should be close enough for a stable first real-activation baseline.
