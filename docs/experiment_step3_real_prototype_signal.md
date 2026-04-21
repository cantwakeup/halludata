# Experiment Step 3: Real Prototype Signal Audit

This step tests whether real LLaVA-1.5-7B activations contain a stable factual-vs-counterfactual signal for the four pair-bank subtypes: `cat`, `cnt`, `col`, and `rel`.

It does not run generation, steering, reranking metrics, or model training. It only reads merged activation caches that were already extracted with teacher-forced forward passes.

## Inputs

Expected merged activation caches:

- `data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/merged`
- `data/outputs/activations/llava_v15_7b/v0_mini_seed42/val/merged`
- `data/outputs/activations/llava_v15_7b/v0_mini_seed42/test/merged`

Each cache contains `z_pos` and `z_neg` with shape `[N, L, H, D]`.

## Prototype Signal Audit

For each subtype and layer-head, we normalize activations over the head dimension and compute:

- `mu_pos`: normalized mean positive activation.
- `mu_neg`: normalized mean negative activation.
- `axis`: normalized `mu_pos - mu_neg`.

Two score types are evaluated:

- `axis`: cosine similarity to the contrastive axis.
- `two_proto`: cosine to `mu_pos` minus cosine to `mu_neg`.

For each eval pair:

```text
score_pos = score(z_pos, subtype)
score_neg = score(z_neg, subtype)
pairwise_acc = mean(score_pos > score_neg)
```

Candidate-level metrics are also reported with positive branches as label 1 and negative branches as label 0:

- AUROC
- average precision
- accuracy
- balanced accuracy
- F1

Validation thresholds are chosen on val by maximizing balanced accuracy. Test uses the validation-selected thresholds.

## Label-Shuffle Control

The script also builds shuffled-label train prototypes by randomly swapping positive and negative labels per pair. This control should be close to random on val/test. If it is not, the score may be exploiting an artifact unrelated to factual-vs-counterfactual direction.

## Cross-Subtype Matrix

The cross-subtype matrix evaluates prototype subtype A on eval subtype B. A strong diagonal pattern suggests subtype-specific directions. Strong off-diagonal transfer suggests a more general factuality direction.

## Run Prototype Signal Audit

```bash
python scripts/eval_real_activation_signal.py \
  --train-cache data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/merged \
  --val-cache data/outputs/activations/llava_v15_7b/v0_mini_seed42/val/merged \
  --test-cache data/outputs/activations/llava_v15_7b/v0_mini_seed42/test/merged \
  --out-dir data/outputs/experiments/llava_v15_7b/v0_mini_seed42/prototype_signal \
  --score-types axis two_proto \
  --bootstrap 1000 \
  --seed 42 \
  --overwrite
```

Outputs:

- `prototypes.pt`
- `prototype_summary.json`
- `all_head_eval_val.json`
- `all_head_eval_test.json`
- `label_shuffle_control_val.json`
- `label_shuffle_control_test.json`
- `cross_subtype_matrix_val.json`
- `cross_subtype_matrix_test.json`

## Run Head Ranking

```bash
python scripts/run_real_head_ranking.py \
  --train-cache data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/merged \
  --val-cache data/outputs/activations/llava_v15_7b/v0_mini_seed42/val/merged \
  --test-cache data/outputs/activations/llava_v15_7b/v0_mini_seed42/test/merged \
  --prototype-path data/outputs/experiments/llava_v15_7b/v0_mini_seed42/prototype_signal/prototypes.pt \
  --out-dir data/outputs/experiments/llava_v15_7b/v0_mini_seed42/prototype_signal/head_ranking \
  --topk 8 16 32 64 128 256 \
  --random-repeats 100 \
  --seed 42 \
  --overwrite
```

Outputs:

- `head_ranking.json`
- `all_head_val.json`
- `all_head_test.json`
- `topk_sweep_val.json`
- `topk_eval_test.json`
- `random_topk_baseline_val.json`
- `random_topk_baseline_test.json`
- `topk_overlap_matrix.json`

## Plot Results

```bash
python scripts/plot_real_activation_results.py \
  --experiment-dir data/outputs/experiments/llava_v15_7b/v0_mini_seed42/prototype_signal \
  --out-dir data/outputs/experiments/llava_v15_7b/v0_mini_seed42/prototype_signal/plots \
  --overwrite
```

Plots:

- Per-subtype positive/negative score histograms.
- Per-subtype layer-head heatmaps.
- Top-K subtype overlap matrix.

## How To Interpret Results

Good signs:

- All-head pairwise accuracy is clearly above 0.5.
- AUROC/AP are above random controls.
- Label-shuffle control is near random.
- Head-ranking Top-K beats random Top-K.
- Cross-subtype diagonal is stronger than off-diagonal when using subtype-specific prototypes.

Warning signs:

- Label-shuffle control also performs well.
- All subtypes collapse to chance.
- Top-K does not beat random.
- Only one subtype works and others show no signal.

If the audit is positive, the next step should be a simple detection or reranking baseline before attempting activation steering.
