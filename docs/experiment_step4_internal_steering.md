# Experiment Step 4: Internal Steering Pilot

This step tests whether subtype steering vectors make LLaVA prefer factual candidates over counterfactual candidates on the held-out pair bank.

It does not run free-form generation, external benchmarks, or steering on AMBER/POPE/MME yet. It is a controlled internal check before spending GPU time on broader evaluation.

## What It Scores

For each pair:

```text
image + question + response_pos
image + question + response_neg
```

The script computes average teacher-forced answer-token log probability for both candidates:

```text
margin = log P(response_pos | image, question) - log P(response_neg | image, question)
pairwise_acc = mean(margin > 0)
```

Baseline uses no steering. Steered runs add the same subtype vector configuration to both positive and negative candidate forwards before `self_attn.o_proj`; the evaluator must not flip the direction based on which candidate is factual.

## Steering Vectors

The first steering vector is the subtype prototype axis:

```text
axis = normalize(mu_pos - mu_neg)
```

Heads come from `head_ranking.json`. Validation selects:

- `top_k`
- `alpha`
- `sign`

Test uses the validation-selected config.

## Recommended Smoke Run

Start small on validation/test samples:

```bash
python scripts/run_internal_steering_test.py \
  --val-pairs data/outputs/splits/v0_mini_seed42/pairs_val.jsonl \
  --test-pairs data/outputs/splits/v0_mini_seed42/pairs_test.jsonl \
  --prototype-path data/outputs/experiments/llava_v15_7b/v0_mini_seed42/prototype_signal/prototypes.pt \
  --head-ranking data/outputs/experiments/llava_v15_7b/v0_mini_seed42/prototype_signal/head_ranking/head_ranking.json \
  --out-dir data/outputs/experiments/llava_v15_7b/v0_mini_seed42/internal_steering_smoke \
  --adapter llava \
  --model-id /home/huiwei/sy/models/llava-1.5-7b-hf \
  --image-root /home/huiwei/sy/sy_data/COCO2017/train2017 \
  --instances-json /home/huiwei/sy/sy_data/COCO2017/annotations/instances_train2017.json \
  --device cuda:0 \
  --compute-dtype bfloat16 \
  --topk 16 32 \
  --alphas -0.5 0.5 1.0 \
  --signs 1 -1 \
  --max-val-samples-per-subtype 10 \
  --max-test-samples-per-subtype 10 \
  --progress-every 5 \
  --overwrite
```

## Full Internal Pilot

After the smoke run:

```bash
python scripts/run_internal_steering_test.py \
  --val-pairs data/outputs/splits/v0_mini_seed42/pairs_val.jsonl \
  --test-pairs data/outputs/splits/v0_mini_seed42/pairs_test.jsonl \
  --prototype-path data/outputs/experiments/llava_v15_7b/v0_mini_seed42/prototype_signal/prototypes.pt \
  --head-ranking data/outputs/experiments/llava_v15_7b/v0_mini_seed42/prototype_signal/head_ranking/head_ranking.json \
  --out-dir data/outputs/experiments/llava_v15_7b/v0_mini_seed42/internal_steering \
  --adapter llava \
  --model-id /home/huiwei/sy/models/llava-1.5-7b-hf \
  --image-root /home/huiwei/sy/sy_data/COCO2017/train2017 \
  --instances-json /home/huiwei/sy/sy_data/COCO2017/annotations/instances_train2017.json \
  --device cuda:0 \
  --compute-dtype bfloat16 \
  --topk 8 16 32 64 \
  --alphas -1.0 -0.5 0.5 1.0 \
  --signs 1 -1 \
  --progress-every 20 \
  --overwrite
```

## Outputs

- `steering_config.json`: validation-selected top-K, alpha, and sign per subtype.
- `val_tuning.json`: all validation sweep summaries.
- `test_eval.json`: baseline vs steered held-out results.
- `val_raw_scores.jsonl`: validation candidate scores.
- `test_raw_scores.jsonl`: test candidate scores.

## How To Interpret

Useful signs:

- `steered.pairwise_acc` beats `baseline.pairwise_acc` on test.
- `delta_mean_margin` is positive.
- Gains are not limited to one tiny subtype.

Warning signs:

- Validation improves but test drops.
- Only one sign/alpha works on a tiny sample.
- Margins improve while pairwise accuracy does not.

If the internal pilot is healthy, the next step is an external benchmark such as AMBER, with POPE and MME as secondary checks.
