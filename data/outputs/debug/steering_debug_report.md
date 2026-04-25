# Steering Debug Report

This report is a working debug template for the current typed expert steering run. It is intentionally focused on finding out why POPE Yes/No predictions did not change even though generation text changed.

## 1. Current Vector Artifact

- Vector file: `data/outputs/steering/expert_vectors.pt`
- Stats file: `data/outputs/steering/expert_vectors.stats.json`
- Experts: `cat`, `attr`, `rel`
- Direction: `mean(z_pos - z_neg)`, factual minus counterfactual
- Source activation position: answer last token
- Steering layers: `10-20`
- Shape per expert: `[11, 32, 128]`

Known cloud stats:

| expert | samples | mean_norm | max_norm |
| --- | ---: | ---: | ---: |
| cat | 268 | 0.199621 | 1.622453 |
| attr | 313 | 0.194143 | 1.687536 |
| rel | 77 | 0.041941 | 0.231102 |

## 2. Hook Sanity

Run with real cat vector:

```bash
python scripts/debug_steering_hook_sanity.py \
  --model-path /home/huiwei/sy/models/llava-1.5-7b-hf \
  --image-root /home/huiwei/sy/sy_data/COCO2014/val2014 \
  --pope-file /home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json \
  --sample-index 0 \
  --steer-vector-path data/outputs/steering/expert_vectors.pt \
  --steer-layers 10-20 \
  --steer-router force_cat \
  --steer-enabled-experts cat \
  --steer-alpha 12 \
  --steer-prefill true \
  --steer-decode false \
  --debug-log-hook-delta true
```

Run with random vector sanity control:

```bash
python scripts/debug_steering_hook_sanity.py \
  --model-path /home/huiwei/sy/models/llava-1.5-7b-hf \
  --image-root /home/huiwei/sy/sy_data/COCO2014/val2014 \
  --pope-file /home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json \
  --sample-index 0 \
  --steer-vector-path data/outputs/steering/expert_vectors.pt \
  --steer-layers 10-20 \
  --steer-router force_cat \
  --steer-enabled-experts cat \
  --steer-alpha 20 \
  --steer-prefill true \
  --steer-decode false \
  --debug-log-hook-delta true \
  --debug-random-vector true
```

Interpretation:

- If random vector gives `max_abs_logit_delta > 0`, hook wiring affects logits.
- If random vector does not move logits, inspect hook return, layer path, dtype/device, and layer index mapping.
- If random vector moves logits but cat vector barely moves logits, cat vector direction/scale/head choice is probably weak for first-token decisions.

## 3. First-Token Margin Sweep

Run POPE random 100, force cat, prefill only:

```bash
for a in 1 2 4 8 12; do
  CUDA_VISIBLE_DEVICES=0 python scripts/debug_first_token_margin.py \
    --model-path /home/huiwei/sy/models/llava-1.5-7b-hf \
    --pope-file /home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json \
    --image-root /home/huiwei/sy/sy_data/COCO2014/val2014 \
    --limit 100 \
    --steer-vector-path data/outputs/steering/expert_vectors.pt \
    --steer-layers 10-20 \
    --steer-router force_cat \
    --steer-enabled-experts cat \
    --steer-k-heads 64 \
    --steer-head-select norm \
    --steer-alpha $a \
    --steer-prefill true \
    --steer-decode false \
    --output data/outputs/debug/pope_random100_margin_cat_prefill_alpha${a}.jsonl
done
```

Fill after running:

| alpha | avg_delta_margin_all | avg_delta_margin_yes | avg_delta_margin_no | flip_count | baseline_acc | steered_acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD | TBD | TBD |
| 12 | TBD | TBD | TBD | TBD | TBD | TBD |

## 4. Prefill vs Decode

Run three modes on POPE random 100:

```bash
# prefill only
CUDA_VISIBLE_DEVICES=0 python scripts/debug_first_token_margin.py \
  --model-path /home/huiwei/sy/models/llava-1.5-7b-hf \
  --pope-file /home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json \
  --image-root /home/huiwei/sy/sy_data/COCO2014/val2014 \
  --limit 100 \
  --steer-vector-path data/outputs/steering/expert_vectors.pt \
  --steer-layers 10-20 \
  --steer-router force_cat \
  --steer-enabled-experts cat \
  --steer-alpha 8 \
  --steer-prefill true \
  --steer-decode false \
  --output data/outputs/debug/pope_random100_margin_prefill_only_alpha8.jsonl

# decode only, should not affect prompt-only first-token margin because no decode step runs
CUDA_VISIBLE_DEVICES=0 python scripts/debug_first_token_margin.py \
  --model-path /home/huiwei/sy/models/llava-1.5-7b-hf \
  --pope-file /home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json \
  --image-root /home/huiwei/sy/sy_data/COCO2014/val2014 \
  --limit 100 \
  --steer-vector-path data/outputs/steering/expert_vectors.pt \
  --steer-layers 10-20 \
  --steer-router force_cat \
  --steer-enabled-experts cat \
  --steer-alpha 8 \
  --steer-prefill false \
  --steer-decode true \
  --output data/outputs/debug/pope_random100_margin_decode_only_alpha8.jsonl

# both
CUDA_VISIBLE_DEVICES=0 python scripts/debug_first_token_margin.py \
  --model-path /home/huiwei/sy/models/llava-1.5-7b-hf \
  --pope-file /home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_random.json \
  --image-root /home/huiwei/sy/sy_data/COCO2014/val2014 \
  --limit 100 \
  --steer-vector-path data/outputs/steering/expert_vectors.pt \
  --steer-layers 10-20 \
  --steer-router force_cat \
  --steer-enabled-experts cat \
  --steer-alpha 8 \
  --steer-prefill true \
  --steer-decode true \
  --output data/outputs/debug/pope_random100_margin_both_alpha8.jsonl
```

Expected:

- Prefill-only and both can change first-token margin.
- Decode-only should not change prompt-only first-token margin.
- Decode-only can still change later generated wording in full generation.

## 5. Held-Out Activation Direction

Run offline direction check:

```bash
python scripts/eval_expert_vector_direction.py \
  --vector-path data/outputs/steering/expert_vectors.pt \
  --val-cache data/outputs/activations/llava_v15_7b/v0_mini_seed42/val/merged \
  --test-cache data/outputs/activations/llava_v15_7b/v0_mini_seed42/test/merged \
  --layers 10-20 \
  --output data/outputs/debug/expert_vector_direction_eval.json
```

Fill after running:

| split | group | n | dot_positive_rate | mean_dot | mean_cos |
| --- | --- | ---: | ---: | ---: | ---: |
| val | cat | TBD | TBD | TBD | TBD |
| val | attr_cnt | TBD | TBD | TBD | TBD |
| val | attr_col | TBD | TBD | TBD | TBD |
| val | rel | TBD | TBD | TBD | TBD |
| test | cat | TBD | TBD | TBD | TBD |
| test | attr_cnt | TBD | TBD | TBD | TBD |
| test | attr_col | TBD | TBD | TBD | TBD |
| test | rel | TBD | TBD | TBD | TBD |

Interpretation:

- If held-out dot positive rate is well above 0.5, the vector is directionally meaningful in activation space.
- If held-out direction is good but POPE logits do not move, injection timing/layer/head/scale is the likely issue.
- If held-out direction is weak, rebuild vectors or revisit activation position.

## 6. Generation Benchmark Sweep

Run baseline for each POPE split:

```bash
for split in random popular adversarial; do
  CUDA_VISIBLE_DEVICES=0 python scripts/run_steered_benchmark.py \
    --benchmark-data /home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_${split}.json \
    --benchmark-name pope_${split} \
    --out-dir data/outputs/runs/pope_${split}_baseline_500 \
    --adapter llava \
    --model-id /home/huiwei/sy/models/llava-1.5-7b-hf \
    --image-root /home/huiwei/sy/sy_data/COCO2014/val2014 \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --limit 500 \
    --progress-every 50 \
    --overwrite
done
```

Run steered sweep:

```bash
for split in random popular adversarial; do
  for a in 1 2 4 8 12; do
    CUDA_VISIBLE_DEVICES=0 python scripts/run_steered_benchmark.py \
      --benchmark-data /home/huiwei/sy/benchmarks/POPE/output/coco/coco_pope_${split}.json \
      --benchmark-name pope_${split} \
      --out-dir data/outputs/runs/pope_${split}_steer_cat_prefill_decode_alpha${a}_500 \
      --adapter llava \
      --model-id /home/huiwei/sy/models/llava-1.5-7b-hf \
      --image-root /home/huiwei/sy/sy_data/COCO2014/val2014 \
      --device cuda:0 \
      --compute-dtype bfloat16 \
      --limit 500 \
      --progress-every 50 \
      --steer-enable \
      --steer-vector-path data/outputs/steering/expert_vectors.pt \
      --steer-layers 10-20 \
      --steer-alpha $a \
      --steer-k-heads 64 \
      --steer-head-select norm \
      --steer-router force_cat \
      --steer-enabled-experts cat \
      --steer-prefill true \
      --steer-decode true \
      --overwrite
  done
done
```

## 7. Current Read

Known result before this debug pass:

- POPE random 100 baseline accuracy: `0.87`
- Cat steering alpha `0.25,0.5,1,2,4` with decode-only-style setup changed text increasingly often.
- Prediction flips remained `0`.
- This is consistent with steering changing later wording but not the first Yes/No decision.

Next decision:

- If prefill steering moves first-token margin, expand POPE popular/adversarial and tune alpha.
- If random-vector sanity moves logits but cat vector does not, inspect vector scale/head selection and consider top-K from real head ranking.
- If held-out direction is weak, rebuild vector using a better position such as diff span instead of answer-last-token.
- If rel remains weak, increase rel data before trying relation steering benchmark claims.
