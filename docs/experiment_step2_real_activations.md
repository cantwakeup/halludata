# Experiment Step 2: Real LVLM Activation Extraction

This step prepares real activation caches for prototype, head-ranking, detection, reranking, and later steering experiments. It does not run generation, detection metrics, reranking metrics, steering, or model training.

## What This Step Does

For each pair row, the extraction script runs two teacher-forced forwards:

- `image + question + response_pos`
- `image + question + response_neg`

For LLaVA, the prompt is:

```text
USER: <image>
{question}
ASSISTANT: {response}
```

The adapter captures each decoder layer's attention-head output at the final non-padding answer token. The hook is registered on `self_attn.o_proj` as a forward pre-hook, so it stores the concatenated per-head values before the output projection and reshapes them to:

```text
[num_layers, num_heads, head_dim]
```

Across N pair rows, the cache stores:

```text
z_pos: [N, L, H, D]
z_neg: [N, L, H, D]
```

## Why This Is Needed

The current `MockActivationAdapter` only proves the engineering path. It cannot support paper claims. Real LVLM activations are required before we can evaluate prototype separation, subtype-specific head ranking, hallucination detection, reranking, or steering.

## Output Files

Each extraction directory contains:

- `activations.pt`: pair IDs, row indices, image IDs, subtypes, `z_pos`, and `z_neg`.
- `metadata.jsonl`: one row per pair with prompt text, image path, target token indices, and L/H/D metadata.
- `activation_manifest.json`: model, input, split, dtype, shard, timestamp, and reproducibility metadata.

## Single-GPU Smoke Run

```bash
python scripts/extract_activations.py \
  --pairs data/outputs/splits/v0_mini_seed42/pairs_train.jsonl \
  --image-root /home/huiwei/sy/sy_data/COCO2017/train2017 \
  --instances-json /home/huiwei/sy/sy_data/COCO2017/annotations/instances_train2017.json \
  --out-dir data/outputs/activations/llava_v15_7b/v0_mini_seed42/smoke_train_8 \
  --adapter llava \
  --model-id llava-hf/llava-1.5-7b-hf \
  --split train \
  --device cuda:0 \
  --compute-dtype bfloat16 \
  --storage-dtype float16 \
  --max-samples 8 \
  --overwrite
```

After the smoke run, check:

- `activations.pt` exists.
- `metadata.jsonl` has 8 rows.
- `z_pos` and `z_neg` have shape `[8, L, H, D]`.
- L/H/D match the model config.
- There are no NaNs.
- Positive and negative activations are not identical.

## 4xA100 Sharding

Each GPU runs one independent modulo shard. Use `CUDA_VISIBLE_DEVICES=$i` and keep the script device as `cuda:0` inside each process.

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/extract_activations.py \
    --pairs data/outputs/splits/v0_mini_seed42/pairs_train.jsonl \
    --image-root /home/huiwei/sy/sy_data/COCO2017/train2017 \
    --instances-json /home/huiwei/sy/sy_data/COCO2017/annotations/instances_train2017.json \
    --out-dir data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/shard_$i \
    --adapter llava \
    --model-id llava-hf/llava-1.5-7b-hf \
    --split train \
    --device cuda:0 \
    --compute-dtype bfloat16 \
    --storage-dtype float16 \
    --num-shards 4 \
    --shard-index $i \
    --overwrite &
done
wait
```

There is no DataParallel or distributed communication in this step. It is pure offline extraction.

## Merge Shards

```bash
python scripts/merge_activation_shards.py \
  --shard-dirs \
    data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/shard_0 \
    data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/shard_1 \
    data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/shard_2 \
    data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/shard_3 \
  --out-dir data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/merged \
  --overwrite
```

The merge script checks consistent L/H/D shapes, duplicate `row_index`, duplicate `pair_id`, and then sorts by original `row_index`.

## Later Use

- Train activations: build subtype prototypes and initial head rankings.
- Val activations: tune thresholds, top-K heads, or reranking rules.
- Test activations: report final detection or reranking results.

## Explicit Non-Goals

This step does not:

- Generate model answers.
- Compute detection metrics.
- Compute reranking metrics.
- Apply steering or intervention.
- Train or fine-tune a model.
