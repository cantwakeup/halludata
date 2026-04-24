# Step 6: External Benchmark Steering Runner

This step adds the first minimal benchmark loop for additive typed expert steering.

It supports a lightweight yes/no benchmark format that can cover POPE and the MME hallucination subset after converting or pointing to JSONL rows with common fields:

```json
{"image_id": 123, "question": "Is there a dog in the image?", "answer": "yes"}
{"image_path": "/path/to/image.jpg", "question": "Is the cup left of the plate?", "label": "no"}
```

Accepted field aliases:

```text
question/query/prompt/text
answer/label/gt_answer/ground_truth/target
image_id/id/coco_id
image_path/image/img/file_name
```

## Build Vectors

```bash
python scripts/build_expert_steering_vectors.py \
  --train-cache data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/merged \
  --output-path data/outputs/steering/expert_vectors.pt \
  --layers 10-20 \
  --max-samples-per-type 2000 \
  --normalize false \
  --overwrite
```

Current vector source:

```text
cat  = mean(z_pos - z_neg) over cat rows
attr = mean(z_pos - z_neg) over cnt + col rows
rel  = mean(z_pos - z_neg) over rel rows
```

## Baseline Benchmark

```bash
python scripts/run_steered_benchmark.py \
  --benchmark-data /path/to/pope_or_mme_yesno.jsonl \
  --benchmark-name pope \
  --out-dir data/outputs/runs/pope_baseline \
  --adapter llava \
  --model-id /home/huiwei/sy/models/llava-1.5-7b-hf \
  --image-root /home/huiwei/sy/sy_data/COCO2017/train2017 \
  --instances-json /home/huiwei/sy/sy_data/COCO2017/annotations/instances_train2017.json \
  --device cuda:0 \
  --compute-dtype bfloat16 \
  --limit 100 \
  --overwrite
```

## Steered Benchmark

```bash
python scripts/run_steered_benchmark.py \
  --benchmark-data /path/to/pope_or_mme_yesno.jsonl \
  --benchmark-name pope \
  --out-dir data/outputs/runs/pope_steer_alpha1 \
  --adapter llava \
  --model-id /home/huiwei/sy/models/llava-1.5-7b-hf \
  --image-root /home/huiwei/sy/sy_data/COCO2017/train2017 \
  --instances-json /home/huiwei/sy/sy_data/COCO2017/annotations/instances_train2017.json \
  --device cuda:0 \
  --compute-dtype bfloat16 \
  --limit 100 \
  --steer-enable \
  --steer-vector-path data/outputs/steering/expert_vectors.pt \
  --steer-layers 10-20 \
  --steer-alpha 1.0 \
  --steer-k-heads 64 \
  --steer-head-select norm \
  --steer-router no_filter \
  --steer-enabled-experts cat,attr,rel \
  --steer-prefill false \
  --overwrite
```

The runner also supports `--steer-router rule`, `force_cat`, `force_attr`, and `force_rel`.

## What The Hook Edits

The controller registers forward-pre hooks at selected layers' `self_attn.o_proj`.

For generation with cache enabled:

```text
hidden: [B, T, hidden_size]
reshape -> [B, T, num_heads, head_dim]
edit selected heads at the last decoding token
activation[layer][head] += alpha * expert_vector[layer][head]
reshape back -> [B, T, hidden_size]
```

Default intervention:

```text
layers: 10-20
head selection: top 64 layer-head pairs by vector norm
router: no_filter, so cat + attr + rel are summed
alpha: sweep 0.25, 0.5, 1.0, 2.0, 4.0
apply_to: last_token
steer_prefill: false
```

## Outputs

Each run writes:

```text
predictions.jsonl
metrics.json
config.json
```

Metrics include:

```text
accuracy
F1 for yes
yes-rate
average output length
number of samples
steering diagnostics when steering is enabled
```

## Current Limitations

This is still the simplest additive baseline. It does not implement a learned router, confidence gate, PCA, spherical rotation, or semantic retrieval. It also currently uses vectors built from answer-last-token activations. A stricter follow-up should rebuild vectors from diff-span activations.
