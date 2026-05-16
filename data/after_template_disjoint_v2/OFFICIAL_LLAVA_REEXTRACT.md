# Official LLaVA Cat Activation Re-Extraction

This note pins the official-LLaVA path for rebuilding the COCO category expert
vector before running the aligned POPE CatExpert experiment.

## Goal

Re-extract the AFTER-template `cat` activations with:

- Official LLaVA repo: `/home/huiwei/sy/LLaVA-official-clean`
- Official checkpoint: `/home/huiwei/sy/models/llava-v1.5-7b-official-clean`
- Trusted branch: question + factual text prompt, text-only
- Untrusted branch: image + question prompt
- Direction: `z_text - z_visual`

The new extractor writes the same cache schema as the old HF extractor, so
`scripts/build_after_template_vectors.py` can consume it directly.

## 1. Smoke Extract

```bash
conda activate llava_official
cd /home/huiwei/sy/halludata

python scripts/extract_after_template_activations_official_llava.py \
  --model-path /home/huiwei/sy/models/llava-v1.5-7b-official-clean \
  --llava-repo-path /home/huiwei/sy/LLaVA-official-clean \
  --conv-mode llava_v1 \
  --pair-file data/after_template_disjoint_v2/pairs/train.jsonl \
  --image-root /home/huiwei/sy/sy_data/COCO2014/train2014 \
  --instances-json /home/huiwei/sy/sy_data/COCO2014/annotations/instances_train2014.json \
  --types cat \
  --trusted-input-mode text_only \
  --output data/after_template_disjoint_v2/official_llava_activations/cat_train_smoke.pt \
  --metadata-output data/after_template_disjoint_v2/official_llava_activations/cat_train_smoke.meta.jsonl \
  --max-samples 100 \
  --progress-every 10 \
  --seed 42 \
  --overwrite
```

## 2. Build Smoke Vector

```bash
python scripts/build_after_template_vectors.py \
  --activation-cache data/after_template_disjoint_v2/official_llava_activations/cat_train_smoke.pt \
  --metadata data/after_template_disjoint_v2/official_llava_activations/cat_train_smoke.meta.jsonl \
  --output data/after_template_disjoint_v2/official_llava_steering/official_llava_cat_expert_vectors_smoke.pt \
  --stats-output data/after_template_disjoint_v2/official_llava_steering/official_llava_cat_expert_vectors_smoke.stats.json \
  --layers 5-25 \
  --normalize false \
  --overwrite
```

## 3. POPE Smoke With Re-Extracted Cat Vector

```bash
RUN_ROOT=data/pope_cat_expert_eval/official_reextract_cat_smoke_seed42 \
CAT_VECTOR_PATH=data/after_template_disjoint_v2/official_llava_steering/official_llava_cat_expert_vectors_smoke.pt \
CAT_VECTOR_SOURCE=official_llava_after_template_disjoint_v2_cat_smoke \
DATASETS="MSCOCO GQA" \
SETTINGS="random popular adversarial" \
METHODS="regular cat" \
ALPHAS="0 0.1 0.3 0.5 1.0" \
LIMIT=300 \
PROMPT_SUFFIX="Please answer this question with one word." \
PARSER_MODE=contains_yes_no_octopus_like \
DO_SAMPLE=true \
TEMPERATURE=1.0 \
TOP_P=1.0 \
NUM_BEAMS=1 \
MAX_NEW_TOKENS=1024 \
SEED=42 \
GPU_POOL=0,1,2,3 \
FORCE_PARALLEL=true \
SKIP_COMPLETED=true \
SKIP_EXISTING_FILES=true \
OVERWRITE=true \
bash scripts/run_pope_official_cat_expert_sweep.sh
```

## 4. Full Extract

After the smoke run confirms shape and nonzero effect, run full `cat`
extraction. Prefer the 4-GPU sharded version below.

```bash
mkdir -p data/after_template_disjoint_v2/official_llava_activations/logs

for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${shard} python scripts/extract_after_template_activations_official_llava.py \
    --model-path /home/huiwei/sy/models/llava-v1.5-7b-official-clean \
    --llava-repo-path /home/huiwei/sy/LLaVA-official-clean \
    --conv-mode llava_v1 \
    --pair-file data/after_template_disjoint_v2/pairs/train.jsonl \
    --image-root /home/huiwei/sy/sy_data/COCO2014/train2014 \
    --instances-json /home/huiwei/sy/sy_data/COCO2014/annotations/instances_train2014.json \
    --types cat \
    --trusted-input-mode text_only \
    --output data/after_template_disjoint_v2/official_llava_activations/cat_train_shard${shard}.pt \
    --metadata-output data/after_template_disjoint_v2/official_llava_activations/cat_train_shard${shard}.meta.jsonl \
    --num-shards 4 \
    --shard-index ${shard} \
    --progress-every 20 \
    --seed 42 \
    --overwrite \
    > data/after_template_disjoint_v2/official_llava_activations/logs/cat_train_shard${shard}.log 2>&1 &
done
wait
```

Merge the four shards:

```bash
python scripts/merge_after_template_activation_files.py \
  --activation-files \
    data/after_template_disjoint_v2/official_llava_activations/cat_train_shard0.pt \
    data/after_template_disjoint_v2/official_llava_activations/cat_train_shard1.pt \
    data/after_template_disjoint_v2/official_llava_activations/cat_train_shard2.pt \
    data/after_template_disjoint_v2/official_llava_activations/cat_train_shard3.pt \
  --output data/after_template_disjoint_v2/official_llava_activations/cat_train.pt \
  --metadata-output data/after_template_disjoint_v2/official_llava_activations/cat_train.meta.jsonl \
  --overwrite
```

Single-GPU fallback:

```bash
python scripts/extract_after_template_activations_official_llava.py \
  --model-path /home/huiwei/sy/models/llava-v1.5-7b-official-clean \
  --llava-repo-path /home/huiwei/sy/LLaVA-official-clean \
  --conv-mode llava_v1 \
  --pair-file data/after_template_disjoint_v2/pairs/train.jsonl \
  --image-root /home/huiwei/sy/sy_data/COCO2014/train2014 \
  --instances-json /home/huiwei/sy/sy_data/COCO2014/annotations/instances_train2014.json \
  --types cat \
  --trusted-input-mode text_only \
  --output data/after_template_disjoint_v2/official_llava_activations/cat_train.pt \
  --metadata-output data/after_template_disjoint_v2/official_llava_activations/cat_train.meta.jsonl \
  --progress-every 20 \
  --seed 42 \
  --overwrite
```

Then build the full vector:

```bash
python scripts/build_after_template_vectors.py \
  --activation-cache data/after_template_disjoint_v2/official_llava_activations/cat_train.pt \
  --metadata data/after_template_disjoint_v2/official_llava_activations/cat_train.meta.jsonl \
  --output data/after_template_disjoint_v2/official_llava_steering/official_llava_cat_expert_vectors.pt \
  --stats-output data/after_template_disjoint_v2/official_llava_steering/official_llava_cat_expert_vectors.stats.json \
  --layers 5-25 \
  --normalize false \
  --overwrite
```

Use that vector file for the final aligned full POPE CatExpert run.
