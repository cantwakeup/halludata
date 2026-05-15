# AFTER-Style Caption Alignment

This note documents the caption-alignment layer used to move the typed
`cat` / `attr` / `rel` pair bank closer to AFTER's factual-text construction.

## Why

The original `after_template_disjoint_v2` trusted side is mostly
rule/template text:

```text
base_scene + type_specific_fact
```

AFTER instead builds:

- `t+`: an image-level factual description generated from annotation facts.
- `t*`: a query-focused factual description extracted from `t+`.

`scripts/build_after_style_caption_pairs.py` prepares those two stages while
keeping the existing visual side unchanged.

## Step 1: Prepare Prompts And Template Smoke Pairs

```bash
python scripts/build_after_style_caption_pairs.py \
  --input-dir data/after_template_disjoint_v2/pairs \
  --output-dir data/after_template_disjoint_v2/after_style_caption_pairs_template \
  --splits train,val,test \
  --backend template \
  --trusted-caption-mode query \
  --overwrite
```

Important outputs:

- `image_caption_prompts.jsonl`: one AFTER `Ifst` prompt per image for generating `t+`.
- `query_caption_prompts.jsonl`: one AFTER `Iqst` prompt per pair for generating `t*`.
- `caption_cache_template.jsonl`: deterministic local `best_cap` fallback.
- `train.jsonl`, `val.jsonl`, `test.jsonl`: pair files with rewritten
  `trusted_factual_text` and `trusted_prompt`.

The `template` backend is only a smoke path. Use generated captions for the
closest AFTER alignment.

## Step 2: Generate Captions Externally

The script can call the OpenAI API directly without installing the `openai`
package. Put the key in an environment variable, not in code:

```bash
export OPENAI_API_KEY='sk-...'
```

Then run a small smoke first:

```bash
python scripts/build_after_style_caption_pairs.py \
  --input-dir data/after_template_disjoint_v2/pairs \
  --output-dir data/after_template_disjoint_v2/after_style_caption_pairs_openai_smoke \
  --splits train \
  --backend openai \
  --openai-model gpt-4o-mini \
  --limit-images 5 \
  --trusted-caption-mode query \
  --overwrite
```

For full image-level captions:

```bash
python scripts/build_after_style_caption_pairs.py \
  --input-dir data/after_template_disjoint_v2/pairs \
  --output-dir data/after_template_disjoint_v2/after_style_caption_pairs_openai \
  --splits train,val,test \
  --backend openai \
  --openai-model gpt-4o-mini \
  --trusted-caption-mode query \
  --overwrite
```

This generates `generated_best_caps.jsonl`. If you also want GPT/LVLM
query-focused captions `t*`, add:

```bash
--openai-generate-query-captions
```

That is more AFTER-faithful but much more expensive because it calls the API per
pair, not per image.

Alternatively, feed `image_caption_prompts.jsonl` to GPT-4o-mini, GPT-4o, or a
local LVLM yourself and write a JSONL cache like:

```json
{"image_key": "12345", "best_cap": "A complete factual description ..."}
```

Optionally feed `query_caption_prompts.jsonl` to GPT/LVLM and write:

```json
{"pair_id": "after_template_disjoint_v2_cat_present_...", "query_cap": "Object-related description ..."}
```

If no query cache is provided, the script falls back to the pair's current
type-specific fact, or a no-object sentence for negative category/attribute
queries.

## Step 3: Rebuild Pair Files From Caption Cache

```bash
python scripts/build_after_style_caption_pairs.py \
  --input-dir data/after_template_disjoint_v2/pairs \
  --output-dir data/after_template_disjoint_v2/after_style_caption_pairs \
  --splits train,val,test \
  --backend cache \
  --caption-cache data/after_template_disjoint_v2/after_style_caption_pairs/generated_best_caps.jsonl \
  --query-cache data/after_template_disjoint_v2/after_style_caption_pairs/generated_query_caps.jsonl \
  --trusted-caption-mode query \
  --overwrite
```

For a pure FAS-style common direction, use `--trusted-caption-mode best`.
For our typed static expert vectors, `query` is usually the better first choice
because each row keeps its query-specific trusted side.

## Step 4: Re-Extract Activations And Rebuild Vectors

Use the existing activation extractor on the new pair files:

```bash
python scripts/extract_after_template_activations.py \
  --pair-file data/after_template_disjoint_v2/after_style_caption_pairs/train.jsonl \
  --output data/after_template_disjoint_v2/after_style_caption_activations/train.pt \
  --metadata-output data/after_template_disjoint_v2/after_style_caption_activations/train.meta.jsonl \
  --adapter llava \
  --image-root /home/huiwei/sy/sy_data/COCO2014/train2014
```

Then build vectors with the same vector builder:

```bash
python scripts/build_after_template_vectors.py \
  --activation-cache data/after_template_disjoint_v2/after_style_caption_activations/train.pt \
  --metadata data/after_template_disjoint_v2/after_style_caption_activations/train.meta.jsonl \
  --output data/after_template_disjoint_v2/after_style_caption_vectors/after_style_caption_expert_vectors.pt \
  --stats-output data/after_template_disjoint_v2/after_style_caption_vectors/stats.json \
  --layers 5-25 \
  --normalize true \
  --overwrite
```

## Expected Checks

- Inspect `REPORT.md` from the pair build.
- Compare raw-vector cosine matrices against the old template vectors.
- Check whether `cat/attr/rel` residuals become less dominated by shared
  factual-template direction.
- Run a small POPE/GQA sanity before any full sweep.
