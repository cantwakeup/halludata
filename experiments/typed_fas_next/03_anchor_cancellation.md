# Anchor Cancellation

## Experiment A Status

Not executed in this pass.

## Hypothesis

Attr and rel vectors may be too similar to cat because their factual text includes object grounding. Subtracting object-anchor activations may isolate type-private attribute/relation components.

## What Was Checked

Existing facts and pairs can construct anchor texts:

- attribute color/shape:
  - typed: `The cup is gray.`
  - anchor: `There is a cup in the image.`
- attribute count:
  - typed: `There are two cups in the image.`
  - anchor: `There are cups in the image.`
- relation:
  - typed: `The tv is to the right of the laptop.`
  - anchor: `There is a tv and a laptop in the image.`
- category:
  - typed: `There is a dog in the image.`
  - anchor: `There is an object in the image.`

Relevant existing fields:

- `facts.category_facts`
- `facts.attribute_facts`
- `facts.relation_facts`
- `target_object`
- `target_attribute`
- `target_relation`
- `source_fact_id`

## Why It Was Not Run

The required anchor text activations are not cached. Existing raw vectors are `mean(z_text - z_visual)` for the current trusted text; they do not include separate anchor text branches.

Running anchor cancellation requires a new activation extraction pass for typed statement and anchor statement text. That is a GPU job, but does not require an external API.

At decision time all four GPUs were heavily occupied:

| gpu | used/total MiB | utilization |
| --- | --- | --- |
| 0 | 58738/81920 | 100% |
| 1 | 57793/81920 | 99% |
| 2 | 58345/81920 | 100% |
| 3 | 57793/81920 | 91% |

Starting the run in that state would risk OOM/headroom failures and would not follow the requested low-cost-first workflow.

## Recommended Implementation

1. Add an anchor JSONL builder under `experiments/typed_fas_next/` or `scripts/` that emits:
   - `id`
   - `image_id`
   - `expert_type`
   - `subtype`
   - `typed_statement`
   - `anchor_statement`
   - `source_pair_id`
2. Reuse official-LLaVA text-only activation extraction, adding branch names:
   - `z_typed_text`
   - `z_anchor_text`
3. Build vectors:
   - `cat_anchor_delta = mean(z_typed - z_anchor)`
   - `attr_anchor_delta = mean(z_typed - z_anchor)`
   - `rel_anchor_delta = mean(z_typed - z_anchor)`
4. Run the same diagnostics produced by `run_vector_only_diagnostics.py`.
5. Only then run small dev benchmarks if representation metrics improve without collapsing vector norms.

## Next-Step Command Sketch

This is not executed yet:

```bash
python experiments/typed_fas_next/build_anchor_text_rows.py \
  --input-root data/after_fas_type_v1_gpt4omini_typed250_text \
  --output data/typed_fas_next/anchor_cancellation/anchor_rows.jsonl
```

Then extract text activations and build vectors once GPUs are available.
