# Correct-vs-Wrong Minimal Pair

## Experiment D Status

No new activation extraction was run in this pass. The existing clean minimal-pair v2 artifacts were audited and incorporated into the next-step recommendation.

## Hypothesis

`z_text - z_visual` may mostly capture visual-text alignment or general factuality. A more direct hallucination-correction direction is:

`v_type = mean(z_correct_statement - z_wrong_statement)`

## Existing Data

`data/clean_type_minpair_v2/minimal_pairs/train.jsonl` already has the intended schema:

- `base_scene`
- `target_fact`
- `target_counterfact`
- `fact_text`
- `counterfact_text`
- `trusted_prompt_fact`
- `trusted_prompt_counterfact`
- `condition_key`
- subtype labels

Examples:

- color: `There is a pant in the image. The pant is white.` vs `There is a pant in the image. The pant is brown.`
- relation: left/right, above/below, holding/wearing, sitting/riding pairs.

## Existing Vectors

`data/clean_type_minpair_v2/vectors/condition_vectors.pt` already contains condition-balanced correct-vs-counterfact vectors:

- coarse: `g_all_clean`, `g_attr_clean`, `g_rel_clean`
- subtype: `s_attr_color_clean_clean`, `s_attr_count_clean_clean`, `s_attr_shape_clean_clean`, `s_rel_left_right_clean_clean`, etc.

The vector-only diagnostic recomputed a subtype mean absolute cosine of `0.0880`, much lower than raw typed FAS cosine (`0.8137` mean absolute pairwise among cat/attr/rel).

## Prior Eval Signal

`data/clean_type_minpair_v2/eval/heldout_mask_limit100_seed42/MASK_EVAL_REPORT.md` reports:

- credible positive signal: `attr_count_clean`, `attr_shape_clean`
- weak or failed signal: most other attr subtypes
- relation subtypes: often affected by yes-rate drift or mismatched/random wins

## Does This Support the Hypothesis?

Partially.

- Supported at representation level: clean correct-vs-counterfact subtype vectors are more separated than raw FAS vectors.
- Not broadly supported at benchmark level: prior heldout eval did not establish stable selectivity across most subtypes.

## Recommended Next Step

Do not rebuild the same vectors. Instead run targeted dev reruns:

1. `attr_count_clean` and `attr_shape_clean` with larger heldout samples, negative alpha, and multiple masks.
2. relation subtypes split into:
   - spatial: `rel_left_right_clean`, `rel_above_below_clean`
   - semantic: `rel_holding_wearing_clean`, `rel_sitting_riding_clean`
3. report yes-rate and selectivity margin before considering routing.

No external API is needed.
