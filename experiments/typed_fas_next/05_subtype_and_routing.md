# Subtype Experts and Routing

## Experiment E/F Status

No new routing benchmark was run in this pass.

## Reason

The routing objective requires candidate experts that already show type/subtype selectivity on dev. Current evidence is not strong enough:

- raw cat/attr/rel vectors fail clean selectivity on large benchmarks;
- clean minimal-pair subtype vectors are representationally separated, but prior 100-sample heldout eval only supports `attr_count_clean` and `attr_shape_clean`;
- relation masks show yes-rate drift and off-diagonal wins.

Running routing now would likely test routing over weak or biased experts rather than testing the routing idea.

## Existing Routing Support

`src/expert_data/steering.py` has a simple rule router for `cat`, `attr`, and `rel`:

- attr keywords: `how many`, `number`, `count`, `color`, `what color`, `many`
- rel keywords: `left`, `right`, `above`, `below`, `under`, `next to`, `behind`, `in front of`, `position`, `where`
- cat keywords: `object`, `is there`, `does image contain`, `what is in`, `what objects`, `contain`

The current router is coarse. It does not yet route to subtype keys such as `attr_count_clean` or `rel_left_right_clean`.

## Recommended Routing Sequence

Only after dev selectivity is positive:

1. Family oracle routing:
   - POPE/category -> best cat/category candidate
   - AMBER attribute -> best attr/subtype candidate
   - GQA relation -> best relation candidate
2. Keyword routing:
   - `how many` -> count
   - `what color` -> color
   - `left/right/above/below` -> spatial relation
   - `holding/wearing/riding/sitting` -> semantic relation
3. Mixture routing:
   - object+attribute question -> `0.5 * cat + 0.5 * attr`
   - relation question with object grounding -> `0.3 * cat + 0.7 * rel`

## Minimum Success Criteria Before Large Routing

- matched dev selectivity margin > 0;
- matched yes-rate not abnormal on balanced yes/no sets;
- random masks do not match the effect;
- at least two families improve or hold baseline without off-diagonal winner dominance.

## Candidate Inputs Prepared This Pass

The vector-only run produced candidate vector bundles and head maps:

- `experiments/typed_fas_next/vector_only/vector_only_variants.pt`
- `experiments/typed_fas_next/vector_only/head_maps/*.json`

These are sufficient to run small dev steering once GPUs are free.
