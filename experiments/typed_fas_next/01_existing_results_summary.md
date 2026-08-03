# Existing Results Summary

## Supported Conclusions

The current raw typed vectors do not support a clean cat/attr/rel expert claim.

Representation diagnostics from `raw_type_vectors.REPORT.md`:

| pair | cosine |
| --- | ---: |
| cat-attr | 0.881471 |
| cat-rel | 0.752529 |
| attr-rel | 0.807162 |

Top-64 norm-head overlap is also high:

| pair | top64 jaccard |
| --- | ---: |
| cat-attr | 0.729730 |
| cat-rel | 0.662338 |
| attr-rel | 0.662338 |

This supports the interpretation that raw typed vectors share a strong general factuality/FAS component.

## Large Raw Benchmark Selectivity

Derived from `data/after_fas_type_v1_gpt4omini_typed250_text/bench_sanity_raw_injection_v2_large/*/summary.csv`.

| benchmark | family | matched | baseline_acc | winner | matched_delta | best_mismatch_delta | global_delta | selectivity_margin | baseline_yes_rate | matched_yes_rate |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| POPE MSCOCO random 1000 | category | cat | 0.8710 | attr | 0.0070 | 0.0090 | 0.0060 | -0.0020 | 0.3930 | 0.4040 |
| POPE MSCOCO popular 1000 | category | cat | 0.8590 | cat | 0.0070 | 0.0060 | 0.0060 | 0.0010 | 0.4050 | 0.4160 |
| POPE MSCOCO adversarial 1000 | category | cat | 0.8330 | cat | 0.0050 | 0.0040 | 0.0040 | 0.0010 | 0.4310 | 0.4440 |
| AMBER attribute 1000 | attribute | attr | 0.7910 | attr | 0.0030 | 0.0020 | 0.0030 | 0.0000 | 0.5110 | 0.5580 |
| GQA relation full 802 | relation | rel | 0.5860 | cat | -0.0012 | 0.0000 | -0.0012 | -0.0012 | 0.7581 | 0.7768 |

Definition:

`selectivity_margin = matched_delta - max(best_mismatch_delta, global_delta)`

Only POPE popular/adversarial show a tiny positive margin (`+0.001`), too small to treat as robust expert selectivity. AMBER ties global, and GQA relation is negative.

## Clean Minimal-Pair v2 Context

The clean minimal-pair v2 data passed audit and contains balanced GQA-derived subtype pairs:

| group | examples | train count |
| --- | --- | ---: |
| attr | color, count, state, material, shape, action | 4200 |
| rel | left/right, above/below, holding/wearing, sitting/riding | 1938 |

Condition-balanced vectors in `CONDITION_VECTOR_REPORT.md` are much more separated than raw FAS directions. The subtype mean absolute cosine recomputed in this pass is `0.0880`.

Prior heldout mask eval (`heldout_mask_limit100_seed42`) found only limited credible selectivity:

- `attr_count_clean`: passed current success criteria.
- `attr_shape_clean`: passed current success criteria.
- most other attr subtypes did not beat mismatched/random/g_type baselines.
- relation masks frequently showed abnormal yes-rate or mismatched/random wins.

## Operational Note

A real dev benchmark is still needed for new contrast/mask variants. This pass did not start one because all GPUs were already heavily occupied (`91-100%` utilization) and the goal explicitly prioritizes low-cost diagnostics before GPU-heavy work.
