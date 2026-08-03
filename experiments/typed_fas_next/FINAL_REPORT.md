# Typed FAS Next: Final Report

Date: 2026-07-08

## Executive Judgment

This pass makes progress on the representation side, but not yet on benchmark/routing success.

The current raw typed vectors should not be scaled further as-is. Prior large benchmarks already show that raw `cat`, `attr`, and `rel` do not produce stable matched-vector selectivity. The best next direction is a controlled dev benchmark of type-vs-other contrast vectors plus discriminative head masks, especially `contrast_l0p5`, `contrast_l0p75`, and `headnorm_contrast_l0p75` with `specificity_cos`, `contrast_norm`, or `ratio` masks.

Anchor cancellation is plausible and well motivated, but it needs a new text-only activation pass for anchor statements. Correct-vs-wrong minimal pairs are already available and are still the best source for subtype-level directions, but prior evals only support `attr_count_clean` and `attr_shape_clean` robustly enough to continue.

## What Was Done

| phase | status | output |
| --- | --- | --- |
| Repo audit | completed | `experiments/typed_fas_next/00_repo_audit.md` |
| Existing result reconstruction | completed | `experiments/typed_fas_next/01_existing_results_summary.md` |
| Vector-only experiments | completed | `experiments/typed_fas_next/02_vector_only_experiments.md` |
| Anchor cancellation | not run; feasibility audited | `experiments/typed_fas_next/03_anchor_cancellation.md` |
| Correct-vs-wrong minpair | existing artifacts audited | `experiments/typed_fas_next/04_correct_wrong_minpair.md` |
| Subtype/routing | not run; prerequisites stated | `experiments/typed_fas_next/05_subtype_and_routing.md` |

Main generated artifacts:

- `experiments/typed_fas_next/vector_only/VECTOR_ONLY_DIAGNOSTICS.md`
- `experiments/typed_fas_next/vector_only/vector_only_diagnostics.json`
- `experiments/typed_fas_next/vector_only/vector_family_summary.csv`
- `experiments/typed_fas_next/vector_only/head_mask_summary.csv`
- `experiments/typed_fas_next/vector_only/vector_only_variants.pt`
- `experiments/typed_fas_next/vector_only/head_maps/*.json`
- `experiments/typed_fas_next/results_summary.csv`
- `experiments/typed_fas_next/results_summary.json`
- `experiments/typed_fas_next/commands.sh`

## Existing Evidence Rechecked

Raw representation diagnostics remain strongly shared:

| vector pair | cosine |
| --- | ---: |
| cat-attr | 0.8815 |
| cat-rel | 0.7525 |
| attr-rel | 0.8072 |

Large raw benchmark selectivity margins:

| benchmark | matched | winner | selectivity_margin |
| --- | --- | --- | ---: |
| POPE random 1000 | cat | attr | -0.0020 |
| POPE popular 1000 | cat | cat | 0.0010 |
| POPE adversarial 1000 | cat | cat | 0.0010 |
| AMBER attribute 1000 | attr | attr | 0.0000 |
| GQA relation full 802 | rel | cat | -0.0012 |

Conclusion: raw typed vectors do not meet the project's expert-vector standard. At best, they behave like weak general FAS variants.

## New Vector-Only Results

The new script `run_vector_only_diagnostics.py` built contrast/PCA/head-normalized variants and discriminative masks from cached tensors only.

Best representation candidates:

| family | cat_attr | cat_rel | attr_rel | mean_abs_pairwise_cos | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| headnorm_contrast_l0p75 | 0.1989 | -0.3206 | -0.1849 | 0.2348 | best moderate-norm separation |
| contrast_l0p75 | 0.1228 | -0.4488 | -0.2655 | 0.2790 | direct contrast, promising but aggressive |
| contrast_l0p5 | 0.5306 | 0.1127 | 0.2660 | 0.3031 | safer compromise |
| contrast_l1 | -0.1765 | -0.7305 | -0.5433 | 0.4834 | highly anti-correlated; may erase useful shared FAS |
| global_residual | -0.1787 | -0.7292 | -0.5430 | 0.4836 | similar risk to prior residual attempts |

Discriminative top-64 masks can reduce triple overlap to zero for several contrast/ratio/hybrid combinations. This is a real representation-level improvement over norm-topK, but it is not a benchmark result.

## Why No New GPU Dev Benchmark Was Started

The goal asked for low-cost diagnostics first and to avoid blind large benchmark runs. A real small/dev benchmark still requires loading official LLaVA. At the decision point, all GPUs were busy:

| gpu | used/total MiB | utilization |
| --- | --- | --- |
| 0 | 58738/81920 | 100% |
| 1 | 57793/81920 | 99% |
| 2 | 58345/81920 | 100% |
| 3 | 57793/81920 | 91% |

Given prior OOM/headroom issues in this project, starting another LLaVA run in that state would likely produce operational noise rather than clean evidence. The dev command template is recorded in `commands.sh`.

## What Has Positive Signal

1. `headnorm_contrast_l0p75` and `contrast_l0p75` lower raw pairwise cosine substantially while keeping nontrivial norms.
2. Discriminative masks (`ratio`, `hybrid`, `specificity_cos`, `contrast_norm`) can remove most shared topK overlap.
3. Clean minimal-pair subtype vectors remain the best source of truly separated directions; subtype mean absolute cosine is `0.0880`.
4. Prior eval still leaves `attr_count_clean` and `attr_shape_clean` worth following up.

## What Failed Or Should Stop

1. Stop scaling raw `cat/attr/rel` vectors without transformation; large evidence is already negative.
2. Do not claim routing success from coarse raw vectors; off-diagonal winners dominate or tie.
3. Do not treat relation subtype results as reliable yet; prior evals show yes-rate drift and mismatched/random wins.
4. Do not rely on PCA/common-subspace removal alone; it can produce low cosine by collapsing or over-rotating directions.

## Recommended Next Round

Run a small, controlled dev benchmark when GPUs are actually available:

| candidate | vector keys | masks | K | alpha |
| --- | --- | --- | --- | --- |
| safer contrast | `contrast_l0p5_cat/attr/rel` | `specificity_cos`, `contrast_norm` | 16, 32, 64 | -0.25, -0.1, 0.1, 0.25 |
| stronger contrast | `contrast_l0p75_cat/attr/rel` | `specificity_cos`, `ratio` | 16, 32, 64 | -0.25, -0.1, 0.1, 0.25 |
| normalized contrast | `headnorm_contrast_l0p75_cat/attr/rel` | `ratio`, `hybrid` | 16, 32, 64 | -0.25, -0.1, 0.1, 0.25 |

Use three dev families:

- POPE/category dev
- AMBER attribute dev
- GQA relation dev

Report:

- accuracy
- delta over baseline
- selectivity margin
- yes-rate
- changed predictions
- wrong-to-right and right-to-wrong

Only expand to large benchmark if at least one candidate has positive selectivity margin without abnormal yes-rate.

## Anchor Cancellation Next Step

Build anchor rows from existing facts, then run a text-only activation pass:

- attr: typed attribute statement minus object existence anchor
- count: count statement minus plural object anchor
- rel: relation statement minus object-pair anchor
- cat: object existence statement minus generic object anchor

This is likely the most conceptually aligned next construction, but it cannot be completed from existing cached activations alone.

## Are We Closer To Type-Private Routing?

Slightly closer, but not yet at the final goal.

Closer because this pass identifies concrete vector/mask variants that are more type-private than raw vectors and produces runnable artifacts for dev benchmarking. Not fully closer because no new benchmark result yet shows `matched > mismatched/global`, and routing should not be tested until that condition appears on dev.

Main blocker: the project has representation transformations that can create separation, but it has not yet shown that those separated directions preserve useful intervention behavior on matched benchmarks without answer-bias side effects.
