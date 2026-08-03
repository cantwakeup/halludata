# Vector-Only Typed-FAS Diagnostics

## Inputs

- runtime vectors: `/home/huiwei/sy/halludata/data/after_fas_type_v1_gpt4omini_typed250_text/official_llava_vectors/runtime_raw_type_vectors.pt`
- condition vectors: `/home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt`
- computation: cached CPU tensor diagnostics only; no API calls, activation extraction, or benchmark generation.

## Raw Baseline Recheck

| family | cat_attr_cos | cat_rel_cos | attr_rel_cos | mean_abs_pairwise_cos | cat_norm | attr_norm | rel_norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.8815 | 0.7525 | 0.8072 | 0.8137 | 26.7621 | 25.8483 | 26.0618 |

This matches the prior conclusion: the raw cat/attr/rel directions are highly aligned.

## Lowest Pairwise-Cosine Vector Variants

| family | cat_attr_cos | cat_rel_cos | attr_rel_cos | mean_abs_pairwise_cos | cat_global_cos | attr_global_cos | rel_global_cos | cat_norm | attr_norm | rel_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| headnorm_contrast_l0p75 | 0.1989 | -0.3206 | -0.1849 | 0.2348 | 0.5293 | 0.5949 | 0.4125 | 14.7512 | 13.7933 | 16.7319 |
| pca_remove3 | 0.1887 | 0.0851 | 0.4831 | 0.2523 | 0.3842 | 0.7616 | 0.8918 | 0.0009 | 0.0013 | 0.0024 |
| pca_remove5 | 0.1887 | 0.0851 | 0.4831 | 0.2523 | 0.3842 | 0.7616 | 0.8918 | 0.0009 | 0.0013 | 0.0024 |
| contrast_l0p75 | 0.1228 | -0.4488 | -0.2655 | 0.2790 | 0.4829 | 0.5292 | 0.3223 | 14.4098 | 12.1184 | 15.6182 |
| contrast_l0p5 | 0.5306 | 0.1127 | 0.2660 | 0.3031 | 0.7503 | 0.8014 | 0.6425 | 17.3515 | 15.6270 | 17.5401 |
| headnorm_contrast_l0p5 | 0.6242 | 0.2992 | 0.3940 | 0.4391 | 0.8049 | 0.8430 | 0.7255 | 19.1724 | 18.7353 | 20.1339 |
| contrast_l1 | -0.1765 | -0.7305 | -0.5433 | 0.4834 | 0.0074 | 0.0141 | -0.0161 | 13.7948 | 11.2213 | 16.1733 |
| global_residual | -0.1787 | -0.7292 | -0.5430 | 0.4836 | 0.0002 | 0.0003 | -0.0004 | 9.1770 | 7.4781 | 10.7524 |
| headnorm_contrast_l1 | -0.1692 | -0.6971 | -0.5887 | 0.4850 | -0.0050 | 0.0028 | 0.0021 | 13.6539 | 12.1103 | 16.6474 |
| pca_remove1 | -0.2166 | -0.7227 | -0.5182 | 0.4858 | -0.8337 | -0.3586 | 0.9842 | 9.0003 | 7.4012 | 10.9480 |

Lower cosine alone is not enough. PCA removal and head-normalized contrast can make vectors look private while also changing norms and possibly removing the shared effect that made FAS work.

## Lowest Top-64 Mask Overlap

| family | strategy | top_k | triple_intersection | triple_union | triple_jaccard | cat_attr_jaccard | cat_rel_jaccard | attr_rel_jaccard | cat_attr_restricted_cos | cat_rel_restricted_cos | attr_rel_restricted_cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contrast_l0p75 | ratio | 64 | 0 | 192 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0791 | -0.3257 | -0.2717 |
| contrast_l0p5 | ratio | 64 | 0 | 191 | 0.0000 | 0.0000 | 0.0000 | 0.0079 | 0.5294 | 0.1043 | 0.1676 |
| contrast_l1 | hybrid | 64 | 0 | 191 | 0.0000 | 0.0079 | 0.0000 | 0.0000 | -0.5561 | -0.7874 | -0.6699 |
| global_residual | hybrid | 64 | 0 | 191 | 0.0000 | 0.0079 | 0.0000 | 0.0000 | -0.5648 | -0.7799 | -0.6709 |
| pca_remove1 | hybrid | 64 | 0 | 191 | 0.0000 | 0.0079 | 0.0000 | 0.0000 | -0.5772 | -0.7748 | -0.6528 |
| global_residual | ratio | 64 | 0 | 190 | 0.0000 | 0.0079 | 0.0000 | 0.0079 | -0.5277 | -0.8406 | -0.7025 |
| pca_remove1 | ratio | 64 | 0 | 190 | 0.0000 | 0.0079 | 0.0000 | 0.0079 | -0.5592 | -0.8342 | -0.6862 |
| headnorm_contrast_l0p75 | ratio | 64 | 0 | 190 | 0.0000 | 0.0000 | 0.0000 | 0.0159 | -0.1988 | -0.6010 | -0.4984 |
| headnorm_contrast_l1 | ratio | 64 | 0 | 189 | 0.0000 | 0.0079 | 0.0079 | 0.0079 | -0.4544 | -0.8103 | -0.7047 |
| contrast_l1 | ratio | 64 | 0 | 189 | 0.0000 | 0.0159 | 0.0000 | 0.0079 | -0.5151 | -0.8450 | -0.6997 |
| headnorm_contrast_l0p5 | ratio | 64 | 0 | 189 | 0.0000 | 0.0000 | 0.0079 | 0.0159 | 0.2480 | -0.1632 | -0.0507 |
| headnorm_contrast_l0p25 | ratio | 64 | 0 | 188 | 0.0000 | 0.0000 | 0.0159 | 0.0159 | 0.5796 | 0.2622 | 0.3585 |

The discriminative strategies can reduce mask overlap, so they are better dev-benchmark candidates than norm-topK. The benchmark claim still needs actual dev runs.

## Clean Minimal-Pair Context

- subtype mean absolute cosine: `0.0880`
- Cached condition vectors are much less aligned than the raw FAS vectors, but previous heldout mask evals showed only limited credible selectivity.

## Decision

- Do not scale raw vectors again; the existing large run already falsifies clean expert behavior.
- Best cheap next dev candidates: `contrast_l1` or `global_residual` with `specificity_cos`/`contrast_norm` masks at K=16/32/64, plus negative-alpha checks.
- Anchor cancellation remains unexecuted here because the necessary anchor text activations are not cached; it requires a new text-only activation pass, not an API call.
- Correct-vs-wrong minimal pairs are already cached in `clean_type_minpair_v2`; the useful next step is targeted dev reruns for attr_count/attr_shape and relation bias checks, not another vector-only pass.
