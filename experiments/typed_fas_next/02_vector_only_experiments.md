# Vector-Only Experiments

## Experiment B/C: Contrast Vectors and Discriminative Head Selection

### Hypothesis

Raw typed vectors are dominated by a shared general factuality direction. Type-vs-other contrast and discriminative head selection may reduce shared components enough to create better dev-benchmark candidates.

### Implementation Changes

Added:

- `experiments/typed_fas_next/run_vector_only_diagnostics.py`

The script:

1. loads cached `runtime_raw_type_vectors.pt`;
2. constructs vector families:
   - `raw`
   - `global_residual`
   - `contrast_l0p25`, `contrast_l0p5`, `contrast_l0p75`, `contrast_l1`
   - `headnorm_contrast_l0p25`, `headnorm_contrast_l0p5`, `headnorm_contrast_l0p75`, `headnorm_contrast_l1`
   - `pca_remove1`, `pca_remove2`, `pca_remove3`, `pca_remove5` (`pca_remove5` saturates at rank 3 because only three raw vectors are available)
3. computes representation diagnostics:
   - pairwise cosine
   - cosine to family global
   - vector norms
   - topK mask overlap for K in 16, 32, 64, 128
   - restricted cosine on selected heads
4. generates head maps for:
   - `norm`
   - `specificity_cos`
   - `contrast_norm`
   - `ratio`
   - `hybrid`
   - `shared`
   - `random`

### Data

- Raw runtime vectors: `data/after_fas_type_v1_gpt4omini_typed250_text/official_llava_vectors/runtime_raw_type_vectors.pt`
- Clean minimal-pair context vectors: `data/clean_type_minpair_v2/vectors/condition_vectors.pt`

### Command

```bash
python experiments/typed_fas_next/run_vector_only_diagnostics.py --overwrite
```

### Outputs

| file | purpose |
| --- | --- |
| `experiments/typed_fas_next/vector_only/VECTOR_ONLY_DIAGNOSTICS.md` | Human-readable report. |
| `experiments/typed_fas_next/vector_only/vector_only_diagnostics.json` | Full diagnostics payload. |
| `experiments/typed_fas_next/vector_only/vector_family_summary.csv` | Pairwise cosine/norm summary. |
| `experiments/typed_fas_next/vector_only/head_mask_summary.csv` | Mask overlap and restricted cosine summary. |
| `experiments/typed_fas_next/vector_only/vector_only_variants.pt` | Runtime vector bundle for candidate dev benchmarks. |
| `experiments/typed_fas_next/vector_only/head_maps/*.json` | Expert-map files usable by `--steer-head-select expert_map`. |

### Key Results

Raw baseline recheck:

| family | cat_attr_cos | cat_rel_cos | attr_rel_cos | mean_abs_pairwise_cos |
| --- | ---: | ---: | ---: | ---: |
| raw | 0.8815 | 0.7525 | 0.8072 | 0.8137 |

Best representation-only decoupling candidates:

| family | cat_attr_cos | cat_rel_cos | attr_rel_cos | mean_abs_pairwise_cos | note |
| --- | ---: | ---: | ---: | ---: | --- |
| headnorm_contrast_l0p75 | 0.1989 | -0.3206 | -0.1849 | 0.2348 | strongest moderate-norm decoupling |
| contrast_l0p75 | 0.1228 | -0.4488 | -0.2655 | 0.2790 | direct type-vs-other contrast |
| contrast_l0p5 | 0.5306 | 0.1127 | 0.2660 | 0.3031 | less aggressive, likely safer than lambda 1 |
| contrast_l1 | -0.1765 | -0.7305 | -0.5433 | 0.4834 | strongly anti-correlated with near-zero family-global cosine |
| global_residual | -0.1787 | -0.7292 | -0.5430 | 0.4836 | similar to lambda-1 contrast, previous residual warning applies |

Discriminative head selection can sharply reduce overlap. Several top-64 `ratio`/`hybrid` masks have zero triple intersection and near-zero pairwise jaccard, whereas raw norm-topK had high overlap.

### Does This Support the Hypothesis?

Partially, at representation level only.

- Supported: type-vs-other contrast and discriminative mask scoring can reduce pairwise cosine and topK overlap.
- Not yet supported: no benchmark selectivity claim can be made from these diagnostics.
- Risk: aggressive contrast or PCA removal may remove the shared FAS direction that caused the small raw benchmark gains.

### Next Step

Run a small dev benchmark only when GPUs are available:

- vectors: `contrast_l0p5`, `contrast_l0p75`, `headnorm_contrast_l0p75`, `global_residual`
- masks: `specificity_cos`, `contrast_norm`, and `ratio`
- K: 16, 32, 64
- alpha: `[-0.25, -0.1, 0.1, 0.25]`
- benchmarks: POPE category dev, AMBER attribute dev, GQA relation dev
- required metrics: accuracy, selectivity margin, yes-rate, changed predictions
