# GQA Type-Aware Diagnostic Eval Summary

- Summary CSV: `data/gqa_typeaware_v1/shared_private_sanity_eval/summary.csv`
- Runs summarized: 51

## Best Steered Runs By Delta

| eval_subset | method | vector | alpha | accuracy | baseline_accuracy | delta_acc | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_attr_val | steered | attr_raw | 0.1000 | 0.6800 | 0.6600 | 0.0200 | 0.6200 | 1 | 0 | 1 | 50 |
| gqa_attr_val | steered | attr_raw | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | attr_res | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | attr_res | 0.1000 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | cat_raw | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | cat_raw | 0.1000 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | cat_res | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | global_type_svd | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | global_type_svd | 0.1000 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | global_type_svd,cat_res | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | global_type_svd,cat_res | 0.1000 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | rel_raw | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |

## All Runs

| eval_subset | method | vector | alpha | accuracy | baseline_accuracy | delta_acc | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_attr_val | baseline |  |  | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | attr_raw | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | attr_raw | 0.1000 | 0.6800 | 0.6600 | 0.0200 | 0.6200 | 1 | 0 | 1 | 50 |
| gqa_attr_val | steered | attr_res | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | attr_res | 0.1000 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | cat_raw | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | cat_raw | 0.1000 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | cat_res | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | cat_res | 0.1000 | 0.6400 | 0.6600 | -0.0200 | 0.6600 | 0 | 1 | 1 | 50 |
| gqa_attr_val | steered | global_type_svd | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | global_type_svd | 0.1000 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | global_type_svd,cat_res | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | global_type_svd,cat_res | 0.1000 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | rel_raw | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | rel_raw | 0.1000 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | rel_res | 0.0500 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_attr_val | steered | rel_res | 0.1000 | 0.6600 | 0.6600 | 0.0000 | 0.6400 | 0 | 0 | 0 | 50 |
| gqa_cat_val | baseline |  |  | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | attr_raw | 0.0500 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | attr_raw | 0.1000 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | attr_res | 0.0500 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | attr_res | 0.1000 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | cat_raw | 0.0500 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | cat_raw | 0.1000 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | cat_res | 0.0500 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | cat_res | 0.1000 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | global_type_svd | 0.0500 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | global_type_svd | 0.1000 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | global_type_svd,cat_res | 0.0500 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | global_type_svd,cat_res | 0.1000 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | rel_raw | 0.0500 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | rel_raw | 0.1000 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | rel_res | 0.0500 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_cat_val | steered | rel_res | 0.1000 | 0.8600 | 0.8600 | 0.0000 | 0.6800 | 0 | 0 | 0 | 50 |
| gqa_rel_val | baseline |  |  | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | attr_raw | 0.0500 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | attr_raw | 0.1000 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | attr_res | 0.0500 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | attr_res | 0.1000 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | cat_raw | 0.0500 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | cat_raw | 0.1000 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | cat_res | 0.0500 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | cat_res | 0.1000 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | global_type_svd | 0.0500 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | global_type_svd | 0.1000 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | global_type_svd,cat_res | 0.0500 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | global_type_svd,cat_res | 0.1000 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | rel_raw | 0.0500 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | rel_raw | 0.1000 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | rel_res | 0.0500 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |
| gqa_rel_val | steered | rel_res | 0.1000 | 0.6000 | 0.6000 | 0.0000 | 0.8400 | 0 | 0 | 0 | 50 |

## Notes

- `per_type_accuracy` and `per_subtype_accuracy` are also written into each run directory as `gqa_typeaware_metrics.json`.
- Baseline rows have `delta_acc=0`; steered rows compare against the matching subset baseline.
