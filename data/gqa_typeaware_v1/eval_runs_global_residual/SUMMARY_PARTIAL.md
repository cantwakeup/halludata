# GQA Type-Aware Diagnostic Eval Summary

- Summary CSV: `data/gqa_typeaware_v1/eval_runs_global_residual/summary_partial.csv`
- Runs summarized: 15

## Best Steered Runs By Delta

| eval_subset | method | vector | alpha | accuracy | baseline_accuracy | delta_acc | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_cat_val | steered | global_all | 1.0000 | 0.8660 | 0.8560 | 0.0100 | 0.5400 | 31 | 21 | 52 | 1000 |
| gqa_cat_val | steered | global_all | 0.5000 | 0.8640 | 0.8560 | 0.0080 | 0.5660 | 15 | 7 | 22 | 1000 |
| gqa_cat_val | steered | cat | 1.0000 | 0.8610 | 0.8560 | 0.0050 | 0.5770 | 21 | 16 | 37 | 1000 |
| gqa_cat_val | steered | global_all | 0.2500 | 0.8610 | 0.8560 | 0.0050 | 0.5730 | 11 | 6 | 17 | 1000 |
| gqa_cat_val | steered | cat | 0.5000 | 0.8600 | 0.8560 | 0.0040 | 0.5760 | 12 | 8 | 20 | 1000 |
| gqa_cat_val | steered | cat | 0.2500 | 0.8570 | 0.8560 | 0.0010 | 0.5790 | 6 | 5 | 11 | 1000 |
| gqa_cat_val | steered | cat | 0.0500 | 0.8550 | 0.8560 | -0.0010 | 0.5850 | 2 | 3 | 5 | 1000 |
| gqa_cat_val | steered | cat | 0.1000 | 0.8540 | 0.8560 | -0.0020 | 0.5860 | 2 | 4 | 6 | 1000 |
| gqa_cat_val | steered | global_all | 0.0500 | 0.8540 | 0.8560 | -0.0020 | 0.5880 | 1 | 3 | 4 | 1000 |
| gqa_cat_val | steered | global_all | 0.1000 | 0.8540 | 0.8560 | -0.0020 | 0.5860 | 2 | 4 | 6 | 1000 |
| gqa_cat_val | steered | cat_res | 0.1000 | 0.8530 | 0.8560 | -0.0030 | 0.5910 | 2 | 5 | 7 | 1000 |
| gqa_cat_val | steered | cat_res | 0.0500 | 0.8520 | 0.8560 | -0.0040 | 0.5880 | 0 | 4 | 4 | 1000 |

## All Runs

| eval_subset | method | vector | alpha | accuracy | baseline_accuracy | delta_acc | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_cat_val | baseline |  |  | 0.8560 | 0.8560 | 0.0000 | 0.5880 | 0 | 0 | 0 | 1000 |
| gqa_cat_val | steered | cat | 0.0500 | 0.8550 | 0.8560 | -0.0010 | 0.5850 | 2 | 3 | 5 | 1000 |
| gqa_cat_val | steered | cat | 0.1000 | 0.8540 | 0.8560 | -0.0020 | 0.5860 | 2 | 4 | 6 | 1000 |
| gqa_cat_val | steered | cat | 0.2500 | 0.8570 | 0.8560 | 0.0010 | 0.5790 | 6 | 5 | 11 | 1000 |
| gqa_cat_val | steered | cat | 0.5000 | 0.8600 | 0.8560 | 0.0040 | 0.5760 | 12 | 8 | 20 | 1000 |
| gqa_cat_val | steered | cat | 1.0000 | 0.8610 | 0.8560 | 0.0050 | 0.5770 | 21 | 16 | 37 | 1000 |
| gqa_cat_val | steered | cat_res | 0.0500 | 0.8520 | 0.8560 | -0.0040 | 0.5880 | 0 | 4 | 4 | 1000 |
| gqa_cat_val | steered | cat_res | 0.1000 | 0.8530 | 0.8560 | -0.0030 | 0.5910 | 2 | 5 | 7 | 1000 |
| gqa_cat_val | steered | cat_res | 0.2500 | 0.8510 | 0.8560 | -0.0050 | 0.5970 | 2 | 7 | 9 | 1000 |
| gqa_cat_val | steered | cat_res | 0.5000 | 0.8480 | 0.8560 | -0.0080 | 0.5980 | 3 | 11 | 14 | 1000 |
| gqa_cat_val | steered | global_all | 0.0500 | 0.8540 | 0.8560 | -0.0020 | 0.5880 | 1 | 3 | 4 | 1000 |
| gqa_cat_val | steered | global_all | 0.1000 | 0.8540 | 0.8560 | -0.0020 | 0.5860 | 2 | 4 | 6 | 1000 |
| gqa_cat_val | steered | global_all | 0.2500 | 0.8610 | 0.8560 | 0.0050 | 0.5730 | 11 | 6 | 17 | 1000 |
| gqa_cat_val | steered | global_all | 0.5000 | 0.8640 | 0.8560 | 0.0080 | 0.5660 | 15 | 7 | 22 | 1000 |
| gqa_cat_val | steered | global_all | 1.0000 | 0.8660 | 0.8560 | 0.0100 | 0.5400 | 31 | 21 | 52 | 1000 |

## Notes

- `per_type_accuracy` and `per_subtype_accuracy` are also written into each run directory as `gqa_typeaware_metrics.json`.
- Baseline rows have `delta_acc=0`; steered rows compare against the matching subset baseline.
