# GQA Type-Aware Diagnostic Eval Summary

- Summary CSV: `data/gqa_typeaware_v1/eval_runs_global_residual/summary.csv`
- Runs summarized: 36

## Best Steered Runs By Delta

| eval_subset | method | vector | alpha | accuracy | baseline_accuracy | delta_acc | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_rel_val | steered | rel_res | 1.0000 | 0.6260 | 0.5940 | 0.0320 | 0.8520 | 19 | 3 | 22 | 500 |
| gqa_rel_val | steered | rel_res | 0.5000 | 0.6140 | 0.5940 | 0.0200 | 0.8760 | 10 | 0 | 10 | 500 |
| gqa_rel_val | steered | global_all,rel_res | 1.0000 | 0.6120 | 0.5940 | 0.0180 | 0.8500 | 20 | 11 | 31 | 500 |
| gqa_rel_val | steered | rel | 1.0000 | 0.6120 | 0.5940 | 0.0180 | 0.8500 | 21 | 12 | 33 | 500 |
| gqa_cat_val | steered | global_all | 1.0000 | 0.8460 | 0.8340 | 0.0120 | 0.5220 | 17 | 11 | 28 | 500 |
| gqa_cat_val | steered | cat | 1.0000 | 0.8440 | 0.8340 | 0.0100 | 0.5720 | 12 | 7 | 19 | 500 |
| gqa_attr_val | steered | attr_res | 1.0000 | 0.6920 | 0.6860 | 0.0060 | 0.6560 | 11 | 8 | 19 | 500 |
| gqa_attr_val | steered | global_all | 1.0000 | 0.6920 | 0.6860 | 0.0060 | 0.6520 | 17 | 14 | 31 | 500 |
| gqa_cat_val | steered | global_all | 0.2500 | 0.8610 | 0.8560 | 0.0050 | 0.5730 | 11 | 6 | 17 | 1000 |
| gqa_cat_val | steered | global_all,cat_res | 0.5000 | 0.8380 | 0.8340 | 0.0040 | 0.5740 | 5 | 3 | 8 | 500 |
| gqa_attr_val | steered | attr | 1.0000 | 0.6900 | 0.6860 | 0.0040 | 0.6060 | 26 | 24 | 50 | 500 |
| gqa_attr_val | steered | global_all,attr_res | 0.5000 | 0.6900 | 0.6860 | 0.0040 | 0.6660 | 12 | 10 | 22 | 500 |

## All Runs

| eval_subset | method | vector | alpha | accuracy | baseline_accuracy | delta_acc | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_attr_val | baseline |  |  | 0.6860 | 0.6860 | 0.0000 | 0.6860 | 0 | 0 | 0 | 500 |
| gqa_attr_val | steered | attr | 0.5000 | 0.6880 | 0.6860 | 0.0020 | 0.6680 | 12 | 11 | 23 | 500 |
| gqa_attr_val | steered | attr | 1.0000 | 0.6900 | 0.6860 | 0.0040 | 0.6060 | 26 | 24 | 50 | 500 |
| gqa_attr_val | steered | attr_res | 0.5000 | 0.6860 | 0.6860 | 0.0000 | 0.6740 | 5 | 5 | 10 | 500 |
| gqa_attr_val | steered | attr_res | 1.0000 | 0.6920 | 0.6860 | 0.0060 | 0.6560 | 11 | 8 | 19 | 500 |
| gqa_attr_val | steered | global_all | 0.5000 | 0.6780 | 0.6860 | -0.0080 | 0.6820 | 7 | 11 | 18 | 500 |
| gqa_attr_val | steered | global_all | 1.0000 | 0.6920 | 0.6860 | 0.0060 | 0.6520 | 17 | 14 | 31 | 500 |
| gqa_attr_val | steered | global_all,attr_res | 0.5000 | 0.6900 | 0.6860 | 0.0040 | 0.6660 | 12 | 10 | 22 | 500 |
| gqa_attr_val | steered | global_all,attr_res | 1.0000 | 0.6880 | 0.6860 | 0.0020 | 0.6080 | 25 | 24 | 49 | 500 |
| gqa_cat_val | baseline |  |  | 0.8340 | 0.8340 | 0.0000 | 0.5780 | 0 | 0 | 0 | 500 |
| gqa_cat_val | steered | cat | 0.0500 | 0.8550 | 0.8560 | -0.0010 | 0.5850 | 2 | 3 | 5 | 1000 |
| gqa_cat_val | steered | cat | 0.1000 | 0.8540 | 0.8560 | -0.0020 | 0.5860 | 2 | 4 | 6 | 1000 |
| gqa_cat_val | steered | cat | 0.2500 | 0.8570 | 0.8560 | 0.0010 | 0.5790 | 6 | 5 | 11 | 1000 |
| gqa_cat_val | steered | cat | 0.5000 | 0.8360 | 0.8340 | 0.0020 | 0.5640 | 5 | 4 | 9 | 500 |
| gqa_cat_val | steered | cat | 1.0000 | 0.8440 | 0.8340 | 0.0100 | 0.5720 | 12 | 7 | 19 | 500 |
| gqa_cat_val | steered | cat_res | 0.0500 | 0.8520 | 0.8560 | -0.0040 | 0.5880 | 0 | 4 | 4 | 1000 |
| gqa_cat_val | steered | cat_res | 0.1000 | 0.8530 | 0.8560 | -0.0030 | 0.5910 | 2 | 5 | 7 | 1000 |
| gqa_cat_val | steered | cat_res | 0.2500 | 0.8510 | 0.8560 | -0.0050 | 0.5970 | 2 | 7 | 9 | 1000 |
| gqa_cat_val | steered | cat_res | 0.5000 | 0.8280 | 0.8340 | -0.0060 | 0.5920 | 3 | 6 | 9 | 500 |
| gqa_cat_val | steered | cat_res | 1.0000 | 0.8200 | 0.8340 | -0.0140 | 0.6000 | 2 | 9 | 11 | 500 |
| gqa_cat_val | steered | global_all | 0.0500 | 0.8540 | 0.8560 | -0.0020 | 0.5880 | 1 | 3 | 4 | 1000 |
| gqa_cat_val | steered | global_all | 0.1000 | 0.8540 | 0.8560 | -0.0020 | 0.5860 | 2 | 4 | 6 | 1000 |
| gqa_cat_val | steered | global_all | 0.2500 | 0.8610 | 0.8560 | 0.0050 | 0.5730 | 11 | 6 | 17 | 1000 |
| gqa_cat_val | steered | global_all | 0.5000 | 0.8360 | 0.8340 | 0.0020 | 0.5560 | 6 | 5 | 11 | 500 |
| gqa_cat_val | steered | global_all | 1.0000 | 0.8460 | 0.8340 | 0.0120 | 0.5220 | 17 | 11 | 28 | 500 |
| gqa_cat_val | steered | global_all,cat_res | 0.5000 | 0.8380 | 0.8340 | 0.0040 | 0.5740 | 5 | 3 | 8 | 500 |
| gqa_cat_val | steered | global_all,cat_res | 1.0000 | 0.8220 | 0.8340 | -0.0120 | 0.5940 | 6 | 12 | 18 | 500 |
| gqa_rel_val | baseline |  |  | 0.5940 | 0.5940 | 0.0000 | 0.8960 | 0 | 0 | 0 | 500 |
| gqa_rel_val | steered | global_all | 0.5000 | 0.5760 | 0.5940 | -0.0180 | 0.9140 | 1 | 10 | 11 | 500 |
| gqa_rel_val | steered | global_all | 1.0000 | 0.5680 | 0.5940 | -0.0260 | 0.9220 | 3 | 16 | 19 | 500 |
| gqa_rel_val | steered | global_all,rel_res | 0.5000 | 0.5840 | 0.5940 | -0.0100 | 0.9060 | 2 | 7 | 9 | 500 |
| gqa_rel_val | steered | global_all,rel_res | 1.0000 | 0.6120 | 0.5940 | 0.0180 | 0.8500 | 20 | 11 | 31 | 500 |
| gqa_rel_val | steered | rel | 0.5000 | 0.5860 | 0.5940 | -0.0080 | 0.9040 | 3 | 7 | 10 | 500 |
| gqa_rel_val | steered | rel | 1.0000 | 0.6120 | 0.5940 | 0.0180 | 0.8500 | 21 | 12 | 33 | 500 |
| gqa_rel_val | steered | rel_res | 0.5000 | 0.6140 | 0.5940 | 0.0200 | 0.8760 | 10 | 0 | 10 | 500 |
| gqa_rel_val | steered | rel_res | 1.0000 | 0.6260 | 0.5940 | 0.0320 | 0.8520 | 19 | 3 | 22 | 500 |

## Notes

- `per_type_accuracy` and `per_subtype_accuracy` are also written into each run directory as `gqa_typeaware_metrics.json`.
- Baseline rows have `delta_acc=0`; steered rows compare against the matching subset baseline.
