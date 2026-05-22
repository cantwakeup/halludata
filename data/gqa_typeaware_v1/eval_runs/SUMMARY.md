# GQA Type-Aware Diagnostic Eval Summary

- Summary CSV: `data/gqa_typeaware_v1/eval_runs/summary.csv`
- Runs summarized: 48

## Best Steered Runs By Delta

| eval_subset | method | vector | alpha | accuracy | baseline_accuracy | delta_acc | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_attr_val | steered | rel | 1.0000 | 0.6935 | 0.6778 | 0.0157 | 0.6172 | 49 | 34 | 83 | 956 |
| gqa_cat_val | steered | attr | 0.5000 | 0.8700 | 0.8560 | 0.0140 | 0.5580 | 22 | 8 | 30 | 1000 |
| gqa_cat_val | steered | attr | 1.0000 | 0.8690 | 0.8560 | 0.0130 | 0.5290 | 36 | 23 | 59 | 1000 |
| gqa_cat_val | steered | rel | 0.5000 | 0.8690 | 0.8560 | 0.0130 | 0.5550 | 23 | 10 | 33 | 1000 |
| gqa_cat_val | steered | rel | 1.0000 | 0.8680 | 0.8560 | 0.0120 | 0.5240 | 38 | 26 | 64 | 1000 |
| gqa_attr_val | steered | attr | 1.0000 | 0.6893 | 0.6778 | 0.0115 | 0.6088 | 51 | 40 | 91 | 956 |
| gqa_rel_val | steered | attr | 1.0000 | 0.5711 | 0.5611 | 0.0100 | 0.8628 | 25 | 17 | 42 | 802 |
| gqa_rel_val | steered | rel | 1.0000 | 0.5711 | 0.5611 | 0.0100 | 0.8529 | 30 | 22 | 52 | 802 |
| gqa_cat_val | steered | rel | 0.2500 | 0.8620 | 0.8560 | 0.0060 | 0.5740 | 10 | 4 | 14 | 1000 |
| gqa_cat_val | steered | attr | 0.2500 | 0.8610 | 0.8560 | 0.0050 | 0.5750 | 9 | 4 | 13 | 1000 |
| gqa_cat_val | steered | cat | 1.0000 | 0.8610 | 0.8560 | 0.0050 | 0.5770 | 21 | 16 | 37 | 1000 |
| gqa_cat_val | steered | cat | 0.5000 | 0.8600 | 0.8560 | 0.0040 | 0.5760 | 12 | 8 | 20 | 1000 |

## All Runs

| eval_subset | method | vector | alpha | accuracy | baseline_accuracy | delta_acc | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_attr_val | baseline |  |  | 0.6778 | 0.6778 | 0.0000 | 0.6935 | 0 | 0 | 0 | 956 |
| gqa_attr_val | steered | attr | 0.0500 | 0.6695 | 0.6778 | -0.0084 | 0.6893 | 4 | 12 | 16 | 956 |
| gqa_attr_val | steered | attr | 0.1000 | 0.6705 | 0.6778 | -0.0073 | 0.6967 | 5 | 12 | 17 | 956 |
| gqa_attr_val | steered | attr | 0.2500 | 0.6653 | 0.6778 | -0.0126 | 0.6893 | 5 | 17 | 22 | 956 |
| gqa_attr_val | steered | attr | 0.5000 | 0.6726 | 0.6778 | -0.0052 | 0.6799 | 15 | 20 | 35 | 956 |
| gqa_attr_val | steered | attr | 1.0000 | 0.6893 | 0.6778 | 0.0115 | 0.6088 | 51 | 40 | 91 | 956 |
| gqa_attr_val | steered | cat | 0.0500 | 0.6663 | 0.6778 | -0.0115 | 0.6946 | 5 | 16 | 21 | 956 |
| gqa_attr_val | steered | cat | 0.1000 | 0.6653 | 0.6778 | -0.0126 | 0.6935 | 5 | 17 | 22 | 956 |
| gqa_attr_val | steered | cat | 0.2500 | 0.6663 | 0.6778 | -0.0115 | 0.7008 | 5 | 16 | 21 | 956 |
| gqa_attr_val | steered | cat | 0.5000 | 0.6642 | 0.6778 | -0.0136 | 0.6967 | 9 | 22 | 31 | 956 |
| gqa_attr_val | steered | cat | 1.0000 | 0.6674 | 0.6778 | -0.0105 | 0.7186 | 22 | 32 | 54 | 956 |
| gqa_attr_val | steered | rel | 0.0500 | 0.6726 | 0.6778 | -0.0052 | 0.6925 | 5 | 10 | 15 | 956 |
| gqa_attr_val | steered | rel | 0.1000 | 0.6715 | 0.6778 | -0.0063 | 0.6872 | 7 | 13 | 20 | 956 |
| gqa_attr_val | steered | rel | 0.2500 | 0.6747 | 0.6778 | -0.0031 | 0.6904 | 6 | 9 | 15 | 956 |
| gqa_attr_val | steered | rel | 0.5000 | 0.6726 | 0.6778 | -0.0052 | 0.6820 | 12 | 17 | 29 | 956 |
| gqa_attr_val | steered | rel | 1.0000 | 0.6935 | 0.6778 | 0.0157 | 0.6172 | 49 | 34 | 83 | 956 |
| gqa_cat_val | baseline |  |  | 0.8560 | 0.8560 | 0.0000 | 0.5880 | 0 | 0 | 0 | 1000 |
| gqa_cat_val | steered | attr | 0.0500 | 0.8540 | 0.8560 | -0.0020 | 0.5880 | 0 | 2 | 2 | 1000 |
| gqa_cat_val | steered | attr | 0.1000 | 0.8560 | 0.8560 | 0.0000 | 0.5820 | 5 | 5 | 10 | 1000 |
| gqa_cat_val | steered | attr | 0.2500 | 0.8610 | 0.8560 | 0.0050 | 0.5750 | 9 | 4 | 13 | 1000 |
| gqa_cat_val | steered | attr | 0.5000 | 0.8700 | 0.8560 | 0.0140 | 0.5580 | 22 | 8 | 30 | 1000 |
| gqa_cat_val | steered | attr | 1.0000 | 0.8690 | 0.8560 | 0.0130 | 0.5290 | 36 | 23 | 59 | 1000 |
| gqa_cat_val | steered | cat | 0.0500 | 0.8550 | 0.8560 | -0.0010 | 0.5850 | 2 | 3 | 5 | 1000 |
| gqa_cat_val | steered | cat | 0.1000 | 0.8540 | 0.8560 | -0.0020 | 0.5860 | 2 | 4 | 6 | 1000 |
| gqa_cat_val | steered | cat | 0.2500 | 0.8570 | 0.8560 | 0.0010 | 0.5790 | 6 | 5 | 11 | 1000 |
| gqa_cat_val | steered | cat | 0.5000 | 0.8600 | 0.8560 | 0.0040 | 0.5760 | 12 | 8 | 20 | 1000 |
| gqa_cat_val | steered | cat | 1.0000 | 0.8610 | 0.8560 | 0.0050 | 0.5770 | 21 | 16 | 37 | 1000 |
| gqa_cat_val | steered | rel | 0.0500 | 0.8550 | 0.8560 | -0.0010 | 0.5850 | 2 | 3 | 5 | 1000 |
| gqa_cat_val | steered | rel | 0.1000 | 0.8550 | 0.8560 | -0.0010 | 0.5850 | 3 | 4 | 7 | 1000 |
| gqa_cat_val | steered | rel | 0.2500 | 0.8620 | 0.8560 | 0.0060 | 0.5740 | 10 | 4 | 14 | 1000 |
| gqa_cat_val | steered | rel | 0.5000 | 0.8690 | 0.8560 | 0.0130 | 0.5550 | 23 | 10 | 33 | 1000 |
| gqa_cat_val | steered | rel | 1.0000 | 0.8680 | 0.8560 | 0.0120 | 0.5240 | 38 | 26 | 64 | 1000 |
| gqa_rel_val | baseline |  |  | 0.5611 | 0.5611 | 0.0000 | 0.8978 | 0 | 0 | 0 | 802 |
| gqa_rel_val | steered | attr | 0.0500 | 0.5561 | 0.5611 | -0.0050 | 0.9002 | 2 | 6 | 8 | 802 |
| gqa_rel_val | steered | attr | 0.1000 | 0.5549 | 0.5611 | -0.0062 | 0.9040 | 3 | 8 | 11 | 802 |
| gqa_rel_val | steered | attr | 0.2500 | 0.5549 | 0.5611 | -0.0062 | 0.9040 | 3 | 8 | 11 | 802 |
| gqa_rel_val | steered | attr | 0.5000 | 0.5561 | 0.5611 | -0.0050 | 0.9027 | 6 | 10 | 16 | 802 |
| gqa_rel_val | steered | attr | 1.0000 | 0.5711 | 0.5611 | 0.0100 | 0.8628 | 25 | 17 | 42 | 802 |
| gqa_rel_val | steered | cat | 0.0500 | 0.5524 | 0.5611 | -0.0087 | 0.9065 | 1 | 8 | 9 | 802 |
| gqa_rel_val | steered | cat | 0.1000 | 0.5486 | 0.5611 | -0.0125 | 0.9102 | 0 | 10 | 10 | 802 |
| gqa_rel_val | steered | cat | 0.2500 | 0.5449 | 0.5611 | -0.0162 | 0.9190 | 2 | 15 | 17 | 802 |
| gqa_rel_val | steered | cat | 0.5000 | 0.5362 | 0.5611 | -0.0249 | 0.9277 | 3 | 23 | 26 | 802 |
| gqa_rel_val | steered | cat | 1.0000 | 0.5100 | 0.5611 | -0.0511 | 0.9539 | 4 | 45 | 49 | 802 |
| gqa_rel_val | steered | rel | 0.0500 | 0.5611 | 0.5611 | 0.0000 | 0.8978 | 3 | 3 | 6 | 802 |
| gqa_rel_val | steered | rel | 0.1000 | 0.5561 | 0.5611 | -0.0050 | 0.9002 | 3 | 7 | 10 | 802 |
| gqa_rel_val | steered | rel | 0.2500 | 0.5549 | 0.5611 | -0.0062 | 0.9090 | 5 | 10 | 15 | 802 |
| gqa_rel_val | steered | rel | 0.5000 | 0.5561 | 0.5611 | -0.0050 | 0.9052 | 7 | 11 | 18 | 802 |
| gqa_rel_val | steered | rel | 1.0000 | 0.5711 | 0.5611 | 0.0100 | 0.8529 | 30 | 22 | 52 | 802 |

## Notes

- `per_type_accuracy` and `per_subtype_accuracy` are also written into each run directory as `gqa_typeaware_metrics.json`.
- Baseline rows have `delta_acc=0`; steered rows compare against the matching subset baseline.
