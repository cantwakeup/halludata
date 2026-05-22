# GQA Type-Aware Diagnostic Eval Summary

- Summary CSV: `data/gqa_typeaware_v1/shared_private_sanity_eval_300/summary.csv`
- Runs summarized: 153

## Best Steered Runs By Delta

| eval_subset | method | vector | alpha | accuracy | baseline_accuracy | delta_acc | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_rel_val | steered | attr_res | 0.7500 | 0.5967 | 0.5900 | 0.0067 | 0.8933 | 3 | 1 | 4 | 300 |
| gqa_rel_val | steered | attr_res | 1.0000 | 0.5967 | 0.5900 | 0.0067 | 0.8933 | 2 | 0 | 2 | 300 |
| gqa_rel_val | steered | attr_res | 1.5000 | 0.5967 | 0.5900 | 0.0067 | 0.8933 | 2 | 0 | 2 | 300 |
| gqa_cat_val | steered | attr_res | 0.5000 | 0.8400 | 0.8333 | 0.0067 | 0.5900 | 2 | 0 | 2 | 300 |
| gqa_cat_val | steered | attr_res | 1.0000 | 0.8400 | 0.8333 | 0.0067 | 0.5900 | 2 | 0 | 2 | 300 |
| gqa_cat_val | steered | global_type_svd,attr_res | 0.2500 | 0.8400 | 0.8333 | 0.0067 | 0.5833 | 2 | 0 | 2 | 300 |
| gqa_rel_val | steered | attr_res | 0.2500 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 2 | 1 | 3 | 300 |
| gqa_rel_val | steered | global_type_svd,attr_res | 0.2500 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 2 | 1 | 3 | 300 |
| gqa_rel_val | steered | global_type_svd,attr_res | 0.7500 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 2 | 1 | 3 | 300 |
| gqa_rel_val | steered | global_type_svd,attr_res | 1.0000 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 2 | 1 | 3 | 300 |
| gqa_rel_val | steered | global_type_svd,attr_res | 1.5000 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 2 | 1 | 3 | 300 |
| gqa_rel_val | steered | global_type_svd,rel_res | 1.0000 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 1 | 0 | 1 | 300 |

## All Runs

| eval_subset | method | vector | alpha | accuracy | baseline_accuracy | delta_acc | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_attr_val | baseline |  |  | 0.6933 | 0.6933 | 0.0000 | 0.6867 | 0 | 0 | 0 | 300 |
| gqa_attr_val | steered | attr_raw | 0.2500 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 0 | 2 | 2 | 300 |
| gqa_attr_val | steered | attr_raw | 0.5000 | 0.6833 | 0.6933 | -0.0100 | 0.6967 | 0 | 3 | 3 | 300 |
| gqa_attr_val | steered | attr_raw | 0.7500 | 0.6833 | 0.6933 | -0.0100 | 0.6967 | 0 | 3 | 3 | 300 |
| gqa_attr_val | steered | attr_raw | 1.0000 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 0 | 2 | 2 | 300 |
| gqa_attr_val | steered | attr_raw | 1.5000 | 0.6833 | 0.6933 | -0.0100 | 0.6967 | 0 | 3 | 3 | 300 |
| gqa_attr_val | steered | attr_res | 0.2500 | 0.6833 | 0.6933 | -0.0100 | 0.6967 | 0 | 3 | 3 | 300 |
| gqa_attr_val | steered | attr_res | 0.5000 | 0.6900 | 0.6933 | -0.0033 | 0.6900 | 0 | 1 | 1 | 300 |
| gqa_attr_val | steered | attr_res | 0.7500 | 0.6900 | 0.6933 | -0.0033 | 0.6833 | 1 | 2 | 3 | 300 |
| gqa_attr_val | steered | attr_res | 1.0000 | 0.6833 | 0.6933 | -0.0100 | 0.6900 | 0 | 3 | 3 | 300 |
| gqa_attr_val | steered | attr_res | 1.5000 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 0 | 2 | 2 | 300 |
| gqa_attr_val | steered | cat_raw | 0.2500 | 0.6933 | 0.6933 | 0.0000 | 0.6867 | 0 | 0 | 0 | 300 |
| gqa_attr_val | steered | cat_raw | 0.5000 | 0.6833 | 0.6933 | -0.0100 | 0.6967 | 0 | 3 | 3 | 300 |
| gqa_attr_val | steered | cat_raw | 0.7500 | 0.6900 | 0.6933 | -0.0033 | 0.6900 | 0 | 1 | 1 | 300 |
| gqa_attr_val | steered | cat_raw | 1.0000 | 0.6833 | 0.6933 | -0.0100 | 0.6967 | 0 | 3 | 3 | 300 |
| gqa_attr_val | steered | cat_raw | 1.5000 | 0.6800 | 0.6933 | -0.0133 | 0.6933 | 0 | 4 | 4 | 300 |
| gqa_attr_val | steered | cat_res | 0.2500 | 0.6900 | 0.6933 | -0.0033 | 0.6900 | 0 | 1 | 1 | 300 |
| gqa_attr_val | steered | cat_res | 0.5000 | 0.6800 | 0.6933 | -0.0133 | 0.7000 | 0 | 4 | 4 | 300 |
| gqa_attr_val | steered | cat_res | 0.7500 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 0 | 2 | 2 | 300 |
| gqa_attr_val | steered | cat_res | 1.0000 | 0.6767 | 0.6933 | -0.0167 | 0.7033 | 0 | 5 | 5 | 300 |
| gqa_attr_val | steered | cat_res | 1.5000 | 0.6800 | 0.6933 | -0.0133 | 0.7000 | 0 | 4 | 4 | 300 |
| gqa_attr_val | steered | global_type_svd | 0.2500 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 0 | 2 | 2 | 300 |
| gqa_attr_val | steered | global_type_svd | 0.5000 | 0.6933 | 0.6933 | 0.0000 | 0.6867 | 1 | 1 | 2 | 300 |
| gqa_attr_val | steered | global_type_svd | 0.7500 | 0.6833 | 0.6933 | -0.0100 | 0.6967 | 0 | 3 | 3 | 300 |
| gqa_attr_val | steered | global_type_svd | 1.0000 | 0.6933 | 0.6933 | 0.0000 | 0.6867 | 1 | 1 | 2 | 300 |
| gqa_attr_val | steered | global_type_svd | 1.5000 | 0.6800 | 0.6933 | -0.0133 | 0.6933 | 0 | 4 | 4 | 300 |
| gqa_attr_val | steered | global_type_svd,attr_res | 0.2500 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 0 | 2 | 2 | 300 |
| gqa_attr_val | steered | global_type_svd,attr_res | 0.5000 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 1 | 3 | 4 | 300 |
| gqa_attr_val | steered | global_type_svd,attr_res | 0.7500 | 0.6900 | 0.6933 | -0.0033 | 0.6900 | 1 | 2 | 3 | 300 |
| gqa_attr_val | steered | global_type_svd,attr_res | 1.0000 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 0 | 2 | 2 | 300 |
| gqa_attr_val | steered | global_type_svd,attr_res | 1.5000 | 0.6833 | 0.6933 | -0.0100 | 0.6833 | 0 | 3 | 3 | 300 |
| gqa_attr_val | steered | global_type_svd,cat_res | 0.2500 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 0 | 2 | 2 | 300 |
| gqa_attr_val | steered | global_type_svd,cat_res | 0.5000 | 0.6800 | 0.6933 | -0.0133 | 0.7000 | 0 | 4 | 4 | 300 |
| gqa_attr_val | steered | global_type_svd,cat_res | 0.7500 | 0.6800 | 0.6933 | -0.0133 | 0.7000 | 0 | 4 | 4 | 300 |
| gqa_attr_val | steered | global_type_svd,cat_res | 1.0000 | 0.6767 | 0.6933 | -0.0167 | 0.6967 | 0 | 5 | 5 | 300 |
| gqa_attr_val | steered | global_type_svd,cat_res | 1.5000 | 0.6700 | 0.6933 | -0.0233 | 0.7033 | 0 | 7 | 7 | 300 |
| gqa_attr_val | steered | global_type_svd,rel_res | 0.2500 | 0.6900 | 0.6933 | -0.0033 | 0.6900 | 1 | 2 | 3 | 300 |
| gqa_attr_val | steered | global_type_svd,rel_res | 0.5000 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 0 | 2 | 2 | 300 |
| gqa_attr_val | steered | global_type_svd,rel_res | 0.7500 | 0.6900 | 0.6933 | -0.0033 | 0.6900 | 0 | 1 | 1 | 300 |
| gqa_attr_val | steered | global_type_svd,rel_res | 1.0000 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 1 | 3 | 4 | 300 |
| gqa_attr_val | steered | global_type_svd,rel_res | 1.5000 | 0.6833 | 0.6933 | -0.0100 | 0.6967 | 1 | 4 | 5 | 300 |
| gqa_attr_val | steered | rel_raw | 0.2500 | 0.6900 | 0.6933 | -0.0033 | 0.6900 | 1 | 2 | 3 | 300 |
| gqa_attr_val | steered | rel_raw | 0.5000 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 0 | 2 | 2 | 300 |
| gqa_attr_val | steered | rel_raw | 0.7500 | 0.6867 | 0.6933 | -0.0067 | 0.6933 | 0 | 2 | 2 | 300 |
| gqa_attr_val | steered | rel_raw | 1.0000 | 0.6900 | 0.6933 | -0.0033 | 0.6900 | 0 | 1 | 1 | 300 |
| gqa_attr_val | steered | rel_raw | 1.5000 | 0.6800 | 0.6933 | -0.0133 | 0.7000 | 0 | 4 | 4 | 300 |
| gqa_attr_val | steered | rel_res | 0.2500 | 0.6900 | 0.6933 | -0.0033 | 0.6900 | 0 | 1 | 1 | 300 |
| gqa_attr_val | steered | rel_res | 0.5000 | 0.6933 | 0.6933 | 0.0000 | 0.6867 | 0 | 0 | 0 | 300 |
| gqa_attr_val | steered | rel_res | 0.7500 | 0.6833 | 0.6933 | -0.0100 | 0.6967 | 0 | 3 | 3 | 300 |
| gqa_attr_val | steered | rel_res | 1.0000 | 0.6900 | 0.6933 | -0.0033 | 0.6900 | 0 | 1 | 1 | 300 |
| gqa_attr_val | steered | rel_res | 1.5000 | 0.6900 | 0.6933 | -0.0033 | 0.6833 | 1 | 2 | 3 | 300 |
| gqa_cat_val | baseline |  |  | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | attr_raw | 0.2500 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | attr_raw | 0.5000 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 1 | 1 | 2 | 300 |
| gqa_cat_val | steered | attr_raw | 0.7500 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | attr_raw | 1.0000 | 0.8300 | 0.8333 | -0.0033 | 0.5933 | 0 | 1 | 1 | 300 |
| gqa_cat_val | steered | attr_raw | 1.5000 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | attr_res | 0.2500 | 0.8367 | 0.8333 | 0.0033 | 0.5933 | 2 | 1 | 3 | 300 |
| gqa_cat_val | steered | attr_res | 0.5000 | 0.8400 | 0.8333 | 0.0067 | 0.5900 | 2 | 0 | 2 | 300 |
| gqa_cat_val | steered | attr_res | 0.7500 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | attr_res | 1.0000 | 0.8400 | 0.8333 | 0.0067 | 0.5900 | 2 | 0 | 2 | 300 |
| gqa_cat_val | steered | attr_res | 1.5000 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | cat_raw | 0.2500 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | cat_raw | 0.5000 | 0.8367 | 0.8333 | 0.0033 | 0.5933 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | cat_raw | 0.7500 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | cat_raw | 1.0000 | 0.8267 | 0.8333 | -0.0067 | 0.5967 | 0 | 2 | 2 | 300 |
| gqa_cat_val | steered | cat_raw | 1.5000 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 1 | 1 | 2 | 300 |
| gqa_cat_val | steered | cat_res | 0.2500 | 0.8333 | 0.8333 | 0.0000 | 0.5967 | 1 | 1 | 2 | 300 |
| gqa_cat_val | steered | cat_res | 0.5000 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 1 | 1 | 2 | 300 |
| gqa_cat_val | steered | cat_res | 0.7500 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 1 | 1 | 2 | 300 |
| gqa_cat_val | steered | cat_res | 1.0000 | 0.8300 | 0.8333 | -0.0033 | 0.5933 | 0 | 1 | 1 | 300 |
| gqa_cat_val | steered | cat_res | 1.5000 | 0.8300 | 0.8333 | -0.0033 | 0.5933 | 0 | 1 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd | 0.2500 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | global_type_svd | 0.5000 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | global_type_svd | 0.7500 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | global_type_svd | 1.0000 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | global_type_svd | 1.5000 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd,attr_res | 0.2500 | 0.8400 | 0.8333 | 0.0067 | 0.5833 | 2 | 0 | 2 | 300 |
| gqa_cat_val | steered | global_type_svd,attr_res | 0.5000 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd,attr_res | 0.7500 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd,attr_res | 1.0000 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 1 | 1 | 2 | 300 |
| gqa_cat_val | steered | global_type_svd,attr_res | 1.5000 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd,cat_res | 0.2500 | 0.8367 | 0.8333 | 0.0033 | 0.5933 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd,cat_res | 0.5000 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | global_type_svd,cat_res | 0.7500 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd,cat_res | 1.0000 | 0.8300 | 0.8333 | -0.0033 | 0.5933 | 0 | 1 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd,cat_res | 1.5000 | 0.8367 | 0.8333 | 0.0033 | 0.5933 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd,rel_res | 0.2500 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 2 | 1 | 3 | 300 |
| gqa_cat_val | steered | global_type_svd,rel_res | 0.5000 | 0.8300 | 0.8333 | -0.0033 | 0.5933 | 0 | 1 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd,rel_res | 0.7500 | 0.8300 | 0.8333 | -0.0033 | 0.5933 | 0 | 1 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd,rel_res | 1.0000 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | global_type_svd,rel_res | 1.5000 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | rel_raw | 0.2500 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | rel_raw | 0.5000 | 0.8367 | 0.8333 | 0.0033 | 0.5933 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | rel_raw | 0.7500 | 0.8367 | 0.8333 | 0.0033 | 0.5867 | 1 | 0 | 1 | 300 |
| gqa_cat_val | steered | rel_raw | 1.0000 | 0.8367 | 0.8333 | 0.0033 | 0.5933 | 2 | 1 | 3 | 300 |
| gqa_cat_val | steered | rel_raw | 1.5000 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 1 | 1 | 2 | 300 |
| gqa_cat_val | steered | rel_res | 0.2500 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | rel_res | 0.5000 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | rel_res | 0.7500 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_cat_val | steered | rel_res | 1.0000 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 1 | 1 | 2 | 300 |
| gqa_cat_val | steered | rel_res | 1.5000 | 0.8333 | 0.8333 | 0.0000 | 0.5900 | 0 | 0 | 0 | 300 |
| gqa_rel_val | baseline |  |  | 0.5900 | 0.5900 | 0.0000 | 0.9000 | 0 | 0 | 0 | 300 |
| gqa_rel_val | steered | attr_raw | 0.2500 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 1 | 2 | 3 | 300 |
| gqa_rel_val | steered | attr_raw | 0.5000 | 0.5800 | 0.5900 | -0.0100 | 0.9100 | 1 | 4 | 5 | 300 |
| gqa_rel_val | steered | attr_raw | 0.7500 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 0 | 1 | 1 | 300 |
| gqa_rel_val | steered | attr_raw | 1.0000 | 0.5833 | 0.5900 | -0.0067 | 0.9067 | 0 | 2 | 2 | 300 |
| gqa_rel_val | steered | attr_raw | 1.5000 | 0.5900 | 0.5900 | 0.0000 | 0.9000 | 1 | 1 | 2 | 300 |
| gqa_rel_val | steered | attr_res | 0.2500 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 2 | 1 | 3 | 300 |
| gqa_rel_val | steered | attr_res | 0.5000 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 1 | 2 | 3 | 300 |
| gqa_rel_val | steered | attr_res | 0.7500 | 0.5967 | 0.5900 | 0.0067 | 0.8933 | 3 | 1 | 4 | 300 |
| gqa_rel_val | steered | attr_res | 1.0000 | 0.5967 | 0.5900 | 0.0067 | 0.8933 | 2 | 0 | 2 | 300 |
| gqa_rel_val | steered | attr_res | 1.5000 | 0.5967 | 0.5900 | 0.0067 | 0.8933 | 2 | 0 | 2 | 300 |
| gqa_rel_val | steered | cat_raw | 0.2500 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 0 | 1 | 1 | 300 |
| gqa_rel_val | steered | cat_raw | 0.5000 | 0.5833 | 0.5900 | -0.0067 | 0.9067 | 0 | 2 | 2 | 300 |
| gqa_rel_val | steered | cat_raw | 0.7500 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 0 | 1 | 1 | 300 |
| gqa_rel_val | steered | cat_raw | 1.0000 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 1 | 2 | 3 | 300 |
| gqa_rel_val | steered | cat_raw | 1.5000 | 0.5800 | 0.5900 | -0.0100 | 0.9100 | 0 | 3 | 3 | 300 |
| gqa_rel_val | steered | cat_res | 0.2500 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 1 | 2 | 3 | 300 |
| gqa_rel_val | steered | cat_res | 0.5000 | 0.5767 | 0.5900 | -0.0133 | 0.9133 | 0 | 4 | 4 | 300 |
| gqa_rel_val | steered | cat_res | 0.7500 | 0.5833 | 0.5900 | -0.0067 | 0.9067 | 0 | 2 | 2 | 300 |
| gqa_rel_val | steered | cat_res | 1.0000 | 0.5800 | 0.5900 | -0.0100 | 0.9100 | 0 | 3 | 3 | 300 |
| gqa_rel_val | steered | cat_res | 1.5000 | 0.5833 | 0.5900 | -0.0067 | 0.9067 | 1 | 3 | 4 | 300 |
| gqa_rel_val | steered | global_type_svd | 0.2500 | 0.5833 | 0.5900 | -0.0067 | 0.9067 | 1 | 3 | 4 | 300 |
| gqa_rel_val | steered | global_type_svd | 0.5000 | 0.5833 | 0.5900 | -0.0067 | 0.9067 | 0 | 2 | 2 | 300 |
| gqa_rel_val | steered | global_type_svd | 0.7500 | 0.5833 | 0.5900 | -0.0067 | 0.9067 | 0 | 2 | 2 | 300 |
| gqa_rel_val | steered | global_type_svd | 1.0000 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 0 | 1 | 1 | 300 |
| gqa_rel_val | steered | global_type_svd | 1.5000 | 0.5833 | 0.5900 | -0.0067 | 0.9067 | 1 | 3 | 4 | 300 |
| gqa_rel_val | steered | global_type_svd,attr_res | 0.2500 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 2 | 1 | 3 | 300 |
| gqa_rel_val | steered | global_type_svd,attr_res | 0.5000 | 0.5833 | 0.5900 | -0.0067 | 0.9067 | 0 | 2 | 2 | 300 |
| gqa_rel_val | steered | global_type_svd,attr_res | 0.7500 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 2 | 1 | 3 | 300 |
| gqa_rel_val | steered | global_type_svd,attr_res | 1.0000 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 2 | 1 | 3 | 300 |
| gqa_rel_val | steered | global_type_svd,attr_res | 1.5000 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 2 | 1 | 3 | 300 |
| gqa_rel_val | steered | global_type_svd,cat_res | 0.2500 | 0.5800 | 0.5900 | -0.0100 | 0.9100 | 1 | 4 | 5 | 300 |
| gqa_rel_val | steered | global_type_svd,cat_res | 0.5000 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 0 | 1 | 1 | 300 |
| gqa_rel_val | steered | global_type_svd,cat_res | 0.7500 | 0.5767 | 0.5900 | -0.0133 | 0.9133 | 0 | 4 | 4 | 300 |
| gqa_rel_val | steered | global_type_svd,cat_res | 1.0000 | 0.5800 | 0.5900 | -0.0100 | 0.9100 | 0 | 3 | 3 | 300 |
| gqa_rel_val | steered | global_type_svd,cat_res | 1.5000 | 0.5800 | 0.5900 | -0.0100 | 0.9100 | 0 | 3 | 3 | 300 |
| gqa_rel_val | steered | global_type_svd,rel_res | 0.2500 | 0.5833 | 0.5900 | -0.0067 | 0.9067 | 1 | 3 | 4 | 300 |
| gqa_rel_val | steered | global_type_svd,rel_res | 0.5000 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 0 | 1 | 1 | 300 |
| gqa_rel_val | steered | global_type_svd,rel_res | 0.7500 | 0.5833 | 0.5900 | -0.0067 | 0.9067 | 0 | 2 | 2 | 300 |
| gqa_rel_val | steered | global_type_svd,rel_res | 1.0000 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 1 | 0 | 1 | 300 |
| gqa_rel_val | steered | global_type_svd,rel_res | 1.5000 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 1 | 2 | 3 | 300 |
| gqa_rel_val | steered | rel_raw | 0.2500 | 0.5900 | 0.5900 | 0.0000 | 0.9000 | 2 | 2 | 4 | 300 |
| gqa_rel_val | steered | rel_raw | 0.5000 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 0 | 1 | 1 | 300 |
| gqa_rel_val | steered | rel_raw | 0.7500 | 0.5767 | 0.5900 | -0.0133 | 0.9133 | 0 | 4 | 4 | 300 |
| gqa_rel_val | steered | rel_raw | 1.0000 | 0.5800 | 0.5900 | -0.0100 | 0.9100 | 0 | 3 | 3 | 300 |
| gqa_rel_val | steered | rel_raw | 1.5000 | 0.5767 | 0.5900 | -0.0133 | 0.9133 | 0 | 4 | 4 | 300 |
| gqa_rel_val | steered | rel_res | 0.2500 | 0.5900 | 0.5900 | 0.0000 | 0.9000 | 1 | 1 | 2 | 300 |
| gqa_rel_val | steered | rel_res | 0.5000 | 0.5933 | 0.5900 | 0.0033 | 0.8967 | 1 | 0 | 1 | 300 |
| gqa_rel_val | steered | rel_res | 0.7500 | 0.5900 | 0.5900 | 0.0000 | 0.9000 | 1 | 1 | 2 | 300 |
| gqa_rel_val | steered | rel_res | 1.0000 | 0.5867 | 0.5900 | -0.0033 | 0.9033 | 1 | 2 | 3 | 300 |
| gqa_rel_val | steered | rel_res | 1.5000 | 0.5900 | 0.5900 | 0.0000 | 0.9000 | 1 | 1 | 2 | 300 |

## Notes

- `per_type_accuracy` and `per_subtype_accuracy` are also written into each run directory as `gqa_typeaware_metrics.json`.
- Baseline rows have `delta_acc=0`; steered rows compare against the matching subset baseline.
