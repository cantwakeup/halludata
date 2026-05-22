# Attribute MME Sanity Report

- Runs root: `/home/huiwei/sy/halludata/data/outputs_attr_sanity_mme/runs`
- Summary CSV: `/home/huiwei/sy/halludata/data/outputs_attr_sanity_mme/summary.csv`
- Data report: `/home/huiwei/sy/halludata/data/outputs_attr_sanity_mme/ATTR_MME_DATA_REPORT.md`
- Runs summarized: `62`

## Experiment Settings

| model/runner | decode | layers | head_select | top_heads | apply_to |
| --- | --- | --- | --- | --- | --- |
| run_steered_benchmark.py HF LLaVA | greedy do_sample=False; max_new_tokens from runner default unless overridden | 5-25 | norm | 64 | prefill+decode last_token |

Vector paths:

- `/home/huiwei/sy/halludata/data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt`
- `/home/huiwei/sy/halludata/data/outputs_after_template_disjoint_v2/steering/after_template_disjoint_v2_bucket_vectors.pt`

Skipped vectors:

| category | vector | reason |
| --- | --- | --- |
| count | global_all_plus_attr_res_separate_alpha_grid | current ExpertSteeringController exposes one shared --steer-alpha for all enabled vectors |
| count | attr_count | ATTR_COUNT_VECTOR_PATH not set |
| color | global_all_plus_attr_res_separate_alpha_grid | current ExpertSteeringController exposes one shared --steer-alpha for all enabled vectors |
| color | attr_color | ATTR_COLOR_VECTOR_PATH not set |

## Baseline

| category | n | accuracy | precision | recall | f1 | yes_rate | tp | tn | fp | fn | label_yes_accuracy | label_no_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| color | 60 | 0.8667 | 0.8056 | 0.9667 | 0.8788 | 0.6000 | 29 | 23 | 7 | 1 | 0.9667 | 0.7667 |
| count | 60 | 0.7167 | 0.8421 | 0.5333 | 0.6531 | 0.3167 | 16 | 27 | 3 | 14 | 0.5333 | 0.9000 |

## Best By F1

| category | vector | enabled_experts | alpha | baseline_accuracy | accuracy | delta_acc | baseline_f1 | f1 | delta_f1 | precision | recall | delta_fp | delta_fn | delta_yes_rate | wrong_to_right | right_to_wrong | changed_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| color | attr_res | attr_res | 0.2500 | 0.8667 | 0.8833 | 0.0167 | 0.8788 | 0.8923 | 0.0135 | 0.8286 | 0.9667 | -1 | 0 | -0.0167 | 1 | 0 | 1 |
| count | attr | attr | 0.5000 | 0.7167 | 0.8333 | 0.1167 | 0.6531 | 0.8214 | 0.1684 | 0.8846 | 0.7667 | 0 | -7 | 0.1167 | 7 | 0 | 7 |

## Best By Accuracy

| category | vector | alpha | baseline_accuracy | accuracy | delta_acc | baseline_f1 | f1 | delta_f1 | delta_fp | delta_fn | delta_yes_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| color | attr_res | 0.2500 | 0.8667 | 0.8833 | 0.0167 | 0.8788 | 0.8923 | 0.0135 | -1 | 0 | -0.0167 |
| count | attr | 0.5000 | 0.7167 | 0.8333 | 0.1167 | 0.6531 | 0.8214 | 0.1684 | 0 | -7 | 0.1167 |

## Automatic Diagnostics

| category | vector | diagnosis |
| --- | --- | --- |
| color | attr | yes-shift risk |
| color | attr_res | unstable single-point gain |
| color | disjoint_v2_attr | weak/no stable effect |
| color | global_all | yes-shift risk |
| color | global_all_plus_attr_res | weak/no stable effect |
| count | attr | yes-shift risk |
| count | attr_res | precise correction |
| count | disjoint_v2_attr | precise correction, yes-shift risk |
| count | global_all | yes-shift risk |
| count | global_all_plus_attr_res | precise correction, yes-shift risk |

## All Runs

| category | method | vector | enabled_experts | alpha | n | accuracy | precision | recall | f1 | yes_rate | tp | tn | fp | fn | delta_acc | delta_f1 | delta_yes_rate | wrong_to_right | right_to_wrong | changed_pred | yes_to_no | no_to_yes | delta_margin_yes | delta_margin_no |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| color | baseline |  | cat,attr,rel |  | 60 | 0.8667 | 0.8056 | 0.9667 | 0.8788 | 0.6000 | 29 | 23 | 7 | 1 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 |
| color | steered | attr | attr | 0.0250 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.0000 | 0.0042 |
| color | steered | attr | attr | 0.0500 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.0375 | 0.0083 |
| color | steered | attr | attr | 0.1000 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.0750 | 0.0250 |
| color | steered | attr | attr | 0.2500 | 60 | 0.8333 | 0.7632 | 0.9667 | 0.8529 | 0.6333 | 29 | 21 | 9 | 1 | -0.0333 | -0.0258 | 0.0333 | 0 | 2 | 2 | 0 | 2 | 0.2375 | 0.0542 |
| color | steered | attr | attr | 0.5000 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.4750 | 0.1792 |
| color | steered | attr | attr | 0.7500 | 60 | 0.8333 | 0.7632 | 0.9667 | 0.8529 | 0.6333 | 29 | 21 | 9 | 1 | -0.0333 | -0.0258 | 0.0333 | 0 | 2 | 2 | 0 | 2 | 0.6500 | 0.3375 |
| color | steered | attr | attr | 1.0000 | 60 | 0.8167 | 0.7317 | 1.0000 | 0.8451 | 0.6833 | 30 | 19 | 11 | 0 | -0.0500 | -0.0337 | 0.0833 | 1 | 4 | 5 | 0 | 5 | 0.7583 | 0.6042 |
| color | steered | attr_res | attr_res | 0.0250 | 60 | 0.8667 | 0.8056 | 0.9667 | 0.8788 | 0.6000 | 29 | 23 | 7 | 1 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | -0.0167 | -0.0083 |
| color | steered | attr_res | attr_res | 0.0500 | 60 | 0.8667 | 0.8056 | 0.9667 | 0.8788 | 0.6000 | 29 | 23 | 7 | 1 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0.0083 | 0.0083 |
| color | steered | attr_res | attr_res | 0.1000 | 60 | 0.8667 | 0.8056 | 0.9667 | 0.8788 | 0.6000 | 29 | 23 | 7 | 1 | 0.0000 | 0.0000 | 0.0000 | 1 | 1 | 2 | 1 | 1 | -0.0042 | -0.0125 |
| color | steered | attr_res | attr_res | 0.2500 | 60 | 0.8833 | 0.8286 | 0.9667 | 0.8923 | 0.5833 | 29 | 24 | 6 | 1 | 0.0167 | 0.0135 | -0.0167 | 1 | 0 | 1 | 1 | 0 | 0.0333 | -0.0250 |
| color | steered | attr_res | attr_res | 0.5000 | 60 | 0.8667 | 0.8056 | 0.9667 | 0.8788 | 0.6000 | 29 | 23 | 7 | 1 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0.0667 | -0.0500 |
| color | steered | attr_res | attr_res | 0.7500 | 60 | 0.8500 | 0.8000 | 0.9333 | 0.8615 | 0.5833 | 28 | 23 | 7 | 2 | -0.0167 | -0.0172 | -0.0167 | 0 | 1 | 1 | 1 | 0 | 0.1083 | -0.1000 |
| color | steered | attr_res | attr_res | 1.0000 | 60 | 0.8500 | 0.8000 | 0.9333 | 0.8615 | 0.5833 | 28 | 23 | 7 | 2 | -0.0167 | -0.0172 | -0.0167 | 0 | 1 | 1 | 1 | 0 | 0.1167 | -0.1167 |
| color | steered | disjoint_v2_attr | attr | 0.0250 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.0083 | -0.0042 |
| color | steered | disjoint_v2_attr | attr | 0.0500 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.0333 | 0.0167 |
| color | steered | disjoint_v2_attr | attr | 0.1000 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.0542 | 0.0417 |
| color | steered | disjoint_v2_attr | attr | 0.2500 | 60 | 0.8333 | 0.7632 | 0.9667 | 0.8529 | 0.6333 | 29 | 21 | 9 | 1 | -0.0333 | -0.0258 | 0.0333 | 0 | 2 | 2 | 0 | 2 | 0.1750 | 0.0750 |
| color | steered | disjoint_v2_attr | attr | 0.5000 | 60 | 0.8333 | 0.7632 | 0.9667 | 0.8529 | 0.6333 | 29 | 21 | 9 | 1 | -0.0333 | -0.0258 | 0.0333 | 0 | 2 | 2 | 0 | 2 | 0.3833 | 0.1542 |
| color | steered | disjoint_v2_attr | attr | 0.7500 | 60 | 0.8333 | 0.7632 | 0.9667 | 0.8529 | 0.6333 | 29 | 21 | 9 | 1 | -0.0333 | -0.0258 | 0.0333 | 0 | 2 | 2 | 0 | 2 | 0.5500 | 0.2750 |
| color | steered | disjoint_v2_attr | attr | 1.0000 | 60 | 0.8333 | 0.7632 | 0.9667 | 0.8529 | 0.6333 | 29 | 21 | 9 | 1 | -0.0333 | -0.0258 | 0.0333 | 0 | 2 | 2 | 0 | 2 | 0.6792 | 0.4333 |
| color | steered | global_all | global_all | 0.0500 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.0208 | 0.0125 |
| color | steered | global_all | global_all | 0.1000 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.0458 | 0.0250 |
| color | steered | global_all | global_all | 0.2500 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.1583 | 0.1000 |
| color | steered | global_all | global_all | 0.5000 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.2917 | 0.2042 |
| color | steered | global_all | global_all | 1.0000 | 60 | 0.8000 | 0.7143 | 1.0000 | 0.8333 | 0.7000 | 30 | 18 | 12 | 0 | -0.0667 | -0.0455 | 0.1000 | 1 | 5 | 6 | 0 | 6 | 0.4667 | 0.5958 |
| color | steered | global_all_plus_attr_res | global_all,attr_res | 0.0500 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.0292 | 0.0083 |
| color | steered | global_all_plus_attr_res | global_all,attr_res | 0.1000 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.0750 | 0.0167 |
| color | steered | global_all_plus_attr_res | global_all,attr_res | 0.2500 | 60 | 0.8333 | 0.7632 | 0.9667 | 0.8529 | 0.6333 | 29 | 21 | 9 | 1 | -0.0333 | -0.0258 | 0.0333 | 0 | 2 | 2 | 0 | 2 | 0.2458 | 0.0625 |
| color | steered | global_all_plus_attr_res | global_all,attr_res | 0.5000 | 60 | 0.8500 | 0.7838 | 0.9667 | 0.8657 | 0.6167 | 29 | 22 | 8 | 1 | -0.0167 | -0.0131 | 0.0167 | 0 | 1 | 1 | 0 | 1 | 0.4500 | 0.1708 |
| count | baseline |  | cat,attr,rel |  | 60 | 0.7167 | 0.8421 | 0.5333 | 0.6531 | 0.3167 | 16 | 27 | 3 | 14 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 |
| count | steered | attr | attr | 0.0250 | 60 | 0.7333 | 0.8500 | 0.5667 | 0.6800 | 0.3333 | 17 | 27 | 3 | 13 | 0.0167 | 0.0269 | 0.0167 | 1 | 0 | 1 | 0 | 1 | 0.0083 | 0.0208 |
| count | steered | attr | attr | 0.0500 | 60 | 0.7500 | 0.8571 | 0.6000 | 0.7059 | 0.3500 | 18 | 27 | 3 | 12 | 0.0333 | 0.0528 | 0.0333 | 2 | 0 | 2 | 0 | 2 | 0.0208 | 0.0125 |
| count | steered | attr | attr | 0.1000 | 60 | 0.7500 | 0.8571 | 0.6000 | 0.7059 | 0.3500 | 18 | 27 | 3 | 12 | 0.0333 | 0.0528 | 0.0333 | 2 | 0 | 2 | 0 | 2 | 0.0583 | 0.0542 |
| count | steered | attr | attr | 0.2500 | 60 | 0.8000 | 0.8750 | 0.7000 | 0.7778 | 0.4000 | 21 | 27 | 3 | 9 | 0.0833 | 0.1247 | 0.0833 | 5 | 0 | 5 | 0 | 5 | 0.2042 | 0.1417 |
| count | steered | attr | attr | 0.5000 | 60 | 0.8333 | 0.8846 | 0.7667 | 0.8214 | 0.4333 | 23 | 27 | 3 | 7 | 0.1167 | 0.1684 | 0.1167 | 7 | 0 | 7 | 0 | 7 | 0.4417 | 0.2917 |
| count | steered | attr | attr | 0.7500 | 60 | 0.8167 | 0.8276 | 0.8000 | 0.8136 | 0.4833 | 24 | 25 | 5 | 6 | 0.1000 | 0.1605 | 0.1667 | 8 | 2 | 10 | 0 | 10 | 0.6333 | 0.4667 |
| count | steered | attr | attr | 1.0000 | 60 | 0.7833 | 0.7297 | 0.9000 | 0.8060 | 0.6167 | 27 | 20 | 10 | 3 | 0.0667 | 0.1529 | 0.3000 | 11 | 7 | 18 | 0 | 18 | 0.8583 | 0.7083 |
| count | steered | attr_res | attr_res | 0.0250 | 60 | 0.7333 | 0.8500 | 0.5667 | 0.6800 | 0.3333 | 17 | 27 | 3 | 13 | 0.0167 | 0.0269 | 0.0167 | 1 | 0 | 1 | 0 | 1 | 0.0000 | 0.0042 |
| count | steered | attr_res | attr_res | 0.0500 | 60 | 0.7500 | 0.8571 | 0.6000 | 0.7059 | 0.3500 | 18 | 27 | 3 | 12 | 0.0333 | 0.0528 | 0.0333 | 2 | 0 | 2 | 0 | 2 | -0.0125 | -0.0125 |
| count | steered | attr_res | attr_res | 0.1000 | 60 | 0.7333 | 0.8500 | 0.5667 | 0.6800 | 0.3333 | 17 | 27 | 3 | 13 | 0.0167 | 0.0269 | 0.0167 | 1 | 0 | 1 | 0 | 1 | -0.0042 | 0.0042 |
| count | steered | attr_res | attr_res | 0.2500 | 60 | 0.7333 | 0.8500 | 0.5667 | 0.6800 | 0.3333 | 17 | 27 | 3 | 13 | 0.0167 | 0.0269 | 0.0167 | 1 | 0 | 1 | 0 | 1 | 0.0042 | -0.0125 |
| count | steered | attr_res | attr_res | 0.5000 | 60 | 0.7500 | 0.8571 | 0.6000 | 0.7059 | 0.3500 | 18 | 27 | 3 | 12 | 0.0333 | 0.0528 | 0.0333 | 2 | 0 | 2 | 0 | 2 | 0.0375 | -0.0500 |
| count | steered | attr_res | attr_res | 0.7500 | 60 | 0.7667 | 0.8636 | 0.6333 | 0.7308 | 0.3667 | 19 | 27 | 3 | 11 | 0.0500 | 0.0777 | 0.0500 | 3 | 0 | 3 | 0 | 3 | 0.0500 | -0.0708 |
| count | steered | attr_res | attr_res | 1.0000 | 60 | 0.7500 | 0.8571 | 0.6000 | 0.7059 | 0.3500 | 18 | 27 | 3 | 12 | 0.0333 | 0.0528 | 0.0333 | 2 | 0 | 2 | 0 | 2 | 0.0792 | -0.0792 |
| count | steered | disjoint_v2_attr | attr | 0.0250 | 60 | 0.7500 | 0.8571 | 0.6000 | 0.7059 | 0.3500 | 18 | 27 | 3 | 12 | 0.0333 | 0.0528 | 0.0333 | 2 | 0 | 2 | 0 | 2 | 0.0042 | 0.0208 |
| count | steered | disjoint_v2_attr | attr | 0.0500 | 60 | 0.7333 | 0.8500 | 0.5667 | 0.6800 | 0.3333 | 17 | 27 | 3 | 13 | 0.0167 | 0.0269 | 0.0167 | 1 | 0 | 1 | 0 | 1 | 0.0000 | 0.0333 |
| count | steered | disjoint_v2_attr | attr | 0.1000 | 60 | 0.7500 | 0.8571 | 0.6000 | 0.7059 | 0.3500 | 18 | 27 | 3 | 12 | 0.0333 | 0.0528 | 0.0333 | 2 | 0 | 2 | 0 | 2 | 0.0375 | 0.0292 |
| count | steered | disjoint_v2_attr | attr | 0.2500 | 60 | 0.7833 | 0.8696 | 0.6667 | 0.7547 | 0.3833 | 20 | 27 | 3 | 10 | 0.0667 | 0.1017 | 0.0667 | 4 | 0 | 4 | 0 | 4 | 0.1000 | 0.0625 |
| count | steered | disjoint_v2_attr | attr | 0.5000 | 60 | 0.8167 | 0.8800 | 0.7333 | 0.8000 | 0.4167 | 22 | 27 | 3 | 8 | 0.1000 | 0.1469 | 0.1000 | 6 | 0 | 6 | 0 | 6 | 0.2750 | 0.1583 |
| count | steered | disjoint_v2_attr | attr | 0.7500 | 60 | 0.8167 | 0.8276 | 0.8000 | 0.8136 | 0.4833 | 24 | 25 | 5 | 6 | 0.1000 | 0.1605 | 0.1667 | 8 | 2 | 10 | 0 | 10 | 0.4167 | 0.3250 |
| count | steered | disjoint_v2_attr | attr | 1.0000 | 60 | 0.8000 | 0.7647 | 0.8667 | 0.8125 | 0.5667 | 26 | 22 | 8 | 4 | 0.0833 | 0.1594 | 0.2500 | 10 | 5 | 15 | 0 | 15 | 0.5917 | 0.4917 |
| count | steered | global_all | global_all | 0.0500 | 60 | 0.7333 | 0.8500 | 0.5667 | 0.6800 | 0.3333 | 17 | 27 | 3 | 13 | 0.0167 | 0.0269 | 0.0167 | 1 | 0 | 1 | 0 | 1 | 0.0167 | 0.0083 |
| count | steered | global_all | global_all | 0.1000 | 60 | 0.7500 | 0.8571 | 0.6000 | 0.7059 | 0.3500 | 18 | 27 | 3 | 12 | 0.0333 | 0.0528 | 0.0333 | 2 | 0 | 2 | 0 | 2 | 0.0333 | 0.0750 |
| count | steered | global_all | global_all | 0.2500 | 60 | 0.7833 | 0.8696 | 0.6667 | 0.7547 | 0.3833 | 20 | 27 | 3 | 10 | 0.0667 | 0.1017 | 0.0667 | 4 | 0 | 4 | 0 | 4 | 0.1542 | 0.1375 |
| count | steered | global_all | global_all | 0.5000 | 60 | 0.8000 | 0.8214 | 0.7667 | 0.7931 | 0.4667 | 23 | 25 | 5 | 7 | 0.0833 | 0.1400 | 0.1500 | 7 | 2 | 9 | 0 | 9 | 0.3042 | 0.2958 |
| count | steered | global_all | global_all | 1.0000 | 60 | 0.8000 | 0.7500 | 0.9000 | 0.8182 | 0.6000 | 27 | 21 | 9 | 3 | 0.0833 | 0.1651 | 0.2833 | 11 | 6 | 17 | 0 | 17 | 0.6458 | 0.7042 |
| count | steered | global_all_plus_attr_res | global_all,attr_res | 0.0500 | 60 | 0.7500 | 0.8571 | 0.6000 | 0.7059 | 0.3500 | 18 | 27 | 3 | 12 | 0.0333 | 0.0528 | 0.0333 | 2 | 0 | 2 | 0 | 2 | 0.0250 | 0.0417 |
| count | steered | global_all_plus_attr_res | global_all,attr_res | 0.1000 | 60 | 0.7500 | 0.8571 | 0.6000 | 0.7059 | 0.3500 | 18 | 27 | 3 | 12 | 0.0333 | 0.0528 | 0.0333 | 2 | 0 | 2 | 0 | 2 | 0.0500 | 0.0458 |
| count | steered | global_all_plus_attr_res | global_all,attr_res | 0.2500 | 60 | 0.8000 | 0.8750 | 0.7000 | 0.7778 | 0.4000 | 21 | 27 | 3 | 9 | 0.0833 | 0.1247 | 0.0833 | 5 | 0 | 5 | 0 | 5 | 0.1958 | 0.1250 |
| count | steered | global_all_plus_attr_res | global_all,attr_res | 0.5000 | 60 | 0.8333 | 0.8846 | 0.7667 | 0.8214 | 0.4333 | 23 | 27 | 3 | 7 | 0.1167 | 0.1684 | 0.1167 | 7 | 0 | 7 | 0 | 7 | 0.4167 | 0.2875 |

## Reading Guide

- `FP` means label=no but pred=yes; rising FP with rising yes_rate is a yes-shift risk.
- `FN` means label=yes but pred=no; rising FN with falling yes_rate is a no-shift risk.
- `delta_margin_yes/no` are first-token Yes-vs-No margin changes when available from fixed steering runs.
- Treat single changed predictions on small MME subsets as sanity signals, not final evidence.
