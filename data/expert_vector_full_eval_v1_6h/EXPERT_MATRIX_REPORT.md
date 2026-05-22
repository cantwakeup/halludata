# Expert Vector Matrix Report

- Summary rows loaded: `136`
- Decision: `FAIL`

## Main 3x3 Matrix

| vector | POPE/category avg | AMBER-attribute | GQA/clean-relation |
| --- | --- | --- | --- |
| baseline | F1=0.8104, Acc=0.8172, a=baseline, yes=0.459, dF1=+0.0000, W2R/R2W=0/0 | F1=0.6867, Acc=0.6867, a=baseline, yes=0.500, dF1=+0.0000, W2R/R2W=0/0 | F1=0.6553, Acc=0.5967, a=baseline, yes=0.673 !, dF1=+0.0000, W2R/R2W=0/0 |
| global | F1=0.8063, Acc=0.8122, a=0.05, yes=0.463, dF1=-0.0041, W2R/R2W=145/154 | F1=0.7120, Acc=0.7033, a=0.1, yes=0.530, dF1=+0.0253, W2R/R2W=51/46 | F1=0.6723, Acc=0.6100, a=0.1, yes=0.693 !, dF1=+0.0170, W2R/R2W=49/45 |
| cat | F1=0.8070, Acc=0.8139, a=0.25, yes=0.459, dF1=-0.0034, W2R/R2W=137/143 | F1=0.7351, Acc=0.7333, a=0.25, yes=0.507, dF1=+0.0484, W2R/R2W=53/39 | F1=0.6893, Acc=0.6333, a=0.05, yes=0.683 !, dF1=+0.0340, W2R/R2W=54/43 |
| attr | F1=0.8126, Acc=0.8206, a=0.5, yes=0.454, dF1=+0.0022, W2R/R2W=150/144 | F1=0.7327, Acc=0.7300, a=0.5, yes=0.510, dF1=+0.0460, W2R/R2W=50/37 | F1=0.6265, Acc=0.5867, a=0.05, yes=0.610, dF1=-0.0288, W2R/R2W=43/46 |
| rel | F1=0.8045, Acc=0.8150, a=0.5, yes=0.444, dF1=-0.0059, W2R/R2W=142/146 | F1=0.7148, Acc=0.7100, a=0.05, yes=0.517, dF1=+0.0281, W2R/R2W=41/34 | F1=0.6571, Acc=0.6000, a=0.5, yes=0.670 !, dF1=+0.0019, W2R/R2W=52/51 |

## Optional Matrix

| vector | AMBER-existence | H-POPE | AMBER-relation | MME-position |
| --- | --- | --- | --- | --- |
| baseline | unavailable | unavailable | unavailable | unavailable |
| global | unavailable | unavailable | unavailable | unavailable |
| cat | unavailable | unavailable | unavailable | unavailable |
| attr | unavailable | unavailable | unavailable | unavailable |
| rel | unavailable | unavailable | unavailable | unavailable |

## Best Rows

| benchmark | vector | alpha | f1 | accuracy | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| POPE/category avg | baseline |  | 0.810387274151568 | 0.8172222222222222 | 0.45944444444444443 | 0 | 0 | 0 | 1800 |
| POPE/category avg | global | 0.05 | 0.806313292814703 | 0.8122222222222222 | 0.4633333333333333 | 145 | 154 | 299 | 1800 |
| POPE/category avg | cat | 0.25 | 0.8070313017562705 | 0.8138888888888889 | 0.45944444444444443 | 137 | 143 | 280 | 1800 |
| POPE/category avg | attr | 0.5 | 0.8125871347476264 | 0.8205555555555556 | 0.4538888888888889 | 150 | 144 | 294 | 1800 |
| POPE/category avg | rel | 0.5 | 0.8044986646318374 | 0.815 | 0.4438888888888889 | 142 | 146 | 288 | 1800 |
| AMBER-attribute | baseline |  | 0.6866666666666666 | 0.6866666666666666 | 0.5 | 0 | 0 | 0 | 300 |
| AMBER-attribute | global | 0.1 | 0.7119741100323624 | 0.7033333333333334 | 0.53 | 51 | 46 | 97 | 300 |
| AMBER-attribute | cat | 0.25 | 0.7350993377483444 | 0.7333333333333333 | 0.5066666666666667 | 53 | 39 | 92 | 300 |
| AMBER-attribute | attr | 0.5 | 0.7326732673267328 | 0.73 | 0.51 | 50 | 37 | 87 | 300 |
| AMBER-attribute | rel | 0.05 | 0.7147540983606557 | 0.71 | 0.5166666666666667 | 41 | 34 | 75 | 300 |
| GQA/clean-relation | baseline |  | 0.6552706552706553 | 0.5966666666666667 | 0.6733333333333333 | 0 | 0 | 0 | 300 |
| GQA/clean-relation | global | 0.1 | 0.6722689075630252 | 0.61 | 0.6933333333333334 | 49 | 45 | 94 | 300 |
| GQA/clean-relation | cat | 0.05 | 0.6892655367231638 | 0.6333333333333333 | 0.6833333333333333 | 54 | 43 | 97 | 300 |
| GQA/clean-relation | attr | 0.05 | 0.6265060240963854 | 0.5866666666666667 | 0.61 | 43 | 46 | 89 | 300 |
| GQA/clean-relation | rel | 0.5 | 0.657142857142857 | 0.6 | 0.67 | 52 | 51 | 103 | 300 |

## Automatic Interpretation

- POPE/category winner: `attr`.
- AMBER-attribute winner: `cat`.
- GQA/clean-relation winner: `cat`.
- `rel` on `GQA/clean-relation` is suspicious due to yes_rate=0.670.
- Off-diagonal or weak results dominate; do not proceed to router/DPO from these vectors.

## Changed Cases

- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/POPE_category_avg_global.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/AMBER-attribute_global.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/GQA_clean-relation_global.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/POPE_category_avg_cat.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/AMBER-attribute_cat.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/GQA_clean-relation_cat.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/POPE_category_avg_attr.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/AMBER-attribute_attr.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/GQA_clean-relation_attr.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/POPE_category_avg_rel.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/AMBER-attribute_rel.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/GQA_clean-relation_rel.jsonl`

## Notes

- `!` beside yes-rate marks a balanced yes/no benchmark suspicious zone (>0.65 or <0.35).
- POPE/category is averaged across completed POPE dataset/setting groups for the same vector/alpha.
