# Cat Domain Vector Comparison

- Run root: `data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/runs`
- Combined CSV: `data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/summary.csv`
- Sources: `coco_cat, gqa_cat, mixed_cat`

## Best Source Per Dataset/Setting

| Source | Dataset | Setting | Best Alpha | Baseline Acc | Best Acc | Delta Acc | Baseline F1 | Best F1 | Delta F1 | Baseline FP | Best FP | Delta FP | Baseline Yes Rate | Best Yes Rate | Delta Yes Rate | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_cat | GQA | adversarial | 0.01 | 76.07 | 76.83 | 0.77 | 76.96 | 77.77 | 0.82 | 417 | 411 | -6 | 53.87 | 54.23 | 0.37 | 3000 |
| gqa_cat | GQA | popular | 0.01 | 79.33 | 78.93 | -0.40 | 79.61 | 79.41 | -0.19 | 330 | 351 | 21 | 51.33 | 52.33 | 1.00 | 3000 |
| coco_cat | GQA | random | 1.0 | 84.73 | 85.50 | 0.77 | 84.09 | 85.71 | 1.63 | 168 | 240 | 72 | 45.93 | 51.50 | 5.57 | 3000 |
| coco_cat | MSCOCO | adversarial | 0.5 | 79.70 | 81.80 | 2.10 | 78.40 | 81.11 | 2.71 | 214 | 218 | 4 | 43.97 | 46.33 | 2.37 | 3000 |
| coco_cat | MSCOCO | popular | 1.0 | 82.57 | 85.13 | 2.57 | 80.86 | 84.27 | 3.41 | 128 | 141 | 13 | 41.10 | 44.53 | 3.43 | 3000 |
| coco_cat | MSCOCO | random | 1.0 | 83.73 | 86.33 | 2.60 | 81.91 | 85.38 | 3.47 | 93 | 107 | 14 | 39.93 | 43.47 | 3.53 | 3000 |

## Full Source Comparison

| Source | Dataset | Setting | Best Alpha | Baseline Acc | Best Acc | Delta Acc | Baseline F1 | Best F1 | Delta F1 | Baseline FP | Best FP | Delta FP | Baseline Yes Rate | Best Yes Rate | Delta Yes Rate | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coco_cat | GQA | adversarial | 0.01 | 76.07 | 76.80 | 0.73 | 76.96 | 77.74 | 0.78 | 417 | 411 | -6 | 53.87 | 54.20 | 0.33 | 3000 |
| gqa_cat | GQA | adversarial | 0.01 | 76.07 | 76.83 | 0.77 | 76.96 | 77.77 | 0.82 | 417 | 411 | -6 | 53.87 | 54.23 | 0.37 | 3000 |
| mixed_cat | GQA | adversarial | 0.01 | 76.07 | 76.80 | 0.73 | 76.96 | 77.74 | 0.78 | 417 | 411 | -6 | 53.87 | 54.20 | 0.33 | 3000 |
| coco_cat | GQA | popular | 0.01 | 79.33 | 78.90 | -0.43 | 79.61 | 79.37 | -0.23 | 330 | 351 | 21 | 51.33 | 52.30 | 0.97 | 3000 |
| gqa_cat | GQA | popular | 0.01 | 79.33 | 78.93 | -0.40 | 79.61 | 79.41 | -0.19 | 330 | 351 | 21 | 51.33 | 52.33 | 1.00 | 3000 |
| mixed_cat | GQA | popular | 0.01 | 79.33 | 78.90 | -0.43 | 79.61 | 79.37 | -0.23 | 330 | 351 | 21 | 51.33 | 52.30 | 0.97 | 3000 |
| coco_cat | GQA | random | 1.0 | 84.73 | 85.50 | 0.77 | 84.09 | 85.71 | 1.63 | 168 | 240 | 72 | 45.93 | 51.50 | 5.57 | 3000 |
| gqa_cat | GQA | random | 0.75 | 84.73 | 85.33 | 0.60 | 84.09 | 85.45 | 1.36 | 168 | 232 | 64 | 45.93 | 50.80 | 4.87 | 3000 |
| mixed_cat | GQA | random | 1.0 | 84.73 | 85.17 | 0.43 | 84.09 | 85.41 | 1.33 | 168 | 248 | 80 | 45.93 | 51.70 | 5.77 | 3000 |
| coco_cat | MSCOCO | adversarial | 0.5 | 79.70 | 81.80 | 2.10 | 78.40 | 81.11 | 2.71 | 214 | 218 | 4 | 43.97 | 46.33 | 2.37 | 3000 |
| gqa_cat | MSCOCO | adversarial | 0.5 | 79.70 | 81.73 | 2.03 | 78.40 | 80.99 | 2.59 | 214 | 215 | 1 | 43.97 | 46.07 | 2.10 | 3000 |
| mixed_cat | MSCOCO | adversarial | 0.5 | 79.70 | 81.63 | 1.93 | 78.40 | 80.89 | 2.49 | 214 | 217 | 3 | 43.97 | 46.10 | 2.13 | 3000 |
| coco_cat | MSCOCO | popular | 1.0 | 82.57 | 85.13 | 2.57 | 80.86 | 84.27 | 3.41 | 128 | 141 | 13 | 41.10 | 44.53 | 3.43 | 3000 |
| gqa_cat | MSCOCO | popular | 1.0 | 82.57 | 84.63 | 2.07 | 80.86 | 83.83 | 2.97 | 128 | 156 | 28 | 41.10 | 45.03 | 3.93 | 3000 |
| mixed_cat | MSCOCO | popular | 1.0 | 82.57 | 84.70 | 2.13 | 80.86 | 83.79 | 2.92 | 128 | 145 | 17 | 41.10 | 44.37 | 3.27 | 3000 |
| coco_cat | MSCOCO | random | 1.0 | 83.73 | 86.33 | 2.60 | 81.91 | 85.38 | 3.47 | 93 | 107 | 14 | 39.93 | 43.47 | 3.53 | 3000 |
| gqa_cat | MSCOCO | random | 1.0 | 83.73 | 85.87 | 2.13 | 81.91 | 84.93 | 3.02 | 93 | 119 | 26 | 39.93 | 43.80 | 3.87 | 3000 |
| mixed_cat | MSCOCO | random | 1.0 | 83.73 | 86.00 | 2.27 | 81.91 | 85.01 | 3.10 | 93 | 111 | 18 | 39.93 | 43.40 | 3.47 | 3000 |

## Reading Guide

- If `coco_cat` mostly improves MSCOCO and not GQA, it is domain-specific.
- If `gqa_cat` mostly improves GQA and weakens MSCOCO, it is domain-specific in the other direction.
- If `mixed_cat` is not always best but stays positive/stable on both, it is the safer shared category direction.
