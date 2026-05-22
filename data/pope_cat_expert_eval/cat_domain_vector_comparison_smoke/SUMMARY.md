# Cat Domain Vector Comparison

- Run root: `data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/runs`
- Combined CSV: `data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/summary.csv`
- Sources: `coco_cat, gqa_cat, mixed_cat`

## Best Source Per Dataset/Setting

| Source | Dataset | Setting | Best Alpha | Baseline Acc | Best Acc | Delta Acc | Baseline F1 | Best F1 | Delta F1 | Baseline FP | Best FP | Delta FP | Baseline Yes Rate | Best Yes Rate | Delta Yes Rate | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coco_cat | GQA | adversarial | 0.1 | 77.33 | 79.33 | 2.00 | 77.78 | 80.50 | 2.73 | 37 | 40 | 3 | 52.00 | 56.00 | 4.00 | 300 |
| mixed_cat | GQA | popular | 2.0 | 78.00 | 81.33 | 3.33 | 78.00 | 81.21 | 3.21 | 33 | 27 | -6 | 50.00 | 49.33 | -0.67 | 300 |
| coco_cat | GQA | random | 0.5 | 85.33 | 89.33 | 4.00 | 84.17 | 88.89 | 4.72 | 11 | 10 | -1 | 42.67 | 46.00 | 3.33 | 300 |
| coco_cat | MSCOCO | adversarial | 0.3 | 79.33 | 81.67 | 2.33 | 78.62 | 81.36 | 2.74 | 26 | 25 | -1 | 46.67 | 48.33 | 1.67 | 300 |
| gqa_cat | MSCOCO | popular | 0.3 | 84.33 | 86.00 | 1.67 | 82.91 | 85.11 | 2.20 | 11 | 12 | 1 | 41.67 | 44.00 | 2.33 | 300 |
| gqa_cat | MSCOCO | random | 1.5 | 83.67 | 85.67 | 2.00 | 82.31 | 85.32 | 3.01 | 13 | 18 | 5 | 42.33 | 47.67 | 5.33 | 300 |

## Full Source Comparison

| Source | Dataset | Setting | Best Alpha | Baseline Acc | Best Acc | Delta Acc | Baseline F1 | Best F1 | Delta F1 | Baseline FP | Best FP | Delta FP | Baseline Yes Rate | Best Yes Rate | Delta Yes Rate | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coco_cat | GQA | adversarial | 0.1 | 77.33 | 79.33 | 2.00 | 77.78 | 80.50 | 2.73 | 37 | 40 | 3 | 52.00 | 56.00 | 4.00 | 300 |
| gqa_cat | GQA | adversarial | 0.1 | 77.33 | 79.33 | 2.00 | 77.78 | 80.50 | 2.73 | 37 | 40 | 3 | 52.00 | 56.00 | 4.00 | 300 |
| mixed_cat | GQA | adversarial | 0.1 | 77.33 | 79.00 | 1.67 | 77.78 | 80.00 | 2.22 | 37 | 39 | 2 | 52.00 | 55.00 | 3.00 | 300 |
| coco_cat | GQA | popular | 0.3 | 78.00 | 79.67 | 1.67 | 78.00 | 80.76 | 2.76 | 33 | 39 | 6 | 50.00 | 55.67 | 5.67 | 300 |
| gqa_cat | GQA | popular | 0.1 | 78.00 | 80.67 | 2.67 | 78.00 | 80.92 | 2.92 | 33 | 31 | -2 | 50.00 | 51.33 | 1.33 | 300 |
| mixed_cat | GQA | popular | 2.0 | 78.00 | 81.33 | 3.33 | 78.00 | 81.21 | 3.21 | 33 | 27 | -6 | 50.00 | 49.33 | -0.67 | 300 |
| coco_cat | GQA | random | 0.5 | 85.33 | 89.33 | 4.00 | 84.17 | 88.89 | 4.72 | 11 | 10 | -1 | 42.67 | 46.00 | 3.33 | 300 |
| gqa_cat | GQA | random | 0.5 | 85.33 | 89.33 | 4.00 | 84.17 | 88.89 | 4.72 | 11 | 10 | -1 | 42.67 | 46.00 | 3.33 | 300 |
| mixed_cat | GQA | random | 0.5 | 85.33 | 87.67 | 2.33 | 84.17 | 86.74 | 2.57 | 11 | 8 | -3 | 42.67 | 43.00 | 0.33 | 300 |
| coco_cat | MSCOCO | adversarial | 0.3 | 79.33 | 81.67 | 2.33 | 78.62 | 81.36 | 2.74 | 26 | 25 | -1 | 46.67 | 48.33 | 1.67 | 300 |
| gqa_cat | MSCOCO | adversarial | 0.3 | 79.33 | 81.67 | 2.33 | 78.62 | 81.36 | 2.74 | 26 | 25 | -1 | 46.67 | 48.33 | 1.67 | 300 |
| mixed_cat | MSCOCO | adversarial | 0.3 | 79.33 | 81.33 | 2.00 | 78.62 | 80.69 | 2.07 | 26 | 23 | -3 | 46.67 | 46.67 | 0.00 | 300 |
| coco_cat | MSCOCO | popular | 0.3 | 84.33 | 85.67 | 1.33 | 82.91 | 84.81 | 1.90 | 11 | 13 | 2 | 41.67 | 44.33 | 2.67 | 300 |
| gqa_cat | MSCOCO | popular | 0.3 | 84.33 | 86.00 | 1.67 | 82.91 | 85.11 | 2.20 | 11 | 12 | 1 | 41.67 | 44.00 | 2.33 | 300 |
| mixed_cat | MSCOCO | popular | 0.3 | 84.33 | 84.67 | 0.33 | 82.91 | 83.57 | 0.66 | 11 | 13 | 2 | 41.67 | 43.33 | 1.67 | 300 |
| coco_cat | MSCOCO | random | 0.3 | 83.67 | 86.00 | 2.33 | 82.31 | 85.11 | 2.80 | 13 | 12 | -1 | 42.33 | 44.00 | 1.67 | 300 |
| gqa_cat | MSCOCO | random | 1.5 | 83.67 | 85.67 | 2.00 | 82.31 | 85.32 | 3.01 | 13 | 18 | 5 | 42.33 | 47.67 | 5.33 | 300 |
| mixed_cat | MSCOCO | random | 0.3 | 83.67 | 84.67 | 1.00 | 82.31 | 83.57 | 1.26 | 13 | 13 | 0 | 42.33 | 43.33 | 1.00 | 300 |

## Reading Guide

- If `coco_cat` mostly improves MSCOCO and not GQA, it is domain-specific.
- If `gqa_cat` mostly improves GQA and weakens MSCOCO, it is domain-specific in the other direction.
- If `mixed_cat` is not always best but stays positive/stable on both, it is the safer shared category direction.
