# POPE CatExpert Evaluation Summary

- Summary CSV: `data/pope_cat_expert_eval/smoke/summary.csv`
- Runs summarized: 12

## Main Table

| Dataset | Setting | Method | Alpha | Accuracy | Precision | Recall | F1 Score | Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 90.00 | 100.00 | 80.00 | 88.89 | 40.00 |
| GQA | adversarial | CatExpert | 1.0 | 80.00 | 75.00 | 90.00 | 81.82 | 60.00 |
| GQA | popular | Regular |  | 90.00 | 100.00 | 81.82 | 90.00 | 45.00 |
| GQA | popular | CatExpert | 1.0 | 80.00 | 76.92 | 90.91 | 83.33 | 65.00 |
| GQA | random | Regular |  | 90.00 | 100.00 | 81.82 | 90.00 | 45.00 |
| GQA | random | CatExpert | 1.0 | 95.00 | 100.00 | 90.91 | 95.24 | 50.00 |
| MSCOCO | adversarial | Regular |  | 90.00 | 90.00 | 90.00 | 90.00 | 50.00 |
| MSCOCO | adversarial | CatExpert | 1.0 | 90.00 | 90.00 | 90.00 | 90.00 | 50.00 |
| MSCOCO | popular | Regular |  | 90.00 | 90.00 | 90.00 | 90.00 | 50.00 |
| MSCOCO | popular | CatExpert | 1.0 | 90.00 | 90.00 | 90.00 | 90.00 | 50.00 |
| MSCOCO | random | Regular |  | 95.00 | 100.00 | 90.00 | 94.74 | 45.00 |
| MSCOCO | random | CatExpert | 1.0 | 95.00 | 100.00 | 90.00 | 94.74 | 45.00 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 8 | 10 | 0 | 2 | 0 | 20 |
| GQA | adversarial | CatExpert | 1.0 | 9 | 7 | 3 | 1 | 0 | 20 |
| GQA | popular | Regular |  | 9 | 9 | 0 | 2 | 0 | 20 |
| GQA | popular | CatExpert | 1.0 | 10 | 6 | 3 | 1 | 0 | 20 |
| GQA | random | Regular |  | 9 | 9 | 0 | 2 | 0 | 20 |
| GQA | random | CatExpert | 1.0 | 10 | 9 | 0 | 1 | 0 | 20 |
| MSCOCO | adversarial | Regular |  | 9 | 9 | 1 | 1 | 0 | 20 |
| MSCOCO | adversarial | CatExpert | 1.0 | 9 | 9 | 1 | 1 | 0 | 20 |
| MSCOCO | popular | Regular |  | 9 | 9 | 1 | 1 | 0 | 20 |
| MSCOCO | popular | CatExpert | 1.0 | 9 | 9 | 1 | 1 | 0 | 20 |
| MSCOCO | random | Regular |  | 9 | 10 | 0 | 1 | 0 | 20 |
| MSCOCO | random | CatExpert | 1.0 | 9 | 10 | 0 | 1 | 0 | 20 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is the object-hallucination count: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
