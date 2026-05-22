# POPE CatExpert Evaluation Summary

- Summary CSV: `data/pope_cat_expert_eval/full_alpha_sweep/summary.csv`
- Runs summarized: 48

## Main Table

| Dataset | Setting | Method | Alpha | Accuracy | Precision | Recall | F1 Score | Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 81.30 | 80.83 | 82.07 | 81.44 | 50.77 |
| GQA | adversarial | CatExpert | 0.25 | 81.10 | 78.71 | 85.27 | 81.86 | 54.17 |
| GQA | adversarial | CatExpert | 0.5 | 79.13 | 75.44 | 86.40 | 80.55 | 57.27 |
| GQA | adversarial | CatExpert | 0.75 | 77.40 | 72.48 | 88.33 | 79.63 | 60.93 |
| GQA | adversarial | CatExpert | 1.0 | 75.17 | 69.51 | 89.67 | 78.31 | 64.50 |
| GQA | adversarial | CatExpert | 1.25 | 73.30 | 67.14 | 91.27 | 77.37 | 67.97 |
| GQA | adversarial | CatExpert | 1.5 | 70.13 | 63.58 | 94.27 | 75.94 | 74.13 |
| GQA | adversarial | CatExpert | 2.0 | 51.53 | 50.78 | 99.73 | 67.30 | 98.20 |
| GQA | popular | Regular |  | 84.17 | 85.66 | 82.07 | 83.83 | 47.90 |
| GQA | popular | CatExpert | 0.25 | 84.23 | 83.54 | 85.27 | 84.39 | 51.03 |
| GQA | popular | CatExpert | 0.5 | 81.97 | 79.36 | 86.40 | 82.73 | 54.43 |
| GQA | popular | CatExpert | 0.75 | 80.10 | 75.84 | 88.33 | 81.61 | 58.23 |
| GQA | popular | CatExpert | 1.0 | 77.33 | 71.93 | 89.67 | 79.82 | 62.33 |
| GQA | popular | CatExpert | 1.25 | 74.43 | 68.28 | 91.27 | 78.12 | 66.83 |
| GQA | popular | CatExpert | 1.5 | 69.63 | 63.15 | 94.27 | 75.64 | 74.63 |
| GQA | popular | CatExpert | 2.0 | 52.50 | 51.29 | 99.73 | 67.74 | 97.23 |
| GQA | random | Regular |  | 88.50 | 94.19 | 82.07 | 87.71 | 43.57 |
| GQA | random | CatExpert | 0.25 | 89.40 | 92.95 | 85.27 | 88.94 | 45.87 |
| GQA | random | CatExpert | 0.5 | 89.10 | 91.33 | 86.40 | 88.80 | 47.30 |
| GQA | random | CatExpert | 0.75 | 88.93 | 89.41 | 88.33 | 88.87 | 49.40 |
| GQA | random | CatExpert | 1.0 | 88.70 | 87.97 | 89.67 | 88.81 | 50.97 |
| GQA | random | CatExpert | 1.25 | 88.17 | 85.94 | 91.27 | 88.52 | 53.10 |
| GQA | random | CatExpert | 1.5 | 86.80 | 82.02 | 94.27 | 87.72 | 57.47 |
| GQA | random | CatExpert | 2.0 | 60.20 | 55.70 | 99.73 | 71.48 | 89.53 |
| MSCOCO | adversarial | Regular |  | 83.70 | 91.13 | 74.67 | 82.08 | 40.97 |
| MSCOCO | adversarial | CatExpert | 0.25 | 83.80 | 90.50 | 75.53 | 82.34 | 41.73 |
| MSCOCO | adversarial | CatExpert | 0.5 | 84.23 | 89.96 | 77.07 | 83.02 | 42.83 |
| MSCOCO | adversarial | CatExpert | 0.75 | 84.20 | 88.51 | 78.60 | 83.26 | 44.40 |
| MSCOCO | adversarial | CatExpert | 1.0 | 84.27 | 87.46 | 80.00 | 83.57 | 45.73 |
| MSCOCO | adversarial | CatExpert | 1.25 | 83.63 | 84.67 | 82.13 | 83.38 | 48.50 |
| MSCOCO | adversarial | CatExpert | 1.5 | 81.90 | 79.67 | 85.67 | 82.56 | 53.77 |
| MSCOCO | adversarial | CatExpert | 2.0 | 53.57 | 51.87 | 99.07 | 68.09 | 95.50 |
| MSCOCO | popular | Regular |  | 85.67 | 95.73 | 74.67 | 83.90 | 39.00 |
| MSCOCO | popular | CatExpert | 0.25 | 85.80 | 95.05 | 75.53 | 84.18 | 39.73 |
| MSCOCO | popular | CatExpert | 0.5 | 86.37 | 94.68 | 77.07 | 84.97 | 40.70 |
| MSCOCO | popular | CatExpert | 0.75 | 86.63 | 93.65 | 78.60 | 85.47 | 41.97 |
| MSCOCO | popular | CatExpert | 1.0 | 87.00 | 93.02 | 80.00 | 86.02 | 43.00 |
| MSCOCO | popular | CatExpert | 1.25 | 87.17 | 91.33 | 82.13 | 86.49 | 44.97 |
| MSCOCO | popular | CatExpert | 1.5 | 87.07 | 88.13 | 85.67 | 86.88 | 48.60 |
| MSCOCO | popular | CatExpert | 2.0 | 56.70 | 53.63 | 99.07 | 69.59 | 92.37 |
| MSCOCO | random | Regular |  | 86.50 | 97.82 | 74.67 | 84.69 | 38.17 |
| MSCOCO | random | CatExpert | 0.25 | 86.73 | 97.34 | 75.53 | 85.06 | 38.80 |
| MSCOCO | random | CatExpert | 0.5 | 87.47 | 97.31 | 77.07 | 86.01 | 39.60 |
| MSCOCO | random | CatExpert | 0.75 | 88.07 | 96.96 | 78.60 | 86.82 | 40.53 |
| MSCOCO | random | CatExpert | 1.0 | 88.53 | 96.46 | 80.00 | 87.46 | 41.47 |
| MSCOCO | random | CatExpert | 1.25 | 89.07 | 95.36 | 82.13 | 88.25 | 43.07 |
| MSCOCO | random | CatExpert | 1.5 | 89.77 | 93.32 | 85.67 | 89.33 | 45.90 |
| MSCOCO | random | CatExpert | 2.0 | 64.33 | 58.46 | 99.07 | 73.53 | 84.73 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 1231 | 1208 | 292 | 269 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.25 | 1279 | 1154 | 346 | 221 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.5 | 1296 | 1078 | 422 | 204 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.75 | 1325 | 997 | 503 | 175 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.0 | 1345 | 910 | 590 | 155 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.25 | 1369 | 830 | 670 | 131 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.5 | 1414 | 690 | 810 | 86 | 0 | 3000 |
| GQA | adversarial | CatExpert | 2.0 | 1496 | 50 | 1450 | 4 | 0 | 3000 |
| GQA | popular | Regular |  | 1231 | 1294 | 206 | 269 | 0 | 3000 |
| GQA | popular | CatExpert | 0.25 | 1279 | 1248 | 252 | 221 | 0 | 3000 |
| GQA | popular | CatExpert | 0.5 | 1296 | 1163 | 337 | 204 | 0 | 3000 |
| GQA | popular | CatExpert | 0.75 | 1325 | 1078 | 422 | 175 | 0 | 3000 |
| GQA | popular | CatExpert | 1.0 | 1345 | 975 | 525 | 155 | 0 | 3000 |
| GQA | popular | CatExpert | 1.25 | 1369 | 864 | 636 | 131 | 0 | 3000 |
| GQA | popular | CatExpert | 1.5 | 1414 | 675 | 825 | 86 | 0 | 3000 |
| GQA | popular | CatExpert | 2.0 | 1496 | 79 | 1421 | 4 | 0 | 3000 |
| GQA | random | Regular |  | 1231 | 1424 | 76 | 269 | 0 | 3000 |
| GQA | random | CatExpert | 0.25 | 1279 | 1403 | 97 | 221 | 0 | 3000 |
| GQA | random | CatExpert | 0.5 | 1296 | 1377 | 123 | 204 | 0 | 3000 |
| GQA | random | CatExpert | 0.75 | 1325 | 1343 | 157 | 175 | 0 | 3000 |
| GQA | random | CatExpert | 1.0 | 1345 | 1316 | 184 | 155 | 0 | 3000 |
| GQA | random | CatExpert | 1.25 | 1369 | 1276 | 224 | 131 | 0 | 3000 |
| GQA | random | CatExpert | 1.5 | 1414 | 1190 | 310 | 86 | 0 | 3000 |
| GQA | random | CatExpert | 2.0 | 1496 | 310 | 1190 | 4 | 0 | 3000 |
| MSCOCO | adversarial | Regular |  | 1120 | 1391 | 109 | 380 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.25 | 1133 | 1381 | 119 | 367 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.5 | 1156 | 1371 | 129 | 344 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.75 | 1179 | 1347 | 153 | 321 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.0 | 1200 | 1328 | 172 | 300 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.25 | 1232 | 1277 | 223 | 268 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.5 | 1285 | 1172 | 328 | 215 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 2.0 | 1486 | 121 | 1379 | 14 | 0 | 3000 |
| MSCOCO | popular | Regular |  | 1120 | 1450 | 50 | 380 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.25 | 1133 | 1441 | 59 | 367 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.5 | 1156 | 1435 | 65 | 344 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.75 | 1179 | 1420 | 80 | 321 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.0 | 1200 | 1410 | 90 | 300 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.25 | 1232 | 1383 | 117 | 268 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.5 | 1285 | 1327 | 173 | 215 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 2.0 | 1486 | 215 | 1285 | 14 | 0 | 3000 |
| MSCOCO | random | Regular |  | 1120 | 1475 | 25 | 380 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.25 | 1133 | 1469 | 31 | 367 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.5 | 1156 | 1468 | 32 | 344 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.75 | 1179 | 1463 | 37 | 321 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.0 | 1200 | 1456 | 44 | 300 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.25 | 1232 | 1440 | 60 | 268 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.5 | 1285 | 1408 | 92 | 215 | 0 | 3000 |
| MSCOCO | random | CatExpert | 2.0 | 1486 | 444 | 1056 | 14 | 0 | 3000 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is the object-hallucination count: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
