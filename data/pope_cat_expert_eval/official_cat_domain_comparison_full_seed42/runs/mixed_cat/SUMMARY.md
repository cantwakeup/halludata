# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/runs/mixed_cat/summary.csv`
- Runs summarized: 108

## Run Config

- Config files found: `6`
- Runner: `official_llava`
- Model path: `/home/huiwei/sy/models/llava-v1.5-7b-official-clean`
- LLaVA repo: `/home/huiwei/sy/LLaVA-official-clean`
- Conv mode: `llava_v1`
- Prompt template: `{question} Please answer this question with one word.`
- Cat vector source: `mixed_cat`
- Decode: `{"do_sample": true, "temperature": 1.0, "top_p": 1.0, "num_beams": 1, "max_new_tokens": 1024}`
- Steering: `{"layers": "5-25", "topk": 64, "head_select": "norm", "prefill": true, "decode": true, "apply_to": "last_token", "prefill_apply_to": "last_token", "decode_apply_to": "last_token", "enabled_experts": ["cat"]}`

## Best CatExpert By F1

| Dataset | Setting | Best Alpha | Baseline Acc | Best Acc | Delta Acc | Baseline F1 | Best F1 | Delta F1 | Baseline FP | Best FP | FP Delta | Baseline Yes Rate | Best Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | 0.01 | 76.07 | 76.80 | 0.73 | 76.96 | 77.74 | 0.78 | 417 | 411 | -6.00 | 53.87 | 54.20 |
| GQA | popular | 0.01 | 79.33 | 78.90 | -0.43 | 79.61 | 79.37 | -0.23 | 330 | 351 | 21.00 | 51.33 | 52.30 |
| GQA | random | 1.0 | 84.73 | 85.17 | 0.43 | 84.09 | 85.41 | 1.33 | 168 | 248 | 80.00 | 45.93 | 51.70 |
| MSCOCO | adversarial | 0.5 | 79.70 | 81.63 | 1.93 | 78.40 | 80.89 | 2.49 | 214 | 217 | 3.00 | 43.97 | 46.10 |
| MSCOCO | popular | 1.0 | 82.57 | 84.70 | 2.13 | 80.86 | 83.79 | 2.92 | 128 | 145 | 17.00 | 41.10 | 44.37 |
| MSCOCO | random | 1.0 | 83.73 | 86.00 | 2.27 | 81.91 | 85.01 | 3.10 | 93 | 111 | 18.00 | 39.93 | 43.40 |

## Alpha 0 Check

| Dataset | Setting | Alpha0 Acc | Regular Acc | Acc Diff | Alpha0 F1 | Regular F1 | F1 Diff | Alpha0 Invalid | Regular Invalid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | 75.17 | 76.07 | -0.90 | 75.99 | 76.96 | -0.97 | 0 | 0 |
| GQA | popular | 78.57 | 79.33 | -0.77 | 78.62 | 79.61 | -0.99 | 0 | 0 |
| GQA | random | 84.17 | 84.73 | -0.57 | 83.27 | 84.09 | -0.82 | 0 | 0 |
| MSCOCO | adversarial | 80.23 | 79.70 | 0.53 | 78.83 | 78.40 | 0.43 | 0 | 0 |
| MSCOCO | popular | 82.60 | 82.57 | 0.03 | 80.88 | 80.86 | 0.02 | 0 | 0 |
| MSCOCO | random | 83.97 | 83.73 | 0.23 | 82.11 | 81.91 | 0.20 | 0 | 0 |

## Official Regular vs Old HF Regular

| Dataset | Setting | Official N | HF N | Same N | Official Acc | HF Acc | Acc Diff | Official F1 | HF F1 | F1 Diff | Official FP | HF FP | Official Yes Rate | HF Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | 3000 | 3000 | True | 76.07 | 81.30 | -5.23 | 76.96 | 81.44 | -4.48 | 417 | 292 | 53.87 | 50.77 |
| GQA | popular | 3000 | 3000 | True | 79.33 | 84.17 | -4.83 | 79.61 | 83.83 | -4.22 | 330 | 206 | 51.33 | 47.90 |
| GQA | random | 3000 | 3000 | True | 84.73 | 88.50 | -3.77 | 84.09 | 87.71 | -3.62 | 168 | 76 | 45.93 | 43.57 |
| MSCOCO | adversarial | 3000 | 3000 | True | 79.70 | 83.70 | -4.00 | 78.40 | 82.08 | -3.68 | 214 | 109 | 43.97 | 40.97 |
| MSCOCO | popular | 3000 | 3000 | True | 82.57 | 85.67 | -3.10 | 80.86 | 83.90 | -3.03 | 128 | 50 | 41.10 | 39.00 |
| MSCOCO | random | 3000 | 3000 | True | 83.73 | 86.50 | -2.77 | 81.91 | 84.69 | -2.78 | 93 | 25 | 39.93 | 38.17 |

## Main Table

| Dataset | Setting | Method | Alpha | N | Accuracy | Precision | Recall | F1 Score | Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 3000 | 76.07 | 74.20 | 79.93 | 76.96 | 53.87 |
| GQA | adversarial | CatExpert | 0.0 | 3000 | 75.17 | 73.55 | 78.60 | 75.99 | 53.43 |
| GQA | adversarial | CatExpert | 0.01 | 3000 | 76.80 | 74.72 | 81.00 | 77.74 | 54.20 |
| GQA | adversarial | CatExpert | 0.025 | 3000 | 74.63 | 72.06 | 80.47 | 76.03 | 55.83 |
| GQA | adversarial | CatExpert | 0.05 | 3000 | 76.03 | 73.97 | 80.33 | 77.02 | 54.30 |
| GQA | adversarial | CatExpert | 0.075 | 3000 | 75.10 | 72.40 | 81.13 | 76.52 | 56.03 |
| GQA | adversarial | CatExpert | 0.1 | 3000 | 74.37 | 72.11 | 79.47 | 75.61 | 55.10 |
| GQA | adversarial | CatExpert | 0.15 | 3000 | 75.07 | 72.25 | 81.40 | 76.55 | 56.33 |
| GQA | adversarial | CatExpert | 0.2 | 3000 | 75.07 | 72.51 | 80.73 | 76.40 | 55.67 |
| GQA | adversarial | CatExpert | 0.25 | 3000 | 75.10 | 71.78 | 82.73 | 76.87 | 57.63 |
| GQA | adversarial | CatExpert | 0.3 | 3000 | 75.20 | 72.11 | 82.20 | 76.82 | 57.00 |
| GQA | adversarial | CatExpert | 0.4 | 3000 | 74.70 | 71.21 | 82.93 | 76.62 | 58.23 |
| GQA | adversarial | CatExpert | 0.5 | 3000 | 75.20 | 71.28 | 84.40 | 77.29 | 59.20 |
| GQA | adversarial | CatExpert | 0.75 | 3000 | 73.97 | 69.44 | 85.60 | 76.68 | 61.63 |
| GQA | adversarial | CatExpert | 1.0 | 3000 | 72.10 | 67.20 | 86.33 | 75.58 | 64.23 |
| GQA | adversarial | CatExpert | 1.25 | 3000 | 71.43 | 66.02 | 88.33 | 75.56 | 66.90 |
| GQA | adversarial | CatExpert | 1.5 | 3000 | 67.97 | 62.62 | 89.13 | 73.56 | 71.17 |
| GQA | adversarial | CatExpert | 2.0 | 3000 | 56.40 | 53.66 | 93.93 | 68.30 | 87.53 |
| GQA | popular | Regular |  | 3000 | 79.33 | 78.57 | 80.67 | 79.61 | 51.33 |
| GQA | popular | CatExpert | 0.0 | 3000 | 78.57 | 78.43 | 78.80 | 78.62 | 50.23 |
| GQA | popular | CatExpert | 0.01 | 3000 | 78.90 | 77.63 | 81.20 | 79.37 | 52.30 |
| GQA | popular | CatExpert | 0.025 | 3000 | 78.17 | 76.76 | 80.80 | 78.73 | 52.63 |
| GQA | popular | CatExpert | 0.05 | 3000 | 77.87 | 77.50 | 78.53 | 78.01 | 50.67 |
| GQA | popular | CatExpert | 0.075 | 3000 | 77.57 | 76.35 | 79.87 | 78.07 | 52.30 |
| GQA | popular | CatExpert | 0.1 | 3000 | 78.57 | 77.28 | 80.93 | 79.06 | 52.37 |
| GQA | popular | CatExpert | 0.15 | 3000 | 78.43 | 77.46 | 80.20 | 78.81 | 51.77 |
| GQA | popular | CatExpert | 0.2 | 3000 | 77.33 | 75.59 | 80.73 | 78.08 | 53.40 |
| GQA | popular | CatExpert | 0.25 | 3000 | 78.03 | 76.20 | 81.53 | 78.78 | 53.50 |
| GQA | popular | CatExpert | 0.3 | 3000 | 78.07 | 75.96 | 82.13 | 78.92 | 54.07 |
| GQA | popular | CatExpert | 0.4 | 3000 | 77.40 | 74.44 | 83.47 | 78.69 | 56.07 |
| GQA | popular | CatExpert | 0.5 | 3000 | 76.90 | 73.83 | 83.33 | 78.30 | 56.43 |
| GQA | popular | CatExpert | 0.75 | 3000 | 75.33 | 70.95 | 85.80 | 77.67 | 60.47 |
| GQA | popular | CatExpert | 1.0 | 3000 | 75.03 | 70.24 | 86.87 | 77.68 | 61.83 |
| GQA | popular | CatExpert | 1.25 | 3000 | 73.30 | 67.97 | 88.13 | 76.75 | 64.83 |
| GQA | popular | CatExpert | 1.5 | 3000 | 67.93 | 62.51 | 89.60 | 73.64 | 71.67 |
| GQA | popular | CatExpert | 2.0 | 3000 | 57.53 | 54.36 | 93.93 | 68.87 | 86.40 |
| GQA | random | Regular |  | 3000 | 84.73 | 87.81 | 80.67 | 84.09 | 45.93 |
| GQA | random | CatExpert | 0.0 | 3000 | 84.17 | 88.27 | 78.80 | 83.27 | 44.63 |
| GQA | random | CatExpert | 0.01 | 3000 | 85.00 | 87.88 | 81.20 | 84.41 | 46.20 |
| GQA | random | CatExpert | 0.025 | 3000 | 84.23 | 86.76 | 80.80 | 83.67 | 46.57 |
| GQA | random | CatExpert | 0.05 | 3000 | 83.40 | 87.00 | 78.53 | 82.55 | 45.13 |
| GQA | random | CatExpert | 0.075 | 3000 | 83.37 | 85.88 | 79.87 | 82.76 | 46.50 |
| GQA | random | CatExpert | 0.1 | 3000 | 84.63 | 87.40 | 80.93 | 84.04 | 46.30 |
| GQA | random | CatExpert | 0.15 | 3000 | 84.17 | 87.11 | 80.20 | 83.51 | 46.03 |
| GQA | random | CatExpert | 0.2 | 3000 | 84.10 | 86.56 | 80.73 | 83.55 | 46.63 |
| GQA | random | CatExpert | 0.25 | 3000 | 84.53 | 86.74 | 81.53 | 84.05 | 47.00 |
| GQA | random | CatExpert | 0.3 | 3000 | 85.47 | 88.00 | 82.13 | 84.97 | 46.67 |
| GQA | random | CatExpert | 0.4 | 3000 | 85.47 | 86.94 | 83.47 | 85.17 | 48.00 |
| GQA | random | CatExpert | 0.5 | 3000 | 84.53 | 85.38 | 83.33 | 84.35 | 48.80 |
| GQA | random | CatExpert | 0.75 | 3000 | 85.20 | 84.78 | 85.80 | 85.29 | 50.60 |
| GQA | random | CatExpert | 1.0 | 3000 | 85.17 | 84.01 | 86.87 | 85.41 | 51.70 |
| GQA | random | CatExpert | 1.25 | 3000 | 83.77 | 81.05 | 88.13 | 84.45 | 54.37 |
| GQA | random | CatExpert | 1.5 | 3000 | 80.93 | 76.36 | 89.60 | 82.45 | 58.67 |
| GQA | random | CatExpert | 2.0 | 3000 | 64.90 | 59.37 | 94.40 | 72.90 | 79.50 |
| MSCOCO | adversarial | Regular |  | 3000 | 79.70 | 83.78 | 73.67 | 78.40 | 43.97 |
| MSCOCO | adversarial | CatExpert | 0.0 | 3000 | 80.23 | 84.86 | 73.60 | 78.83 | 43.37 |
| MSCOCO | adversarial | CatExpert | 0.01 | 3000 | 80.43 | 84.04 | 75.13 | 79.34 | 44.70 |
| MSCOCO | adversarial | CatExpert | 0.025 | 3000 | 78.73 | 81.97 | 73.67 | 77.60 | 44.93 |
| MSCOCO | adversarial | CatExpert | 0.05 | 3000 | 79.63 | 83.85 | 73.40 | 78.28 | 43.77 |
| MSCOCO | adversarial | CatExpert | 0.075 | 3000 | 80.80 | 84.32 | 75.67 | 79.76 | 44.87 |
| MSCOCO | adversarial | CatExpert | 0.1 | 3000 | 79.20 | 82.74 | 73.80 | 78.01 | 44.60 |
| MSCOCO | adversarial | CatExpert | 0.15 | 3000 | 79.00 | 83.62 | 72.13 | 77.45 | 43.13 |
| MSCOCO | adversarial | CatExpert | 0.2 | 3000 | 79.93 | 83.26 | 74.93 | 78.88 | 45.00 |
| MSCOCO | adversarial | CatExpert | 0.25 | 3000 | 80.30 | 84.51 | 74.20 | 79.02 | 43.90 |
| MSCOCO | adversarial | CatExpert | 0.3 | 3000 | 80.53 | 84.38 | 74.93 | 79.38 | 44.40 |
| MSCOCO | adversarial | CatExpert | 0.4 | 3000 | 80.60 | 83.85 | 75.80 | 79.62 | 45.20 |
| MSCOCO | adversarial | CatExpert | 0.5 | 3000 | 81.63 | 84.31 | 77.73 | 80.89 | 46.10 |
| MSCOCO | adversarial | CatExpert | 0.75 | 3000 | 81.03 | 83.27 | 77.67 | 80.37 | 46.63 |
| MSCOCO | adversarial | CatExpert | 1.0 | 3000 | 80.17 | 80.59 | 79.47 | 80.03 | 49.30 |
| MSCOCO | adversarial | CatExpert | 1.25 | 3000 | 80.07 | 79.79 | 80.53 | 80.16 | 50.47 |
| MSCOCO | adversarial | CatExpert | 1.5 | 3000 | 78.00 | 75.18 | 83.60 | 79.17 | 55.60 |
| MSCOCO | adversarial | CatExpert | 2.0 | 3000 | 62.70 | 57.96 | 92.47 | 71.26 | 79.77 |
| MSCOCO | popular | Regular |  | 3000 | 82.57 | 89.62 | 73.67 | 80.86 | 41.10 |
| MSCOCO | popular | CatExpert | 0.0 | 3000 | 82.60 | 89.76 | 73.60 | 80.88 | 41.00 |
| MSCOCO | popular | CatExpert | 0.01 | 3000 | 83.40 | 90.02 | 75.13 | 81.90 | 41.73 |
| MSCOCO | popular | CatExpert | 0.025 | 3000 | 81.93 | 88.26 | 73.67 | 80.31 | 41.73 |
| MSCOCO | popular | CatExpert | 0.05 | 3000 | 82.90 | 90.62 | 73.40 | 81.10 | 40.50 |
| MSCOCO | popular | CatExpert | 0.075 | 3000 | 83.67 | 90.08 | 75.67 | 82.25 | 42.00 |
| MSCOCO | popular | CatExpert | 0.1 | 3000 | 82.43 | 89.20 | 73.80 | 80.77 | 41.37 |
| MSCOCO | popular | CatExpert | 0.15 | 3000 | 81.47 | 88.69 | 72.13 | 79.56 | 40.67 |
| MSCOCO | popular | CatExpert | 0.2 | 3000 | 83.07 | 89.49 | 74.93 | 81.57 | 41.87 |
| MSCOCO | popular | CatExpert | 0.25 | 3000 | 82.93 | 89.90 | 74.20 | 81.30 | 41.27 |
| MSCOCO | popular | CatExpert | 0.3 | 3000 | 83.70 | 90.86 | 74.93 | 82.13 | 41.23 |
| MSCOCO | popular | CatExpert | 0.4 | 3000 | 83.13 | 88.77 | 75.87 | 81.81 | 42.73 |
| MSCOCO | popular | CatExpert | 0.5 | 3000 | 83.53 | 88.99 | 76.53 | 82.29 | 43.00 |
| MSCOCO | popular | CatExpert | 0.75 | 3000 | 84.17 | 88.56 | 78.47 | 83.21 | 44.30 |
| MSCOCO | popular | CatExpert | 1.0 | 3000 | 84.70 | 89.11 | 79.07 | 83.79 | 44.37 |
| MSCOCO | popular | CatExpert | 1.25 | 3000 | 83.53 | 85.62 | 80.60 | 83.04 | 47.07 |
| MSCOCO | popular | CatExpert | 1.5 | 3000 | 81.77 | 81.25 | 82.60 | 81.92 | 50.83 |
| MSCOCO | popular | CatExpert | 2.0 | 3000 | 66.17 | 60.56 | 92.73 | 73.27 | 76.57 |
| MSCOCO | random | Regular |  | 3000 | 83.73 | 92.24 | 73.67 | 81.91 | 39.93 |
| MSCOCO | random | CatExpert | 0.0 | 3000 | 83.97 | 92.85 | 73.60 | 82.11 | 39.63 |
| MSCOCO | random | CatExpert | 0.01 | 3000 | 84.83 | 93.22 | 75.13 | 83.20 | 40.30 |
| MSCOCO | random | CatExpert | 0.025 | 3000 | 83.57 | 91.85 | 73.67 | 81.76 | 40.10 |
| MSCOCO | random | CatExpert | 0.05 | 3000 | 83.63 | 92.29 | 73.40 | 81.77 | 39.77 |
| MSCOCO | random | CatExpert | 0.075 | 3000 | 84.63 | 92.20 | 75.67 | 83.12 | 41.03 |
| MSCOCO | random | CatExpert | 0.1 | 3000 | 83.53 | 91.64 | 73.80 | 81.76 | 40.27 |
| MSCOCO | random | CatExpert | 0.15 | 3000 | 83.20 | 92.64 | 72.13 | 81.11 | 38.93 |
| MSCOCO | random | CatExpert | 0.2 | 3000 | 84.53 | 92.74 | 74.93 | 82.89 | 40.40 |
| MSCOCO | random | CatExpert | 0.25 | 3000 | 84.30 | 92.98 | 74.20 | 82.54 | 39.90 |
| MSCOCO | random | CatExpert | 0.3 | 3000 | 84.63 | 92.97 | 74.93 | 82.98 | 40.30 |
| MSCOCO | random | CatExpert | 0.4 | 3000 | 84.73 | 92.29 | 75.80 | 83.24 | 41.07 |
| MSCOCO | random | CatExpert | 0.5 | 3000 | 85.77 | 92.61 | 77.73 | 84.52 | 41.97 |
| MSCOCO | random | CatExpert | 0.75 | 3000 | 85.83 | 92.03 | 78.47 | 84.71 | 42.63 |
| MSCOCO | random | CatExpert | 1.0 | 3000 | 86.00 | 91.47 | 79.40 | 85.01 | 43.40 |
| MSCOCO | random | CatExpert | 1.25 | 3000 | 85.57 | 89.55 | 80.53 | 84.80 | 44.97 |
| MSCOCO | random | CatExpert | 1.5 | 3000 | 83.83 | 83.99 | 83.60 | 83.80 | 49.77 |
| MSCOCO | random | CatExpert | 2.0 | 3000 | 67.77 | 61.92 | 92.27 | 74.11 | 74.50 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 1199 | 1083 | 417 | 301 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.0 | 1179 | 1076 | 424 | 321 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.01 | 1215 | 1089 | 411 | 285 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.025 | 1207 | 1032 | 468 | 293 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.05 | 1205 | 1076 | 424 | 295 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.075 | 1217 | 1036 | 464 | 283 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.1 | 1192 | 1039 | 461 | 308 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.15 | 1221 | 1031 | 469 | 279 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.2 | 1211 | 1041 | 459 | 289 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.25 | 1241 | 1012 | 488 | 259 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.3 | 1233 | 1023 | 477 | 267 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.4 | 1244 | 997 | 503 | 256 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.5 | 1266 | 990 | 510 | 234 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.75 | 1284 | 935 | 565 | 216 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.0 | 1295 | 868 | 632 | 205 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.25 | 1325 | 818 | 682 | 175 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.5 | 1337 | 702 | 798 | 163 | 0 | 3000 |
| GQA | adversarial | CatExpert | 2.0 | 1409 | 283 | 1217 | 91 | 0 | 3000 |
| GQA | popular | Regular |  | 1210 | 1170 | 330 | 290 | 0 | 3000 |
| GQA | popular | CatExpert | 0.0 | 1182 | 1175 | 325 | 318 | 0 | 3000 |
| GQA | popular | CatExpert | 0.01 | 1218 | 1149 | 351 | 282 | 0 | 3000 |
| GQA | popular | CatExpert | 0.025 | 1212 | 1133 | 367 | 288 | 0 | 3000 |
| GQA | popular | CatExpert | 0.05 | 1178 | 1158 | 342 | 322 | 0 | 3000 |
| GQA | popular | CatExpert | 0.075 | 1198 | 1129 | 371 | 302 | 0 | 3000 |
| GQA | popular | CatExpert | 0.1 | 1214 | 1143 | 357 | 286 | 0 | 3000 |
| GQA | popular | CatExpert | 0.15 | 1203 | 1150 | 350 | 297 | 0 | 3000 |
| GQA | popular | CatExpert | 0.2 | 1211 | 1109 | 391 | 289 | 0 | 3000 |
| GQA | popular | CatExpert | 0.25 | 1223 | 1118 | 382 | 277 | 0 | 3000 |
| GQA | popular | CatExpert | 0.3 | 1232 | 1110 | 390 | 268 | 0 | 3000 |
| GQA | popular | CatExpert | 0.4 | 1252 | 1070 | 430 | 248 | 0 | 3000 |
| GQA | popular | CatExpert | 0.5 | 1250 | 1057 | 443 | 250 | 0 | 3000 |
| GQA | popular | CatExpert | 0.75 | 1287 | 973 | 527 | 213 | 0 | 3000 |
| GQA | popular | CatExpert | 1.0 | 1303 | 948 | 552 | 197 | 0 | 3000 |
| GQA | popular | CatExpert | 1.25 | 1322 | 877 | 623 | 178 | 0 | 3000 |
| GQA | popular | CatExpert | 1.5 | 1344 | 694 | 806 | 156 | 0 | 3000 |
| GQA | popular | CatExpert | 2.0 | 1409 | 317 | 1183 | 91 | 0 | 3000 |
| GQA | random | Regular |  | 1210 | 1332 | 168 | 290 | 0 | 3000 |
| GQA | random | CatExpert | 0.0 | 1182 | 1343 | 157 | 318 | 0 | 3000 |
| GQA | random | CatExpert | 0.01 | 1218 | 1332 | 168 | 282 | 0 | 3000 |
| GQA | random | CatExpert | 0.025 | 1212 | 1315 | 185 | 288 | 0 | 3000 |
| GQA | random | CatExpert | 0.05 | 1178 | 1324 | 176 | 322 | 0 | 3000 |
| GQA | random | CatExpert | 0.075 | 1198 | 1303 | 197 | 302 | 0 | 3000 |
| GQA | random | CatExpert | 0.1 | 1214 | 1325 | 175 | 286 | 0 | 3000 |
| GQA | random | CatExpert | 0.15 | 1203 | 1322 | 178 | 297 | 0 | 3000 |
| GQA | random | CatExpert | 0.2 | 1211 | 1312 | 188 | 289 | 0 | 3000 |
| GQA | random | CatExpert | 0.25 | 1223 | 1313 | 187 | 277 | 0 | 3000 |
| GQA | random | CatExpert | 0.3 | 1232 | 1332 | 168 | 268 | 0 | 3000 |
| GQA | random | CatExpert | 0.4 | 1252 | 1312 | 188 | 248 | 0 | 3000 |
| GQA | random | CatExpert | 0.5 | 1250 | 1286 | 214 | 250 | 0 | 3000 |
| GQA | random | CatExpert | 0.75 | 1287 | 1269 | 231 | 213 | 0 | 3000 |
| GQA | random | CatExpert | 1.0 | 1303 | 1252 | 248 | 197 | 0 | 3000 |
| GQA | random | CatExpert | 1.25 | 1322 | 1191 | 309 | 178 | 0 | 3000 |
| GQA | random | CatExpert | 1.5 | 1344 | 1084 | 416 | 156 | 0 | 3000 |
| GQA | random | CatExpert | 2.0 | 1416 | 531 | 969 | 84 | 0 | 3000 |
| MSCOCO | adversarial | Regular |  | 1105 | 1286 | 214 | 395 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.0 | 1104 | 1303 | 197 | 396 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.01 | 1127 | 1286 | 214 | 373 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.025 | 1105 | 1257 | 243 | 395 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.05 | 1101 | 1288 | 212 | 399 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.075 | 1135 | 1289 | 211 | 365 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.1 | 1107 | 1269 | 231 | 393 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.15 | 1082 | 1288 | 212 | 418 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.2 | 1124 | 1274 | 226 | 376 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.25 | 1113 | 1296 | 204 | 387 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.3 | 1124 | 1292 | 208 | 376 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.4 | 1137 | 1281 | 219 | 363 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.5 | 1166 | 1283 | 217 | 334 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.75 | 1165 | 1266 | 234 | 335 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.0 | 1192 | 1213 | 287 | 308 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.25 | 1208 | 1194 | 306 | 292 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.5 | 1254 | 1086 | 414 | 246 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 2.0 | 1387 | 494 | 1006 | 113 | 0 | 3000 |
| MSCOCO | popular | Regular |  | 1105 | 1372 | 128 | 395 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.0 | 1104 | 1374 | 126 | 396 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.01 | 1127 | 1375 | 125 | 373 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.025 | 1105 | 1353 | 147 | 395 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.05 | 1101 | 1386 | 114 | 399 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.075 | 1135 | 1375 | 125 | 365 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.1 | 1107 | 1366 | 134 | 393 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.15 | 1082 | 1362 | 138 | 418 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.2 | 1124 | 1368 | 132 | 376 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.25 | 1113 | 1375 | 125 | 387 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.3 | 1124 | 1387 | 113 | 376 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.4 | 1138 | 1356 | 144 | 362 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.5 | 1148 | 1358 | 142 | 352 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.75 | 1177 | 1348 | 152 | 323 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.0 | 1186 | 1355 | 145 | 314 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.25 | 1209 | 1297 | 203 | 291 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.5 | 1239 | 1214 | 286 | 261 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 2.0 | 1391 | 594 | 906 | 109 | 0 | 3000 |
| MSCOCO | random | Regular |  | 1105 | 1407 | 93 | 395 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.0 | 1104 | 1415 | 85 | 396 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.01 | 1127 | 1418 | 82 | 373 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.025 | 1105 | 1402 | 98 | 395 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.05 | 1101 | 1408 | 92 | 399 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.075 | 1135 | 1404 | 96 | 365 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.1 | 1107 | 1399 | 101 | 393 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.15 | 1082 | 1414 | 86 | 418 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.2 | 1124 | 1412 | 88 | 376 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.25 | 1113 | 1416 | 84 | 387 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.3 | 1124 | 1415 | 85 | 376 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.4 | 1137 | 1405 | 95 | 363 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.5 | 1166 | 1407 | 93 | 334 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.75 | 1177 | 1398 | 102 | 323 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.0 | 1191 | 1389 | 111 | 309 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.25 | 1208 | 1359 | 141 | 292 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.5 | 1254 | 1261 | 239 | 246 | 0 | 3000 |
| MSCOCO | random | CatExpert | 2.0 | 1384 | 649 | 851 | 116 | 0 | 3000 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- In `Official Regular vs Old HF Regular`, count fields such as FP are only directly comparable when `Same N=True`.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
