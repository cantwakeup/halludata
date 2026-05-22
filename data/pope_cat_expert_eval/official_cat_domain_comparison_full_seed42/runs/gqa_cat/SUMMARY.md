# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/runs/gqa_cat/summary.csv`
- Runs summarized: 108

## Run Config

- Config files found: `6`
- Runner: `official_llava`
- Model path: `/home/huiwei/sy/models/llava-v1.5-7b-official-clean`
- LLaVA repo: `/home/huiwei/sy/LLaVA-official-clean`
- Conv mode: `llava_v1`
- Prompt template: `{question} Please answer this question with one word.`
- Cat vector source: `gqa_cat`
- Decode: `{"do_sample": true, "temperature": 1.0, "top_p": 1.0, "num_beams": 1, "max_new_tokens": 1024}`
- Steering: `{"layers": "5-25", "topk": 64, "head_select": "norm", "prefill": true, "decode": true, "apply_to": "last_token", "prefill_apply_to": "last_token", "decode_apply_to": "last_token", "enabled_experts": ["cat"]}`

## Best CatExpert By F1

| Dataset | Setting | Best Alpha | Baseline Acc | Best Acc | Delta Acc | Baseline F1 | Best F1 | Delta F1 | Baseline FP | Best FP | FP Delta | Baseline Yes Rate | Best Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | 0.01 | 76.07 | 76.83 | 0.77 | 76.96 | 77.77 | 0.82 | 417 | 411 | -6.00 | 53.87 | 54.23 |
| GQA | popular | 0.01 | 79.33 | 78.93 | -0.40 | 79.61 | 79.41 | -0.19 | 330 | 351 | 21.00 | 51.33 | 52.33 |
| GQA | random | 0.75 | 84.73 | 85.33 | 0.60 | 84.09 | 85.45 | 1.36 | 168 | 232 | 64.00 | 45.93 | 50.80 |
| MSCOCO | adversarial | 0.5 | 79.70 | 81.73 | 2.03 | 78.40 | 80.99 | 2.59 | 214 | 215 | 1.00 | 43.97 | 46.07 |
| MSCOCO | popular | 1.0 | 82.57 | 84.63 | 2.07 | 80.86 | 83.83 | 2.97 | 128 | 156 | 28.00 | 41.10 | 45.03 |
| MSCOCO | random | 1.0 | 83.73 | 85.87 | 2.13 | 81.91 | 84.93 | 3.02 | 93 | 119 | 26.00 | 39.93 | 43.80 |

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
| GQA | adversarial | CatExpert | 0.01 | 3000 | 76.83 | 74.74 | 81.07 | 77.77 | 54.23 |
| GQA | adversarial | CatExpert | 0.025 | 3000 | 74.63 | 72.06 | 80.47 | 76.03 | 55.83 |
| GQA | adversarial | CatExpert | 0.05 | 3000 | 75.87 | 73.83 | 80.13 | 76.85 | 54.27 |
| GQA | adversarial | CatExpert | 0.075 | 3000 | 75.03 | 72.36 | 81.00 | 76.44 | 55.97 |
| GQA | adversarial | CatExpert | 0.1 | 3000 | 74.30 | 72.08 | 79.33 | 75.53 | 55.03 |
| GQA | adversarial | CatExpert | 0.15 | 3000 | 75.03 | 72.28 | 81.20 | 76.48 | 56.17 |
| GQA | adversarial | CatExpert | 0.2 | 3000 | 75.00 | 72.48 | 80.60 | 76.33 | 55.60 |
| GQA | adversarial | CatExpert | 0.25 | 3000 | 75.07 | 71.78 | 82.60 | 76.81 | 57.53 |
| GQA | adversarial | CatExpert | 0.3 | 3000 | 75.17 | 72.09 | 82.13 | 76.78 | 56.97 |
| GQA | adversarial | CatExpert | 0.4 | 3000 | 74.67 | 71.14 | 83.00 | 76.62 | 58.33 |
| GQA | adversarial | CatExpert | 0.5 | 3000 | 75.30 | 71.38 | 84.47 | 77.37 | 59.17 |
| GQA | adversarial | CatExpert | 0.75 | 3000 | 73.90 | 69.22 | 86.07 | 76.73 | 62.17 |
| GQA | adversarial | CatExpert | 1.0 | 3000 | 71.50 | 66.50 | 86.67 | 75.25 | 65.17 |
| GQA | adversarial | CatExpert | 1.25 | 3000 | 70.43 | 64.80 | 89.47 | 75.16 | 69.03 |
| GQA | adversarial | CatExpert | 1.5 | 3000 | 65.23 | 60.11 | 90.60 | 72.27 | 75.37 |
| GQA | adversarial | CatExpert | 2.0 | 3000 | 55.00 | 52.78 | 94.93 | 67.84 | 89.93 |
| GQA | popular | Regular |  | 3000 | 79.33 | 78.57 | 80.67 | 79.61 | 51.33 |
| GQA | popular | CatExpert | 0.0 | 3000 | 78.57 | 78.43 | 78.80 | 78.62 | 50.23 |
| GQA | popular | CatExpert | 0.01 | 3000 | 78.93 | 77.64 | 81.27 | 79.41 | 52.33 |
| GQA | popular | CatExpert | 0.025 | 3000 | 78.10 | 76.69 | 80.73 | 78.66 | 52.63 |
| GQA | popular | CatExpert | 0.05 | 3000 | 77.83 | 77.49 | 78.47 | 77.97 | 50.63 |
| GQA | popular | CatExpert | 0.075 | 3000 | 77.50 | 76.32 | 79.73 | 77.99 | 52.23 |
| GQA | popular | CatExpert | 0.1 | 3000 | 78.53 | 77.23 | 80.93 | 79.04 | 52.40 |
| GQA | popular | CatExpert | 0.15 | 3000 | 78.30 | 77.33 | 80.07 | 78.68 | 51.77 |
| GQA | popular | CatExpert | 0.2 | 3000 | 77.23 | 75.48 | 80.67 | 77.99 | 53.43 |
| GQA | popular | CatExpert | 0.25 | 3000 | 77.87 | 76.03 | 81.40 | 78.62 | 53.53 |
| GQA | popular | CatExpert | 0.3 | 3000 | 77.77 | 75.69 | 81.80 | 78.63 | 54.03 |
| GQA | popular | CatExpert | 0.4 | 3000 | 77.30 | 74.30 | 83.47 | 78.62 | 56.17 |
| GQA | popular | CatExpert | 0.5 | 3000 | 76.70 | 73.71 | 83.00 | 78.08 | 56.30 |
| GQA | popular | CatExpert | 0.75 | 3000 | 75.07 | 70.52 | 86.13 | 77.55 | 61.07 |
| GQA | popular | CatExpert | 1.0 | 3000 | 74.07 | 69.06 | 87.20 | 77.08 | 63.13 |
| GQA | popular | CatExpert | 1.25 | 3000 | 71.53 | 65.90 | 89.27 | 75.82 | 67.73 |
| GQA | popular | CatExpert | 1.5 | 3000 | 65.47 | 60.25 | 90.93 | 72.48 | 75.47 |
| GQA | popular | CatExpert | 2.0 | 3000 | 54.57 | 52.51 | 95.47 | 67.75 | 90.90 |
| GQA | random | Regular |  | 3000 | 84.73 | 87.81 | 80.67 | 84.09 | 45.93 |
| GQA | random | CatExpert | 0.0 | 3000 | 84.17 | 88.27 | 78.80 | 83.27 | 44.63 |
| GQA | random | CatExpert | 0.01 | 3000 | 85.03 | 87.89 | 81.27 | 84.45 | 46.23 |
| GQA | random | CatExpert | 0.025 | 3000 | 84.17 | 86.69 | 80.73 | 83.60 | 46.57 |
| GQA | random | CatExpert | 0.05 | 3000 | 83.37 | 86.99 | 78.47 | 82.51 | 45.10 |
| GQA | random | CatExpert | 0.075 | 3000 | 83.27 | 85.80 | 79.73 | 82.65 | 46.47 |
| GQA | random | CatExpert | 0.1 | 3000 | 84.70 | 87.53 | 80.93 | 84.10 | 46.23 |
| GQA | random | CatExpert | 0.15 | 3000 | 84.10 | 87.09 | 80.07 | 83.43 | 45.97 |
| GQA | random | CatExpert | 0.2 | 3000 | 84.07 | 86.55 | 80.67 | 83.51 | 46.60 |
| GQA | random | CatExpert | 0.25 | 3000 | 84.43 | 86.66 | 81.40 | 83.95 | 46.97 |
| GQA | random | CatExpert | 0.3 | 3000 | 85.30 | 87.96 | 81.80 | 84.77 | 46.50 |
| GQA | random | CatExpert | 0.4 | 3000 | 85.37 | 86.76 | 83.47 | 85.08 | 48.10 |
| GQA | random | CatExpert | 0.5 | 3000 | 84.30 | 85.22 | 83.00 | 84.09 | 48.70 |
| GQA | random | CatExpert | 0.75 | 3000 | 85.33 | 84.78 | 86.13 | 85.45 | 50.80 |
| GQA | random | CatExpert | 1.0 | 3000 | 84.67 | 82.99 | 87.20 | 85.05 | 52.53 |
| GQA | random | CatExpert | 1.25 | 3000 | 83.13 | 79.51 | 89.27 | 84.11 | 56.13 |
| GQA | random | CatExpert | 1.5 | 3000 | 78.23 | 72.51 | 90.93 | 80.69 | 62.70 |
| GQA | random | CatExpert | 2.0 | 3000 | 60.67 | 56.34 | 94.73 | 70.66 | 84.07 |
| MSCOCO | adversarial | Regular |  | 3000 | 79.70 | 83.78 | 73.67 | 78.40 | 43.97 |
| MSCOCO | adversarial | CatExpert | 0.0 | 3000 | 80.23 | 84.86 | 73.60 | 78.83 | 43.37 |
| MSCOCO | adversarial | CatExpert | 0.01 | 3000 | 80.43 | 84.04 | 75.13 | 79.34 | 44.70 |
| MSCOCO | adversarial | CatExpert | 0.025 | 3000 | 78.77 | 82.03 | 73.67 | 77.63 | 44.90 |
| MSCOCO | adversarial | CatExpert | 0.05 | 3000 | 79.60 | 83.79 | 73.40 | 78.25 | 43.80 |
| MSCOCO | adversarial | CatExpert | 0.075 | 3000 | 80.80 | 84.32 | 75.67 | 79.76 | 44.87 |
| MSCOCO | adversarial | CatExpert | 0.1 | 3000 | 79.17 | 82.72 | 73.73 | 77.97 | 44.57 |
| MSCOCO | adversarial | CatExpert | 0.15 | 3000 | 78.97 | 83.60 | 72.07 | 77.41 | 43.10 |
| MSCOCO | adversarial | CatExpert | 0.2 | 3000 | 79.83 | 83.22 | 74.73 | 78.75 | 44.90 |
| MSCOCO | adversarial | CatExpert | 0.25 | 3000 | 80.13 | 84.35 | 74.00 | 78.84 | 43.87 |
| MSCOCO | adversarial | CatExpert | 0.3 | 3000 | 80.40 | 84.23 | 74.80 | 79.24 | 44.40 |
| MSCOCO | adversarial | CatExpert | 0.4 | 3000 | 80.27 | 83.38 | 75.60 | 79.30 | 45.33 |
| MSCOCO | adversarial | CatExpert | 0.5 | 3000 | 81.73 | 84.44 | 77.80 | 80.99 | 46.07 |
| MSCOCO | adversarial | CatExpert | 0.75 | 3000 | 80.90 | 83.08 | 77.60 | 80.25 | 46.70 |
| MSCOCO | adversarial | CatExpert | 1.0 | 3000 | 79.93 | 79.81 | 80.13 | 79.97 | 50.20 |
| MSCOCO | adversarial | CatExpert | 1.25 | 3000 | 79.60 | 78.21 | 82.07 | 80.09 | 52.47 |
| MSCOCO | adversarial | CatExpert | 1.5 | 3000 | 75.67 | 71.46 | 85.47 | 77.84 | 59.80 |
| MSCOCO | adversarial | CatExpert | 2.0 | 3000 | 58.53 | 55.04 | 93.13 | 69.19 | 84.60 |
| MSCOCO | popular | Regular |  | 3000 | 82.57 | 89.62 | 73.67 | 80.86 | 41.10 |
| MSCOCO | popular | CatExpert | 0.0 | 3000 | 82.60 | 89.76 | 73.60 | 80.88 | 41.00 |
| MSCOCO | popular | CatExpert | 0.01 | 3000 | 83.37 | 89.94 | 75.13 | 81.87 | 41.77 |
| MSCOCO | popular | CatExpert | 0.025 | 3000 | 81.97 | 88.33 | 73.67 | 80.33 | 41.70 |
| MSCOCO | popular | CatExpert | 0.05 | 3000 | 82.90 | 90.62 | 73.40 | 81.10 | 40.50 |
| MSCOCO | popular | CatExpert | 0.075 | 3000 | 83.67 | 90.08 | 75.67 | 82.25 | 42.00 |
| MSCOCO | popular | CatExpert | 0.1 | 3000 | 82.40 | 89.19 | 73.73 | 80.73 | 41.33 |
| MSCOCO | popular | CatExpert | 0.15 | 3000 | 81.50 | 88.82 | 72.07 | 79.57 | 40.57 |
| MSCOCO | popular | CatExpert | 0.2 | 3000 | 82.90 | 89.32 | 74.73 | 81.38 | 41.83 |
| MSCOCO | popular | CatExpert | 0.25 | 3000 | 82.83 | 89.88 | 74.00 | 81.17 | 41.17 |
| MSCOCO | popular | CatExpert | 0.3 | 3000 | 83.63 | 90.85 | 74.80 | 82.05 | 41.17 |
| MSCOCO | popular | CatExpert | 0.4 | 3000 | 83.07 | 88.87 | 75.60 | 81.70 | 42.53 |
| MSCOCO | popular | CatExpert | 0.5 | 3000 | 84.63 | 90.12 | 77.80 | 83.51 | 43.17 |
| MSCOCO | popular | CatExpert | 0.75 | 3000 | 83.60 | 88.53 | 77.20 | 82.48 | 43.60 |
| MSCOCO | popular | CatExpert | 1.0 | 3000 | 84.63 | 88.45 | 79.67 | 83.83 | 45.03 |
| MSCOCO | popular | CatExpert | 1.25 | 3000 | 83.27 | 84.18 | 81.93 | 83.04 | 48.67 |
| MSCOCO | popular | CatExpert | 1.5 | 3000 | 79.87 | 77.55 | 84.07 | 80.68 | 54.20 |
| MSCOCO | popular | CatExpert | 2.0 | 3000 | 60.00 | 56.02 | 93.07 | 69.94 | 83.07 |
| MSCOCO | random | Regular |  | 3000 | 83.73 | 92.24 | 73.67 | 81.91 | 39.93 |
| MSCOCO | random | CatExpert | 0.0 | 3000 | 83.97 | 92.85 | 73.60 | 82.11 | 39.63 |
| MSCOCO | random | CatExpert | 0.01 | 3000 | 84.83 | 93.22 | 75.13 | 83.20 | 40.30 |
| MSCOCO | random | CatExpert | 0.025 | 3000 | 83.63 | 92.01 | 73.67 | 81.82 | 40.03 |
| MSCOCO | random | CatExpert | 0.05 | 3000 | 83.67 | 92.37 | 73.40 | 81.80 | 39.73 |
| MSCOCO | random | CatExpert | 0.075 | 3000 | 84.63 | 92.20 | 75.67 | 83.12 | 41.03 |
| MSCOCO | random | CatExpert | 0.1 | 3000 | 83.47 | 91.56 | 73.73 | 81.68 | 40.27 |
| MSCOCO | random | CatExpert | 0.15 | 3000 | 83.17 | 92.63 | 72.07 | 81.06 | 38.90 |
| MSCOCO | random | CatExpert | 0.2 | 3000 | 84.40 | 92.64 | 74.73 | 82.73 | 40.33 |
| MSCOCO | random | CatExpert | 0.25 | 3000 | 84.17 | 92.89 | 74.00 | 82.37 | 39.83 |
| MSCOCO | random | CatExpert | 0.3 | 3000 | 84.57 | 92.96 | 74.80 | 82.90 | 40.23 |
| MSCOCO | random | CatExpert | 0.4 | 3000 | 84.60 | 92.20 | 75.60 | 83.08 | 41.00 |
| MSCOCO | random | CatExpert | 0.5 | 3000 | 85.77 | 92.55 | 77.80 | 84.53 | 42.03 |
| MSCOCO | random | CatExpert | 0.75 | 3000 | 85.93 | 92.04 | 78.67 | 84.83 | 42.73 |
| MSCOCO | random | CatExpert | 1.0 | 3000 | 85.87 | 90.94 | 79.67 | 84.93 | 43.80 |
| MSCOCO | random | CatExpert | 1.25 | 3000 | 84.57 | 86.59 | 81.80 | 84.13 | 47.23 |
| MSCOCO | random | CatExpert | 1.5 | 3000 | 81.93 | 79.75 | 85.60 | 82.57 | 53.67 |
| MSCOCO | random | CatExpert | 2.0 | 3000 | 63.43 | 58.34 | 93.93 | 71.98 | 80.50 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 1199 | 1083 | 417 | 301 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.0 | 1179 | 1076 | 424 | 321 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.01 | 1216 | 1089 | 411 | 284 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.025 | 1207 | 1032 | 468 | 293 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.05 | 1202 | 1074 | 426 | 298 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.075 | 1215 | 1036 | 464 | 285 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.1 | 1190 | 1039 | 461 | 310 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.15 | 1218 | 1033 | 467 | 282 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.2 | 1209 | 1041 | 459 | 291 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.25 | 1239 | 1013 | 487 | 261 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.3 | 1232 | 1023 | 477 | 268 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.4 | 1245 | 995 | 505 | 255 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.5 | 1267 | 992 | 508 | 233 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.75 | 1291 | 926 | 574 | 209 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.0 | 1300 | 845 | 655 | 200 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.25 | 1342 | 771 | 729 | 158 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.5 | 1359 | 598 | 902 | 141 | 0 | 3000 |
| GQA | adversarial | CatExpert | 2.0 | 1424 | 226 | 1274 | 76 | 0 | 3000 |
| GQA | popular | Regular |  | 1210 | 1170 | 330 | 290 | 0 | 3000 |
| GQA | popular | CatExpert | 0.0 | 1182 | 1175 | 325 | 318 | 0 | 3000 |
| GQA | popular | CatExpert | 0.01 | 1219 | 1149 | 351 | 281 | 0 | 3000 |
| GQA | popular | CatExpert | 0.025 | 1211 | 1132 | 368 | 289 | 0 | 3000 |
| GQA | popular | CatExpert | 0.05 | 1177 | 1158 | 342 | 323 | 0 | 3000 |
| GQA | popular | CatExpert | 0.075 | 1196 | 1129 | 371 | 304 | 0 | 3000 |
| GQA | popular | CatExpert | 0.1 | 1214 | 1142 | 358 | 286 | 0 | 3000 |
| GQA | popular | CatExpert | 0.15 | 1201 | 1148 | 352 | 299 | 0 | 3000 |
| GQA | popular | CatExpert | 0.2 | 1210 | 1107 | 393 | 290 | 0 | 3000 |
| GQA | popular | CatExpert | 0.25 | 1221 | 1115 | 385 | 279 | 0 | 3000 |
| GQA | popular | CatExpert | 0.3 | 1227 | 1106 | 394 | 273 | 0 | 3000 |
| GQA | popular | CatExpert | 0.4 | 1252 | 1067 | 433 | 248 | 0 | 3000 |
| GQA | popular | CatExpert | 0.5 | 1245 | 1056 | 444 | 255 | 0 | 3000 |
| GQA | popular | CatExpert | 0.75 | 1292 | 960 | 540 | 208 | 0 | 3000 |
| GQA | popular | CatExpert | 1.0 | 1308 | 914 | 586 | 192 | 0 | 3000 |
| GQA | popular | CatExpert | 1.25 | 1339 | 807 | 693 | 161 | 0 | 3000 |
| GQA | popular | CatExpert | 1.5 | 1364 | 600 | 900 | 136 | 0 | 3000 |
| GQA | popular | CatExpert | 2.0 | 1432 | 205 | 1295 | 68 | 0 | 3000 |
| GQA | random | Regular |  | 1210 | 1332 | 168 | 290 | 0 | 3000 |
| GQA | random | CatExpert | 0.0 | 1182 | 1343 | 157 | 318 | 0 | 3000 |
| GQA | random | CatExpert | 0.01 | 1219 | 1332 | 168 | 281 | 0 | 3000 |
| GQA | random | CatExpert | 0.025 | 1211 | 1314 | 186 | 289 | 0 | 3000 |
| GQA | random | CatExpert | 0.05 | 1177 | 1324 | 176 | 323 | 0 | 3000 |
| GQA | random | CatExpert | 0.075 | 1196 | 1302 | 198 | 304 | 0 | 3000 |
| GQA | random | CatExpert | 0.1 | 1214 | 1327 | 173 | 286 | 0 | 3000 |
| GQA | random | CatExpert | 0.15 | 1201 | 1322 | 178 | 299 | 0 | 3000 |
| GQA | random | CatExpert | 0.2 | 1210 | 1312 | 188 | 290 | 0 | 3000 |
| GQA | random | CatExpert | 0.25 | 1221 | 1312 | 188 | 279 | 0 | 3000 |
| GQA | random | CatExpert | 0.3 | 1227 | 1332 | 168 | 273 | 0 | 3000 |
| GQA | random | CatExpert | 0.4 | 1252 | 1309 | 191 | 248 | 0 | 3000 |
| GQA | random | CatExpert | 0.5 | 1245 | 1284 | 216 | 255 | 0 | 3000 |
| GQA | random | CatExpert | 0.75 | 1292 | 1268 | 232 | 208 | 0 | 3000 |
| GQA | random | CatExpert | 1.0 | 1308 | 1232 | 268 | 192 | 0 | 3000 |
| GQA | random | CatExpert | 1.25 | 1339 | 1155 | 345 | 161 | 0 | 3000 |
| GQA | random | CatExpert | 1.5 | 1364 | 983 | 517 | 136 | 0 | 3000 |
| GQA | random | CatExpert | 2.0 | 1421 | 399 | 1101 | 79 | 0 | 3000 |
| MSCOCO | adversarial | Regular |  | 1105 | 1286 | 214 | 395 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.0 | 1104 | 1303 | 197 | 396 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.01 | 1127 | 1286 | 214 | 373 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.025 | 1105 | 1258 | 242 | 395 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.05 | 1101 | 1287 | 213 | 399 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.075 | 1135 | 1289 | 211 | 365 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.1 | 1106 | 1269 | 231 | 394 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.15 | 1081 | 1288 | 212 | 419 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.2 | 1121 | 1274 | 226 | 379 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.25 | 1110 | 1294 | 206 | 390 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.3 | 1122 | 1290 | 210 | 378 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.4 | 1134 | 1274 | 226 | 366 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.5 | 1167 | 1285 | 215 | 333 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.75 | 1164 | 1263 | 237 | 336 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.0 | 1202 | 1196 | 304 | 298 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.25 | 1231 | 1157 | 343 | 269 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.5 | 1282 | 988 | 512 | 218 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 2.0 | 1397 | 359 | 1141 | 103 | 0 | 3000 |
| MSCOCO | popular | Regular |  | 1105 | 1372 | 128 | 395 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.0 | 1104 | 1374 | 126 | 396 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.01 | 1127 | 1374 | 126 | 373 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.025 | 1105 | 1354 | 146 | 395 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.05 | 1101 | 1386 | 114 | 399 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.075 | 1135 | 1375 | 125 | 365 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.1 | 1106 | 1366 | 134 | 394 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.15 | 1081 | 1364 | 136 | 419 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.2 | 1121 | 1366 | 134 | 379 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.25 | 1110 | 1375 | 125 | 390 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.3 | 1122 | 1387 | 113 | 378 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.4 | 1134 | 1358 | 142 | 366 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.5 | 1167 | 1372 | 128 | 333 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.75 | 1158 | 1350 | 150 | 342 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.0 | 1195 | 1344 | 156 | 305 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.25 | 1229 | 1269 | 231 | 271 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.5 | 1261 | 1135 | 365 | 239 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 2.0 | 1396 | 404 | 1096 | 104 | 0 | 3000 |
| MSCOCO | random | Regular |  | 1105 | 1407 | 93 | 395 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.0 | 1104 | 1415 | 85 | 396 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.01 | 1127 | 1418 | 82 | 373 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.025 | 1105 | 1404 | 96 | 395 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.05 | 1101 | 1409 | 91 | 399 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.075 | 1135 | 1404 | 96 | 365 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.1 | 1106 | 1398 | 102 | 394 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.15 | 1081 | 1414 | 86 | 419 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.2 | 1121 | 1411 | 89 | 379 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.25 | 1110 | 1415 | 85 | 390 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.3 | 1122 | 1415 | 85 | 378 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.4 | 1134 | 1404 | 96 | 366 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.5 | 1167 | 1406 | 94 | 333 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.75 | 1180 | 1398 | 102 | 320 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.0 | 1195 | 1381 | 119 | 305 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.25 | 1227 | 1310 | 190 | 273 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.5 | 1284 | 1174 | 326 | 216 | 0 | 3000 |
| MSCOCO | random | CatExpert | 2.0 | 1409 | 494 | 1006 | 91 | 0 | 3000 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- In `Official Regular vs Old HF Regular`, count fields such as FP are only directly comparable when `Same N=True`.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
