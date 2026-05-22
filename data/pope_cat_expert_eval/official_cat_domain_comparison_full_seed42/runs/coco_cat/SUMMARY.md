# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/runs/coco_cat/summary.csv`
- Runs summarized: 108

## Run Config

- Config files found: `6`
- Runner: `official_llava`
- Model path: `/home/huiwei/sy/models/llava-v1.5-7b-official-clean`
- LLaVA repo: `/home/huiwei/sy/LLaVA-official-clean`
- Conv mode: `llava_v1`
- Prompt template: `{question} Please answer this question with one word.`
- Cat vector source: `coco_cat`
- Decode: `{"do_sample": true, "temperature": 1.0, "top_p": 1.0, "num_beams": 1, "max_new_tokens": 1024}`
- Steering: `{"layers": "5-25", "topk": 64, "head_select": "norm", "prefill": true, "decode": true, "apply_to": "last_token", "prefill_apply_to": "last_token", "decode_apply_to": "last_token", "enabled_experts": ["cat"]}`

## Best CatExpert By F1

| Dataset | Setting | Best Alpha | Baseline Acc | Best Acc | Delta Acc | Baseline F1 | Best F1 | Delta F1 | Baseline FP | Best FP | FP Delta | Baseline Yes Rate | Best Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | 0.01 | 76.07 | 76.80 | 0.73 | 76.96 | 77.74 | 0.78 | 417 | 411 | -6.00 | 53.87 | 54.20 |
| GQA | popular | 0.01 | 79.33 | 78.90 | -0.43 | 79.61 | 79.37 | -0.23 | 330 | 351 | 21.00 | 51.33 | 52.30 |
| GQA | random | 1.0 | 84.73 | 85.50 | 0.77 | 84.09 | 85.71 | 1.63 | 168 | 240 | 72.00 | 45.93 | 51.50 |
| MSCOCO | adversarial | 0.5 | 79.70 | 81.80 | 2.10 | 78.40 | 81.11 | 2.71 | 214 | 218 | 4.00 | 43.97 | 46.33 |
| MSCOCO | popular | 1.0 | 82.57 | 85.13 | 2.57 | 80.86 | 84.27 | 3.41 | 128 | 141 | 13.00 | 41.10 | 44.53 |
| MSCOCO | random | 1.0 | 83.73 | 86.33 | 2.60 | 81.91 | 85.38 | 3.47 | 93 | 107 | 14.00 | 39.93 | 43.47 |

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
| GQA | adversarial | CatExpert | 0.025 | 3000 | 74.67 | 72.08 | 80.53 | 76.07 | 55.87 |
| GQA | adversarial | CatExpert | 0.05 | 3000 | 75.93 | 73.87 | 80.27 | 76.93 | 54.33 |
| GQA | adversarial | CatExpert | 0.075 | 3000 | 75.17 | 72.46 | 81.20 | 76.58 | 56.03 |
| GQA | adversarial | CatExpert | 0.1 | 3000 | 74.33 | 72.04 | 79.53 | 75.60 | 55.20 |
| GQA | adversarial | CatExpert | 0.15 | 3000 | 75.07 | 72.22 | 81.47 | 76.57 | 56.40 |
| GQA | adversarial | CatExpert | 0.2 | 3000 | 75.23 | 72.65 | 80.93 | 76.57 | 55.70 |
| GQA | adversarial | CatExpert | 0.25 | 3000 | 75.17 | 71.76 | 83.00 | 76.97 | 57.83 |
| GQA | adversarial | CatExpert | 0.3 | 3000 | 75.27 | 72.11 | 82.40 | 76.91 | 57.13 |
| GQA | adversarial | CatExpert | 0.4 | 3000 | 74.93 | 71.37 | 83.27 | 76.86 | 58.33 |
| GQA | adversarial | CatExpert | 0.5 | 3000 | 75.17 | 71.15 | 84.67 | 77.32 | 59.50 |
| GQA | adversarial | CatExpert | 0.75 | 3000 | 73.93 | 69.22 | 86.20 | 76.78 | 62.27 |
| GQA | adversarial | CatExpert | 1.0 | 3000 | 72.80 | 67.85 | 86.67 | 76.11 | 63.87 |
| GQA | adversarial | CatExpert | 1.25 | 3000 | 72.50 | 67.21 | 87.87 | 76.16 | 65.37 |
| GQA | adversarial | CatExpert | 1.5 | 3000 | 70.20 | 64.94 | 87.80 | 74.66 | 67.60 |
| GQA | adversarial | CatExpert | 2.0 | 3000 | 60.40 | 56.34 | 92.47 | 70.02 | 82.07 |
| GQA | popular | Regular |  | 3000 | 79.33 | 78.57 | 80.67 | 79.61 | 51.33 |
| GQA | popular | CatExpert | 0.0 | 3000 | 78.57 | 78.43 | 78.80 | 78.62 | 50.23 |
| GQA | popular | CatExpert | 0.01 | 3000 | 78.90 | 77.63 | 81.20 | 79.37 | 52.30 |
| GQA | popular | CatExpert | 0.025 | 3000 | 78.07 | 76.61 | 80.80 | 78.65 | 52.73 |
| GQA | popular | CatExpert | 0.05 | 3000 | 77.87 | 77.50 | 78.53 | 78.01 | 50.67 |
| GQA | popular | CatExpert | 0.075 | 3000 | 77.60 | 76.40 | 79.87 | 78.10 | 52.27 |
| GQA | popular | CatExpert | 0.1 | 3000 | 78.57 | 77.21 | 81.07 | 79.09 | 52.50 |
| GQA | popular | CatExpert | 0.15 | 3000 | 78.53 | 77.54 | 80.33 | 78.91 | 51.80 |
| GQA | popular | CatExpert | 0.2 | 3000 | 77.37 | 75.54 | 80.93 | 78.15 | 53.57 |
| GQA | popular | CatExpert | 0.25 | 3000 | 78.07 | 76.18 | 81.67 | 78.83 | 53.60 |
| GQA | popular | CatExpert | 0.3 | 3000 | 78.27 | 76.17 | 82.27 | 79.10 | 54.00 |
| GQA | popular | CatExpert | 0.4 | 3000 | 77.77 | 74.69 | 84.00 | 79.07 | 56.23 |
| GQA | popular | CatExpert | 0.5 | 3000 | 77.13 | 74.05 | 83.53 | 78.51 | 56.40 |
| GQA | popular | CatExpert | 0.75 | 3000 | 75.90 | 71.45 | 86.27 | 78.16 | 60.37 |
| GQA | popular | CatExpert | 1.0 | 3000 | 76.00 | 71.31 | 87.00 | 78.38 | 61.00 |
| GQA | popular | CatExpert | 1.25 | 3000 | 74.93 | 69.75 | 88.07 | 77.84 | 63.13 |
| GQA | popular | CatExpert | 1.5 | 3000 | 71.03 | 65.66 | 88.20 | 75.28 | 67.17 |
| GQA | popular | CatExpert | 2.0 | 3000 | 61.73 | 57.27 | 92.47 | 70.73 | 80.73 |
| GQA | random | Regular |  | 3000 | 84.73 | 87.81 | 80.67 | 84.09 | 45.93 |
| GQA | random | CatExpert | 0.0 | 3000 | 84.17 | 88.27 | 78.80 | 83.27 | 44.63 |
| GQA | random | CatExpert | 0.01 | 3000 | 85.00 | 87.88 | 81.20 | 84.41 | 46.20 |
| GQA | random | CatExpert | 0.025 | 3000 | 84.23 | 86.76 | 80.80 | 83.67 | 46.57 |
| GQA | random | CatExpert | 0.05 | 3000 | 83.40 | 87.00 | 78.53 | 82.55 | 45.13 |
| GQA | random | CatExpert | 0.075 | 3000 | 83.37 | 85.88 | 79.87 | 82.76 | 46.50 |
| GQA | random | CatExpert | 0.1 | 3000 | 84.77 | 87.54 | 81.07 | 84.18 | 46.30 |
| GQA | random | CatExpert | 0.15 | 3000 | 84.23 | 87.13 | 80.33 | 83.59 | 46.10 |
| GQA | random | CatExpert | 0.2 | 3000 | 84.17 | 86.53 | 80.93 | 83.64 | 46.77 |
| GQA | random | CatExpert | 0.25 | 3000 | 84.60 | 86.76 | 81.67 | 84.13 | 47.07 |
| GQA | random | CatExpert | 0.3 | 3000 | 85.47 | 87.89 | 82.27 | 84.99 | 46.80 |
| GQA | random | CatExpert | 0.4 | 3000 | 85.77 | 87.08 | 84.00 | 85.51 | 48.23 |
| GQA | random | CatExpert | 0.5 | 3000 | 84.63 | 85.41 | 83.53 | 84.46 | 48.90 |
| GQA | random | CatExpert | 0.75 | 3000 | 85.57 | 85.08 | 86.27 | 85.67 | 50.70 |
| GQA | random | CatExpert | 1.0 | 3000 | 85.50 | 84.47 | 87.00 | 85.71 | 51.50 |
| GQA | random | CatExpert | 1.25 | 3000 | 84.73 | 82.56 | 88.07 | 85.23 | 53.33 |
| GQA | random | CatExpert | 1.5 | 3000 | 82.70 | 79.46 | 88.20 | 83.60 | 55.50 |
| GQA | random | CatExpert | 2.0 | 3000 | 69.67 | 63.64 | 91.73 | 75.15 | 72.07 |
| MSCOCO | adversarial | Regular |  | 3000 | 79.70 | 83.78 | 73.67 | 78.40 | 43.97 |
| MSCOCO | adversarial | CatExpert | 0.0 | 3000 | 80.23 | 84.86 | 73.60 | 78.83 | 43.37 |
| MSCOCO | adversarial | CatExpert | 0.01 | 3000 | 80.47 | 84.05 | 75.20 | 79.38 | 44.73 |
| MSCOCO | adversarial | CatExpert | 0.025 | 3000 | 78.73 | 81.97 | 73.67 | 77.60 | 44.93 |
| MSCOCO | adversarial | CatExpert | 0.05 | 3000 | 79.63 | 83.85 | 73.40 | 78.28 | 43.77 |
| MSCOCO | adversarial | CatExpert | 0.075 | 3000 | 80.80 | 84.27 | 75.73 | 79.78 | 44.93 |
| MSCOCO | adversarial | CatExpert | 0.1 | 3000 | 79.23 | 82.75 | 73.87 | 78.06 | 44.63 |
| MSCOCO | adversarial | CatExpert | 0.15 | 3000 | 79.07 | 83.64 | 72.27 | 77.54 | 43.20 |
| MSCOCO | adversarial | CatExpert | 0.2 | 3000 | 79.93 | 83.26 | 74.93 | 78.88 | 45.00 |
| MSCOCO | adversarial | CatExpert | 0.25 | 3000 | 80.37 | 84.48 | 74.40 | 79.12 | 44.03 |
| MSCOCO | adversarial | CatExpert | 0.3 | 3000 | 80.73 | 84.45 | 75.33 | 79.63 | 44.60 |
| MSCOCO | adversarial | CatExpert | 0.4 | 3000 | 80.63 | 83.61 | 76.20 | 79.73 | 45.57 |
| MSCOCO | adversarial | CatExpert | 0.5 | 3000 | 81.80 | 84.32 | 78.13 | 81.11 | 46.33 |
| MSCOCO | adversarial | CatExpert | 0.75 | 3000 | 81.20 | 83.24 | 78.13 | 80.61 | 46.93 |
| MSCOCO | adversarial | CatExpert | 1.0 | 3000 | 80.53 | 80.95 | 79.87 | 80.40 | 49.33 |
| MSCOCO | adversarial | CatExpert | 1.25 | 3000 | 80.43 | 80.82 | 79.80 | 80.31 | 49.37 |
| MSCOCO | adversarial | CatExpert | 1.5 | 3000 | 79.40 | 78.27 | 81.40 | 79.80 | 52.00 |
| MSCOCO | adversarial | CatExpert | 2.0 | 3000 | 66.27 | 61.14 | 89.27 | 72.57 | 73.00 |
| MSCOCO | popular | Regular |  | 3000 | 82.57 | 89.62 | 73.67 | 80.86 | 41.10 |
| MSCOCO | popular | CatExpert | 0.0 | 3000 | 82.60 | 89.76 | 73.60 | 80.88 | 41.00 |
| MSCOCO | popular | CatExpert | 0.01 | 3000 | 83.43 | 90.02 | 75.20 | 81.95 | 41.77 |
| MSCOCO | popular | CatExpert | 0.025 | 3000 | 81.93 | 88.26 | 73.67 | 80.31 | 41.73 |
| MSCOCO | popular | CatExpert | 0.05 | 3000 | 82.90 | 90.62 | 73.40 | 81.10 | 40.50 |
| MSCOCO | popular | CatExpert | 0.075 | 3000 | 83.67 | 90.02 | 75.73 | 82.26 | 42.07 |
| MSCOCO | popular | CatExpert | 0.1 | 3000 | 82.50 | 89.28 | 73.87 | 80.85 | 41.37 |
| MSCOCO | popular | CatExpert | 0.15 | 3000 | 81.50 | 88.63 | 72.27 | 79.62 | 40.77 |
| MSCOCO | popular | CatExpert | 0.2 | 3000 | 83.07 | 89.49 | 74.93 | 81.57 | 41.87 |
| MSCOCO | popular | CatExpert | 0.25 | 3000 | 83.03 | 89.93 | 74.40 | 81.43 | 41.37 |
| MSCOCO | popular | CatExpert | 0.3 | 3000 | 83.90 | 90.91 | 75.33 | 82.39 | 41.43 |
| MSCOCO | popular | CatExpert | 0.4 | 3000 | 83.30 | 88.81 | 76.20 | 82.02 | 42.90 |
| MSCOCO | popular | CatExpert | 0.5 | 3000 | 84.83 | 90.22 | 78.13 | 83.74 | 43.30 |
| MSCOCO | popular | CatExpert | 0.75 | 3000 | 83.90 | 88.79 | 77.60 | 82.82 | 43.70 |
| MSCOCO | popular | CatExpert | 1.0 | 3000 | 85.13 | 89.45 | 79.67 | 84.27 | 44.53 |
| MSCOCO | popular | CatExpert | 1.25 | 3000 | 84.20 | 86.96 | 80.47 | 83.59 | 46.27 |
| MSCOCO | popular | CatExpert | 1.5 | 3000 | 83.77 | 85.54 | 81.27 | 83.35 | 47.50 |
| MSCOCO | popular | CatExpert | 2.0 | 3000 | 71.00 | 65.13 | 90.40 | 75.71 | 69.40 |
| MSCOCO | random | Regular |  | 3000 | 83.73 | 92.24 | 73.67 | 81.91 | 39.93 |
| MSCOCO | random | CatExpert | 0.0 | 3000 | 83.97 | 92.85 | 73.60 | 82.11 | 39.63 |
| MSCOCO | random | CatExpert | 0.01 | 3000 | 84.87 | 93.22 | 75.20 | 83.25 | 40.33 |
| MSCOCO | random | CatExpert | 0.025 | 3000 | 83.60 | 91.93 | 73.67 | 81.79 | 40.07 |
| MSCOCO | random | CatExpert | 0.05 | 3000 | 83.63 | 92.29 | 73.40 | 81.77 | 39.77 |
| MSCOCO | random | CatExpert | 0.075 | 3000 | 84.63 | 92.13 | 75.73 | 83.13 | 41.10 |
| MSCOCO | random | CatExpert | 0.1 | 3000 | 83.53 | 91.57 | 73.87 | 81.77 | 40.33 |
| MSCOCO | random | CatExpert | 0.15 | 3000 | 83.27 | 92.65 | 72.27 | 81.20 | 39.00 |
| MSCOCO | random | CatExpert | 0.2 | 3000 | 84.53 | 92.74 | 74.93 | 82.89 | 40.40 |
| MSCOCO | random | CatExpert | 0.25 | 3000 | 84.40 | 93.00 | 74.40 | 82.67 | 40.00 |
| MSCOCO | random | CatExpert | 0.3 | 3000 | 84.80 | 92.93 | 75.33 | 83.21 | 40.53 |
| MSCOCO | random | CatExpert | 0.4 | 3000 | 84.90 | 92.25 | 76.20 | 83.46 | 41.30 |
| MSCOCO | random | CatExpert | 0.5 | 3000 | 86.00 | 92.72 | 78.13 | 84.80 | 42.13 |
| MSCOCO | random | CatExpert | 0.75 | 3000 | 85.93 | 92.04 | 78.67 | 84.83 | 42.73 |
| MSCOCO | random | CatExpert | 1.0 | 3000 | 86.33 | 91.79 | 79.80 | 85.38 | 43.47 |
| MSCOCO | random | CatExpert | 1.25 | 3000 | 85.90 | 90.89 | 79.80 | 84.98 | 43.90 |
| MSCOCO | random | CatExpert | 1.5 | 3000 | 84.60 | 86.97 | 81.40 | 84.09 | 46.80 |
| MSCOCO | random | CatExpert | 2.0 | 3000 | 72.90 | 67.39 | 88.73 | 76.60 | 65.83 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 1199 | 1083 | 417 | 301 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.0 | 1179 | 1076 | 424 | 321 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.01 | 1215 | 1089 | 411 | 285 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.025 | 1208 | 1032 | 468 | 292 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.05 | 1204 | 1074 | 426 | 296 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.075 | 1218 | 1037 | 463 | 282 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.1 | 1193 | 1037 | 463 | 307 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.15 | 1222 | 1030 | 470 | 278 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.2 | 1214 | 1043 | 457 | 286 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.25 | 1245 | 1010 | 490 | 255 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.3 | 1236 | 1022 | 478 | 264 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.4 | 1249 | 999 | 501 | 251 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.5 | 1270 | 985 | 515 | 230 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.75 | 1293 | 925 | 575 | 207 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.0 | 1300 | 884 | 616 | 200 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.25 | 1318 | 857 | 643 | 182 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.5 | 1317 | 789 | 711 | 183 | 0 | 3000 |
| GQA | adversarial | CatExpert | 2.0 | 1387 | 425 | 1075 | 113 | 0 | 3000 |
| GQA | popular | Regular |  | 1210 | 1170 | 330 | 290 | 0 | 3000 |
| GQA | popular | CatExpert | 0.0 | 1182 | 1175 | 325 | 318 | 0 | 3000 |
| GQA | popular | CatExpert | 0.01 | 1218 | 1149 | 351 | 282 | 0 | 3000 |
| GQA | popular | CatExpert | 0.025 | 1212 | 1130 | 370 | 288 | 0 | 3000 |
| GQA | popular | CatExpert | 0.05 | 1178 | 1158 | 342 | 322 | 0 | 3000 |
| GQA | popular | CatExpert | 0.075 | 1198 | 1130 | 370 | 302 | 0 | 3000 |
| GQA | popular | CatExpert | 0.1 | 1216 | 1141 | 359 | 284 | 0 | 3000 |
| GQA | popular | CatExpert | 0.15 | 1205 | 1151 | 349 | 295 | 0 | 3000 |
| GQA | popular | CatExpert | 0.2 | 1214 | 1107 | 393 | 286 | 0 | 3000 |
| GQA | popular | CatExpert | 0.25 | 1225 | 1117 | 383 | 275 | 0 | 3000 |
| GQA | popular | CatExpert | 0.3 | 1234 | 1114 | 386 | 266 | 0 | 3000 |
| GQA | popular | CatExpert | 0.4 | 1260 | 1073 | 427 | 240 | 0 | 3000 |
| GQA | popular | CatExpert | 0.5 | 1253 | 1061 | 439 | 247 | 0 | 3000 |
| GQA | popular | CatExpert | 0.75 | 1294 | 983 | 517 | 206 | 0 | 3000 |
| GQA | popular | CatExpert | 1.0 | 1305 | 975 | 525 | 195 | 0 | 3000 |
| GQA | popular | CatExpert | 1.25 | 1321 | 927 | 573 | 179 | 0 | 3000 |
| GQA | popular | CatExpert | 1.5 | 1323 | 808 | 692 | 177 | 0 | 3000 |
| GQA | popular | CatExpert | 2.0 | 1387 | 465 | 1035 | 113 | 0 | 3000 |
| GQA | random | Regular |  | 1210 | 1332 | 168 | 290 | 0 | 3000 |
| GQA | random | CatExpert | 0.0 | 1182 | 1343 | 157 | 318 | 0 | 3000 |
| GQA | random | CatExpert | 0.01 | 1218 | 1332 | 168 | 282 | 0 | 3000 |
| GQA | random | CatExpert | 0.025 | 1212 | 1315 | 185 | 288 | 0 | 3000 |
| GQA | random | CatExpert | 0.05 | 1178 | 1324 | 176 | 322 | 0 | 3000 |
| GQA | random | CatExpert | 0.075 | 1198 | 1303 | 197 | 302 | 0 | 3000 |
| GQA | random | CatExpert | 0.1 | 1216 | 1327 | 173 | 284 | 0 | 3000 |
| GQA | random | CatExpert | 0.15 | 1205 | 1322 | 178 | 295 | 0 | 3000 |
| GQA | random | CatExpert | 0.2 | 1214 | 1311 | 189 | 286 | 0 | 3000 |
| GQA | random | CatExpert | 0.25 | 1225 | 1313 | 187 | 275 | 0 | 3000 |
| GQA | random | CatExpert | 0.3 | 1234 | 1330 | 170 | 266 | 0 | 3000 |
| GQA | random | CatExpert | 0.4 | 1260 | 1313 | 187 | 240 | 0 | 3000 |
| GQA | random | CatExpert | 0.5 | 1253 | 1286 | 214 | 247 | 0 | 3000 |
| GQA | random | CatExpert | 0.75 | 1294 | 1273 | 227 | 206 | 0 | 3000 |
| GQA | random | CatExpert | 1.0 | 1305 | 1260 | 240 | 195 | 0 | 3000 |
| GQA | random | CatExpert | 1.25 | 1321 | 1221 | 279 | 179 | 0 | 3000 |
| GQA | random | CatExpert | 1.5 | 1323 | 1158 | 342 | 177 | 0 | 3000 |
| GQA | random | CatExpert | 2.0 | 1376 | 714 | 786 | 124 | 0 | 3000 |
| MSCOCO | adversarial | Regular |  | 1105 | 1286 | 214 | 395 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.0 | 1104 | 1303 | 197 | 396 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.01 | 1128 | 1286 | 214 | 372 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.025 | 1105 | 1257 | 243 | 395 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.05 | 1101 | 1288 | 212 | 399 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.075 | 1136 | 1288 | 212 | 364 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.1 | 1108 | 1269 | 231 | 392 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.15 | 1084 | 1288 | 212 | 416 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.2 | 1124 | 1274 | 226 | 376 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.25 | 1116 | 1295 | 205 | 384 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.3 | 1130 | 1292 | 208 | 370 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.4 | 1143 | 1276 | 224 | 357 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.5 | 1172 | 1282 | 218 | 328 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.75 | 1172 | 1264 | 236 | 328 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.0 | 1198 | 1218 | 282 | 302 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.25 | 1197 | 1216 | 284 | 303 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.5 | 1221 | 1161 | 339 | 279 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 2.0 | 1339 | 649 | 851 | 161 | 0 | 3000 |
| MSCOCO | popular | Regular |  | 1105 | 1372 | 128 | 395 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.0 | 1104 | 1374 | 126 | 396 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.01 | 1128 | 1375 | 125 | 372 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.025 | 1105 | 1353 | 147 | 395 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.05 | 1101 | 1386 | 114 | 399 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.075 | 1136 | 1374 | 126 | 364 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.1 | 1108 | 1367 | 133 | 392 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.15 | 1084 | 1361 | 139 | 416 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.2 | 1124 | 1368 | 132 | 376 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.25 | 1116 | 1375 | 125 | 384 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.3 | 1130 | 1387 | 113 | 370 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.4 | 1143 | 1356 | 144 | 357 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.5 | 1172 | 1373 | 127 | 328 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.75 | 1164 | 1353 | 147 | 336 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.0 | 1195 | 1359 | 141 | 305 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.25 | 1207 | 1319 | 181 | 293 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.5 | 1219 | 1294 | 206 | 281 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 2.0 | 1356 | 774 | 726 | 144 | 0 | 3000 |
| MSCOCO | random | Regular |  | 1105 | 1407 | 93 | 395 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.0 | 1104 | 1415 | 85 | 396 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.01 | 1128 | 1418 | 82 | 372 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.025 | 1105 | 1403 | 97 | 395 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.05 | 1101 | 1408 | 92 | 399 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.075 | 1136 | 1403 | 97 | 364 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.1 | 1108 | 1398 | 102 | 392 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.15 | 1084 | 1414 | 86 | 416 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.2 | 1124 | 1412 | 88 | 376 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.25 | 1116 | 1416 | 84 | 384 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.3 | 1130 | 1414 | 86 | 370 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.4 | 1143 | 1404 | 96 | 357 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.5 | 1172 | 1408 | 92 | 328 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.75 | 1180 | 1398 | 102 | 320 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.0 | 1197 | 1393 | 107 | 303 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.25 | 1197 | 1380 | 120 | 303 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.5 | 1221 | 1317 | 183 | 279 | 0 | 3000 |
| MSCOCO | random | CatExpert | 2.0 | 1331 | 856 | 644 | 169 | 0 | 3000 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- In `Official Regular vs Old HF Regular`, count fields such as FP are only directly comparable when `Same N=True`.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
