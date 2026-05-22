# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/aligned_official_cat_expert_full_seed42/summary.csv`
- Runs summarized: 108

## Run Config

- Config files found: `6`
- Runner: `official_llava`
- Model path: `/home/huiwei/sy/models/llava-v1.5-7b-official-clean`
- LLaVA repo: `/home/huiwei/sy/LLaVA-official-clean`
- Conv mode: `llava_v1`
- Prompt template: `{question} Please answer this question with one word.`
- Cat vector source: `hf_after_template_disjoint_v2`
- Decode: `{"do_sample": true, "temperature": 1.0, "top_p": 1.0, "num_beams": 1, "max_new_tokens": 1024}`
- Steering: `{"layers": "5-25", "topk": 64, "head_select": "norm", "prefill": true, "decode": true, "apply_to": "last_token", "prefill_apply_to": "last_token", "decode_apply_to": "last_token", "enabled_experts": ["cat"]}`

## Best CatExpert By F1

| Dataset | Setting | Best Alpha | Baseline Acc | Best Acc | Delta Acc | Baseline F1 | Best F1 | Delta F1 | Baseline FP | Best FP | FP Delta | Baseline Yes Rate | Best Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | 0.01 | 76.07 | 76.80 | 0.73 | 76.96 | 77.74 | 0.78 | 417 | 411 | -6.00 | 53.87 | 54.20 |
| GQA | popular | 0.01 | 79.33 | 78.93 | -0.40 | 79.61 | 79.40 | -0.21 | 330 | 350 | 20.00 | 51.33 | 52.27 |
| GQA | random | 1.0 | 84.73 | 86.00 | 1.27 | 84.09 | 86.07 | 1.99 | 168 | 218 | 50.00 | 45.93 | 50.53 |
| MSCOCO | adversarial | 0.5 | 79.70 | 81.93 | 2.23 | 78.40 | 81.21 | 2.81 | 214 | 213 | -1.00 | 43.97 | 46.13 |
| MSCOCO | popular | 1.0 | 82.57 | 85.03 | 2.47 | 80.86 | 84.05 | 3.19 | 128 | 132 | 4.00 | 41.10 | 43.83 |
| MSCOCO | random | 1.0 | 83.73 | 86.27 | 2.53 | 81.91 | 85.22 | 3.31 | 93 | 100 | 7.00 | 39.93 | 42.93 |

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
| GQA | adversarial | CatExpert | 0.05 | 3000 | 75.97 | 73.85 | 80.40 | 76.99 | 54.43 |
| GQA | adversarial | CatExpert | 0.075 | 3000 | 75.17 | 72.46 | 81.20 | 76.58 | 56.03 |
| GQA | adversarial | CatExpert | 0.1 | 3000 | 74.37 | 72.08 | 79.53 | 75.63 | 55.17 |
| GQA | adversarial | CatExpert | 0.15 | 3000 | 75.23 | 72.38 | 81.60 | 76.72 | 56.37 |
| GQA | adversarial | CatExpert | 0.2 | 3000 | 75.20 | 72.61 | 80.93 | 76.54 | 55.73 |
| GQA | adversarial | CatExpert | 0.25 | 3000 | 75.20 | 71.77 | 83.07 | 77.01 | 57.87 |
| GQA | adversarial | CatExpert | 0.3 | 3000 | 75.40 | 72.23 | 82.53 | 77.04 | 57.13 |
| GQA | adversarial | CatExpert | 0.4 | 3000 | 75.17 | 71.56 | 83.53 | 77.08 | 58.37 |
| GQA | adversarial | CatExpert | 0.5 | 3000 | 75.63 | 71.61 | 84.93 | 77.71 | 59.30 |
| GQA | adversarial | CatExpert | 0.75 | 3000 | 74.17 | 69.58 | 85.87 | 76.87 | 61.70 |
| GQA | adversarial | CatExpert | 1.0 | 3000 | 73.33 | 68.54 | 86.27 | 76.39 | 62.93 |
| GQA | adversarial | CatExpert | 1.25 | 3000 | 73.13 | 68.11 | 87.00 | 76.41 | 63.87 |
| GQA | adversarial | CatExpert | 1.5 | 3000 | 71.20 | 66.09 | 87.07 | 75.14 | 65.87 |
| GQA | adversarial | CatExpert | 2.0 | 3000 | 64.17 | 59.29 | 90.40 | 71.61 | 76.23 |
| GQA | popular | Regular |  | 3000 | 79.33 | 78.57 | 80.67 | 79.61 | 51.33 |
| GQA | popular | CatExpert | 0.0 | 3000 | 78.57 | 78.43 | 78.80 | 78.62 | 50.23 |
| GQA | popular | CatExpert | 0.01 | 3000 | 78.93 | 77.68 | 81.20 | 79.40 | 52.27 |
| GQA | popular | CatExpert | 0.025 | 3000 | 78.17 | 76.76 | 80.80 | 78.73 | 52.63 |
| GQA | popular | CatExpert | 0.05 | 3000 | 77.90 | 77.55 | 78.53 | 78.04 | 50.63 |
| GQA | popular | CatExpert | 0.075 | 3000 | 77.63 | 76.45 | 79.87 | 78.12 | 52.23 |
| GQA | popular | CatExpert | 0.1 | 3000 | 78.77 | 77.43 | 81.20 | 79.27 | 52.43 |
| GQA | popular | CatExpert | 0.15 | 3000 | 78.60 | 77.64 | 80.33 | 78.96 | 51.73 |
| GQA | popular | CatExpert | 0.2 | 3000 | 77.50 | 75.73 | 80.93 | 78.25 | 53.43 |
| GQA | popular | CatExpert | 0.25 | 3000 | 78.27 | 76.34 | 81.93 | 79.04 | 53.67 |
| GQA | popular | CatExpert | 0.3 | 3000 | 78.27 | 76.17 | 82.27 | 79.10 | 54.00 |
| GQA | popular | CatExpert | 0.4 | 3000 | 78.03 | 74.99 | 84.13 | 79.30 | 56.10 |
| GQA | popular | CatExpert | 0.5 | 3000 | 77.57 | 74.54 | 83.73 | 78.87 | 56.17 |
| GQA | popular | CatExpert | 0.75 | 3000 | 76.40 | 72.12 | 86.07 | 78.48 | 59.67 |
| GQA | popular | CatExpert | 1.0 | 3000 | 76.63 | 72.23 | 86.53 | 78.74 | 59.90 |
| GQA | popular | CatExpert | 1.25 | 3000 | 76.03 | 71.30 | 87.13 | 78.43 | 61.10 |
| GQA | popular | CatExpert | 1.5 | 3000 | 73.40 | 68.38 | 87.07 | 76.60 | 63.67 |
| GQA | popular | CatExpert | 2.0 | 3000 | 63.67 | 59.07 | 89.00 | 71.01 | 75.33 |
| GQA | random | Regular |  | 3000 | 84.73 | 87.81 | 80.67 | 84.09 | 45.93 |
| GQA | random | CatExpert | 0.0 | 3000 | 84.17 | 88.27 | 78.80 | 83.27 | 44.63 |
| GQA | random | CatExpert | 0.01 | 3000 | 85.00 | 87.88 | 81.20 | 84.41 | 46.20 |
| GQA | random | CatExpert | 0.025 | 3000 | 84.23 | 86.76 | 80.80 | 83.67 | 46.57 |
| GQA | random | CatExpert | 0.05 | 3000 | 83.40 | 87.00 | 78.53 | 82.55 | 45.13 |
| GQA | random | CatExpert | 0.075 | 3000 | 83.43 | 86.00 | 79.87 | 82.82 | 46.43 |
| GQA | random | CatExpert | 0.1 | 3000 | 84.83 | 87.56 | 81.20 | 84.26 | 46.37 |
| GQA | random | CatExpert | 0.15 | 3000 | 84.27 | 87.19 | 80.33 | 83.62 | 46.07 |
| GQA | random | CatExpert | 0.2 | 3000 | 84.20 | 86.59 | 80.93 | 83.67 | 46.73 |
| GQA | random | CatExpert | 0.25 | 3000 | 84.87 | 87.04 | 81.93 | 84.41 | 47.07 |
| GQA | random | CatExpert | 0.3 | 3000 | 85.73 | 88.40 | 82.27 | 85.22 | 46.53 |
| GQA | random | CatExpert | 0.4 | 3000 | 86.00 | 87.40 | 84.13 | 85.73 | 48.13 |
| GQA | random | CatExpert | 0.5 | 3000 | 85.00 | 85.91 | 83.73 | 84.81 | 48.73 |
| GQA | random | CatExpert | 0.75 | 3000 | 85.97 | 85.89 | 86.07 | 85.98 | 50.10 |
| GQA | random | CatExpert | 1.0 | 3000 | 86.00 | 85.62 | 86.53 | 86.07 | 50.53 |
| GQA | random | CatExpert | 1.25 | 3000 | 85.47 | 84.32 | 87.13 | 85.70 | 51.67 |
| GQA | random | CatExpert | 1.5 | 3000 | 84.77 | 82.99 | 87.47 | 85.17 | 52.70 |
| GQA | random | CatExpert | 2.0 | 3000 | 74.43 | 68.71 | 89.73 | 77.83 | 65.30 |
| MSCOCO | adversarial | Regular |  | 3000 | 79.70 | 83.78 | 73.67 | 78.40 | 43.97 |
| MSCOCO | adversarial | CatExpert | 0.0 | 3000 | 80.23 | 84.86 | 73.60 | 78.83 | 43.37 |
| MSCOCO | adversarial | CatExpert | 0.01 | 3000 | 80.50 | 84.12 | 75.20 | 79.41 | 44.70 |
| MSCOCO | adversarial | CatExpert | 0.025 | 3000 | 78.77 | 81.99 | 73.73 | 77.64 | 44.97 |
| MSCOCO | adversarial | CatExpert | 0.05 | 3000 | 79.63 | 83.91 | 73.33 | 78.26 | 43.70 |
| MSCOCO | adversarial | CatExpert | 0.075 | 3000 | 80.77 | 84.26 | 75.67 | 79.73 | 44.90 |
| MSCOCO | adversarial | CatExpert | 0.1 | 3000 | 79.27 | 82.76 | 73.93 | 78.10 | 44.67 |
| MSCOCO | adversarial | CatExpert | 0.15 | 3000 | 79.10 | 83.86 | 72.07 | 77.52 | 42.97 |
| MSCOCO | adversarial | CatExpert | 0.2 | 3000 | 79.83 | 83.07 | 74.93 | 78.79 | 45.10 |
| MSCOCO | adversarial | CatExpert | 0.25 | 3000 | 80.33 | 84.68 | 74.07 | 79.02 | 43.73 |
| MSCOCO | adversarial | CatExpert | 0.3 | 3000 | 80.83 | 84.70 | 75.27 | 79.70 | 44.43 |
| MSCOCO | adversarial | CatExpert | 0.4 | 3000 | 80.90 | 84.26 | 76.00 | 79.92 | 45.10 |
| MSCOCO | adversarial | CatExpert | 0.5 | 3000 | 81.93 | 84.61 | 78.07 | 81.21 | 46.13 |
| MSCOCO | adversarial | CatExpert | 0.75 | 3000 | 80.97 | 82.83 | 78.13 | 80.41 | 47.17 |
| MSCOCO | adversarial | CatExpert | 1.0 | 3000 | 81.23 | 83.78 | 77.47 | 80.50 | 46.23 |
| MSCOCO | adversarial | CatExpert | 1.25 | 3000 | 81.37 | 82.79 | 79.20 | 80.95 | 47.83 |
| MSCOCO | adversarial | CatExpert | 1.5 | 3000 | 78.57 | 78.82 | 78.13 | 78.47 | 49.57 |
| MSCOCO | adversarial | CatExpert | 2.0 | 3000 | 71.27 | 66.38 | 86.20 | 75.00 | 64.93 |
| MSCOCO | popular | Regular |  | 3000 | 82.57 | 89.62 | 73.67 | 80.86 | 41.10 |
| MSCOCO | popular | CatExpert | 0.0 | 3000 | 82.60 | 89.76 | 73.60 | 80.88 | 41.00 |
| MSCOCO | popular | CatExpert | 0.01 | 3000 | 83.43 | 90.02 | 75.20 | 81.95 | 41.77 |
| MSCOCO | popular | CatExpert | 0.025 | 3000 | 82.00 | 88.34 | 73.73 | 80.38 | 41.73 |
| MSCOCO | popular | CatExpert | 0.05 | 3000 | 82.90 | 90.68 | 73.33 | 81.09 | 40.43 |
| MSCOCO | popular | CatExpert | 0.075 | 3000 | 83.63 | 90.01 | 75.67 | 82.22 | 42.03 |
| MSCOCO | popular | CatExpert | 0.1 | 3000 | 82.57 | 89.36 | 73.93 | 80.92 | 41.37 |
| MSCOCO | popular | CatExpert | 0.15 | 3000 | 81.50 | 88.82 | 72.07 | 79.57 | 40.57 |
| MSCOCO | popular | CatExpert | 0.2 | 3000 | 83.07 | 89.49 | 74.93 | 81.57 | 41.87 |
| MSCOCO | popular | CatExpert | 0.25 | 3000 | 83.03 | 90.25 | 74.07 | 81.36 | 41.03 |
| MSCOCO | popular | CatExpert | 0.3 | 3000 | 83.93 | 91.05 | 75.27 | 82.41 | 41.33 |
| MSCOCO | popular | CatExpert | 0.4 | 3000 | 83.47 | 89.34 | 76.00 | 82.13 | 42.53 |
| MSCOCO | popular | CatExpert | 0.5 | 3000 | 84.97 | 90.56 | 78.07 | 83.85 | 43.10 |
| MSCOCO | popular | CatExpert | 0.75 | 3000 | 84.10 | 89.62 | 77.13 | 82.91 | 43.03 |
| MSCOCO | popular | CatExpert | 1.0 | 3000 | 85.03 | 89.96 | 78.87 | 84.05 | 43.83 |
| MSCOCO | popular | CatExpert | 1.25 | 3000 | 84.27 | 89.72 | 77.40 | 83.11 | 43.13 |
| MSCOCO | popular | CatExpert | 1.5 | 3000 | 84.30 | 87.97 | 79.47 | 83.50 | 45.17 |
| MSCOCO | popular | CatExpert | 2.0 | 3000 | 74.33 | 69.50 | 86.73 | 77.16 | 62.40 |
| MSCOCO | random | Regular |  | 3000 | 83.73 | 92.24 | 73.67 | 81.91 | 39.93 |
| MSCOCO | random | CatExpert | 0.0 | 3000 | 83.97 | 92.85 | 73.60 | 82.11 | 39.63 |
| MSCOCO | random | CatExpert | 0.01 | 3000 | 84.87 | 93.22 | 75.20 | 83.25 | 40.33 |
| MSCOCO | random | CatExpert | 0.025 | 3000 | 83.67 | 92.01 | 73.73 | 81.87 | 40.07 |
| MSCOCO | random | CatExpert | 0.05 | 3000 | 83.60 | 92.28 | 73.33 | 81.72 | 39.73 |
| MSCOCO | random | CatExpert | 0.075 | 3000 | 84.63 | 92.20 | 75.67 | 83.12 | 41.03 |
| MSCOCO | random | CatExpert | 0.1 | 3000 | 83.70 | 91.88 | 73.93 | 81.94 | 40.23 |
| MSCOCO | random | CatExpert | 0.15 | 3000 | 83.37 | 93.11 | 72.07 | 81.25 | 38.70 |
| MSCOCO | random | CatExpert | 0.2 | 3000 | 84.53 | 92.74 | 74.93 | 82.89 | 40.40 |
| MSCOCO | random | CatExpert | 0.25 | 3000 | 84.30 | 93.13 | 74.07 | 82.51 | 39.77 |
| MSCOCO | random | CatExpert | 0.3 | 3000 | 84.93 | 93.31 | 75.27 | 83.32 | 40.33 |
| MSCOCO | random | CatExpert | 0.4 | 3000 | 85.07 | 92.83 | 76.00 | 83.58 | 40.93 |
| MSCOCO | random | CatExpert | 0.5 | 3000 | 86.03 | 92.86 | 78.07 | 84.82 | 42.03 |
| MSCOCO | random | CatExpert | 0.75 | 3000 | 85.67 | 93.01 | 77.13 | 84.33 | 41.47 |
| MSCOCO | random | CatExpert | 1.0 | 3000 | 86.27 | 92.24 | 79.20 | 85.22 | 42.93 |
| MSCOCO | random | CatExpert | 1.25 | 3000 | 85.90 | 91.58 | 79.07 | 84.87 | 43.17 |
| MSCOCO | random | CatExpert | 1.5 | 3000 | 85.53 | 90.44 | 79.47 | 84.60 | 43.93 |
| MSCOCO | random | CatExpert | 2.0 | 3000 | 74.70 | 70.99 | 83.53 | 76.75 | 58.83 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 1199 | 1083 | 417 | 301 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.0 | 1179 | 1076 | 424 | 321 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.01 | 1215 | 1089 | 411 | 285 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.025 | 1208 | 1032 | 468 | 292 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.05 | 1206 | 1073 | 427 | 294 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.075 | 1218 | 1037 | 463 | 282 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.1 | 1193 | 1038 | 462 | 307 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.15 | 1224 | 1033 | 467 | 276 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.2 | 1214 | 1042 | 458 | 286 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.25 | 1246 | 1010 | 490 | 254 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.3 | 1238 | 1024 | 476 | 262 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.4 | 1253 | 1002 | 498 | 247 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.5 | 1274 | 995 | 505 | 226 | 0 | 3000 |
| GQA | adversarial | CatExpert | 0.75 | 1288 | 937 | 563 | 212 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.0 | 1294 | 906 | 594 | 206 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.25 | 1305 | 889 | 611 | 195 | 0 | 3000 |
| GQA | adversarial | CatExpert | 1.5 | 1306 | 830 | 670 | 194 | 0 | 3000 |
| GQA | adversarial | CatExpert | 2.0 | 1356 | 569 | 931 | 144 | 0 | 3000 |
| GQA | popular | Regular |  | 1210 | 1170 | 330 | 290 | 0 | 3000 |
| GQA | popular | CatExpert | 0.0 | 1182 | 1175 | 325 | 318 | 0 | 3000 |
| GQA | popular | CatExpert | 0.01 | 1218 | 1150 | 350 | 282 | 0 | 3000 |
| GQA | popular | CatExpert | 0.025 | 1212 | 1133 | 367 | 288 | 0 | 3000 |
| GQA | popular | CatExpert | 0.05 | 1178 | 1159 | 341 | 322 | 0 | 3000 |
| GQA | popular | CatExpert | 0.075 | 1198 | 1131 | 369 | 302 | 0 | 3000 |
| GQA | popular | CatExpert | 0.1 | 1218 | 1145 | 355 | 282 | 0 | 3000 |
| GQA | popular | CatExpert | 0.15 | 1205 | 1153 | 347 | 295 | 0 | 3000 |
| GQA | popular | CatExpert | 0.2 | 1214 | 1111 | 389 | 286 | 0 | 3000 |
| GQA | popular | CatExpert | 0.25 | 1229 | 1119 | 381 | 271 | 0 | 3000 |
| GQA | popular | CatExpert | 0.3 | 1234 | 1114 | 386 | 266 | 0 | 3000 |
| GQA | popular | CatExpert | 0.4 | 1262 | 1079 | 421 | 238 | 0 | 3000 |
| GQA | popular | CatExpert | 0.5 | 1256 | 1071 | 429 | 244 | 0 | 3000 |
| GQA | popular | CatExpert | 0.75 | 1291 | 1001 | 499 | 209 | 0 | 3000 |
| GQA | popular | CatExpert | 1.0 | 1298 | 1001 | 499 | 202 | 0 | 3000 |
| GQA | popular | CatExpert | 1.25 | 1307 | 974 | 526 | 193 | 0 | 3000 |
| GQA | popular | CatExpert | 1.5 | 1306 | 896 | 604 | 194 | 0 | 3000 |
| GQA | popular | CatExpert | 2.0 | 1335 | 575 | 925 | 165 | 0 | 3000 |
| GQA | random | Regular |  | 1210 | 1332 | 168 | 290 | 0 | 3000 |
| GQA | random | CatExpert | 0.0 | 1182 | 1343 | 157 | 318 | 0 | 3000 |
| GQA | random | CatExpert | 0.01 | 1218 | 1332 | 168 | 282 | 0 | 3000 |
| GQA | random | CatExpert | 0.025 | 1212 | 1315 | 185 | 288 | 0 | 3000 |
| GQA | random | CatExpert | 0.05 | 1178 | 1324 | 176 | 322 | 0 | 3000 |
| GQA | random | CatExpert | 0.075 | 1198 | 1305 | 195 | 302 | 0 | 3000 |
| GQA | random | CatExpert | 0.1 | 1218 | 1327 | 173 | 282 | 0 | 3000 |
| GQA | random | CatExpert | 0.15 | 1205 | 1323 | 177 | 295 | 0 | 3000 |
| GQA | random | CatExpert | 0.2 | 1214 | 1312 | 188 | 286 | 0 | 3000 |
| GQA | random | CatExpert | 0.25 | 1229 | 1317 | 183 | 271 | 0 | 3000 |
| GQA | random | CatExpert | 0.3 | 1234 | 1338 | 162 | 266 | 0 | 3000 |
| GQA | random | CatExpert | 0.4 | 1262 | 1318 | 182 | 238 | 0 | 3000 |
| GQA | random | CatExpert | 0.5 | 1256 | 1294 | 206 | 244 | 0 | 3000 |
| GQA | random | CatExpert | 0.75 | 1291 | 1288 | 212 | 209 | 0 | 3000 |
| GQA | random | CatExpert | 1.0 | 1298 | 1282 | 218 | 202 | 0 | 3000 |
| GQA | random | CatExpert | 1.25 | 1307 | 1257 | 243 | 193 | 0 | 3000 |
| GQA | random | CatExpert | 1.5 | 1312 | 1231 | 269 | 188 | 0 | 3000 |
| GQA | random | CatExpert | 2.0 | 1346 | 887 | 613 | 154 | 0 | 3000 |
| MSCOCO | adversarial | Regular |  | 1105 | 1286 | 214 | 395 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.0 | 1104 | 1303 | 197 | 396 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.01 | 1128 | 1287 | 213 | 372 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.025 | 1106 | 1257 | 243 | 394 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.05 | 1100 | 1289 | 211 | 400 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.075 | 1135 | 1288 | 212 | 365 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.1 | 1109 | 1269 | 231 | 391 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.15 | 1081 | 1292 | 208 | 419 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.2 | 1124 | 1271 | 229 | 376 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.25 | 1111 | 1299 | 201 | 389 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.3 | 1129 | 1296 | 204 | 371 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.4 | 1140 | 1287 | 213 | 360 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.5 | 1171 | 1287 | 213 | 329 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 0.75 | 1172 | 1257 | 243 | 328 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.0 | 1162 | 1275 | 225 | 338 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.25 | 1188 | 1253 | 247 | 312 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 1.5 | 1172 | 1185 | 315 | 328 | 0 | 3000 |
| MSCOCO | adversarial | CatExpert | 2.0 | 1293 | 845 | 655 | 207 | 0 | 3000 |
| MSCOCO | popular | Regular |  | 1105 | 1372 | 128 | 395 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.0 | 1104 | 1374 | 126 | 396 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.01 | 1128 | 1375 | 125 | 372 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.025 | 1106 | 1354 | 146 | 394 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.05 | 1100 | 1387 | 113 | 400 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.075 | 1135 | 1374 | 126 | 365 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.1 | 1109 | 1368 | 132 | 391 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.15 | 1081 | 1364 | 136 | 419 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.2 | 1124 | 1368 | 132 | 376 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.25 | 1111 | 1380 | 120 | 389 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.3 | 1129 | 1389 | 111 | 371 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.4 | 1140 | 1364 | 136 | 360 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.5 | 1171 | 1378 | 122 | 329 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 0.75 | 1157 | 1366 | 134 | 343 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.0 | 1183 | 1368 | 132 | 317 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.25 | 1161 | 1367 | 133 | 339 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 1.5 | 1192 | 1337 | 163 | 308 | 0 | 3000 |
| MSCOCO | popular | CatExpert | 2.0 | 1301 | 929 | 571 | 199 | 0 | 3000 |
| MSCOCO | random | Regular |  | 1105 | 1407 | 93 | 395 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.0 | 1104 | 1415 | 85 | 396 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.01 | 1128 | 1418 | 82 | 372 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.025 | 1106 | 1404 | 96 | 394 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.05 | 1100 | 1408 | 92 | 400 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.075 | 1135 | 1404 | 96 | 365 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.1 | 1109 | 1402 | 98 | 391 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.15 | 1081 | 1420 | 80 | 419 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.2 | 1124 | 1412 | 88 | 376 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.25 | 1111 | 1418 | 82 | 389 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.3 | 1129 | 1419 | 81 | 371 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.4 | 1140 | 1412 | 88 | 360 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.5 | 1171 | 1410 | 90 | 329 | 0 | 3000 |
| MSCOCO | random | CatExpert | 0.75 | 1157 | 1413 | 87 | 343 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.0 | 1188 | 1400 | 100 | 312 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.25 | 1186 | 1391 | 109 | 314 | 0 | 3000 |
| MSCOCO | random | CatExpert | 1.5 | 1192 | 1374 | 126 | 308 | 0 | 3000 |
| MSCOCO | random | CatExpert | 2.0 | 1253 | 988 | 512 | 247 | 0 | 3000 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- In `Official Regular vs Old HF Regular`, count fields such as FP are only directly comparable when `Same N=True`.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
