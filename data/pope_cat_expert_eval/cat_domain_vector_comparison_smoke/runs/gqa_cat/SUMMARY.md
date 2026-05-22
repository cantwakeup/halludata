# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/runs/gqa_cat/summary.csv`
- Runs summarized: 48

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
| GQA | adversarial | 0.1 | 77.33 | 79.33 | 2.00 | 77.78 | 80.50 | 2.73 | 37 | 40 | 3.00 | 52.00 | 56.00 |
| GQA | popular | 0.1 | 78.00 | 80.67 | 2.67 | 78.00 | 80.92 | 2.92 | 33 | 31 | -2.00 | 50.00 | 51.33 |
| GQA | random | 0.5 | 85.33 | 89.33 | 4.00 | 84.17 | 88.89 | 4.72 | 11 | 10 | -1.00 | 42.67 | 46.00 |
| MSCOCO | adversarial | 0.3 | 79.33 | 81.67 | 2.33 | 78.62 | 81.36 | 2.74 | 26 | 25 | -1.00 | 46.67 | 48.33 |
| MSCOCO | popular | 0.3 | 84.33 | 86.00 | 1.67 | 82.91 | 85.11 | 2.20 | 11 | 12 | 1.00 | 41.67 | 44.00 |
| MSCOCO | random | 1.5 | 83.67 | 85.67 | 2.00 | 82.31 | 85.32 | 3.01 | 13 | 18 | 5.00 | 42.33 | 47.67 |

## Alpha 0 Check

| Dataset | Setting | Alpha0 Acc | Regular Acc | Acc Diff | Alpha0 F1 | Regular F1 | F1 Diff | Alpha0 Invalid | Regular Invalid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | 76.67 | 77.33 | -0.67 | 76.97 | 77.78 | -0.80 | 0 | 0 |
| GQA | popular | 78.00 | 78.00 | 0.00 | 78.15 | 78.00 | 0.15 | 0 | 0 |
| GQA | random | 84.00 | 85.33 | -1.33 | 83.10 | 84.17 | -1.07 | 0 | 0 |
| MSCOCO | adversarial | 77.67 | 79.33 | -1.67 | 76.98 | 78.62 | -1.64 | 0 | 0 |
| MSCOCO | popular | 82.00 | 84.33 | -2.33 | 80.58 | 82.91 | -2.33 | 0 | 0 |
| MSCOCO | random | 82.33 | 83.67 | -1.33 | 80.87 | 82.31 | -1.44 | 0 | 0 |

## Official Regular vs Old HF Regular

| Dataset | Setting | Official N | HF N | Same N | Official Acc | HF Acc | Acc Diff | Official F1 | HF F1 | F1 Diff | Official FP | HF FP | Official Yes Rate | HF Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | 300 | 3000 | False | 77.33 | 81.30 | -3.97 | 77.78 | 81.44 | -3.66 | 37 | 292 | 52.00 | 50.77 |
| GQA | popular | 300 | 3000 | False | 78.00 | 84.17 | -6.17 | 78.00 | 83.83 | -5.83 | 33 | 206 | 50.00 | 47.90 |
| GQA | random | 300 | 3000 | False | 85.33 | 88.50 | -3.17 | 84.17 | 87.71 | -3.54 | 11 | 76 | 42.67 | 43.57 |
| MSCOCO | adversarial | 300 | 3000 | False | 79.33 | 83.70 | -4.37 | 78.62 | 82.08 | -3.46 | 26 | 109 | 46.67 | 40.97 |
| MSCOCO | popular | 300 | 3000 | False | 84.33 | 85.67 | -1.33 | 82.91 | 83.90 | -0.99 | 11 | 50 | 41.67 | 39.00 |
| MSCOCO | random | 300 | 3000 | False | 83.67 | 86.50 | -2.83 | 82.31 | 84.69 | -2.38 | 13 | 25 | 42.33 | 38.17 |

## Main Table

| Dataset | Setting | Method | Alpha | N | Accuracy | Precision | Recall | F1 Score | Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 300 | 77.33 | 76.28 | 79.33 | 77.78 | 52.00 |
| GQA | adversarial | CatExpert | 0.0 | 300 | 76.67 | 75.97 | 78.00 | 76.97 | 51.33 |
| GQA | adversarial | CatExpert | 0.1 | 300 | 79.33 | 76.19 | 85.33 | 80.50 | 56.00 |
| GQA | adversarial | CatExpert | 0.3 | 300 | 77.67 | 74.27 | 84.67 | 79.13 | 57.00 |
| GQA | adversarial | CatExpert | 0.5 | 300 | 77.33 | 73.56 | 85.33 | 79.01 | 58.00 |
| GQA | adversarial | CatExpert | 1.0 | 300 | 74.33 | 69.95 | 85.33 | 76.88 | 61.00 |
| GQA | adversarial | CatExpert | 1.5 | 300 | 70.00 | 64.42 | 89.33 | 74.86 | 69.33 |
| GQA | adversarial | CatExpert | 2.0 | 300 | 57.00 | 54.09 | 92.67 | 68.30 | 85.67 |
| GQA | popular | Regular |  | 300 | 78.00 | 78.00 | 78.00 | 78.00 | 50.00 |
| GQA | popular | CatExpert | 0.0 | 300 | 78.00 | 77.63 | 78.67 | 78.15 | 50.67 |
| GQA | popular | CatExpert | 0.1 | 300 | 80.67 | 79.87 | 82.00 | 80.92 | 51.33 |
| GQA | popular | CatExpert | 0.3 | 300 | 79.67 | 76.65 | 85.33 | 80.76 | 55.67 |
| GQA | popular | CatExpert | 0.5 | 300 | 77.33 | 73.56 | 85.33 | 79.01 | 58.00 |
| GQA | popular | CatExpert | 1.0 | 300 | 75.00 | 70.72 | 85.33 | 77.34 | 60.33 |
| GQA | popular | CatExpert | 1.5 | 300 | 69.33 | 63.81 | 89.33 | 74.44 | 70.00 |
| GQA | popular | CatExpert | 2.0 | 300 | 57.33 | 54.40 | 90.67 | 68.00 | 83.33 |
| GQA | random | Regular |  | 300 | 85.33 | 91.41 | 78.00 | 84.17 | 42.67 |
| GQA | random | CatExpert | 0.0 | 300 | 84.00 | 88.06 | 78.67 | 83.10 | 44.67 |
| GQA | random | CatExpert | 0.1 | 300 | 86.00 | 89.13 | 82.00 | 85.42 | 46.00 |
| GQA | random | CatExpert | 0.3 | 300 | 89.00 | 92.09 | 85.33 | 88.58 | 46.33 |
| GQA | random | CatExpert | 0.5 | 300 | 89.33 | 92.75 | 85.33 | 88.89 | 46.00 |
| GQA | random | CatExpert | 1.0 | 300 | 87.00 | 88.28 | 85.33 | 86.78 | 48.33 |
| GQA | random | CatExpert | 1.5 | 300 | 86.33 | 84.28 | 89.33 | 86.73 | 53.00 |
| GQA | random | CatExpert | 2.0 | 300 | 66.67 | 61.26 | 90.67 | 73.12 | 74.00 |
| MSCOCO | adversarial | Regular |  | 300 | 79.33 | 81.43 | 76.00 | 78.62 | 46.67 |
| MSCOCO | adversarial | CatExpert | 0.0 | 300 | 77.67 | 79.43 | 74.67 | 76.98 | 47.00 |
| MSCOCO | adversarial | CatExpert | 0.1 | 300 | 80.00 | 82.14 | 76.67 | 79.31 | 46.67 |
| MSCOCO | adversarial | CatExpert | 0.3 | 300 | 81.67 | 82.76 | 80.00 | 81.36 | 48.33 |
| MSCOCO | adversarial | CatExpert | 0.5 | 300 | 78.33 | 80.58 | 74.67 | 77.51 | 46.33 |
| MSCOCO | adversarial | CatExpert | 1.0 | 300 | 79.67 | 81.12 | 77.33 | 79.18 | 47.67 |
| MSCOCO | adversarial | CatExpert | 1.5 | 300 | 76.33 | 73.10 | 83.33 | 77.88 | 57.00 |
| MSCOCO | adversarial | CatExpert | 2.0 | 300 | 61.67 | 57.45 | 90.00 | 70.13 | 78.33 |
| MSCOCO | popular | Regular |  | 300 | 84.33 | 91.20 | 76.00 | 82.91 | 41.67 |
| MSCOCO | popular | CatExpert | 0.0 | 300 | 82.00 | 87.50 | 74.67 | 80.58 | 42.67 |
| MSCOCO | popular | CatExpert | 0.1 | 300 | 83.67 | 89.15 | 76.67 | 82.44 | 43.00 |
| MSCOCO | popular | CatExpert | 0.3 | 300 | 86.00 | 90.91 | 80.00 | 85.11 | 44.00 |
| MSCOCO | popular | CatExpert | 0.5 | 300 | 84.33 | 92.56 | 74.67 | 82.66 | 40.33 |
| MSCOCO | popular | CatExpert | 1.0 | 300 | 84.00 | 89.23 | 77.33 | 82.86 | 43.33 |
| MSCOCO | popular | CatExpert | 1.5 | 300 | 85.00 | 86.21 | 83.33 | 84.75 | 48.33 |
| MSCOCO | popular | CatExpert | 2.0 | 300 | 66.00 | 60.81 | 90.00 | 72.58 | 74.00 |
| MSCOCO | random | Regular |  | 300 | 83.67 | 89.76 | 76.00 | 82.31 | 42.33 |
| MSCOCO | random | CatExpert | 0.0 | 300 | 82.33 | 88.19 | 74.67 | 80.87 | 42.33 |
| MSCOCO | random | CatExpert | 0.1 | 300 | 83.67 | 89.15 | 76.67 | 82.44 | 43.00 |
| MSCOCO | random | CatExpert | 0.3 | 300 | 85.67 | 90.23 | 80.00 | 84.81 | 44.33 |
| MSCOCO | random | CatExpert | 0.5 | 300 | 82.00 | 87.50 | 74.67 | 80.58 | 42.67 |
| MSCOCO | random | CatExpert | 1.0 | 300 | 84.00 | 89.23 | 77.33 | 82.86 | 43.33 |
| MSCOCO | random | CatExpert | 1.5 | 300 | 85.67 | 87.41 | 83.33 | 85.32 | 47.67 |
| MSCOCO | random | CatExpert | 2.0 | 300 | 67.67 | 62.21 | 90.00 | 73.57 | 72.33 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 119 | 113 | 37 | 31 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.0 | 117 | 113 | 37 | 33 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.1 | 128 | 110 | 40 | 22 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.3 | 127 | 106 | 44 | 23 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.5 | 128 | 104 | 46 | 22 | 0 | 300 |
| GQA | adversarial | CatExpert | 1.0 | 128 | 95 | 55 | 22 | 0 | 300 |
| GQA | adversarial | CatExpert | 1.5 | 134 | 76 | 74 | 16 | 0 | 300 |
| GQA | adversarial | CatExpert | 2.0 | 139 | 32 | 118 | 11 | 0 | 300 |
| GQA | popular | Regular |  | 117 | 117 | 33 | 33 | 0 | 300 |
| GQA | popular | CatExpert | 0.0 | 118 | 116 | 34 | 32 | 0 | 300 |
| GQA | popular | CatExpert | 0.1 | 123 | 119 | 31 | 27 | 0 | 300 |
| GQA | popular | CatExpert | 0.3 | 128 | 111 | 39 | 22 | 0 | 300 |
| GQA | popular | CatExpert | 0.5 | 128 | 104 | 46 | 22 | 0 | 300 |
| GQA | popular | CatExpert | 1.0 | 128 | 97 | 53 | 22 | 0 | 300 |
| GQA | popular | CatExpert | 1.5 | 134 | 74 | 76 | 16 | 0 | 300 |
| GQA | popular | CatExpert | 2.0 | 136 | 36 | 114 | 14 | 0 | 300 |
| GQA | random | Regular |  | 117 | 139 | 11 | 33 | 0 | 300 |
| GQA | random | CatExpert | 0.0 | 118 | 134 | 16 | 32 | 0 | 300 |
| GQA | random | CatExpert | 0.1 | 123 | 135 | 15 | 27 | 0 | 300 |
| GQA | random | CatExpert | 0.3 | 128 | 139 | 11 | 22 | 0 | 300 |
| GQA | random | CatExpert | 0.5 | 128 | 140 | 10 | 22 | 0 | 300 |
| GQA | random | CatExpert | 1.0 | 128 | 133 | 17 | 22 | 0 | 300 |
| GQA | random | CatExpert | 1.5 | 134 | 125 | 25 | 16 | 0 | 300 |
| GQA | random | CatExpert | 2.0 | 136 | 64 | 86 | 14 | 0 | 300 |
| MSCOCO | adversarial | Regular |  | 114 | 124 | 26 | 36 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.0 | 112 | 121 | 29 | 38 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.1 | 115 | 125 | 25 | 35 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.3 | 120 | 125 | 25 | 30 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.5 | 112 | 123 | 27 | 38 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 1.0 | 116 | 123 | 27 | 34 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 1.5 | 125 | 104 | 46 | 25 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 2.0 | 135 | 50 | 100 | 15 | 0 | 300 |
| MSCOCO | popular | Regular |  | 114 | 139 | 11 | 36 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.0 | 112 | 134 | 16 | 38 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.1 | 115 | 136 | 14 | 35 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.3 | 120 | 138 | 12 | 30 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.5 | 112 | 141 | 9 | 38 | 0 | 300 |
| MSCOCO | popular | CatExpert | 1.0 | 116 | 136 | 14 | 34 | 0 | 300 |
| MSCOCO | popular | CatExpert | 1.5 | 125 | 130 | 20 | 25 | 0 | 300 |
| MSCOCO | popular | CatExpert | 2.0 | 135 | 63 | 87 | 15 | 0 | 300 |
| MSCOCO | random | Regular |  | 114 | 137 | 13 | 36 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.0 | 112 | 135 | 15 | 38 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.1 | 115 | 136 | 14 | 35 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.3 | 120 | 137 | 13 | 30 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.5 | 112 | 134 | 16 | 38 | 0 | 300 |
| MSCOCO | random | CatExpert | 1.0 | 116 | 136 | 14 | 34 | 0 | 300 |
| MSCOCO | random | CatExpert | 1.5 | 125 | 132 | 18 | 25 | 0 | 300 |
| MSCOCO | random | CatExpert | 2.0 | 135 | 68 | 82 | 15 | 0 | 300 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- In `Official Regular vs Old HF Regular`, count fields such as FP are only directly comparable when `Same N=True`.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
