# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/runs/mixed_cat/summary.csv`
- Runs summarized: 48

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
| GQA | adversarial | 0.1 | 77.33 | 79.00 | 1.67 | 77.78 | 80.00 | 2.22 | 37 | 39 | 2.00 | 52.00 | 55.00 |
| GQA | popular | 2.0 | 78.00 | 81.33 | 3.33 | 78.00 | 81.21 | 3.21 | 33 | 27 | -6.00 | 50.00 | 49.33 |
| GQA | random | 0.5 | 85.33 | 87.67 | 2.33 | 84.17 | 86.74 | 2.57 | 11 | 8 | -3.00 | 42.67 | 43.00 |
| MSCOCO | adversarial | 0.3 | 79.33 | 81.33 | 2.00 | 78.62 | 80.69 | 2.07 | 26 | 23 | -3.00 | 46.67 | 46.67 |
| MSCOCO | popular | 0.3 | 84.33 | 84.67 | 0.33 | 82.91 | 83.57 | 0.66 | 11 | 13 | 2.00 | 41.67 | 43.33 |
| MSCOCO | random | 0.3 | 83.67 | 84.67 | 1.00 | 82.31 | 83.57 | 1.26 | 13 | 13 | 0.00 | 42.33 | 43.33 |

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
| GQA | adversarial | CatExpert | 0.1 | 300 | 79.00 | 76.36 | 84.00 | 80.00 | 55.00 |
| GQA | adversarial | CatExpert | 0.3 | 300 | 77.67 | 74.85 | 83.33 | 78.86 | 55.67 |
| GQA | adversarial | CatExpert | 0.5 | 300 | 75.67 | 73.62 | 80.00 | 76.68 | 54.33 |
| GQA | adversarial | CatExpert | 1.0 | 300 | 77.00 | 75.16 | 80.67 | 77.81 | 53.67 |
| GQA | adversarial | CatExpert | 1.5 | 300 | 78.00 | 77.63 | 78.67 | 78.15 | 50.67 |
| GQA | adversarial | CatExpert | 2.0 | 300 | 78.00 | 75.93 | 82.00 | 78.85 | 54.00 |
| GQA | popular | Regular |  | 300 | 78.00 | 78.00 | 78.00 | 78.00 | 50.00 |
| GQA | popular | CatExpert | 0.0 | 300 | 78.00 | 77.63 | 78.67 | 78.15 | 50.67 |
| GQA | popular | CatExpert | 0.1 | 300 | 80.67 | 79.87 | 82.00 | 80.92 | 51.33 |
| GQA | popular | CatExpert | 0.3 | 300 | 80.00 | 79.61 | 80.67 | 80.13 | 50.67 |
| GQA | popular | CatExpert | 0.5 | 300 | 77.00 | 75.16 | 80.67 | 77.81 | 53.67 |
| GQA | popular | CatExpert | 1.0 | 300 | 75.67 | 74.84 | 77.33 | 76.07 | 51.67 |
| GQA | popular | CatExpert | 1.5 | 300 | 79.67 | 80.69 | 78.00 | 79.32 | 48.33 |
| GQA | popular | CatExpert | 2.0 | 300 | 81.33 | 81.76 | 80.67 | 81.21 | 49.33 |
| GQA | random | Regular |  | 300 | 85.33 | 91.41 | 78.00 | 84.17 | 42.67 |
| GQA | random | CatExpert | 0.0 | 300 | 84.00 | 88.06 | 78.67 | 83.10 | 44.67 |
| GQA | random | CatExpert | 0.1 | 300 | 86.33 | 89.78 | 82.00 | 85.71 | 45.67 |
| GQA | random | CatExpert | 0.3 | 300 | 86.67 | 91.67 | 80.67 | 85.82 | 44.00 |
| GQA | random | CatExpert | 0.5 | 300 | 87.67 | 93.80 | 80.67 | 86.74 | 43.00 |
| GQA | random | CatExpert | 1.0 | 300 | 85.00 | 91.34 | 77.33 | 83.75 | 42.33 |
| GQA | random | CatExpert | 1.5 | 300 | 85.33 | 91.41 | 78.00 | 84.17 | 42.67 |
| GQA | random | CatExpert | 2.0 | 300 | 87.00 | 92.37 | 80.67 | 86.12 | 43.67 |
| MSCOCO | adversarial | Regular |  | 300 | 79.33 | 81.43 | 76.00 | 78.62 | 46.67 |
| MSCOCO | adversarial | CatExpert | 0.0 | 300 | 77.67 | 79.43 | 74.67 | 76.98 | 47.00 |
| MSCOCO | adversarial | CatExpert | 0.1 | 300 | 80.33 | 82.73 | 76.67 | 79.58 | 46.33 |
| MSCOCO | adversarial | CatExpert | 0.3 | 300 | 81.33 | 83.57 | 78.00 | 80.69 | 46.67 |
| MSCOCO | adversarial | CatExpert | 0.5 | 300 | 77.33 | 81.06 | 71.33 | 75.89 | 44.00 |
| MSCOCO | adversarial | CatExpert | 1.0 | 300 | 80.33 | 84.73 | 74.00 | 79.00 | 43.67 |
| MSCOCO | adversarial | CatExpert | 1.5 | 300 | 80.67 | 84.85 | 74.67 | 79.43 | 44.00 |
| MSCOCO | adversarial | CatExpert | 2.0 | 300 | 78.33 | 82.95 | 71.33 | 76.70 | 43.00 |
| MSCOCO | popular | Regular |  | 300 | 84.33 | 91.20 | 76.00 | 82.91 | 41.67 |
| MSCOCO | popular | CatExpert | 0.0 | 300 | 82.00 | 87.50 | 74.67 | 80.58 | 42.67 |
| MSCOCO | popular | CatExpert | 0.1 | 300 | 83.67 | 89.15 | 76.67 | 82.44 | 43.00 |
| MSCOCO | popular | CatExpert | 0.3 | 300 | 84.67 | 90.00 | 78.00 | 83.57 | 43.33 |
| MSCOCO | popular | CatExpert | 0.5 | 300 | 82.33 | 91.45 | 71.33 | 80.15 | 39.00 |
| MSCOCO | popular | CatExpert | 1.0 | 300 | 84.33 | 93.28 | 74.00 | 82.53 | 39.67 |
| MSCOCO | popular | CatExpert | 1.5 | 300 | 83.33 | 90.32 | 74.67 | 81.75 | 41.33 |
| MSCOCO | popular | CatExpert | 2.0 | 300 | 81.67 | 89.92 | 71.33 | 79.55 | 39.67 |
| MSCOCO | random | Regular |  | 300 | 83.67 | 89.76 | 76.00 | 82.31 | 42.33 |
| MSCOCO | random | CatExpert | 0.0 | 300 | 82.33 | 88.19 | 74.67 | 80.87 | 42.33 |
| MSCOCO | random | CatExpert | 0.1 | 300 | 83.67 | 89.15 | 76.67 | 82.44 | 43.00 |
| MSCOCO | random | CatExpert | 0.3 | 300 | 84.67 | 90.00 | 78.00 | 83.57 | 43.33 |
| MSCOCO | random | CatExpert | 0.5 | 300 | 81.33 | 89.17 | 71.33 | 79.26 | 40.00 |
| MSCOCO | random | CatExpert | 1.0 | 300 | 82.67 | 89.52 | 74.00 | 81.02 | 41.33 |
| MSCOCO | random | CatExpert | 1.5 | 300 | 84.67 | 93.33 | 74.67 | 82.96 | 40.00 |
| MSCOCO | random | CatExpert | 2.0 | 300 | 83.00 | 93.04 | 71.33 | 80.75 | 38.33 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 119 | 113 | 37 | 31 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.0 | 117 | 113 | 37 | 33 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.1 | 126 | 111 | 39 | 24 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.3 | 125 | 108 | 42 | 25 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.5 | 120 | 107 | 43 | 30 | 0 | 300 |
| GQA | adversarial | CatExpert | 1.0 | 121 | 110 | 40 | 29 | 0 | 300 |
| GQA | adversarial | CatExpert | 1.5 | 118 | 116 | 34 | 32 | 0 | 300 |
| GQA | adversarial | CatExpert | 2.0 | 123 | 111 | 39 | 27 | 0 | 300 |
| GQA | popular | Regular |  | 117 | 117 | 33 | 33 | 0 | 300 |
| GQA | popular | CatExpert | 0.0 | 118 | 116 | 34 | 32 | 0 | 300 |
| GQA | popular | CatExpert | 0.1 | 123 | 119 | 31 | 27 | 0 | 300 |
| GQA | popular | CatExpert | 0.3 | 121 | 119 | 31 | 29 | 0 | 300 |
| GQA | popular | CatExpert | 0.5 | 121 | 110 | 40 | 29 | 0 | 300 |
| GQA | popular | CatExpert | 1.0 | 116 | 111 | 39 | 34 | 0 | 300 |
| GQA | popular | CatExpert | 1.5 | 117 | 122 | 28 | 33 | 0 | 300 |
| GQA | popular | CatExpert | 2.0 | 121 | 123 | 27 | 29 | 0 | 300 |
| GQA | random | Regular |  | 117 | 139 | 11 | 33 | 0 | 300 |
| GQA | random | CatExpert | 0.0 | 118 | 134 | 16 | 32 | 0 | 300 |
| GQA | random | CatExpert | 0.1 | 123 | 136 | 14 | 27 | 0 | 300 |
| GQA | random | CatExpert | 0.3 | 121 | 139 | 11 | 29 | 0 | 300 |
| GQA | random | CatExpert | 0.5 | 121 | 142 | 8 | 29 | 0 | 300 |
| GQA | random | CatExpert | 1.0 | 116 | 139 | 11 | 34 | 0 | 300 |
| GQA | random | CatExpert | 1.5 | 117 | 139 | 11 | 33 | 0 | 300 |
| GQA | random | CatExpert | 2.0 | 121 | 140 | 10 | 29 | 0 | 300 |
| MSCOCO | adversarial | Regular |  | 114 | 124 | 26 | 36 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.0 | 112 | 121 | 29 | 38 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.1 | 115 | 126 | 24 | 35 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.3 | 117 | 127 | 23 | 33 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.5 | 107 | 125 | 25 | 43 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 1.0 | 111 | 130 | 20 | 39 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 1.5 | 112 | 130 | 20 | 38 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 2.0 | 107 | 128 | 22 | 43 | 0 | 300 |
| MSCOCO | popular | Regular |  | 114 | 139 | 11 | 36 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.0 | 112 | 134 | 16 | 38 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.1 | 115 | 136 | 14 | 35 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.3 | 117 | 137 | 13 | 33 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.5 | 107 | 140 | 10 | 43 | 0 | 300 |
| MSCOCO | popular | CatExpert | 1.0 | 111 | 142 | 8 | 39 | 0 | 300 |
| MSCOCO | popular | CatExpert | 1.5 | 112 | 138 | 12 | 38 | 0 | 300 |
| MSCOCO | popular | CatExpert | 2.0 | 107 | 138 | 12 | 43 | 0 | 300 |
| MSCOCO | random | Regular |  | 114 | 137 | 13 | 36 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.0 | 112 | 135 | 15 | 38 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.1 | 115 | 136 | 14 | 35 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.3 | 117 | 137 | 13 | 33 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.5 | 107 | 137 | 13 | 43 | 0 | 300 |
| MSCOCO | random | CatExpert | 1.0 | 111 | 137 | 13 | 39 | 0 | 300 |
| MSCOCO | random | CatExpert | 1.5 | 112 | 142 | 8 | 38 | 0 | 300 |
| MSCOCO | random | CatExpert | 2.0 | 107 | 142 | 8 | 43 | 0 | 300 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- In `Official Regular vs Old HF Regular`, count fields such as FP are only directly comparable when `Same N=True`.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
