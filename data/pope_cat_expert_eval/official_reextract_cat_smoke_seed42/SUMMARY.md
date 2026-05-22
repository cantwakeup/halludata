# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/official_reextract_cat_smoke_seed42/summary.csv`
- Runs summarized: 36

## Run Config

- Config files found: `6`
- Runner: `official_llava`
- Model path: `/home/huiwei/sy/models/llava-v1.5-7b-official-clean`
- LLaVA repo: `/home/huiwei/sy/LLaVA-official-clean`
- Conv mode: `llava_v1`
- Prompt template: `{question} Please answer this question with one word.`
- Cat vector source: `official_llava_after_template_disjoint_v2_cat_smoke`
- Decode: `{"do_sample": true, "temperature": 1.0, "top_p": 1.0, "num_beams": 1, "max_new_tokens": 1024}`
- Steering: `{"layers": "5-25", "topk": 64, "head_select": "norm", "prefill": true, "decode": true, "apply_to": "last_token", "prefill_apply_to": "last_token", "decode_apply_to": "last_token", "enabled_experts": ["cat"]}`

## Best CatExpert By F1

| Dataset | Setting | Best Alpha | Baseline Acc | Best Acc | Delta Acc | Baseline F1 | Best F1 | Delta F1 | Baseline FP | Best FP | FP Delta | Baseline Yes Rate | Best Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | 0.1 | 77.33 | 79.33 | 2.00 | 77.78 | 80.62 | 2.85 | 37 | 41 | 4.00 | 52.00 | 56.67 |
| GQA | popular | 0.1 | 78.00 | 80.33 | 2.33 | 78.00 | 80.66 | 2.66 | 33 | 32 | -1.00 | 50.00 | 51.67 |
| GQA | random | 0.5 | 85.33 | 89.33 | 4.00 | 84.17 | 88.89 | 4.72 | 11 | 10 | -1.00 | 42.67 | 46.00 |
| MSCOCO | adversarial | 1.0 | 79.33 | 82.33 | 3.00 | 78.62 | 82.27 | 3.65 | 26 | 26 | 0.00 | 46.67 | 49.67 |
| MSCOCO | popular | 1.0 | 84.33 | 87.33 | 3.00 | 82.91 | 86.62 | 3.71 | 11 | 11 | 0.00 | 41.67 | 44.67 |
| MSCOCO | random | 1.0 | 83.67 | 86.00 | 2.33 | 82.31 | 85.42 | 3.11 | 13 | 15 | 2.00 | 42.33 | 46.00 |

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
| GQA | adversarial | CatExpert | 0.1 | 300 | 79.33 | 75.88 | 86.00 | 80.62 | 56.67 |
| GQA | adversarial | CatExpert | 0.3 | 300 | 77.00 | 73.41 | 84.67 | 78.64 | 57.67 |
| GQA | adversarial | CatExpert | 0.5 | 300 | 77.33 | 73.03 | 86.67 | 79.27 | 59.33 |
| GQA | adversarial | CatExpert | 1.0 | 300 | 75.33 | 70.21 | 88.00 | 78.11 | 62.67 |
| GQA | popular | Regular |  | 300 | 78.00 | 78.00 | 78.00 | 78.00 | 50.00 |
| GQA | popular | CatExpert | 0.0 | 300 | 78.00 | 77.63 | 78.67 | 78.15 | 50.67 |
| GQA | popular | CatExpert | 0.1 | 300 | 80.33 | 79.35 | 82.00 | 80.66 | 51.67 |
| GQA | popular | CatExpert | 0.3 | 300 | 79.33 | 75.88 | 86.00 | 80.62 | 56.67 |
| GQA | popular | CatExpert | 0.5 | 300 | 76.00 | 71.91 | 85.33 | 78.05 | 59.33 |
| GQA | popular | CatExpert | 1.0 | 300 | 75.33 | 70.21 | 88.00 | 78.11 | 62.67 |
| GQA | random | Regular |  | 300 | 85.33 | 91.41 | 78.00 | 84.17 | 42.67 |
| GQA | random | CatExpert | 0.0 | 300 | 84.00 | 88.06 | 78.67 | 83.10 | 44.67 |
| GQA | random | CatExpert | 0.1 | 300 | 86.00 | 89.13 | 82.00 | 85.42 | 46.00 |
| GQA | random | CatExpert | 0.3 | 300 | 88.67 | 90.85 | 86.00 | 88.36 | 47.33 |
| GQA | random | CatExpert | 0.5 | 300 | 89.33 | 92.75 | 85.33 | 88.89 | 46.00 |
| GQA | random | CatExpert | 1.0 | 300 | 88.33 | 88.59 | 88.00 | 88.29 | 49.67 |
| MSCOCO | adversarial | Regular |  | 300 | 79.33 | 81.43 | 76.00 | 78.62 | 46.67 |
| MSCOCO | adversarial | CatExpert | 0.0 | 300 | 77.67 | 79.43 | 74.67 | 76.98 | 47.00 |
| MSCOCO | adversarial | CatExpert | 0.1 | 300 | 80.00 | 82.14 | 76.67 | 79.31 | 46.67 |
| MSCOCO | adversarial | CatExpert | 0.3 | 300 | 81.33 | 82.19 | 80.00 | 81.08 | 48.67 |
| MSCOCO | adversarial | CatExpert | 0.5 | 300 | 79.00 | 80.42 | 76.67 | 78.50 | 47.67 |
| MSCOCO | adversarial | CatExpert | 1.0 | 300 | 82.33 | 82.55 | 82.00 | 82.27 | 49.67 |
| MSCOCO | popular | Regular |  | 300 | 84.33 | 91.20 | 76.00 | 82.91 | 41.67 |
| MSCOCO | popular | CatExpert | 0.0 | 300 | 82.00 | 87.50 | 74.67 | 80.58 | 42.67 |
| MSCOCO | popular | CatExpert | 0.1 | 300 | 83.67 | 89.15 | 76.67 | 82.44 | 43.00 |
| MSCOCO | popular | CatExpert | 0.3 | 300 | 85.67 | 90.23 | 80.00 | 84.81 | 44.33 |
| MSCOCO | popular | CatExpert | 0.5 | 300 | 85.00 | 92.00 | 76.67 | 83.64 | 41.67 |
| MSCOCO | popular | CatExpert | 1.0 | 300 | 87.33 | 91.79 | 82.00 | 86.62 | 44.67 |
| MSCOCO | random | Regular |  | 300 | 83.67 | 89.76 | 76.00 | 82.31 | 42.33 |
| MSCOCO | random | CatExpert | 0.0 | 300 | 82.33 | 88.19 | 74.67 | 80.87 | 42.33 |
| MSCOCO | random | CatExpert | 0.1 | 300 | 83.67 | 89.15 | 76.67 | 82.44 | 43.00 |
| MSCOCO | random | CatExpert | 0.3 | 300 | 85.33 | 89.55 | 80.00 | 84.51 | 44.67 |
| MSCOCO | random | CatExpert | 0.5 | 300 | 83.00 | 87.79 | 76.67 | 81.85 | 43.67 |
| MSCOCO | random | CatExpert | 1.0 | 300 | 86.00 | 89.13 | 82.00 | 85.42 | 46.00 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 119 | 113 | 37 | 31 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.0 | 117 | 113 | 37 | 33 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.1 | 129 | 109 | 41 | 21 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.3 | 127 | 104 | 46 | 23 | 0 | 300 |
| GQA | adversarial | CatExpert | 0.5 | 130 | 102 | 48 | 20 | 0 | 300 |
| GQA | adversarial | CatExpert | 1.0 | 132 | 94 | 56 | 18 | 0 | 300 |
| GQA | popular | Regular |  | 117 | 117 | 33 | 33 | 0 | 300 |
| GQA | popular | CatExpert | 0.0 | 118 | 116 | 34 | 32 | 0 | 300 |
| GQA | popular | CatExpert | 0.1 | 123 | 118 | 32 | 27 | 0 | 300 |
| GQA | popular | CatExpert | 0.3 | 129 | 109 | 41 | 21 | 0 | 300 |
| GQA | popular | CatExpert | 0.5 | 128 | 100 | 50 | 22 | 0 | 300 |
| GQA | popular | CatExpert | 1.0 | 132 | 94 | 56 | 18 | 0 | 300 |
| GQA | random | Regular |  | 117 | 139 | 11 | 33 | 0 | 300 |
| GQA | random | CatExpert | 0.0 | 118 | 134 | 16 | 32 | 0 | 300 |
| GQA | random | CatExpert | 0.1 | 123 | 135 | 15 | 27 | 0 | 300 |
| GQA | random | CatExpert | 0.3 | 129 | 137 | 13 | 21 | 0 | 300 |
| GQA | random | CatExpert | 0.5 | 128 | 140 | 10 | 22 | 0 | 300 |
| GQA | random | CatExpert | 1.0 | 132 | 133 | 17 | 18 | 0 | 300 |
| MSCOCO | adversarial | Regular |  | 114 | 124 | 26 | 36 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.0 | 112 | 121 | 29 | 38 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.1 | 115 | 125 | 25 | 35 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.3 | 120 | 124 | 26 | 30 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 0.5 | 115 | 122 | 28 | 35 | 0 | 300 |
| MSCOCO | adversarial | CatExpert | 1.0 | 123 | 124 | 26 | 27 | 0 | 300 |
| MSCOCO | popular | Regular |  | 114 | 139 | 11 | 36 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.0 | 112 | 134 | 16 | 38 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.1 | 115 | 136 | 14 | 35 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.3 | 120 | 137 | 13 | 30 | 0 | 300 |
| MSCOCO | popular | CatExpert | 0.5 | 115 | 140 | 10 | 35 | 0 | 300 |
| MSCOCO | popular | CatExpert | 1.0 | 123 | 139 | 11 | 27 | 0 | 300 |
| MSCOCO | random | Regular |  | 114 | 137 | 13 | 36 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.0 | 112 | 135 | 15 | 38 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.1 | 115 | 136 | 14 | 35 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.3 | 120 | 136 | 14 | 30 | 0 | 300 |
| MSCOCO | random | CatExpert | 0.5 | 115 | 134 | 16 | 35 | 0 | 300 |
| MSCOCO | random | CatExpert | 1.0 | 123 | 135 | 15 | 27 | 0 | 300 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- In `Official Regular vs Old HF Regular`, count fields such as FP are only directly comparable when `Same N=True`.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
