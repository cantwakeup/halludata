# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/official_llava_regular_full/summary.csv`
- Runs summarized: 6

## Run Config

- Config files found: `6`
- Runner: `official_llava`
- Model path: `/home/huiwei/sy/models/llava-v1.5-7b-official-clean`
- LLaVA repo: `/home/huiwei/sy/LLaVA-official-clean`
- Conv mode: `llava_v1`
- Prompt template: `{question} Please answer this question in one word.`
- Cat vector source: `unknown`
- Decode: `{"do_sample": false, "temperature": 0.0, "top_p": 1.0, "num_beams": 1, "max_new_tokens": 5}`
- Steering: `{"layers": "5-25", "topk": 64, "head_select": "norm", "prefill": true, "decode": true, "apply_to": "last_token", "prefill_apply_to": "last_token", "decode_apply_to": "last_token", "enabled_experts": ["cat"]}`

## Official Regular vs Old HF Regular

| Dataset | Setting | Official N | HF N | Same N | Official Acc | HF Acc | Acc Diff | Official F1 | HF F1 | F1 Diff | Official FP | HF FP | Official Yes Rate | HF Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | 3000 | 3000 | True | 80.80 | 81.30 | -0.50 | 81.55 | 81.44 | 0.11 | 349 | 292 | 54.07 | 50.77 |
| GQA | popular | 3000 | 3000 | True | 84.00 | 84.17 | -0.17 | 84.14 | 83.83 | 0.31 | 253 | 206 | 50.87 | 47.90 |
| GQA | random | 3000 | 3000 | True | 89.50 | 88.50 | 1.00 | 88.99 | 87.71 | 1.28 | 88 | 76 | 45.37 | 43.57 |
| MSCOCO | adversarial | 3000 | 3000 | True | 83.57 | 83.70 | -0.13 | 82.26 | 82.08 | 0.18 | 136 | 109 | 42.63 | 40.97 |
| MSCOCO | popular | 3000 | 3000 | True | 85.80 | 85.67 | 0.13 | 84.29 | 83.90 | 0.40 | 69 | 50 | 40.40 | 39.00 |
| MSCOCO | random | 3000 | 3000 | True | 87.07 | 86.50 | 0.57 | 85.49 | 84.69 | 0.80 | 31 | 25 | 39.13 | 38.17 |

## Main Table

| Dataset | Setting | Method | Alpha | N | Accuracy | Precision | Recall | F1 Score | Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 3000 | 80.80 | 78.48 | 84.87 | 81.55 | 54.07 |
| GQA | popular | Regular |  | 3000 | 84.00 | 83.42 | 84.87 | 84.14 | 50.87 |
| GQA | random | Regular |  | 3000 | 89.50 | 93.53 | 84.87 | 88.99 | 45.37 |
| MSCOCO | adversarial | Regular |  | 3000 | 83.57 | 89.37 | 76.20 | 82.26 | 42.63 |
| MSCOCO | popular | Regular |  | 3000 | 85.80 | 94.31 | 76.20 | 84.29 | 40.40 |
| MSCOCO | random | Regular |  | 3000 | 87.07 | 97.36 | 76.20 | 85.49 | 39.13 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA | adversarial | Regular |  | 1273 | 1151 | 349 | 227 | 0 | 3000 |
| GQA | popular | Regular |  | 1273 | 1247 | 253 | 227 | 0 | 3000 |
| GQA | random | Regular |  | 1273 | 1412 | 88 | 227 | 0 | 3000 |
| MSCOCO | adversarial | Regular |  | 1143 | 1364 | 136 | 357 | 0 | 3000 |
| MSCOCO | popular | Regular |  | 1143 | 1431 | 69 | 357 | 0 | 3000 |
| MSCOCO | random | Regular |  | 1143 | 1469 | 31 | 357 | 0 | 3000 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- In `Official Regular vs Old HF Regular`, count fields such as FP are only directly comparable when `Same N=True`.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
