# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/alignment_debug_full/ours_pope_ours_decode/summary.csv`
- Runs summarized: 3

## Run Config

- Config files found: `1`
- Runner: `official_llava`
- Model path: `/home/huiwei/sy/models/llava-v1.5-7b-official-clean`
- LLaVA repo: `/home/huiwei/sy/LLaVA-official-clean`
- Conv mode: `llava_v1`
- Prompt template: `{question} Please answer this question in one word.`
- Cat vector source: `unknown`
- Decode: `{"do_sample": false, "temperature": 0.0, "top_p": 1.0, "num_beams": 1, "max_new_tokens": 5}`
- Steering: `{"layers": "5-25", "topk": 64, "head_select": "norm", "prefill": true, "decode": true, "apply_to": "last_token", "prefill_apply_to": "last_token", "decode_apply_to": "last_token", "enabled_experts": ["cat"]}`

## Main Table

| Dataset | Setting | Method | Alpha | N | Accuracy | Precision | Recall | F1 Score | Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSCOCO | adversarial | Regular |  | 3000 | 83.57 | 89.37 | 76.20 | 82.26 | 42.63 |
| MSCOCO | popular | Regular |  | 3000 | 85.80 | 94.31 | 76.20 | 84.29 | 40.40 |
| MSCOCO | random | Regular |  | 3000 | 87.07 | 97.36 | 76.20 | 85.49 | 39.13 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
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
