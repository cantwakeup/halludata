# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/alignment_debug/ours_pope_ours_decode/summary.csv`
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
| MSCOCO | adversarial | Regular |  | 300 | 83.67 | 88.55 | 77.33 | 82.56 | 43.67 |
| MSCOCO | popular | Regular |  | 300 | 87.00 | 95.87 | 77.33 | 85.61 | 40.33 |
| MSCOCO | random | Regular |  | 300 | 86.33 | 94.31 | 77.33 | 84.98 | 41.00 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSCOCO | adversarial | Regular |  | 116 | 135 | 15 | 34 | 0 | 300 |
| MSCOCO | popular | Regular |  | 116 | 145 | 5 | 34 | 0 | 300 |
| MSCOCO | random | Regular |  | 116 | 143 | 7 | 34 | 0 | 300 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- In `Official Regular vs Old HF Regular`, count fields such as FP are only directly comparable when `Same N=True`.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
