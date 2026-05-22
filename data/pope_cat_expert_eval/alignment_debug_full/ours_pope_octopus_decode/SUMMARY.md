# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/alignment_debug_full/ours_pope_octopus_decode/summary.csv`
- Runs summarized: 3

## Run Config

- Config files found: `1`
- Runner: `official_llava`
- Model path: `/home/huiwei/sy/models/llava-v1.5-7b-official-clean`
- LLaVA repo: `/home/huiwei/sy/LLaVA-official-clean`
- Conv mode: `llava_v1`
- Prompt template: `{question} Please answer this question with one word.`
- Cat vector source: `unknown`
- Decode: `{"do_sample": true, "temperature": 1.0, "top_p": 1.0, "num_beams": 1, "max_new_tokens": 1024}`
- Steering: `{"layers": "5-25", "topk": 64, "head_select": "norm", "prefill": true, "decode": true, "apply_to": "last_token", "prefill_apply_to": "last_token", "decode_apply_to": "last_token", "enabled_experts": ["cat"]}`

## Main Table

| Dataset | Setting | Method | Alpha | N | Accuracy | Precision | Recall | F1 Score | Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSCOCO | adversarial | Regular |  | 3000 | 80.40 | 84.03 | 75.07 | 79.30 | 44.67 |
| MSCOCO | popular | Regular |  | 3000 | 82.60 | 89.76 | 73.60 | 80.88 | 41.00 |
| MSCOCO | random | Regular |  | 3000 | 83.73 | 92.24 | 73.67 | 81.91 | 39.93 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSCOCO | adversarial | Regular |  | 1126 | 1286 | 214 | 374 | 0 | 3000 |
| MSCOCO | popular | Regular |  | 1104 | 1374 | 126 | 396 | 0 | 3000 |
| MSCOCO | random | Regular |  | 1105 | 1407 | 93 | 395 | 0 | 3000 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- In `Official Regular vs Old HF Regular`, count fields such as FP are only directly comparable when `Same N=True`.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
