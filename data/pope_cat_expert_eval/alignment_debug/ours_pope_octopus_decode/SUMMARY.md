# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/alignment_debug/ours_pope_octopus_decode/summary.csv`
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
| MSCOCO | adversarial | Regular |  | 300 | 80.33 | 82.73 | 76.67 | 79.58 | 46.33 |
| MSCOCO | popular | Regular |  | 300 | 82.00 | 87.50 | 74.67 | 80.58 | 42.67 |
| MSCOCO | random | Regular |  | 300 | 83.67 | 89.76 | 76.00 | 82.31 | 42.33 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSCOCO | adversarial | Regular |  | 115 | 126 | 24 | 35 | 0 | 300 |
| MSCOCO | popular | Regular |  | 112 | 134 | 16 | 38 | 0 | 300 |
| MSCOCO | random | Regular |  | 114 | 137 | 13 | 36 | 0 | 300 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- In `Official Regular vs Old HF Regular`, count fields such as FP are only directly comparable when `Same N=True`.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
