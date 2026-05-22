# Official LLaVA POPE CatExpert Summary

- Summary CSV: `data/pope_cat_expert_eval/official_llava_regular_smoke_mscoco_adv100/summary.csv`
- Runs summarized: 1

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

## Official Regular vs Old HF Regular

| Dataset | Setting | Official Acc | HF Acc | Acc Diff | Official F1 | HF F1 | F1 Diff | Official FP | HF FP | Official Yes Rate | HF Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSCOCO | adversarial | 83.00 | 83.70 | -0.70 | 82.83 | 82.08 | 0.75 | 8 | 109 | 49.00 | 40.97 |

## Main Table

| Dataset | Setting | Method | Alpha | N | Accuracy | Precision | Recall | F1 Score | Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSCOCO | adversarial | Regular |  | 100 | 83.00 | 83.67 | 82.00 | 82.83 | 49.00 |

## Debug Counts

| Dataset | Setting | Method | Alpha | TP | TN | FP | FN | Invalid | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSCOCO | adversarial | Regular |  | 41 | 42 | 8 | 9 | 0 | 100 |

## Notes

- Positive class is `yes`, meaning the queried object exists.
- `FP` is object hallucination: label=no, pred=yes.
- Invalid predictions are counted as wrong in Accuracy/Precision/Recall/F1.
- `Alpha 0 Check` should be exactly or nearly identical to Regular if the hook adds a zero vector.
- If the cat vector was built from the old HF activation space, treat CatExpert results as diagnostic until an official-loader vector is rebuilt.
