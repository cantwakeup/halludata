# Official LLaVA POPE Regular Diagnostic

- Summary CSV: `data/pope_cat_expert_eval/official_llava_diagnostics_cleanenv_coco_adv100/summary.csv`
- Limit per dataset/setting: `100`
- HF comparison root: `data/pope_cat_expert_eval/full_alpha_sweep`
- Prompt suffix: `Please answer this question in one word.`
- Conversation mode: `llava_v1`
- Steering/hooks: disabled; Regular baseline only.
- Decode: `temperature=0`, `top_p=1.0`, `do_sample=False`, `num_beams=1`, `max_new_tokens=5` unless overridden.

## Main Comparison

| Dataset | Setting | Method | N | Accuracy | Precision | Recall | F1 Score | Yes Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSCOCO | adversarial | HFRunner-Regular | 100 | 82.00 | 84.78 | 78.00 | 81.25 | 46.00 |
| MSCOCO | adversarial | OfficialLLaVA-Regular | 100 | 80.00 | 81.25 | 78.00 | 79.59 | 48.00 |
| MSCOCO | adversarial | Official-minus-HF | 100 | -2.00 | -3.53 | 0.00 | -1.66 | 2.00 |

## Debug Counts

| Dataset | Setting | Method | TP | TN | FP | FN | Invalid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MSCOCO | adversarial | HFRunner-Regular | 39 | 43 | 7 | 11 | 0 |
| MSCOCO | adversarial | OfficialLLaVA-Regular | 39 | 41 | 9 | 11 | 0 |
| MSCOCO | adversarial | Official-minus-HF | 0 | -2 | 2 | 0 | 0 |

## Prompt Examples

```json
[
  {
    "dataset": "MSCOCO",
    "setting": "adversarial",
    "index": 0,
    "original_question": "Is there a snowboard in the image?",
    "final_question": "Is there a snowboard in the image? Please answer this question in one word.",
    "question_with_image": "<image>\nIs there a snowboard in the image? Please answer this question in one word.",
    "full_prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nIs there a snowboard in the image? Please answer this question in one word. ASSISTANT:",
    "template_info": {
      "raw_full_output": "No",
      "prompt_token_len": 60,
      "output_token_len": 3,
      "conv_mode": "llava_v1",
      "roles": [
        "USER",
        "ASSISTANT"
      ],
      "sep": " ",
      "sep2": "</s>",
      "sep_style": "SeparatorStyle.TWO",
      "system": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
      "stop_str": "</s>",
      "mm_use_im_start_end": false
    }
  },
  {
    "dataset": "MSCOCO",
    "setting": "adversarial",
    "index": 1,
    "original_question": "Is there a backpack in the image?",
    "final_question": "Is there a backpack in the image? Please answer this question in one word.",
    "question_with_image": "<image>\nIs there a backpack in the image? Please answer this question in one word.",
    "full_prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nIs there a backpack in the image? Please answer this question in one word. ASSISTANT:",
    "template_info": {
      "raw_full_output": "No",
      "prompt_token_len": 60,
      "output_token_len": 3,
      "conv_mode": "llava_v1",
      "roles": [
        "USER",
        "ASSISTANT"
      ],
      "sep": " ",
      "sep2": "</s>",
      "sep_style": "SeparatorStyle.TWO",
      "system": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
      "stop_str": "</s>",
      "mm_use_im_start_end": false
    }
  },
  {
    "dataset": "MSCOCO",
    "setting": "adversarial",
    "index": 2,
    "original_question": "Is there a person in the image?",
    "final_question": "Is there a person in the image? Please answer this question in one word.",
    "question_with_image": "<image>\nIs there a person in the image? Please answer this question in one word.",
    "full_prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nIs there a person in the image? Please answer this question in one word. ASSISTANT:",
    "template_info": {
      "raw_full_output": "Yes",
      "prompt_token_len": 59,
      "output_token_len": 3,
      "conv_mode": "llava_v1",
      "roles": [
        "USER",
        "ASSISTANT"
      ],
      "sep": " ",
      "sep2": "</s>",
      "sep_style": "SeparatorStyle.TWO",
      "system": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
      "stop_str": "</s>",
      "mm_use_im_start_end": false
    }
  }
]
```

## Interpretation Hint

- If `OfficialLLaVA-Regular` has higher `FP` and higher `Yes Rate` than `HFRunner-Regular`, the previous high baseline likely comes from the HF loader/processor or prompt wrapper being more conservative.
- If `OfficialLLaVA-Regular` is similar to `HFRunner-Regular`, the main gap is more likely POPE annotation version / negative sampling, model checkpoint, or parser rather than conversation template alone.
