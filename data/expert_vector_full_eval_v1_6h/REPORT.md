# Expert Vector Full Eval V1

## Goal

Simplified vector-only full benchmark evaluation. Subtype masks, expert masks, and routing are intentionally disabled.

## Evaluation Settings

- Direction: vector itself (`global`, `cat`, `attr`, `rel`).
- Head selection: vector norm top64 over all available 32 layers.
- Hook: official LLaVA decoder self-attention `o_proj` forward pre-hook.
- Apply: prefill=true, decode=true, apply_to=last_token.
- Decoding: do_sample=true, temperature=1.0, top_p=1.0, num_beams=1, max_new_tokens=1024, seed=42.

## Matrix

See `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/EXPERT_MATRIX_REPORT.md`.

## Decision

`FAIL`

- POPE/category winner: `attr`.
- AMBER-attribute winner: `cat`.
- GQA/clean-relation winner: `cat`.
- `rel` on `GQA/clean-relation` is suspicious due to yes_rate=0.670.
- Off-diagonal or weak results dominate; do not proceed to router/DPO from these vectors.

## Changed Cases

- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/POPE_category_avg_global.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/AMBER-attribute_global.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/GQA_clean-relation_global.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/POPE_category_avg_cat.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/AMBER-attribute_cat.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/GQA_clean-relation_cat.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/POPE_category_avg_attr.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/AMBER-attribute_attr.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/GQA_clean-relation_attr.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/POPE_category_avg_rel.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/AMBER-attribute_rel.jsonl`
- `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1_6h/changed_cases/GQA_clean-relation_rel.jsonl`
