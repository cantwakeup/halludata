# Repo Audit: Typed FAS Next

Date: 2026-07-08

## Scope

This audit covers the current AFTER/FAS typed-vector work in `/home/huiwei/sy/halludata`, with emphasis on cached artifacts that can support low-cost next experiments without API calls or fresh benchmark generation.

## High-Level Layout

| area | relevant paths | notes |
| --- | --- | --- |
| Base typed FAS data | `data/after_fas_type_v1/` | Structured COCO facts, deterministic template text, verbalizer prompts. No activations in this original root. |
| Typed250 LVLM text run | `data/after_fas_type_v1_gpt4omini_typed250_text/` | Main current typed-caption run: 250 disjoint images per cat/attr/rel, GPT-4o-mini factual descriptions applied, official LLaVA activations/vectors. |
| Current raw vectors | `data/after_fas_type_v1_gpt4omini_typed250_text/official_llava_vectors/` | `raw_type_vectors.pt` has `catraw`, `attrraw`, `relraw`; `runtime_raw_type_vectors.pt` has `cat`, `attr`, `rel`, `global`. |
| Raw benchmark results | `data/after_fas_type_v1_gpt4omini_typed250_text/bench_sanity_raw_injection_v2_large/` | Large POPE/AMBER/GQA sanity run for raw vectors. |
| Clean minimal-pair v2 | `data/clean_type_minpair_v2/` | GQA-derived correct/counterfact minimal pairs, cached activations, condition-balanced vectors, heldout mask eval. |
| Subtype mask work | `data/subtype_mask_steering_v1/`, `data/subtype_minpair_v1/` | Prior subtype mask artifacts and larger heldout subtype eval residue. |
| Benchmarks | `data/benchmarks/`, runner scripts under `scripts/` | POPE/AMBER/GQA inputs and steering/eval helpers. |

## Current Typed250 Data

`data/after_fas_type_v1_gpt4omini_typed250_text/REPORT.md` says:

| mode | rows |
| --- | ---: |
| fas_cat | 1387 |
| fas_attr | 2000 |
| fas_rel | 3584 |

The pair JSONL files include fields needed for future anchor construction, such as `target_object`, `target_attribute`, `target_relation`, `source_fact_id`, `facts`, `trusted_text`, `visual_prompt`, and `trusted_prompt`.

The facts cache has object, attribute, and relation statements:

- category facts: object and count with sentences such as `There is one cup in the image.`
- attribute facts: color/shape/count with object-specific sentences such as `The cup is gray.`
- relation facts: subject, relation, object, bbox geometry, and sentences such as `The cup is to the right of the cake.`

This is enough to build anchor text rows, but not enough to build anchor-delta vectors without a new text-only activation extraction pass.

## Vector Format

`runtime_raw_type_vectors.pt` is a torch mapping with:

- `vectors`: keys `cat`, `attr`, `rel`, `global`
- each tensor shape: `[32, 32, 128]`
- `layers`: 32 layer ids
- `num_heads`: 32
- `head_dim`: 128
- `hidden_size`: 4096

`raw_type_vectors.pt` is similar, but uses raw keys `catraw`, `attrraw`, `relraw`.

`data/clean_type_minpair_v2/vectors/condition_vectors.pt` contains coarse and subtype correct-vs-counterfact directions:

- coarse: `g_all_clean`, `g_attr_clean`, `g_rel_clean`
- subtype: `s_attr_color_clean_clean`, `s_attr_count_clean_clean`, `s_attr_shape_clean_clean`, etc.
- `yesno_direction`
- same `[32, 32, 128]` vector shape.

## Important Scripts

| task | script | notes |
| --- | --- | --- |
| Build typed facts/pairs | `scripts/build_after_fas_type_data.py` | COCO fact extraction and type-separated FAS data. |
| Build disjoint splits | `scripts/build_after_fas_type_disjoint_splits.py` | 250-image disjoint type split. |
| Verbalizer inputs | `scripts/build_after_fas_type_verbalizer_inputs.py` | Creates prompts for factual descriptions. |
| API verbalization | `scripts/run_after_fas_type_api_verbalization.py` | External API helper; not used in this pass. |
| Local LLaVA verbalization | `scripts/run_after_fas_type_official_llava_verbalization.py` | Local captioning helper. |
| Apply verbalized text | `scripts/apply_after_fas_type_verbalized_texts.py` | Creates LVLM-text pair root. |
| Build raw vectors | `scripts/build_after_fas_type_raw_vectors.py` | Mean `z_text - z_visual`. |
| Runtime vectors | `scripts/build_after_fas_type_runtime_vectors.py` | Converts raw keys to runtime keys and adds global. |
| Raw diagnostics | `scripts/diagnose_after_fas_type_heads.py` | Pairwise cosine, topK norm overlap, restricted cosine. |
| Benchmark runner | `scripts/run_steered_benchmark.py` | HF LLaVA yes/no runner with `norm`, `random`, `all`, and `expert_map` head selection. |
| Benchmark summarizer | `scripts/summarize_after_fas_type_bench_sanity.py` | Aggregates raw-vector sanity results. |
| Clean minpair data | `scripts/build_clean_type_minpair_v2.py` | GQA minimal-pair construction. |
| Clean minpair activations | `scripts/extract_clean_type_minpair_activations.py` | Extracts `z_visual`, `z_fact_text`, `z_counterfact_text`. |
| Clean minpair vectors | `scripts/build_clean_type_condition_vectors.py` | Condition-balanced correct-vs-counterfact vectors and masks. |
| Subtype mask eval | `scripts/eval_subtype_mask_steering.py` | Heldout subtype mask benchmark. |

## Head Selection Logic

The benchmark controller in `src/expert_data/steering.py` supports:

- `norm`: top-K by active vector head norm.
- `random`: deterministic shuffled layer/head pairs.
- `all`: all requested heads.
- `expert_map`: external JSON mapping expert keys to `[layer, head]` rows.

This means new head selection strategies can be generated as JSON maps without modifying the benchmark runner.

## Current Result Generation Locations

| output | path |
| --- | --- |
| raw vector report | `data/after_fas_type_v1_gpt4omini_typed250_text/official_llava_vectors/raw_type_vectors.REPORT.md` |
| raw top64 report | `data/after_fas_type_v1_gpt4omini_typed250_text/official_llava_vectors/raw_type_vectors.head_diagnostics.top64.REPORT.md` |
| raw large benchmark | `data/after_fas_type_v1_gpt4omini_typed250_text/bench_sanity_raw_injection_v2_large/LARGE_RUN_REPORT.md` |
| clean minpair data report | `data/clean_type_minpair_v2/minimal_pairs/DATA_REPORT.md` |
| clean minpair vector report | `data/clean_type_minpair_v2/vectors/CONDITION_VECTOR_REPORT.md` |
| clean minpair heldout mask eval | `data/clean_type_minpair_v2/eval/heldout_mask_limit100_seed42/MASK_EVAL_REPORT.md` |

## Dev/Small Benchmark Availability

Existing small/dev-style evaluation artifacts exist:

- `data/clean_type_minpair_v2/eval/heldout_mask_limit100_seed42/summary.csv`
- `data/clean_type_minpair_v2/eval/heldout_mask_smoke20_seed42/run_specs.clean_v2.jsonl`
- prior 100-sample heldout mask report in `MASK_EVAL_REPORT.md`

The general benchmark runner supports small limits via `--limit`, but a real run still requires loading LLaVA on GPU. At audit time, `nvidia-smi` showed all four GPUs at high utilization (`91-100%`) with about `58 GB` already used on each card, so this pass did not start a fragile GPU dev benchmark.

## Existing Clean Minimal-Pair v2 Data

`data/clean_type_minpair_v2/minimal_pairs/train.jsonl` and `val.jsonl` use a useful schema:

- `base_scene`
- `target_fact`
- `target_counterfact`
- `fact_text`
- `counterfact_text`
- `trusted_prompt_fact`
- `trusted_prompt_counterfact`
- `condition_key`
- subtype labels such as `attr_color_clean`, `attr_count_clean`, `rel_left_right_clean`

This already supports correct-vs-wrong direction construction and has cached activations/vectors.

## Audit Conclusion

The codebase already contains the necessary cached artifacts for vector-only contrast, PCA/common-subspace removal, discriminative head selection, and clean minimal-pair review. Anchor cancellation is feasible from existing facts/pairs, but requires a new text-only activation extraction pass for typed and anchor statements before vectors can be constructed.
