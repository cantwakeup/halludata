# Cloud Run Checklist

## A. Local Smoke Run Order

1. Run `python scripts/build_shell_bank.py`
2. Run `python scripts/build_cat_neg_bank.py`
3. Run `python scripts/build_coco_fact_index.py`
4. Run `python scripts/render_pairs.py --config configs/v0_mini.yaml`
5. Run `python -m unittest discover -s tests -p "test*.py"`

## B. Cloud 50-Image Smoke Run Order

1. Set `configs/v0_mini.yaml` or CLI flags for real `instances_json`
2. Optionally set `panoptic_json`, `panoptic_root`, and `image_root`
3. Run `python scripts/build_coco_fact_index.py --instances-json <path> --max-images 50`
4. Run `python scripts/render_pairs.py --config configs/v0_mini.yaml`
5. Inspect `fact_index_v0.jsonl`, `atomic_facts_v0.jsonl`, `pairs_balanced_v0.jsonl`, and `pair_stats_v0.json`

## C. Cloud 500-Image v0-Mini Run Order

1. Confirm shell bank and category negative bank are already built
2. Run `python scripts/build_coco_fact_index.py --instances-json <path> --max-images 500 --resume`
3. Run `python scripts/render_pairs.py --config configs/v0_mini.yaml`
4. Re-run `python scripts/render_pairs.py --config configs/v0_mini.yaml --stats-only` after any manual inspection edits

## D. Output Files To Check

- `data/outputs/fact_index_v0.jsonl`
- `data/outputs/atomic_facts_v0.jsonl`
- `data/outputs/pairs_unbalanced_v0.jsonl`
- `data/outputs/pairs_balanced_v0.jsonl`
- `data/outputs/pair_stats_v0.json`

## E. Stats To Watch Closely

- `counts_before_filter`
- `counts_after_filter`
- `counts_unbalanced`
- `counts_balanced`
- `dropped_by_reason`
- `per_image_pair_counts`
- `template_usage`

## F. Debugging Tips

If `col=0`:

- Check whether `use_panoptic` and `use_images_for_color` are enabled
- Confirm `panoptic_json`, `panoptic_root`, and `image_root` all point to real files
- Inspect `dropped_by_reason.missing_color`

If `rel` is very small:

- Inspect `dropped_by_reason.same_category_relation`
- Inspect `dropped_by_reason.ambiguous_rel_anchor`
- Check whether object counts are mostly greater than `1`, which will block pure relation anchors

If `cat` dominates everything:

- Compare `counts_before_filter.cat` vs `counts_after_filter.cat`
- Inspect `per_label_cap` and subtype `targets`
- Remember that unique-category filtering reduces ambiguity, but not all prompt ambiguity without crop or referring-expression support
