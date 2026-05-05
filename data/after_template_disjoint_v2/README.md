# AFTER-Template Disjoint V2

This dataset line is a more AFTER-aligned replacement for `after_template_disjoint_v1`.
It keeps image-level disjointness across `cat`, `attr`, and `rel`, but changes the prompt and trusted-text construction.

## Main Changes

- The visual side is still image + question.
- The trusted side is now an AFTER-style image description prompt:
  `The given image depicts the following scene: ... Please directly answer ...`
- `cat` and `attr` are built from COCO annotations.
- `rel` can be built from an external relation source such as VG/GQA/AMBER-style JSON/JSONL.
- If no external relation source is provided, `rel` falls back to the existing high-confidence COCO bbox relation builder.
- External `rel` rows filter self-referential pairs such as `clouds` vs `clouds`.
- External `rel` rows are sampled with per-image bucket balancing so left/right relations do not dominate all selected rows.

## Intended Command

```bash
python scripts/build_after_template_disjoint_v2_pairs.py \
  --coco-instances /home/huiwei/sy/sy_data/COCO2014/annotations/instances_train2014.json \
  --image-root /home/huiwei/sy/sy_data/COCO2014/train2014 \
  --relation-source /path/to/relation_annotations.jsonl \
  --relation-image-root /path/to/relation/images \
  --output-dir data/after_template_disjoint_v2/pairs \
  --num-images 5000 \
  --type-image-ratio cat=0.3,attr=0.3,rel=0.4 \
  --seed 42 \
  --split-ratio 0.6,0.2,0.2 \
  --relation-bucket-ratio horizontal=0.5,vertical=0.1,depth=0.15,contact=0.15,interaction=0.1,semantic=0.0 \
  --overwrite
```

If `--relation-source` is omitted, the script uses COCO bbox fallback relation rows.
