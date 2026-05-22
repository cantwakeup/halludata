# Clean Type Minimal-Pair v2 Inspection

## Data Sources

- GQA train scene graph: `/home/huiwei/sy/sy_data/GQA/raw/sceneGraphs/train_sceneGraphs.json`
- GQA val scene graph: `/home/huiwei/sy/sy_data/GQA/raw/sceneGraphs/val_sceneGraphs.json`
- GQA image root argument: `/home/huiwei/sy/sy_data/GQA/raw/images/images`
- Resolved image roots: `['/home/huiwei/sy/sy_data/GQA/raw/images/images', '/home/huiwei/sy/sy_data/GQA/raw/images']`
- Train records with images: `74289`
- Val records with images: `10568`

## GQA Object Field Example

```json
{
  "object_id": "681267",
  "name": "banana",
  "attributes": [
    "small",
    "yellow"
  ],
  "relations": [
    {
      "object_id": "681262",
      "relation": "left of"
    }
  ],
  "bbox": {
    "x": 248.0,
    "y": 55.0,
    "w": 64.0,
    "h": 34.0
  }
}
```

## GQA Attribute Field Example

```json
[
  "small",
  "yellow"
]
```

## GQA Relation Field Example

```json
{
  "object_id": "681262",
  "relation": "left of"
}
```

## BBox Format

```json
{
  "x": 248.0,
  "y": 55.0,
  "w": 64.0,
  "h": 34.0
}
```

## Reusable Scripts

- `scripts/analyze_amber_attribute_subtypes.py`
- `scripts/analyze_gqa_typeaware_diagnostic.py`
- `scripts/assemble_subtype_mask_report.py`
- `scripts/build_clean_type_minpair_v2.py`
- `scripts/build_coco_fact_index.py`
- `scripts/build_gqa_global_residual_vectors.py`
- `scripts/build_gqa_typeaware_data.py`
- `scripts/build_subtype_head_masks.py`
- `scripts/build_subtype_minpair_data.py`
- `scripts/build_subtype_vectors.py`
- `scripts/eval_subtype_mask_steering.py`
- `scripts/extract_subtype_minpair_activations.py`
- `scripts/inspect_subtype_mask_inputs.py`
- `scripts/merge_subtype_mask_eval_parts.py`
- `scripts/merge_subtype_minpair_activations.py`
- `scripts/run_subtype_minpair_eval.py`
- `scripts/summarize_gqa_typeaware_eval.py`
- `scripts/summarize_subtype_mask_eval.py`
- `scripts/summarize_subtype_minpair_eval_shards.py`
- `scripts/summarize_subtype_minpair_experiment.py`
- `scripts/write_subtype_mask_data_quality_notes.py`

## Data Quality Risks

- GQA attributes can be sparse or ambiguous for material/shape/state.
- Bbox-derived relations need strict margin filtering; ambiguous pairs are filtered.
- Interaction counterfacts are conservative to avoid unnatural pairs such as wearing umbrella.
- Count questions rely on object instance names; pluralization is explicitly audited.

## Loader Audit Counters

| item | count |
| --- | --- |
| empty_objects_after_filter | 781 |