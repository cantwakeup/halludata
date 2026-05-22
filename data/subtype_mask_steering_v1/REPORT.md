# Type/Subtype-Specific Mask Steering Report

## Goal

This experiment tests whether expert separation is better expressed as type/subtype-specific intervention masks rather than distinct steering directions.

The intervention is:

```text
h[l,h] <- h[l,h] + alpha * M_subtype[l,h] * g_type[l,h]
```

- Direction: `g_all_clean` or `g_type_clean` from subtype minimal-pair vectors.
- Mask: top64 heads over all 32 layers, selected by subtype sample-level `s_delta = z_fact_text - z_counterfact_text` energy.
- Loader: official LLaVA only; no old HF runner.

## Inspection Summary

# Subtype Mask Steering Input Inspection

## Paths

- Train JSONL: `data/subtype_minpair_v1/minimal_pairs/train.jsonl`
- Val JSONL: `data/subtype_minpair_v1/minimal_pairs/val.jsonl`
- Activations: `data/subtype_minpair_v1/activations/train_activations.pt`
- Vectors: `data/subtype_minpair_v1/vectors/subtype_vectors.pt`

## Train Counts

| subtype | count | yes | no | sources |
| --- | --- | --- | --- | --- |
| attr_color | 500 | 250 | 250 | {'gqa': 500} |
| attr_count | 500 | 250 | 250 | {'gqa': 500} |
| cat_hard | 600 | 300 | 300 | {'coco': 600} |
| cat_popular | 600 | 300 | 300 | {'coco': 600} |
| cat_random | 600 | 300 | 300 | {'coco': 600} |
| rel_contact | 500 | 250 | 250 | {'gqa': 500} |
| rel_spatial | 500 | 250 | 250 | {'gqa_bbox_derived': 500} |

## Val Counts

| subtype | count | yes | no | sources |
| --- | --- | --- | --- | --- |
| attr_color | 200 | 100 | 100 | {'gqa': 200} |
| attr_count | 200 | 100 | 100 | {'gqa': 200} |
| cat_hard | 200 | 100 | 100 | {'coco': 200} |
| cat_popular | 200 | 100 | 100 | {'coco': 200} |
| cat_random | 200 | 100 | 100 | {'coco': 200} |
| rel_contact | 200 | 100 | 100 | {'gqa': 200} |
| rel_spatial | 200 | 100 | 100 | {'gqa_bbox_derived': 200} |

## Expert-Type Image Overlap In Train

| type_a | type_b | overlap |
| --- | --- | --- |
| attr | attr | 496 |
| attr | cat | 0 |
| attr | rel | 0 |
| cat | attr | 0 |
| cat | cat | 300 |
| cat | rel | 0 |
| rel | attr | 0 |
| rel | cat | 0 |
| rel | rel | 494 |

## Activation Payload Schema

```json
{
  "z_visual": {
    "type": "Tensor",
    "shape": [
      3800,
      32,
      32,
      128
    ],
    "dtype": "torch.float16"
  },
  "z_fact_text": {
    "type": "Tensor",
    "shape": [
      3800,
      32,
      32,
      128
    ],
    "dtype": "torch.float16"
  },
  "z_counterfact_text": {
    "type": "Tensor",
    "shape": [
      3800,
      32,
      32,
      128
    ],
    "dtype": "torch.float16"
  },
  "metadata": {
    "type": "list",
    "len": 3800
  },
  "schema": {
    "type": "dict",
    "len": 18,
    "keys": [
      "script",
      "model_path",
      "model_name",
      "context_len",
      "llava_repo_path",
      "conv_mode",
      "storage_dtype",
      "num_layers",
      "num_heads",
      "head_dim",
      "shape",
      "counts_by_subtype",
      "num_shards",
      "shard_index",
      "source_jsonl",
      "branch_definitions",
      "num_shards_merged",
      "shard_files"
    ]
  }
}
```

## Vector Payload Schema

```json
{
  "g_all_raw": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "g_cat_raw": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "g_attr_raw": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "g_rel_raw": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_cat_random_raw": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_cat_popular_raw": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_cat_hard_raw": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_attr_color_raw": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_attr_count_raw": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_rel_spatial_raw": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_rel_contact_raw": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "g_all_clean": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "g_cat_clean": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "g_attr_clean": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "g_rel_clean": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_cat_random_clean": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_cat_popular_clean": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_cat_hard_clean": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_attr_color_clean": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_attr_count_clean": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_rel_spatial_clean": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "s_rel_contact_clean": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "yesno_direction": {
    "type": "Tensor",
    "shape": [
      32,
      32,
      128
    ],
    "dtype": "torch.float32"
  },
  "composed": {
    "type": "dict",
    "len": 35,
    "keys": [
      "d_cat_random_g_only_clean",
      "d_cat_random_s_only_clean",
      "d_cat_random_g1_s025_clean",
      "d_cat_random_g1_s05_clean",
      "d_cat_random_g1_s1_clean",
      "d_cat_popular_g_only_clean",
      "d_cat_popular_s_only_clean",
      "d_cat_popular_g1_s025_clean",
      "d_cat_popular_g1_s05_clean",
      "d_cat_popular_g1_s1_clean",
      "d_cat_hard_g_only_clean",
      "d_cat_hard_s_only_clean",
      "d_cat_hard_g1_s025_clean",
      "d_cat_hard_g1_s05_clean",
      "d_cat_hard_g1_s1_clean",
      "d_attr_color_g_only_clean",
      "d_attr_color_s_only_clean",
      "d_attr_color_g1_s025_clean",
      "d_attr_color_g1_s05_clean",
      "d_attr_color_g1_s1_clean",
      "d_attr_count_g_only_clean",
      "d_attr_count_s_only_clean",
      "d_attr_count_g1_s025_clean",
      "d_attr_count_g1_s05_clean",
      "d_attr_count_g1_s1_clean",
      "d_rel_spatial_g_only_clean",
      "d_rel_spatial_s_only_clean",
      "d_rel_spatial_g1_s025_clean",
      "d_rel_spatial_g1_s05_clean",
      "d_rel_spatial_g1_s1_clean",
      "d_rel_contact_g_only_clean",
      "d_rel_contact_s_only_clean",
      "d_rel_contact_g1_s025_clean",
      "d_rel_contact_g1_s05_clean",
      "d_rel_contact_g1_s1_clean"
    ]
  },
  "vectors": {
    "type": "dict",
    "len": 104,
    "keys": [
      "g_all_raw",
      "g_cat_raw",
      "g_attr_raw",
      "g_rel_raw",
      "s_cat_random_raw",
      "s_cat_popular_raw",
      "s_cat_hard_raw",
      "s_attr_color_raw",
      "s_attr_count_raw",
      "s_rel_spatial_raw",
      "s_rel_contact_raw",
      "g_all_clean",
      "g_cat_clean",
      "g_attr_clean",
      "g_rel_clean",
      "s_cat_random_clean",
      "s_cat_popular_clean",
      "s_cat_hard_clean",
      "s_attr_color_clean",
      "s_attr_count_clean",
      "s_rel_spatial_clean",
      "s_rel_contact_clean",
      "g_all_mean_only",
      "g_cat_mean_only",
      "g_attr_mean_only",
      "g_rel_mean_only",
      "s_cat_random_mean_only",
      "s_cat_popular_mean_only",
      "s_cat_hard_mean_only",
      "s_attr_color_mean_only",
      "s_attr_count_mean_only",
      "s_rel_spatial_mean_only",
      "s_rel_contact_mean_only",
      "d_cat_random_g_only_clean",
      "d_cat_random_s_only_clean",
      "d_cat_random_g1_s025_clean",
      "d_cat_random_g1_s05_clean",
      "d_cat_random_g1_s1_clean",
      "d_cat_popular_g_only_clean",
      "d_cat_popular_s_only_clean"
    ]
  },
  "layers": {
    "type": "list",
    "len": 32
  },
  "num_heads": {
    "type": "int"
  },
  "head_dim": {
    "type": "int"
  },
  "hidden_size": {
    "type": "int"
  },
  "metadata": {
    "type": "dict",
    "len": 11,
    "keys": [
      "created_by",
      "source_activations",
      "sample_normalize",
      "denoise_method",
      "svd_k",
      "remove_yesno",
      "yesno_mode",
      "shuffle_subtype_labels",
      "counts_by_subtype",
      "counts_by_expert_type",
      "vector_shape"
    ]
  },
  "diagnostics": {
    "type": "dict",
    "len": 3,
    "keys": [
      "denoise",
      "yesno_projection",
      "yesno_info"
    ]
  }
}
```

## Vector Keys

| key | shape | dtype | flat_norm |
| --- | --- | --- | --- |
| d_attr_color_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.4023 |
| d_attr_color_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_attr_color_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.4023 |
| d_attr_color_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_attr_color_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.4023 |
| d_attr_color_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_attr_color_g_only_clean | [32, 32, 128] | torch.float32 | 0.4023 |
| d_attr_color_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_attr_color_s_only_clean | [32, 32, 128] | torch.float32 | 0.4023 |
| d_attr_color_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_attr_count_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.5894 |
| d_attr_count_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_attr_count_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.5894 |
| d_attr_count_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_attr_count_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.5894 |
| d_attr_count_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_attr_count_g_only_clean | [32, 32, 128] | torch.float32 | 0.5894 |
| d_attr_count_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_attr_count_s_only_clean | [32, 32, 128] | torch.float32 | 0.5894 |
| d_attr_count_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_hard_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.4604 |
| d_cat_hard_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_hard_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.4604 |
| d_cat_hard_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_hard_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.4604 |
| d_cat_hard_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_hard_g_only_clean | [32, 32, 128] | torch.float32 | 0.4604 |
| d_cat_hard_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_hard_s_only_clean | [32, 32, 128] | torch.float32 | 0.4604 |
| d_cat_hard_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_popular_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.4713 |
| d_cat_popular_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_popular_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.4713 |
| d_cat_popular_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_popular_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.4713 |
| d_cat_popular_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_popular_g_only_clean | [32, 32, 128] | torch.float32 | 0.4713 |
| d_cat_popular_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_popular_s_only_clean | [32, 32, 128] | torch.float32 | 0.4713 |
| d_cat_popular_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_random_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.4666 |
| d_cat_random_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_random_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.4666 |
| d_cat_random_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_random_g1_s1_clean | [32, 32, 128] | tor

_Section truncated in aggregate report; open the source artifact for full details._

## Mask Construction

# Subtype Head Mask Report

- Activations: `data/subtype_minpair_v1/activations/train_activations.pt`
- Vectors: `data/subtype_minpair_v1/vectors/subtype_vectors.pt`
- TopK: `64`
- Masks written: `data/subtype_mask_steering_v1/masks/subtype_head_masks.pt`

## Activation Schema

```json
{
  "z_visual": {
    "type": "Tensor",
    "shape": [
      3800,
      32,
      32,
      128
    ],
    "dtype": "torch.float16"
  },
  "z_fact_text": {
    "type": "Tensor",
    "shape": [
      3800,
      32,
      32,
      128
    ],
    "dtype": "torch.float16"
  },
  "z_counterfact_text": {
    "type": "Tensor",
    "shape": [
      3800,
      32,
      32,
      128
    ],
    "dtype": "torch.float16"
  },
  "metadata": {
    "type": "list",
    "len": 3800
  },
  "schema": {
    "type": "dict",
    "keys": [
      "script",
      "model_path",
      "model_name",
      "context_len",
      "llava_repo_path",
      "conv_mode",
      "storage_dtype",
      "num_layers",
      "num_heads",
      "head_dim",
      "shape",
      "counts_by_subtype",
      "num_shards",
      "shard_index",
      "source_jsonl",
      "branch_definitions",
      "num_shards_merged",
      "shard_files"
    ],
    "len": 18
  }
}
```

## Vector Keys

```json
[
  "d_attr_color_g1_s025_clean",
  "d_attr_color_g1_s025_unit_clean",
  "d_attr_color_g1_s05_clean",
  "d_attr_color_g1_s05_unit_clean",
  "d_attr_color_g1_s1_clean",
  "d_attr_color_g1_s1_unit_clean",
  "d_attr_color_g_only_clean",
  "d_attr_color_g_only_unit_clean",
  "d_attr_color_s_only_clean",
  "d_attr_color_s_only_unit_clean",
  "d_attr_count_g1_s025_clean",
  "d_attr_count_g1_s025_unit_clean",
  "d_attr_count_g1_s05_clean",
  "d_attr_count_g1_s05_unit_clean",
  "d_attr_count_g1_s1_clean",
  "d_attr_count_g1_s1_unit_clean",
  "d_attr_count_g_only_clean",
  "d_attr_count_g_only_unit_clean",
  "d_attr_count_s_only_clean",
  "d_attr_count_s_only_unit_clean",
  "d_cat_hard_g1_s025_clean",
  "d_cat_hard_g1_s025_unit_clean",
  "d_cat_hard_g1_s05_clean",
  "d_cat_hard_g1_s05_unit_clean",
  "d_cat_hard_g1_s1_clean",
  "d_cat_hard_g1_s1_unit_clean",
  "d_cat_hard_g_only_clean",
  "d_cat_hard_g_only_unit_clean",
  "d_cat_hard_s_only_clean",
  "d_cat_hard_s_only_unit_clean",
  "d_cat_popular_g1_s025_clean",
  "d_cat_popular_g1_s025_unit_clean",
  "d_cat_popular_g1_s05_clean",
  "d_cat_popular_g1_s05_unit_clean",
  "d_cat_popular_g1_s1_clean",
  "d_cat_popular_g1_s1_unit_clean",
  "d_cat_popular_g_only_clean",
  "d_cat_popular_g_only_unit_clean",
  "d_cat_popular_s_only_clean",
  "d_cat_popular_s_only_unit_clean",
  "d_cat_random_g1_s025_clean",
  "d_cat_random_g1_s025_unit_clean",
  "d_cat_random_g1_s05_clean",
  "d_cat_random_g1_s05_unit_clean",
  "d_cat_random_g1_s1_clean",
  "d_cat_random_g1_s1_unit_clean",
  "d_cat_random_g_only_clean",
  "d_cat_random_g_only_unit_clean",
  "d_cat_random_s_only_clean",
  "d_cat_random_s_only_unit_clean",
  "d_rel_contact_g1_s025_clean",
  "d_rel_contact_g1_s025_unit_clean",
  "d_rel_contact_g1_s05_clean",
  "d_rel_contact_g1_s05_unit_clean",
  "d_rel_contact_g1_s1_clean",
  "d_rel_contact_g1_s1_unit_clean",
  "d_rel_contact_g_only_clean",
  "d_rel_contact_g_only_unit_clean",
  "d_rel_contact_s_only_clean",
  "d_rel_contact_s_only_unit_clean",
  "d_rel_spatial_g1_s025_clean",
  "d_rel_spatial_g1_s025_unit_clean",
  "d_rel_spatial_g1_s05_clean",
  "d_rel_spatial_g1_s05_unit_clean",
  "d_rel_spatial_g1_s1_clean",
  "d_rel_spatial_g1_s1_unit_clean",
  "d_rel_spatial_g_only_clean",
  "d_rel_spatial_g_only_unit_clean",
  "d_rel_spatial_s_only_clean",
  "d_rel_spatial_s_only_unit_clean",
  "g_all_clean",
  "g_all_mean_only",
  "g_all_raw",
  "g_attr_clean",
  "g_attr_mean_only",
  "g_attr_raw",
  "g_cat_clean",
  "g_cat_mean_only",
  "g_cat_raw",
  "g_rel_clean",
  "g_rel_mean_only",
  "g_rel_raw",
  "s_attr_color_clean",
  "s_attr_color_mean_only",
  "s_attr_color_raw",
  "s_attr_count_clean",
  "s_attr_count_mean_only",
  "s_attr_count_raw",
  "s_cat_hard_clean",
  "s_cat_hard_mean_only",
  "s_cat_hard_raw",
  "s_cat_popular_clean",
  "s_cat_popular_mean_only",
  "s_cat_popular_raw",
  "s_cat_random_clean",
  "s_cat_random_mean_only",
  "s_cat_random_raw",
  "s_rel_contact_clean",
  "s_rel_contact_mean_only",
  "s_rel_contact_raw",
  "s_rel_spatial_clean",
  "s_rel_spatial_mean_only",
  "s_rel_spatial_raw",
  "yesno_direction"
]
```

## Subtype Counts

| subtype | count |
| --- | --- |
| attr_color | 500 |
| attr_count | 500 |
| cat_hard | 600 |
| cat_popular | 600 |
| cat_random | 600 |
| rel_contact | 500 |
| rel_spatial | 500 |

## Score Statistics

| score | min | max | mean | std |
| --- | --- | --- | --- | --- |
| score_g_all_norm | 0.0003 | 0.1239 | 0.0165 | 0.0171 |
| score_g_cat_norm | 0.0004 | 0.1182 | 0.0177 | 0.0186 |
| score_g_attr_norm | 0.0003 | 0.1417 | 0.0165 | 0.0173 |
| score_g_rel_norm | 0.0003 | 0.1353 | 0.0170 | 0.0179 |
| score_s_cat_random_mean | 0.0000 | 0.0280 | 0.0018 | 0.0029 |
| score_s_cat_popular_mean | 0.0000 | 0.0258 | 0.0019 | 0.0032 |
| score_s_cat_hard_mean | 0.0000 | 0.0209 | 0.0016 | 0.0026 |
| score_s_attr_color_mean | 0.0000 | 0.0128 | 0.0006 | 0.0010 |
| score_s_attr_count_mean | 0.0000 | 0.1079 | 0.0067 | 0.0110 |
| score_s_rel_spatial_mean | 0.0000 | 0.0027 | 0.0001 | 0.0002 |
| score_s_rel_contact_mean | 0.0000 | 0.0479 | 0.0034 | 0.0059 |
| score_s_cat_random_energy | 0.0003 | 6.0885 | 0.4028 | 0.6554 |
| score_s_cat_popular_energy | 0.0003 | 5.5441 | 0.3919 | 0.6381 |
| score_s_cat_hard_energy | 0.0003 | 5.5122 | 0.3935 | 0.6411 |
| score_s_attr_color_energy | 0.0003 | 7.6026 | 0.4464 | 0.7323 |
| score_s_attr_count_energy | 0.0004 | 6.2584 | 0.3876 | 0.6180 |
| score_s_rel_spatial_energy | 0.0003 | 5.0839 | 0.3501 | 0.5642 |
| score_s_rel_contact_energy | 0.0004 | 5.4581 | 0.3442 | 0.5671 |

## G Mask Overlap

| mask_a | mask_b | intersection | jaccard |
| --- | --- | --- | --- |
| mask_g_all_norm_top64 | mask_g_cat_norm_top64 | 58 | 0.8286 |
| mask_g_all_norm_top64 | mask_g_attr_norm_top64 | 58 | 0.8286 |
| mask_g_all_norm_top64 | mask_g_rel_norm_top64 | 58 | 0.8286 |
| mask_g_cat_norm_top64 | mask_g_attr_norm_top64 | 52 | 0.6842 |
| mask_g_cat_norm_top64 | mask_g_rel_norm_top64 | 54 | 0.7297 |
| mask_g_attr_norm_top64 | mask_g_rel_norm_top64 | 57 | 0.8028 |

## S Energy Mask Overlap

| mask_a | mask_b | intersection | jaccard |
| --- | --- | --- | --- |
| mask_s_cat_random_energy_top64 | mask_s_cat_popular_energy_top64 | 59 | 0.8551 |
| mask_s_cat_random_energy_top64 | mask_s_cat_hard_energy_top64 | 61 | 0.9104 |
| mask_s_cat_random_energy_top64 | mask_s_attr_color_energy_top64 | 45 | 0.5422 |
| mask_s_cat_random_energy_top64 | mask_s_attr_count_energy_top64 | 44 | 0.5238 |
| mask_s_cat_random_energy_top64 | mask_s_rel_spatial_energy_top64 | 44 | 0.5238 |
| mask_s_cat_random_energy_top64 | mask_s_rel_contact_energy_top64 | 44 | 0.5238 |
| mask_s_cat_popular_energy_top64 | mask_s_cat_hard_energy_top64 | 62 | 0.9394 |
| mask_s_cat_popular_energy_top64 | mask_s_attr_color_energy_top64 | 46 | 0.5610 |
| mask_s_cat_popular_energy_top64 | mask_s_attr_count_energy_top64 | 46 | 0.5610 |
| mask_s_cat_popular_energy_top64 | mask_s_rel_spatial_energy_top64 | 45 | 0.5422 |
| mask_s_cat_popular_energy_top64 | mask_s_rel_contact_energy_top64 | 46 | 0.5610 |
| mask_s_cat_hard_energy_top64 | mask_s_attr_color_energy_top64 | 45 | 0.5422 |
| mask_s_cat_hard_energy_top64 | mask_s_attr_count_energy_top64 | 45 | 0.5422 |
| mask_s_cat_hard_energy_top64 | mask_s_rel_spatial_energy_top64 | 44 | 0.5238 |
| mask_s_cat_hard_energy_top64 | mask_s_rel_contact_energy_top64 | 45 | 0.5422 |
| mask_s_attr_color_energy_top64 | mask_s_attr_count_energy_top64 | 49 | 0.6203 |
| mask_s_attr_color_energy_top64 | mask_s_rel_spatial_energy_top64 | 49 | 0.6203 |
| mask_s_attr_color_energy_top64 | mask_s_rel_contact_energy_top64 | 49 | 0.6203 |
| mask_s_attr_count_energy_top64 | mask_s_rel_spatial_energy_top64 | 51 | 0.6623 |
| mask_s_attr_count_energy_top64 | mask_s_rel_contact_energy_top64 | 52 | 0.6842 |
| mask_s_rel_spatial_energy_top64 | mask_s_rel_contact_energy_top64 | 52 | 0.6842 |

## G vs S Energy Overlap

| mask_a | mask_b | intersection | jaccard |
| --- | --- | --- | --- |
| mask_g_all_norm_top64 | mask_s_cat_random_energy_top64 | 31 | 0.3196 |
| mask_g_all_norm_top64 | mask_s_cat_popular_energy_top64 | 31 | 0.3196 |
| mask_g_all_norm_top64 | mask_s_cat_hard_energy_top64 | 31 | 0.3196 |
| mask_g_all_norm_top64 | mask_s_attr_color_energy_top64 | 26 | 0.2549 |
| mask_g_all_norm_top64 | mask_s_attr_count_energy_top64 | 29 | 0.2929 |
| mask_g_all_norm_top64 | mask_s_rel_spatial_energy_top64 | 30 | 0.3061 |
| mask_g_all_norm_top64 | mask_s_rel_contact_energy_top64 | 30 | 0.3061 |
| mask_g_cat_norm_top64 | mask_s_cat_random_energy_top64 | 31 | 0.3196 |
| mask_g_cat_norm_top64 | mask_s_cat_popular_energy_top64 | 31 | 0.3196 |
| mask_g_cat_norm_top64 | mask_s_cat_hard_energy_top64 | 31 | 0.3196 |
| mask_g_cat_norm_top64 | mask_s_attr_color_energy_top64 | 26 | 0.2549 |
| mask_g_cat_norm_top64 | mask_s_attr_count_energy_top64 | 28 | 0.2800 |
| mask_g_cat_norm_top64 | mask_s_rel_spatial_energy_top64 | 30 | 0.3061 |
| mask_g_cat_norm_top64 | mask_s_rel_contact_energy_top64 | 29 | 0.2929 |
| mask_g_attr_norm_top64 | mask_s_cat_random_energy_top64 | 27 | 0.2673 |
| mask_g_attr_norm_top64 | mask_s_cat_popular_energy_top64 | 27 | 0.2673 |
| mask_g_attr_norm_top64 | mask_s_cat_hard_energy_top64 | 27 | 0.2673 |
| mask_g_attr_norm_top64 | mask_s_attr_color_energy_top64 | 27 | 0.2673 |
| mask_g_attr_norm_top64 | mask_s_attr_count_energy_top64 | 29 | 0.2929 |
| mask_g_attr_norm_top64 | mask_s_rel_spatial_energy_top64 | 30 | 0.3061 |
| mask_g_attr_norm_top64 | mask_s_rel_contact_energy_top64 | 31 | 0.3196 |
| mask_g_rel_norm_top64 | mask_s_cat_random_energy_top64 | 29 | 0.2929 |
| mask_g_rel_norm_top64 | mask_s_cat_popular_energy_top64 | 28 | 0.2800 |
| mask_g_rel_norm_top64 | mask_s_cat_hard_energy_top64 | 28 | 0.2800 |
| mask_g_rel_norm_top64 | mask_s_attr_color_energy_top64 | 26 | 0.2549 |
| mask_g_rel_norm_top64 | mask_s_attr_count_energy_top64 | 28 | 0.2800 |
| mask_g_rel_norm_top64 | mask_s_rel_spatial_energy_top64 | 30 | 0.3061 |
| mask_g_rel_norm_top64 | mask_s_rel_contact_energy_top64 | 30 | 0.3061 |

## S Mean vs S Energy Overlap

| subtype | intersection | jaccard |
| --- | --- | --- |
| cat_random | 47 | 0.5802 |
| cat_popular | 44 | 0.5238 |
| cat_hard | 43 | 0.5059 |
| attr_color | 45 | 0.5422 |
| attr_count | 52 | 0.6842 |
| rel_spatial | 49 | 0.6203 |
| rel_contact | 52 | 0.6842 |

## Random Mask Overlap With Energy Masks

| mask_a | mask_b | intersection | jaccard |
| --- | --- | --- | --- |
| random_mask_top64_seed0 | mask_s_cat_random_energy_top64 | 7 | 0.0579 |
| random_mask_top64_seed0 | mask_s_cat_popular_energy_top64 | 8 | 0.0667 |
| random_mask_top64_seed0 | mask_s_cat_hard_energy_top64 | 8 | 0.0667 |
| random_mask_top64_seed0 | mask_s_attr_color_energy_top64 | 7 | 0.0579 |
| random_mask_top64_seed0 | mask_s_attr_count_energy_top64 | 7 | 0.0579 |
| random_mask_top64_seed0 | mask_s_rel_spatial_energy_top64 | 7 | 0.0579 |
| random_mask_top64_seed0 | mask_s_rel_contact_energy_top64 | 9 | 0.0756 |
| random_mask_top64_seed1 | mask_s_cat_random_energy_top64 | 3 | 0.0240 |
| random_mask_top64_seed1 | mask_s_cat_popular_energy_top64 | 4 | 0.0323 |
| random_mask_top64_seed1 | mask_s_cat_hard_energy_top64 | 3 | 0.0240 |
| random_mask_top64_seed1 | mask_s_attr_color_energy_top64 | 6 | 0.0492 |
| random_mask_top64_seed1 | mask_s_attr_count_energy_top64 | 6 | 0.0492 |
| random_mask_top64_seed1 | mask_s_rel_spatial_energy_top64 | 3 | 0.0240 |
| random_mask_top64_seed1 | mask_s_rel_contact_energy_top64 | 5 | 0.0407 |
| random_mask_top64_seed2 | mask_s_cat_random_energy_top64 | 3 | 0.0240 |
| random_mask_top64_seed2 | mask_s_cat_popular_energy_top64 | 3 | 0.0240 |
| random_mask_top64_seed2 | mask_s_cat_hard_energy_top64 | 3 | 0.0240 |
| random_mask_top64_seed2 | mask_s_attr_color_energy_top64 | 4 | 0.0323 |
| random_mask_top64_seed2 | mask_s_attr_count_energy_top64 | 3 | 0.0240 |
| random_mask_top64_seed2 | mask_s_rel_spatial_energy_top64 | 6 | 0.0492 |
| random_mask_top64_seed2 | mask_s_rel_contact_energy_top64 | 5 | 0.0407 |
| layer_matched_random_cat_random_top64_seed0 | mask_s_cat_random_energy_top64 | 12 | 0.1034 |
| layer_matched_random_cat_random_top64_seed0 | mask_s_cat_popular_energy_top64 | 10 | 0.0847 |
| layer_matched_random_cat_random_top64_seed0 | mask_s_cat_hard_energy_top64 | 12 | 0.1034 |
| layer_matched_random_cat_random_top64_seed0 | mask_s_attr_color_energy_top64 | 9 | 0.0756 |
| layer_matched_random_cat_random_top64_seed0 | mask_s_attr_count_energy_top64 | 8 | 0.0667 |
| layer_matched_random_cat_random_top64_seed0 | mask_s_rel_spatial_energy_top64 | 12 | 0.1034 |
| layer_matched_random_cat_random_top64_seed0 | mask_s_rel_contact_energy_top64 | 10 | 0.0847 |
| layer_matched_random_cat_popular_top64_seed0 | mask_s_cat_random_energy_top64 | 12 | 0.1034 |
| layer_matched_random_cat_popular_top64_seed0 | mask_s_cat_popular_energy_top64 | 10 | 0.0847 |
| layer_matched_random_cat_popular_top64_seed0 | mask_s_cat_hard_energy_top64 | 12 | 0.1034 |
| layer_matched_random_cat_popular_top64_seed0 | mask_s_attr_color_energy_top64 | 10 | 0.0847 |
| layer_matched_random_cat_popular_top64_seed0 | mask_s_attr_count_energy_top64 | 10 | 0.0847 |
| layer_matched_random_cat_popular_top64_seed0 | mask_s_rel_spatial_energy_top64 | 12 | 0.1034 |
| layer_matched_random_cat_popular_top64_seed0 | mask_s_rel_contact_energy_top64 | 11 | 0.0940 |
| layer_matched_random_cat_hard_top64_seed0 | mask_s_cat_random_energy_top64 | 12 | 0.1034 |
| layer_matched_random_cat_hard_top64_seed0 | mask_s_cat_popular_energy_top64 | 10 | 0.0847 |
| layer_matched_random_cat_hard_top64_seed0 | mask_s_cat_hard_energy_top64 | 12 | 0.1034 |
| layer_matched_random_cat_hard_top64_seed0 | mask_s_attr_color_energy_top64 | 9 | 0.0756 |
| layer_matched_random_cat_hard_top64_seed0 | mask_s_attr_count_energy_top64 | 9 | 0.0756 |
| layer_matched_random_cat_hard_top64_seed0 | mask_s_rel_spatial_energy_top64 | 11 | 0.0940 |
| layer_matched_random_cat_hard_top64_seed0 | mask_s_rel_contact_energy_top64 | 10 | 0.0847 |
| layer_matched_random_attr_color_top64_seed0 | mask_s_cat_random_energy_top64 | 14 | 0.1228 |
| layer_matched_random_attr_color_top64_seed0 | mask_s_cat_popular_energy_top64 | 14 | 0.1228 |
| layer_matched_random_attr_color_top64_seed0 | mask_s_cat_hard_energy_top64 | 14 | 0.1228 |
| layer_matched_random_attr_color_top64_seed0 | mask_s_attr_color_energy_top64 | 13 | 0.1130 |
| layer_matched_random_attr_color_top64_seed0 | mask_s_attr_count_energy_top64 | 11 | 0.0940 |
| layer_matched_random_attr_color_top64_seed0 | mask_s_rel_spatial_energy_top64 | 13 | 0.1130 |
| layer_matched_random_attr_color_top64_seed0 | mask_s_rel_contact_energy_top64 | 11 | 0.0940 |
| layer_matched_random_attr_count_top64_seed0 | mask_s_cat_random_energy_top64 | 14 | 0.1228 |
| layer_matched_random_attr_count_top64_seed0 | mask_s_cat_popular_energy_top64 | 14 | 0.1228 |
| layer_matched_random_attr_count_top64_seed0 | mask_s_cat_hard_energy_top64 | 14 | 0.1228 |
| layer_matched_random_attr_count_top64_seed0 | mask_s_attr_color_energy_top64 | 12 | 0.1034 |
| layer_matched_random_attr_count_top64_seed0 | mask_s_attr_count_energy_top64 | 12 | 0.1034 |
| layer_matched_random_attr_count_top64_seed0 | mask_s_rel_spatial_energy_top64 | 12 | 0.1034 |
| layer_matched_random_attr_count_top64_seed0 | mask_s_rel_contact_energy_top64 | 11 | 0.0940 |
| layer_matched_random_rel_spatial_top64_seed0 | mask_s_cat_random_energy_top64 | 14 | 0.1228 |
| layer_matched_random_rel_spatial_top64_seed0 | mask_s_cat_popular_energy_top64 | 14 | 0.1228 |
| layer_matched_random_rel_spatial_top64_seed0 | mask_s_cat_hard_energy_top64 | 14 | 0.1228 |
| layer_matched_random_rel_spatial_top64_seed0 | mask_s_attr_color_energy_top64 | 11 | 0.0940 |
| layer_matched_random_rel_spatial_top64_seed0 | mask_s_attr_count_energy_top64 | 11 | 0.0940 |
| layer_matched_random_rel_spatial_top64_seed0 | mask_s_rel_spatial_energy_top64 | 12 | 0.1034 |
| layer_matched_random_rel_spatial_top64_seed0 | mask_s_rel_contact_energy_top64 | 9 | 0.0756 |
| layer_matched_random_rel_contact_top64_seed0 | mask_s_cat_random_energy_top64 | 11 | 0.0940 |
| layer_matched_random_rel_contact_top64_seed0 | mask_s_cat_popular_energy_top64 | 9 | 0.0756 |
| layer_matched_random_rel_contact_top64_seed0 | mask_s_cat_hard_energy_top64 | 11 | 0.0940 |
| layer_matched_random_rel_contact_top64_seed0 | mask_s_attr_color_energy_top64 | 10 | 0.0847 |
| layer_matched_random_rel_contact_top64_seed0 | mask_s_attr_count_energy_top64 | 11 | 0.0940 |
| layer_matched_random_rel_contact_top64_seed0 | mask_s_rel_spatial_energy_top64 | 14 | 0.1228 |
| layer_matched_random_rel_contact_top64_seed0 | mask_s_rel_contact_energy_top64 | 12 | 0.1034 |

## Top Heads

### layer_matched_random_attr_color_top64_seed0
| rank | layer | head | score |
| --- | --- | --- | --- |
| 1 | 11 | 6 | 1.0000 |
| 2 | 11 | 13 | 1.0000 |
| 3 | 11 | 18 | 1.0000 |
| 4 | 12 | 0 | 1.0000 |
| 5 | 12 | 4 | 1.0000 |
| 6 | 12 | 21 | 1.0000 |
| 7 | 12 | 26 | 1.0000 |
| 8 | 12 | 31 | 1.0000 |
| 9 | 13 | 2 | 1.0000 |
| 10 | 13 | 4 | 1.0000 |
| 11 | 13 | 5 | 1.0000 |
| 12 | 13 | 15 | 1.0000 |
| 13 | 13 | 20 | 1.0000 |
| 14 | 13 | 22 | 1.0000 |
| 15 | 13 | 23 | 1.0000 |
| 16 | 14 | 2 | 1.0000 |
| 17 | 14 | 7 | 1.0000 |
| 18 | 14 | 10 | 1.0000 |
| 19 | 14 | 17 | 1.0000 |
| 20 | 14 | 20 | 1.0000 |
| 21 | 14 | 24 | 1.0000 |
| 22 | 14 | 26 | 1.0000 |
| 23 | 14 | 27 | 1.0000 |
| 24 | 14 | 29 | 1.0000 |
| 25 | 15 | 1 | 1.0000 |
| 26 | 15 | 2 | 1.0000 |
| 27 | 15 | 11 | 1.0000 |
| 28 | 15 | 16 | 1.0000 |
| 29 | 15 | 17 | 1.0000 |
| 30 | 15 | 20 | 1.0000 |
| 31 | 15 | 31 | 1.0000 |
| 32 | 16 | 1 | 1.0000 |
| 33 | 16 | 4 | 1.0000 |
| 34 | 16 | 7 | 1.0000 |
| 35 | 16 | 9 | 1.0000 |
| 36 | 16 | 15 | 1.0000 |
| 37 | 16 | 20 | 1.0000 |
| 38 | 16 | 22 | 1.0000 |
| 39 | 16 | 28 | 1.0000 |
| 40 | 17 | 18 | 1.0000 |
| 41 | 17 | 24 | 1.0000 |
| 42 | 17 | 25 | 1.0000 |
| 43 | 17 | 31 | 1.0000 |
| 44 | 18 | 12 | 1.0000 |
| 45 | 18 | 17 | 1.0000 |
| 46 | 19 | 24 | 1.0000 |
| 47 | 20 | 13 | 1.0000 |
| 48 | 20 | 25 | 1.0000 |
| 49 | 21 | 31 | 1.0000 |
| 50 | 22 | 3 | 1.0000 |
| 51 | 22 | 10 | 1.0000 |
| 52 | 23 | 25 | 1.0000 |
| 53 | 24 | 16 | 1.0000 |
| 54 | 24 | 18 | 1.0000 |
| 55 | 25 | 16 | 1.0000 |
| 56 | 25 | 19 | 1.0000 |
| 57 | 26 | 14 | 1.0000 |
| 58 | 26 | 20 | 1.0000 |
| 59 | 27 | 2 | 1.0000 |
| 60 | 27 | 14 | 1.0000 |
| 61 | 27 | 17 | 1.0000 |
| 62 | 28 | 9 | 1.0000 |
| 63 | 30 | 15 | 1.0000 |
| 64 | 31 | 4 | 1.0000 |

### layer_matched_random_attr_count_top64_seed0
| rank | layer | head | score |
| --- | --- | --- | --- |
| 1 | 11 | 6 | 1.0000 |
| 2 | 11 | 13 | 1.0000 |
| 3 | 11 | 18 | 1.0000 |
| 4 | 12 | 0 | 1.0000 |
| 5 | 12 | 1 | 1.0000 |
| 6 | 12 | 4 | 1.0000 |
| 7 | 12 | 7 | 1.0000 |
| 8 | 12 | 21 | 1.0000 |
| 9 | 12 | 26 | 1.0000 |
| 10 | 12 | 31 | 1.0000 |
| 11 | 13 | 2 | 1.0000 |
| 12 | 13 | 5 | 1.0000 |
| 13 | 13 | 15 | 1.0000 |
| 14 | 13 | 20 | 1.0000 |
| 15 | 13 | 22 | 1.0000 |
| 16 | 14 | 2 | 1.0000 |
| 17 | 14 | 7 | 1.0000 |
| 18 | 14 | 10 | 1.0000 |
| 19 | 14 | 17 | 1.0000 |
| 20 | 14 | 19 | 1.0000 |
| 21 | 14 | 20 | 1.0000 |
| 22 | 14 | 24 | 1.0000 |
| 23 | 14 | 25 | 1.0000 |
| 24 | 14 | 26 | 1.0000 |
| 25 | 14 | 27 | 1.0000 |
| 26 | 14 | 29 | 1.0000 |
| 27 | 15 | 1 | 1.0000 |
| 28 | 15 | 11 | 1.0000 |
| 29 | 15 | 16 | 1.0000 |
| 30 | 15 | 17 | 1.0000 |
| 31 | 15 | 20 | 1.0000 |
| 32 | 15 | 31 | 1.0000 |
| 33 | 16 | 1 | 1.0000 |
| 34 | 16 | 4 | 1.0000 |
| 35 | 16 | 7 | 1.0000 |
| 36 | 16 | 9 | 1.0000 |
| 37 | 16 | 11 | 1.0000 |
| 38 | 16 | 15 | 1.0000 |
| 39 | 16 | 20 | 1.0000 |
| 40 | 16 | 22 | 1.0000 |
| 41 | 16 | 28 | 1.0000 |
| 42 | 17 | 18 | 1.0000 |
| 43 | 17 | 25 | 1.0000 |
| 44 | 17 | 31 | 1.0000 |
| 45 | 18 | 12 | 1.0000 |
| 46 | 18 | 17 | 1.0000 |
| 47 | 18 | 22 | 1.0000 |
| 48 | 19 | 24 | 1.0000 |
| 49 | 20 | 13 | 1.0000

_Section truncated in aggregate report; open the source artifact for full details._

## Held-Out Eval

# Subtype Mask Steering Eval Report

- Summary CSV: `data/subtype_mask_steering_v1/eval/heldout/summary.csv`

## Best Rows By Eval Subset

| group | eval_subset | direction_key | mask_key | alpha | accuracy | f1 | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_any | attr_color | g_all_clean | mask_s_attr_color_energy_top64 | 0.1 | 0.7250 | 0.7236 | 0.4950 | 35 | 22 | 57 | 200 |
| best_any | attr_count | s_attr_count_clean | mask_s_attr_count_energy_top64 | 0.05 | 0.6150 | 0.6244 | 0.5250 | 45 | 26 | 71 | 200 |
| best_any | cat_hard | g_cat_clean | mask_g_cat_norm_top64 | 0.5 | 0.9000 | 0.9000 | 0.5000 | 19 | 10 | 29 | 200 |
| best_any | cat_popular | g_cat_clean | mask_s_attr_count_energy_top64 | 0.5 | 0.9100 | 0.9082 | 0.4800 | 10 | 11 | 21 | 200 |
| best_any | cat_random | g_all_clean | mask_s_cat_random_energy_top64 | 0.05 | 0.9550 | 0.9548 | 0.4950 | 12 | 6 | 18 | 200 |
| best_any | rel_contact | s_rel_contact_clean | mask_s_rel_contact_energy_top64 | 0.1 | 0.7100 | 0.7478 | 0.6500 | 35 | 25 | 60 | 200 |
| best_any | rel_spatial | g_rel_clean | mask_g_rel_norm_top64 | 0.5 | 0.6250 | 0.6988 | 0.7450 | 52 | 27 | 79 | 200 |

## Matched Advantage

| eval_subset | baseline_f1 | matched_f1 | mismatched_f1 | random_f1 | g_type_f1 | g_all_f1 | matched_minus_mismatch_f1 | matched_minus_random_f1 | matched_minus_gtype_f1 | matched_yes_rate | success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attr_color | 0.6731 | 0.7236 | 0.6893 | 0.7083 | 0.7115 | 0.7170 | 0.0343 | 0.0153 | 0.0121 | 0.4950 | yes |
| attr_count | 0.5248 | 0.5905 | 0.5813 | 0.5684 | 0.5524 | 0.5822 | 0.0092 | 0.0221 | 0.0381 | 0.5500 | yes |
| cat_hard | 0.8599 | 0.8889 | 0.8867 | 0.0000 | 0.9000 | 0.8657 | 0.0022 | 0.0000 | -0.0111 | 0.4900 | no |
| cat_popular | 0.9119 | 0.9062 | 0.9082 | 0.0000 | 0.8889 | 0.9005 | -0.0019 | 0.0000 | 0.0174 | 0.4600 | no |
| cat_random | 0.9254 | 0.9548 | 0.9458 | 0.0000 | 0.9192 | 0.9333 | 0.0090 | 0.0000 | 0.0356 | 0.4950 | yes |
| rel_contact | 0.7167 | 0.7368 | 0.7436 | 0.7391 | 0.7448 | 0.7391 | -0.0067 | -0.0023 | -0.0079 | 0.6400 | no |
| rel_spatial | 0.5798 | 0.6877 | 0.6800 | 0.6667 | 0.6988 | 0.6640 | 0.0077 | 0.0211 | -0.0110 | 0.7650 | no |

## Comparison Details

| group | eval_subset | direction_key | mask_key | alpha | accuracy | f1 | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | attr_color |  |  |  | 0.6600 | 0.6731 | 0.5400 | 0 | 0 | 0 | 200 |
| best_matched | attr_color | g_all_clean | mask_s_attr_color_energy_top64 | 0.1 | 0.7250 | 0.7236 | 0.4950 | 35 | 22 | 57 | 200 |
| best_matched_g_type_direction | attr_color | g_attr_clean | mask_s_attr_color_energy_top64 | 0.1 | 0.7150 | 0.7164 | 0.5050 | 33 | 22 | 55 | 200 |
| best_mismatched | attr_color | g_attr_clean | mask_s_rel_spatial_energy_top64 | 0.25 | 0.6800 | 0.6893 | 0.5300 | 35 | 31 | 66 | 200 |
| best_random | attr_color | g_attr_clean | random_mask_top64_seed1 | 0.5 | 0.7200 | 0.7083 | 0.4600 | 39 | 27 | 66 | 200 |
| best_g_type_baseline | attr_color | g_attr_clean | mask_g_attr_norm_top64 | 0.5 | 0.7000 | 0.7115 | 0.5400 | 38 | 30 | 68 | 200 |
| best_g_all_baseline | attr_color | g_all_clean | mask_g_all_norm_top64 | 0.25 | 0.7000 | 0.7170 | 0.5600 | 33 | 25 | 58 | 200 |
| best_s_direction_ablation | attr_color | s_attr_color_clean | mask_s_attr_color_energy_top64 | 0.1 | 0.6450 | 0.6468 | 0.5050 | 31 | 34 | 65 | 200 |
| baseline | attr_count |  |  |  | 0.5200 | 0.5248 | 0.5100 | 0 | 0 | 0 | 200 |
| best_matched | attr_count | g_attr_clean | mask_s_attr_count_energy_top64 | 0.1 | 0.5700 | 0.5905 | 0.5500 | 46 | 36 | 82 | 200 |
| best_matched_g_type_direction | attr_count | g_attr_clean | mask_s_attr_count_energy_top64 | 0.1 | 0.5700 | 0.5905 | 0.5500 | 46 | 36 | 82 | 200 |
| best_mismatched | attr_count | g_attr_clean | mask_s_attr_color_energy_top64 | 0.5 | 0.5750 | 0.5813 | 0.5150 | 41 | 30 | 71 | 200 |
| best_random | attr_count | g_attr_clean | random_mask_top64_seed1 | 0.25 | 0.5900 | 0.5684 | 0.4500 | 45 | 31 | 76 | 200 |
| best_g_type_baseline | attr_count | g_attr_clean | mask_g_attr_norm_top64 | 0.25 | 0.5300 | 0.5524 | 0.5500 | 42 | 40 | 82 | 200 |
| best_g_all_baseline | attr_count | g_all_clean | mask_g_all_norm_top64 | 0.25 | 0.5550 | 0.5822 | 0.5650 | 46 | 39 | 85 | 200 |
| best_s_direction_ablation | attr_count | s_attr_count_clean | mask_s_attr_count_energy_top64 | 0.05 | 0.6150 | 0.6244 | 0.5250 | 45 | 26 | 71 | 200 |
| baseline | cat_hard |  |  |  | 0.8550 | 0.8599 | 0.5350 | 0 | 0 | 0 | 200 |
| best_matched | cat_hard | g_all_clean | mask_s_cat_hard_energy_top64 | 0.25 | 0.8900 | 0.8889 | 0.4900 | 16 | 9 | 25 | 200 |
| best_matched_g_type_direction | cat_hard | g_cat_clean | mask_s_cat_hard_energy_top64 | 0.25 | 0.8850 | 0.8832 | 0.4850 | 13 | 7 | 20 | 200 |
| best_mismatched | cat_hard | g_cat_clean | mask_s_attr_count_energy_top64 | 0.5 | 0.8850 | 0.8867 | 0.5150 | 14 | 8 | 22 | 200 |
| best_random |  |  |  |  |  |  |  |  |  |  |  |
| best_g_type_baseline | cat_hard | g_cat_clean | mask_g_cat_norm_top64 | 0.5 | 0.9000 | 0.9000 | 0.5000 | 19 | 10 | 29 | 200 |
| best_g_all_baseline | cat_hard | g_all_clean | mask_g_all_norm_top64 | 0.1 | 0.8650 | 0.8657 | 0.5050 | 15 | 13 | 28 | 200 |
| best_s_direction_ablation |  |  |  |  |  |  |  |  |  |  |  |
| baseline | cat_popular |  |  |  | 0.9150 | 0.9119 | 0.4650 | 0 | 0 | 0 | 200 |
| best_matched | cat_popular | g_cat_clean | mask_s_cat_popular_energy_top64 | 0.05 | 0.9100 | 0.9062 | 0.4600 | 7 | 8 | 15 | 200 |
| best_matched_g_type_direction | cat_popular | g_cat_clean | mask_s_cat_popular_energy_top64 | 0.05 | 0.9100 | 0.9062 | 0.4600 | 7 | 8 | 15 | 200 |
| best_mismatched | cat_popular | g_cat_clean | mask_s_attr_count_energy_top64 | 0.5 | 0.9100 | 0.9082 | 0.4800 | 10 | 11 | 21 | 200 |
| best_random |  |  |  |  |  |  |  |  |  |  |  |
| best_g_type_baseline | cat_popular | g_cat_clean | mask_g_cat_norm_top64 | 0.1 | 0.8950 | 0.8889 | 0.4450 | 6 | 10 | 16 | 200 |
| best_g_all_baseline | cat_popular | g_all_clean | mask_g_all_norm_top64 | 0.05 | 0.9050 | 0.9005 | 0.4550 | 7 | 9 | 16 | 200 |
| best_s_direction_ablation |  |  |  |  |  |  |  |  |  |  |  |
| baseline | cat_random |  |  |  | 0.9250 | 0.9254 | 0.5050 | 0 | 0 | 0 | 200 |
| best_matched | cat_random | g_all_clean | mask_s_cat_random_energy_top64 | 0.05 | 0.9550 | 0.9548 | 0.4950 | 12 | 6 | 18 | 200 |
| best_matched_g_type_direction | cat_random | g_cat_clean | mask_s_cat_random_energy_top64 | 0.25 | 0.9300 | 0.9286 | 0.4800 | 8 | 7 | 15 | 200 |
| best_mismatched | cat_random | g_cat_clean | mask_s_rel_spatial_energy_top64 | 0.05 | 0.9450 | 0.9458 | 0.5150 | 9 | 5 | 14 | 200 |
| best_random |  |  |  |  |  |  |  |  |  |  |  |
| best_g_type_baseline | cat_random | g_cat_clean | mask_g_cat_norm_top64 | 0.5 | 0.9200 | 0.9192 | 0.4900 | 7 | 8 | 15 | 200 |
| best_g_all_baseline | cat_random | g_all_clean | mask_g_all_norm_top64 | 0.1 | 0.9350 | 0.9333 | 0.4750 | 7 | 5 | 12 | 200 |
| best_s_direction_ablation |  |  |  |  |  |  |  |  |  |  |  |
| baseline | rel_contact |  |  |  | 0.6600 | 0.7167 | 0.7000 | 0 | 0 | 0 | 200 |
| best_matched | rel_contact | g_all_clean | mask_s_rel_contact_energy_top64 | 0.1 | 0.7000 | 0.7368 | 0.6400 | 37 | 29 | 66 | 200 |
| best_matched_g_type_direction | rel_contact | g_rel_clean | mask_s_rel_contact_energy_top64 | 0.5 | 0.6400 | 0.6949 | 0.6800 | 33 | 37 | 70 | 200 |
| best_mismatched | rel_contact | g_rel_clean | mask_s_attr_count_energy_top64 | 0.05 | 0.7000 | 0.7436 | 0.6700 | 35 | 27 | 62 | 200 |
| best_random | rel_contact | g_rel_clean | random_mask_top64_seed1 | 0.25 | 0.7000 | 0.7391 | 0.6500 | 39 | 31 | 70 | 200 |
| best_g_type_baseline | rel_contact | g_rel_clean | mask_g_rel_norm_top64 | 0.1 | 0.6950 | 0.7448 | 0.6950 | 32 | 25 | 57 | 200 |
| best_g_all_baseline | rel_contact | g_all_clean | mask_g_all_norm_top64 | 0.25 | 0.7000 | 0.7391 | 0.6500 | 37 | 29 | 66 | 200 |
| best_s_direction_ablation | rel_contact | s_rel_contact_clean | mask_s_rel_contact_energy_top64 | 0.1 | 0.7100 | 0.7478 | 0.6500 | 35 | 25 | 60 | 200 |
| baseline | rel_spatial |  |  |  | 0.5000 | 0.5798 | 0.6900 | 0 | 0 | 0 | 200 |
| best_matched | rel_spatial | g_all_clean | mask_s_rel_spatial_energy_top64 | 0.1 | 0.6050 | 0.6877 | 0.7650 | 45 | 24 | 69 | 200 |
| best_matched_g_type_direction | rel_spatial | g_rel_clean | mask_s_rel_spatial_energy_top64 | 0.25 | 0.5600 | 0.6333 | 0.7000 | 44 | 32 | 76 | 200 |
| best_mismatched | rel_spatial | g_rel_clean | mask_s_cat_hard_energy_top64 | 0.5 | 0.6000 | 0.6800 | 0.7500 | 48 | 28 | 76 | 200 |
| best_random | rel_spatial | g_rel_clean | random_mask_top64_seed0 | 0.5 | 0.5850 | 0.6667 | 0.7450 | 51 | 34 | 85 | 200 |
| best_g_type_baseline | rel_spatial | g_rel_clean | mask_g_rel_norm_top64 | 0.5 | 0.6250 | 0.6988 | 0.7450 | 52 | 27 | 79 | 200 |
| best_g_all_baseline | rel_spatial | g_all_clean | mask_g_all_norm_top64 | 0.1 | 0.5750 | 0.6640 | 0.7650 | 42 | 27 | 69 | 200 |
| best_s_direction_ablation | rel_spatial | s_rel_spatial_clean | mask_s_rel_spatial_energy_top64 | 0.25 | 0.5850 | 0.6667 | 0.7450 | 44 | 27 | 71 | 200 |

## Yes-Rate Flags

| group | eval_subset | direction_key | mask_key | alpha | accuracy | f1 | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g_all_baseline | rel_spatial | g_all_clean | mask_g_all_norm_top64 | 0.05 | 0.5550 | 0.6454 | 0.7550 | 42 | 31 | 73 | 200 |
| g_all_baseline | rel_spatial | g_all_clean | mask_g_all_norm_top64 | 0.1 | 0.5750 | 0.6640 | 0.7650 | 42 | 27 | 69 | 200 |
| g_all_baseline | rel_spatial | g_all_clean | mask_g_all_norm_top64 | 0.25 | 0.5650 | 0.6390 | 0.7050 | 44 | 31 | 75 | 200 |
| g_all_baseline | rel_spatial | g_all_clean | mask_g_all_norm_top64 | 0.5 | 0.5400 | 0.6260 | 0.7300 | 42 | 34 | 76 | 200 |
| g_type_baseline | rel_spatial | g_rel_clean | mask_g_rel_norm_top64 | 0.05 | 0.5550 | 0.6482 | 0.7650 | 40 | 29 | 69 | 200 |
| g_type_baseline | rel_spatial | g_rel_clean | mask_g_rel_norm_top64 | 0.1 | 0.6050 | 0.6609 | 0.6650 | 51 | 30 | 81 | 200 |
| g_type_baseline | rel_spatial | g_rel_clean | mask_g_rel_norm_top64 | 0.25 | 0.5600 | 0.6364 | 0.7100 | 40 | 28 | 68 | 200 |
| g_type_baseline | rel_spatial | g_rel_clean | mask_g_rel_norm_top64 | 0.5 | 0.6250 | 0.6988 | 0.7450 | 52 | 27 | 79 | 200 |
| matched_energy | rel_spatial | g_rel_clean | mask_s_rel_spatial_energy_top64 | 0.05 | 0.5150 | 0.6104 | 0.7450 | 37 | 34 | 71 | 200 |
| matched_energy | rel_spatial | g_rel_clean | mask_s_rel_spatial_energy_top64 | 0.1 | 0.5350 | 0.6173 | 0.7150 | 41 | 34 | 75 | 200 |
| matched_energy | rel_spatial | g_rel_clean | mask_s_rel_spatial_energy_top64 | 0.25 | 0.5600 | 0.6333 | 0.7000 | 44 | 32 | 76 | 200 |
| matched_energy | rel_spatial | g_rel_clean | mask_s_rel_spatial_energy_top64 | 0.5 | 0.5450 | 0.6255 | 0.7150 | 40 | 31 | 71 | 200 |
| matched_energy_g_all | rel_spatial | g_all_clean | mask_s_rel_spatial_energy_top64 | 0.05 | 0.5250 | 0.6025 | 0.6950 | 41 | 36 | 77 | 200 |
| matched_energy_g_all | rel_spatial | g_all_clean | mask_s_rel_spatial_energy_top64 | 0.1 | 0.6050 | 0.6877 | 0.7650 | 45 | 24 | 69 | 200 |
| matched_energy_g_all | rel_spatial | g_all_clean | mask_s_rel_spatial_energy_top64 | 0.25 | 0.5500 | 0.6250 | 0.7000 | 39 | 29 | 68 | 200 |
| matched_energy_g_all | rel_spatial | g_all_clean | mask_s_rel_spatial_energy_top64 | 0.5 | 0.5750 | 0.6444 | 0.6950 | 43 | 28 | 71 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_rel_contact_energy_top64 | 0.05 | 0.5600 | 0.6333 | 0.7000 | 39 | 27 | 66 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_rel_contact_energy_top64 | 0.1 | 0.5200 | 0.6000 | 0.7000 | 41 | 37 | 78 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_rel_contact_energy_top64 | 0.25 | 0.5850 | 0.6640 | 0.7350 | 43 | 26 | 69 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_rel_contact_energy_top64 | 0.5 | 0.5700 | 0.6325 | 0.6700 | 43 | 29 | 72 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_attr_color_energy_top64 | 0.05 | 0.5250 | 0.6058 | 0.7050 | 37 | 32 | 69 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_attr_color_energy_top64 | 0.1 | 0.5300 | 0.6116 | 0.7100 | 40 | 34 | 74 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_attr_color_energy_top64 | 0.25 | 0.5200 | 0.6190 | 0.7600 | 34 | 30 | 64 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_attr_color_energy_top64 | 0.5 | 0.5950 | 0.6639 | 0.7050 | 49 | 30 | 79 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_cat_hard_energy_top64 | 0.05 | 0.5550 | 0.6180 | 0.6650 | 43 | 32 | 75 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_cat_hard_energy_top64 | 0.1 | 0.5300 | 0.6116 | 0.7100 | 41 | 35 | 76 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_cat_hard_energy_top64 | 0.25 | 0.5250 | 0.6215 | 0.7550 | 39 | 34 | 73 | 200 |
| mismatched_energy | rel_spatial | g_rel_clean | mask_s_cat_hard_energy_top64 | 0.5 | 0.6000 | 0.6800 | 0.7500 | 48 | 28 | 76 | 200 |
| s_direction_ablation | rel_spatial | s_rel_spatial_clean | mask_s_rel_spatial_energy_top64 | 0.05 | 0.5550 | 0.6307 | 0.7050 | 46 | 35 | 81 | 200 |
| s_direction_ablation | rel_spatial | s_rel_spatial_clean | mask_s_rel_spatial_energy_top64 | 0.1 | 0.5400 | 0.6230 | 0.7200 | 41 | 33 | 74 | 200 |
| s_direction_ablation | rel_spatial | s_rel_spatial_clean | mask_s_rel_spatial_energy_top64 | 0.25 | 0.5850 | 0.6667 | 0.7450 | 44 | 27 | 71 | 200 |
| s_direction_ablation | rel_spatial | s_rel_spatial_clean | mask_s_rel_spatial_energy_top64 | 0.5 | 0.5550 | 0.6397 | 0.7350 | 45 | 34 | 79 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed0 | 0.05 | 0.5650 | 0.6420 | 0.7150 | 38 | 25 | 63 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed0 | 0.1 | 0.5100 | 0.5984 | 0.7200 | 40 | 38 | 78 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed0 | 0.25 | 0.5500 | 0.6341 | 0.7300 | 43 | 33 | 76 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed0 | 0.5 | 0.5850 | 0.6667 | 0.7450 | 51 | 34 | 85 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed1 | 0.05 | 0.5000 | 0.5935 | 0.7300 | 33 | 33 | 66 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed1 | 0.1 | 0.5150 | 0.6073 | 0.7350 | 39 | 36 | 75 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed1 | 0.25 | 0.5600 | 0.6364 | 0.7100 | 41 | 29 | 70 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed1 | 0.5 | 0.5300 | 0.6017 | 0.6800 | 41 | 35 | 76 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed2 | 0.05 | 0.5700 | 0.6417 | 0.7000 | 40 | 26 | 66 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed2 | 0.1 | 0.5200 | 0.6129 | 0.7400 | 39 | 35 | 74 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed2 | 0.25 | 0.5000 | 0.5968 | 0.7400 | 35 | 35 | 70 | 200 |
| random_mask | rel_spatial | g_rel_clean | random_mask_top64_seed2 | 0.5 | 0.5450 | 0.6286 | 0.7250 | 41 | 32 | 73 | 200 |
| g_all_baseline | rel_contact | g_all_clean | mask_g_all_norm_top64 | 0.05 | 0.6500 | 0.7059 | 0.6900 | 29 | 31 | 60 | 200 |
| g_all_baseline | rel_contact | g_all_clean | mask_g_all_norm_top64 | 0.1 | 0.6700 | 0.7155 | 0.6600 | 31 | 29 | 60 | 200 |
| g_all_baseline | rel_contact | g_all_clean | mask_g_all_norm_top64 | 0.5 | 0.6750 | 0.7210 | 0.6650 | 31 | 28 | 59 | 200 |
| g_type_baseline | rel_contact | g_rel_clean | mask_g_rel_norm_top64 | 0.05 | 0.6450 | 0.6926 | 0.6550 | 29 | 32 | 61 | 200 |
| g_type_baseline | rel_contact | g_rel_clean | mask_g_rel_norm_top64 | 0.1 | 0.6950 | 0.7448 | 0.6950 | 32 | 25 | 57 | 200 |
| g_type_baseline | rel_contact | g_rel_clean | mask_g_rel_norm_top64 | 0.25 | 0.6550 | 0.7089 | 0.6850 | 29 | 30 | 59 | 200 |
| g_type_baseline | rel_contact | g_rel_clean | mask_g_rel_norm_top64 | 0.5 | 0.6550 | 0.7089 | 0.6850 | 33 | 34 | 67 | 200 |
| matched_energy | rel_contact | g_rel_clean | mask_s_rel_contact_energy_top64 | 0.1 | 0.6200 | 0.6860 | 0.7100 | 26 | 34 | 60 | 200 |
| matched_energy | rel_contact | g_rel_clean | mask_s_rel_contact_energy_top64 | 0.25 | 0.6450 | 0.6926 | 0.6550 | 36 | 39 | 75 | 200 |
| matched_energy | rel_contact | g_rel_clean | mask_s_rel_contact_energy_top64 | 0.5 | 0.6400 | 0.6949 | 0.6800 | 33 | 37 | 70 | 200 |
| matched_energy_g_all | rel_contact | g_all_clean | mask_s_rel_contact_energy_top64 | 0.05 | 0.6800 | 0.7311 | 0.6900 | 29 | 25 | 54 | 200 |
| matched_energy_g_all | rel_contact | g_all_clean | mask_s_rel_contact_energy_top64 | 0.25 | 0.6550 | 0.7039 | 0.6650 | 28 | 29 | 57 | 200 |
| matched_energy_g_all | rel_contact | g_all_clean | mask_s_rel_contact_energy_top64 | 0.5 | 0.6550 | 0.7089 | 0.6850 | 31 | 32 | 63 | 200 |
| mismatched_energy | rel_contact | g_rel_clean | mask_s_rel_spatial_energy_top64 | 0.05 | 0.6800 | 0.7241 | 0.6600 | 37 | 33 | 70 | 200 |
| mismatched_energy | rel_contact | g_rel_clean | mask_s_rel_spatial_energy_top64 | 0.1 | 0.6450 | 0.7149 | 0.7450 | 28 | 31 | 59 | 200 |
| mismatched_energy | rel_contact | g_rel_clean | mask_s_rel_spatial_energy_top64 | 0.25 | 0.6800 | 0.7311 | 0.6900 | 32 | 28 | 60 | 200 |
| mismatched_energy | rel_contact | g_rel_clean | mask_s_attr_count_energy_top64 | 0.05 | 0.7000 | 0.7436 | 0.6700 | 35 | 27 | 62 | 200 |
| mismatched_energy | rel_contact | g_rel_clean | mask_s_attr_count_energy_top64 | 0.1 | 0.6500 | 0.7059 | 0.6900 | 27 | 29 | 56 | 200 |
| mismatched_energy | rel_contact | g_rel_clean | mask_s_attr_count_energy_top64 | 0.25 | 0.6650 | 0.7124 | 0.6650 | 33 | 32 | 65 | 200 |
| mismatched_energy | rel_contact | g_rel_clean | mask_s_attr_count_energy_top64 | 0.5 | 0.6200 | 0.6885 | 0.7200 | 25 | 33 | 58 | 200 |
| mismatched_energy | rel_contact | g_rel_clean | mask_s_cat_hard_energy_top64 | 0.1 | 0.6450 | 0.6979 | 0.6750 | 27 | 30 | 57 | 200 |
| mismatched_energy | rel_contact | g_rel_clean | mask_s_cat_hard_energy_top64 | 0.25 | 0.6450 | 0.6926 | 0.6550 | 34 | 37 | 71 | 200 |
| mismatched_energy | rel_contact | g_rel_clean | mask_s_cat_hard_energy_top64 | 0.5 | 0.6700 | 0.7250 | 0.7000 | 31 | 29 | 60 | 200 |
| s_direction_ablation | rel_contact | s_rel_contact_clean | mask_s_rel_contact_energy_top64 | 0.05 | 0.6550 | 0.7089 | 0.6850 | 32 | 33 | 65 | 200 |
| s_direction_ablation | rel_contact | s_rel_contact_clean | mask_s_rel_contact_energy_top64 | 0.25 | 0.6450 | 0.7054 | 0.7050 | 29 | 32 | 61 | 200 |
| s_direction_ablation | rel_contact | s_rel_contact_clean | mask_s_rel_contact_energy_top64 | 0.5 | 0.6700 | 0.7203 | 0.6800 | 36 | 34 | 70 | 200 |
| random_mask | rel_contact | g_rel_clean | random_mask_top64_seed0 | 0.1 | 0.6450 | 0.6953 | 0.6650 | 35 | 38 | 73 | 200 |
| random_mask | rel_contact | g_rel_clean | random_mask_top64_seed0 | 0.25 | 0.6650 | 0.7149 | 0.6750 | 30 | 29 | 59 | 200 |
| random_mask | rel_contact | g_rel_clean | random_mask_top64_seed1 | 0.05 | 0.6650 | 0.7197 | 0.6950 | 35 | 34 | 69 | 200 |
| random_mask | rel_contact | g_rel_clean | random_mask_top64_seed1 | 0.1 | 0.6500 | 0.6983 | 0.6600 | 33 | 35 | 68 | 200 |
| random_mask | rel_contact | g_rel_clean | random_mask_top64_seed1 | 0.5 | 0.6150 | 0.6751 | 0.6850 | 33 | 42 | 75 | 200 |
| random_mask | rel_contact | g_rel_clean | random_mask_top64_seed2 | 0.05 | 0.6500 | 0.7059 | 0.6900 | 31 | 33 | 64 | 200 |
| random_mask | rel_contact | g_rel_clean | random_mask_top64_seed2 | 0.1 | 0.6700 | 0.7227 | 0.6900 | 32 | 30 | 62 | 200 |
| random_mask | rel_contact | g_rel_clean | random_mask_top64_seed2 | 0.25 | 0.6450 | 0.7004 | 0.6850 | 31 | 34 | 65 | 200 |
| random_mask | rel_contact | g_rel_clean | random_mask_top64_seed2 | 0.5 | 0.6050 | 0.6667 | 0.6850 | 25 | 36 | 61 | 200 |

## Automatic Conclusion

- `attr_color`: matched subtype mask passes the current success criteria.
- `attr_count`: matched subtype mask passes the current success criteria.
- `cat_hard`: matched mask trails the g_type norm baseline; the subtype mask is not adding value yet.
- `cat_popular`: matched mask does not beat mismatched masks; subtype selectivity is not established.
- `cat_random`: matched subtype mask passes the current success criteria.
- `rel_contact`: matched mask does not beat mismatched masks; subtype selectivity is not established.
- `rel_spatial`: matched mask is suspicious because yes_rate=0.77 on a balanced set.

## Success Criteria

- A subtype mask is considered established only if it beats mismatched masks, random masks, and is competitive with the g_type baseline without abnormal yes-rate drift.
- If matched masks do not beat random masks, this argues against moving to router/DPO yet.
- If only category masks work, category is ready while attr/rel likely need cleaner data or value-level masks.

## Data Quality Notes

# Subtype Mask Data Quality Notes

- Train JSONL: `data/subtype_minpair_v1/minimal_pairs/train.jsonl`
- Val JSONL: `data/subtype_minpair_v1/minimal_pairs/val.jsonl`

## Attr Count

### Grammar Warnings
| item | count |
| --- | --- |
| are_there_one | 290 |
| mans | 16 |
| womans | 8 |
| watchs | 8 |
| feets | 4 |

### Count Word Distribution
| item | count |
| --- | --- |
| one | 580 |
| two | 368 |
| three | 65 |
| four | 26 |
| five | 11 |

## Attr Color

### Object Distribution
| item | count |
| --- | --- |
| shirt | 30 |
| sky | 28 |
| tree | 22 |
| wall | 18 |
| pants | 18 |
| hat | 14 |
| t-shirt | 14 |
| pole | 12 |
| letters | 10 |
| leaf | 10 |
| sign | 10 |
| trees | 10 |
| leaves | 10 |
| tail | 10 |
| hair | 8 |
| shoe | 8 |
| floor | 8 |
| car | 6 |
| table | 6 |
| pepper | 6 |

### Part/Stuff-Like Object Ratio
| total | flagged | ratio | examples |
| --- | --- | --- | --- |
| 700 | 160 | 0.2286 | ['train_gqa_attr_color_1591843_3797527_black_white_no: letters', 'train_gqa_attr_color_1591843_3797527_black_white_yes: letters', 'train_gqa_attr_color_1592348_4440619_white_brown_no: hair', 'train_gqa_attr_color_1592348_4440619_white_brown_yes: hair', 'train_gqa_attr_color_1593178_4463277_white_gray_no: wall', 'train_gqa_attr_color_1593178_4463277_white_gray_yes: wall', 'train_gqa_attr_color_2321627_1047437_red_white_no: letters', 'train_gqa_attr_color_2321627_1047437_red_white_yes: letters', 'train_gqa_attr_color_2322859_3415165_blue_black_no: pole', 'train_gqa_attr_color_2322859_3415165_blue_black_yes: pole'] |

## Rel Spatial

### Predicate Distribution
| item | count |
| --- | --- |
| right of | 222 |
| left of | 220 |
| below | 136 |
| above | 122 |

### Part/Stuff-Like Object Ratio
| total | flagged | ratio | examples |
| --- | --- | --- | --- |
| 700 | 222 | 0.3171 | ['train_gqa_rel_spatial_1592653_4448883_above_4448891_no: ground', 'train_gqa_rel_spatial_1592653_4448883_above_4448891_yes: ground', 'train_gqa_rel_spatial_1951_1068367_right_of_1068368_no: pole, sky', 'train_gqa_rel_spatial_1951_1068367_right_of_1068368_yes: pole, sky', 'train_gqa_rel_spatial_2316135_4237199_right_of_4237154_no: wall', 'train_gqa_rel_spatial_2316135_4237199_right_of_4237154_yes: wall', 'train_gqa_rel_spatial_2317572_2957268_right_of_3258312_no: letter', 'train_gqa_rel_spatial_2317572_2957268_right_of_3258312_yes: letter', 'train_gqa_rel_spatial_2318484_2879249_right_of_2748136_no: pole', 'train_gqa_rel_spatial_2318484_2879249_right_of_2748136_yes: pole'] |

## Rel Contact

### Predicate Distribution
| item | count |
| --- | --- |
| wearing | 470 |
| holding | 114 |
| sitting on | 42 |
| carrying | 20 |
| standing on | 18 |
| eating | 14 |
| lying on | 10 |
| riding | 10 |
| leaning on | 2 |

- Wearing count: `470` / `700` (0.6714)

### Potentially Unnatural Counterfacts
No simple `wearing umbrella/plate/table` patterns found.

## Final Decision Guide

- If matched subtype masks beat mismatched and random masks without abnormal yes-rate drift, move toward token-level router / DPO routing.
- If only category masks work, category is ready but attribute/relation data likely needs repair.
- If attr_count/contact work but color/spatial do not, use finer value/predicate-level masks.
- If all matched masks fail against random/mismatched, do not build a router yet; revisit data construction or activation definition.
