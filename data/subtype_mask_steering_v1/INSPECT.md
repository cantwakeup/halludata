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
| d_cat_random_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.4666 |
| d_cat_random_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_random_g_only_clean | [32, 32, 128] | torch.float32 | 0.4666 |
| d_cat_random_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_cat_random_s_only_clean | [32, 32, 128] | torch.float32 | 0.4666 |
| d_cat_random_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_rel_contact_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.5035 |
| d_rel_contact_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_rel_contact_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.5035 |
| d_rel_contact_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_rel_contact_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.5035 |
| d_rel_contact_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_rel_contact_g_only_clean | [32, 32, 128] | torch.float32 | 0.5035 |
| d_rel_contact_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_rel_contact_s_only_clean | [32, 32, 128] | torch.float32 | 0.5035 |
| d_rel_contact_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_rel_spatial_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.3985 |
| d_rel_spatial_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_rel_spatial_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.3985 |
| d_rel_spatial_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_rel_spatial_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.3985 |
| d_rel_spatial_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_rel_spatial_g_only_clean | [32, 32, 128] | torch.float32 | 0.3985 |
| d_rel_spatial_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| d_rel_spatial_s_only_clean | [32, 32, 128] | torch.float32 | 0.3985 |
| d_rel_spatial_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.0000 |
| g_all_clean | [32, 32, 128] | torch.float32 | 0.7594 |
| g_all_mean_only | [32, 32, 128] | torch.float32 | 0.7595 |
| g_all_raw | [32, 32, 128] | torch.float32 | 0.7595 |
| g_attr_clean | [32, 32, 128] | torch.float32 | 0.7661 |
| g_attr_mean_only | [32, 32, 128] | torch.float32 | 0.7662 |
| g_attr_raw | [32, 32, 128] | torch.float32 | 0.7662 |
| g_cat_clean | [32, 32, 128] | torch.float32 | 0.8231 |
| g_cat_mean_only | [32, 32, 128] | torch.float32 | 0.8231 |
| g_cat_raw | [32, 32, 128] | torch.float32 | 0.8231 |
| g_rel_clean | [32, 32, 128] | torch.float32 | 0.7903 |
| g_rel_mean_only | [32, 32, 128] | torch.float32 | 0.7903 |
| g_rel_raw | [32, 32, 128] | torch.float32 | 0.7903 |
| s_attr_color_clean | [32, 32, 128] | torch.float32 | 0.0386 |
| s_attr_color_mean_only | [32, 32, 128] | torch.float32 | 0.0452 |
| s_attr_color_raw | [32, 32, 128] | torch.float32 | 0.0386 |
| s_attr_count_clean | [32, 32, 128] | torch.float32 | 0.4128 |
| s_attr_count_mean_only | [32, 32, 128] | torch.float32 | 0.4129 |
| s_attr_count_raw | [32, 32, 128] | torch.float32 | 0.4128 |
| s_cat_hard_clean | [32, 32, 128] | torch.float32 | 0.0978 |
| s_cat_hard_mean_only | [32, 32, 128] | torch.float32 | 0.1027 |
| s_cat_hard_raw | [32, 32, 128] | torch.float32 | 0.0979 |
| s_cat_popular_clean | [32, 32, 128] | torch.float32 | 0.1195 |
| s_cat_popular_mean_only | [32, 32, 128] | torch.float32 | 0.1259 |
| s_cat_popular_raw | [32, 32, 128] | torch.float32 | 0.1195 |
| s_cat_random_clean | [32, 32, 128] | torch.float32 | 0.1102 |
| s_cat_random_mean_only | [32, 32, 128] | torch.float32 | 0.1129 |
| s_cat_random_raw | [32, 32, 128] | torch.float32 | 0.1102 |
| s_rel_contact_clean | [32, 32, 128] | torch.float32 | 0.2168 |
| s_rel_contact_mean_only | [32, 32, 128] | torch.float32 | 0.2195 |
| s_rel_contact_raw | [32, 32, 128] | torch.float32 | 0.2168 |
| s_rel_spatial_clean | [32, 32, 128] | torch.float32 | 0.0068 |
| s_rel_spatial_mean_only | [32, 32, 128] | torch.float32 | 0.0240 |
| s_rel_spatial_raw | [32, 32, 128] | torch.float32 | 0.0068 |
| yesno_direction | [32, 32, 128] | torch.float32 | 10.7614 |

## Val Sample Preview

| id | subtype | gt_answer | question | image_path |
| --- | --- | --- | --- | --- |
| val_gqa_attr_color_1159654_3704523_yellow_green_no | attr_color | no | Is the bottle green? | /home/huiwei/sy/sy_data/GQA/raw/images/images/1159654.jpg |
| val_gqa_attr_color_1159654_3704523_yellow_green_yes | attr_color | yes | Is the bottle yellow? | /home/huiwei/sy/sy_data/GQA/raw/images/images/1159654.jpg |
| val_gqa_attr_color_1507_1032642_brown_gray_no | attr_color | no | Is the tree trunk gray? | /home/huiwei/sy/sy_data/GQA/raw/images/images/1507.jpg |
| val_gqa_attr_color_1507_1032642_brown_gray_yes | attr_color | yes | Is the tree trunk brown? | /home/huiwei/sy/sy_data/GQA/raw/images/images/1507.jpg |
| val_gqa_attr_color_1592713_4450072_white_blue_no | attr_color | no | Is the vase blue? | /home/huiwei/sy/sy_data/GQA/raw/images/images/1592713.jpg |

## Runner Capability Notes

- `ExpertSteeringController` supports `head_select=expert_map`, so direction vectors and head masks can be separated via an external JSON head map.
- `scripts/eval_subtype_mask_steering.py` writes a runtime vector file keyed by direction/mask pair and a matching expert head map.
- This keeps official LLaVA generation and the existing o_proj pre-hook path.
