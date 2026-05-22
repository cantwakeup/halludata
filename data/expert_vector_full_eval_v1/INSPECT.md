# Expert Vector Full Eval Inspection

## Final Vector Sources

| role | path | key | shape | dtype | norm | covers_32_layers | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | g_all_clean | [32, 32, 128] | torch.float32 | 0.739544 | True | clean_type_minpair_v2:g_all_clean |
| cat | /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_cat_clean | [32, 32, 128] | torch.float32 | 0.823078 | True | subtype_minpair_v1:g_cat_clean |
| attr | /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | g_attr_clean | [32, 32, 128] | torch.float32 | 0.758327 | True | clean_type_minpair_v2:g_attr_clean |
| rel | /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | g_rel_clean | [32, 32, 128] | torch.float32 | 0.774043 | True | clean_type_minpair_v2:g_rel_clean |

## Runtime Bundle

- Runtime vector file: `/home/huiwei/sy/halludata/data/expert_vector_full_eval_v1/vectors/expert_vectors_runtime.pt`
- Runtime keys: `global`, `cat`, `attr`, `rel`
- Runtime layers: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]`
- Head selection for evaluation: vector norm top64 over all runtime layers.

## Candidate Vector Inventory

| path | key | shape | dtype | norm | covers_32_layers |
| --- | --- | --- | --- | --- | --- |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | g_all_clean | [32, 32, 128] | torch.float32 | 0.739544 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | g_all_raw | [32, 32, 128] | torch.float32 | 0.739574 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | g_attr_clean | [32, 32, 128] | torch.float32 | 0.758327 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | g_attr_raw | [32, 32, 128] | torch.float32 | 0.758375 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | g_rel_clean | [32, 32, 128] | torch.float32 | 0.774043 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | g_rel_raw | [32, 32, 128] | torch.float32 | 0.774054 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_action_single_clean_clean | [32, 32, 128] | torch.float32 | 0.172153 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_action_single_clean_raw | [32, 32, 128] | torch.float32 | 0.172169 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_color_clean_clean | [32, 32, 128] | torch.float32 | 0.023405 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_color_clean_raw | [32, 32, 128] | torch.float32 | 0.023409 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_count_clean_clean | [32, 32, 128] | torch.float32 | 0.190953 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_count_clean_raw | [32, 32, 128] | torch.float32 | 0.190987 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_material_clean_clean | [32, 32, 128] | torch.float32 | 0.037708 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_material_clean_raw | [32, 32, 128] | torch.float32 | 0.037708 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_shape_clean_clean | [32, 32, 128] | torch.float32 | 0.065681 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_shape_clean_raw | [32, 32, 128] | torch.float32 | 0.065683 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_state_clean_clean | [32, 32, 128] | torch.float32 | 0.055690 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_attr_state_clean_raw | [32, 32, 128] | torch.float32 | 0.055709 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_rel_above_below_clean_clean | [32, 32, 128] | torch.float32 | 0.031712 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_rel_above_below_clean_raw | [32, 32, 128] | torch.float32 | 0.031712 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_rel_holding_wearing_clean_clean | [32, 32, 128] | torch.float32 | 0.277694 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_rel_holding_wearing_clean_raw | [32, 32, 128] | torch.float32 | 0.277726 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_rel_left_right_clean_clean | [32, 32, 128] | torch.float32 | 0.030165 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_rel_left_right_clean_raw | [32, 32, 128] | torch.float32 | 0.030167 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_rel_sitting_riding_clean_clean | [32, 32, 128] | torch.float32 | 0.291841 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | s_rel_sitting_riding_clean_raw | [32, 32, 128] | torch.float32 | 0.291841 | True |
| /home/huiwei/sy/halludata/data/clean_type_minpair_v2/vectors/condition_vectors.pt | yesno_direction | [32, 32, 128] | torch.float32 | 10.761387 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_color_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.402328 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_color_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_color_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.402328 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_color_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.000002 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_color_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.402328 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_color_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_color_g_only_clean | [32, 32, 128] | torch.float32 | 0.402328 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_color_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_color_s_only_clean | [32, 32, 128] | torch.float32 | 0.402328 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_color_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_count_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.589430 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_count_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_count_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.589431 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_count_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_count_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.589430 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_count_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_count_g_only_clean | [32, 32, 128] | torch.float32 | 0.589430 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_count_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_count_s_only_clean | [32, 32, 128] | torch.float32 | 0.589430 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_attr_count_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_hard_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.460445 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_hard_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_hard_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.460445 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_hard_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_hard_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.460445 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_hard_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_hard_g_only_clean | [32, 32, 128] | torch.float32 | 0.460445 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_hard_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_hard_s_only_clean | [32, 32, 128] | torch.float32 | 0.460445 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_hard_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_popular_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.471266 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_popular_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_popular_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.471267 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_popular_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_popular_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.471267 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_popular_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_popular_g_only_clean | [32, 32, 128] | torch.float32 | 0.471267 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_popular_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_popular_s_only_clean | [32, 32, 128] | torch.float32 | 0.471267 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_popular_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_random_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.466636 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_random_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_random_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.466636 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_random_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_random_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.466636 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_random_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_random_g_only_clean | [32, 32, 128] | torch.float32 | 0.466636 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_random_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_random_s_only_clean | [32, 32, 128] | torch.float32 | 0.466636 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_cat_random_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_contact_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.503528 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_contact_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_contact_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.503528 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_contact_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_contact_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.503528 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_contact_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_contact_g_only_clean | [32, 32, 128] | torch.float32 | 0.503527 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_contact_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_contact_s_only_clean | [32, 32, 128] | torch.float32 | 0.503527 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_contact_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_spatial_g1_s025_clean | [32, 32, 128] | torch.float32 | 0.398531 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_spatial_g1_s025_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_spatial_g1_s05_clean | [32, 32, 128] | torch.float32 | 0.398531 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_spatial_g1_s05_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_spatial_g1_s1_clean | [32, 32, 128] | torch.float32 | 0.398531 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_spatial_g1_s1_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_spatial_g_only_clean | [32, 32, 128] | torch.float32 | 0.398531 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_spatial_g_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_spatial_s_only_clean | [32, 32, 128] | torch.float32 | 0.398531 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | d_rel_spatial_s_only_unit_clean | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_all_clean | [32, 32, 128] | torch.float32 | 0.759436 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_all_mean_only | [32, 32, 128] | torch.float32 | 0.759463 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_all_raw | [32, 32, 128] | torch.float32 | 0.759452 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_attr_clean | [32, 32, 128] | torch.float32 | 0.766101 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_attr_mean_only | [32, 32, 128] | torch.float32 | 0.766162 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_attr_raw | [32, 32, 128] | torch.float32 | 0.766157 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_cat_clean | [32, 32, 128] | torch.float32 | 0.823078 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_cat_mean_only | [32, 32, 128] | torch.float32 | 0.823103 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_cat_raw | [32, 32, 128] | torch.float32 | 0.823098 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_rel_clean | [32, 32, 128] | torch.float32 | 0.790262 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_rel_mean_only | [32, 32, 128] | torch.float32 | 0.790267 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | g_rel_raw | [32, 32, 128] | torch.float32 | 0.790262 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_attr_color_clean | [32, 32, 128] | torch.float32 | 0.038556 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_attr_color_mean_only | [32, 32, 128] | torch.float32 | 0.045186 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_attr_color_raw | [32, 32, 128] | torch.float32 | 0.038558 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_attr_count_clean | [32, 32, 128] | torch.float32 | 0.412760 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_attr_count_mean_only | [32, 32, 128] | torch.float32 | 0.412893 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_attr_count_raw | [32, 32, 128] | torch.float32 | 0.412820 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_cat_hard_clean | [32, 32, 128] | torch.float32 | 0.097812 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_cat_hard_mean_only | [32, 32, 128] | torch.float32 | 0.102691 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_cat_hard_raw | [32, 32, 128] | torch.float32 | 0.097859 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_cat_popular_clean | [32, 32, 128] | torch.float32 | 0.119455 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_cat_popular_mean_only | [32, 32, 128] | torch.float32 | 0.125901 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_cat_popular_raw | [32, 32, 128] | torch.float32 | 0.119502 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_cat_random_clean | [32, 32, 128] | torch.float32 | 0.110194 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_cat_random_mean_only | [32, 32, 128] | torch.float32 | 0.112892 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_cat_random_raw | [32, 32, 128] | torch.float32 | 0.110244 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_rel_contact_clean | [32, 32, 128] | torch.float32 | 0.216793 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_rel_contact_mean_only | [32, 32, 128] | torch.float32 | 0.219511 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_rel_contact_raw | [32, 32, 128] | torch.float32 | 0.216795 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_rel_spatial_clean | [32, 32, 128] | torch.float32 | 0.006801 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_rel_spatial_mean_only | [32, 32, 128] | torch.float32 | 0.023990 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | s_rel_spatial_raw | [32, 32, 128] | torch.float32 | 0.006801 | True |
| /home/huiwei/sy/halludata/data/subtype_minpair_v1/vectors/subtype_vectors.pt | yesno_direction | [32, 32, 128] | torch.float32 | 10.761387 | True |
| /home/huiwei/sy/halludata/data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt | attr | [21, 32, 128] | torch.float32 | 26.363945 | False |
| /home/huiwei/sy/halludata/data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt | cat | [21, 32, 128] | torch.float32 | 27.385265 | False |
| /home/huiwei/sy/halludata/data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt | rel | [21, 32, 128] | torch.float32 | 26.253576 | False |
| /home/huiwei/sy/halludata/data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt | attr | [21, 32, 128] | torch.float32 | 26.363945 | False |
| /home/huiwei/sy/halludata/data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt | attr_res | [21, 32, 128] | torch.float32 | 6.517539 | False |
| /home/huiwei/sy/halludata/data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt | cat | [21, 32, 128] | torch.float32 | 27.385265 | False |
| /home/huiwei/sy/halludata/data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt | cat_res | [21, 32, 128] | torch.float32 | 8.682144 | False |
| /home/huiwei/sy/halludata/data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt | global_all | [21, 32, 128] | torch.float32 | 25.322187 | False |
| /home/huiwei/sy/halludata/data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt | rel | [21, 32, 128] | torch.float32 | 26.253576 | False |
| /home/huiwei/sy/halludata/data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt | rel_res | [21, 32, 128] | torch.float32 | 9.356807 | False |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/official_llava_steering/official_llava_cat_expert_vectors.pt | attr | [21, 32, 128] | torch.float32 | 0.000000 | False |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/official_llava_steering/official_llava_cat_expert_vectors.pt | cat | [21, 32, 128] | torch.float32 | 26.826118 | False |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/official_llava_steering/official_llava_cat_expert_vectors.pt | rel | [21, 32, 128] | torch.float32 | 0.000000 | False |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/official_llava_steering/official_llava_cat_expert_vectors_smoke.pt | attr | [21, 32, 128] | torch.float32 | 0.000000 | False |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/official_llava_steering/official_llava_cat_expert_vectors_smoke.pt | cat | [21, 32, 128] | torch.float32 | 26.850986 | False |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/official_llava_steering/official_llava_cat_expert_vectors_smoke.pt | rel | [21, 32, 128] | torch.float32 | 0.000000 | False |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | attr | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | attr_raw | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | attr_res | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | cat | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | cat_raw | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | cat_res | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | global | [32, 32, 128] | torch.float32 | 1.000002 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | global_all | [32, 32, 128] | torch.float32 | 1.000002 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | global_mean_all | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | global_type_svd | [32, 32, 128] | torch.float32 | 1.000002 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | rel | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | rel_raw | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors.pt | rel_res | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | attr | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | attr_raw | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | attr_res | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | cat | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | cat_raw | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | cat_res | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | global | [32, 32, 128] | torch.float32 | 1.000002 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | global_all | [32, 32, 128] | torch.float32 | 1.000002 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | global_mean_all | [32, 32, 128] | torch.float32 | 1.000001 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | global_type_svd | [32, 32, 128] | torch.float32 | 1.000002 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | rel | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | rel_raw | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/after_template_disjoint_v2/shared_private_vectors/shared_private_vectors_eval_compatible.pt | rel_res | [32, 32, 128] | torch.float32 | 1.000000 | True |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/coco_cat_as_cat.pt | cat | [21, 32, 128] | torch.float32 | 26.940268 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/coco_cat_as_cat.pt | coco_cat | [21, 32, 128] | torch.float32 | 26.940268 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/coco_cat_as_cat.pt | gqa_cat | [21, 32, 128] | torch.float32 | 27.385265 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/coco_cat_as_cat.pt | mixed_cat | [21, 32, 128] | torch.float32 | 1.000000 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/gqa_cat_as_cat.pt | cat | [21, 32, 128] | torch.float32 | 27.385265 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/gqa_cat_as_cat.pt | coco_cat | [21, 32, 128] | torch.float32 | 26.940268 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/gqa_cat_as_cat.pt | gqa_cat | [21, 32, 128] | torch.float32 | 27.385265 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/gqa_cat_as_cat.pt | mixed_cat | [21, 32, 128] | torch.float32 | 1.000000 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/mixed_cat_as_cat.pt | cat | [21, 32, 128] | torch.float32 | 1.000000 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/mixed_cat_as_cat.pt | coco_cat | [21, 32, 128] | torch.float32 | 26.940268 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/mixed_cat_as_cat.pt | gqa_cat | [21, 32, 128] | torch.float32 | 27.385265 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/cat_domain_vector_comparison_smoke/vectors/mixed_cat_as_cat.pt | mixed_cat | [21, 32, 128] | torch.float32 | 1.000000 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/coco_cat_as_cat.pt | cat | [21, 32, 128] | torch.float32 | 26.826118 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/coco_cat_as_cat.pt | coco_cat | [21, 32, 128] | torch.float32 | 26.826118 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/coco_cat_as_cat.pt | gqa_cat | [21, 32, 128] | torch.float32 | 27.270096 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/coco_cat_as_cat.pt | mixed_cat | [21, 32, 128] | torch.float32 | 27.048088 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/gqa_cat_as_cat.pt | cat | [21, 32, 128] | torch.float32 | 27.270096 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/gqa_cat_as_cat.pt | coco_cat | [21, 32, 128] | torch.float32 | 26.826118 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/gqa_cat_as_cat.pt | gqa_cat | [21, 32, 128] | torch.float32 | 27.270096 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/gqa_cat_as_cat.pt | mixed_cat | [21, 32, 128] | torch.float32 | 27.048088 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/mixed_cat_as_cat.pt | cat | [21, 32, 128] | torch.float32 | 27.048088 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/mixed_cat_as_cat.pt | coco_cat | [21, 32, 128] | torch.float32 | 26.826118 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/mixed_cat_as_cat.pt | gqa_cat | [21, 32, 128] | torch.float32 | 27.270096 | False |
| /home/huiwei/sy/halludata/data/pope_cat_expert_eval/official_cat_domain_comparison_full_seed42/vectors/mixed_cat_as_cat.pt | mixed_cat | [21, 32, 128] | torch.float32 | 27.048088 | False |

## Selected Top10 Heads

### global

| rank | layer | head | norm |
| --- | --- | --- | --- |
| 1 | 17 | 5 | 0.122529 |
| 2 | 15 | 17 | 0.103286 |
| 3 | 12 | 11 | 0.095712 |
| 4 | 16 | 0 | 0.091388 |
| 5 | 31 | 22 | 0.089240 |
| 6 | 13 | 4 | 0.088190 |
| 7 | 31 | 4 | 0.084419 |
| 8 | 30 | 31 | 0.082889 |
| 9 | 13 | 2 | 0.076046 |
| 10 | 14 | 0 | 0.072825 |

### cat

| rank | layer | head | norm |
| --- | --- | --- | --- |
| 1 | 17 | 5 | 0.118161 |
| 2 | 13 | 4 | 0.102961 |
| 3 | 15 | 17 | 0.102366 |
| 4 | 12 | 11 | 0.101979 |
| 5 | 12 | 1 | 0.101311 |
| 6 | 13 | 2 | 0.099802 |
| 7 | 31 | 4 | 0.096266 |
| 8 | 14 | 20 | 0.094997 |
| 9 | 16 | 0 | 0.094814 |
| 10 | 11 | 8 | 0.093182 |

### attr

| rank | layer | head | norm |
| --- | --- | --- | --- |
| 1 | 17 | 5 | 0.125387 |
| 2 | 15 | 17 | 0.103661 |
| 3 | 13 | 4 | 0.096441 |
| 4 | 16 | 0 | 0.096381 |
| 5 | 12 | 11 | 0.090523 |
| 6 | 30 | 31 | 0.089786 |
| 7 | 31 | 4 | 0.082218 |
| 8 | 31 | 22 | 0.078916 |
| 9 | 14 | 0 | 0.077245 |
| 10 | 13 | 2 | 0.074533 |

### rel

| rank | layer | head | norm |
| --- | --- | --- | --- |
| 1 | 17 | 5 | 0.125340 |
| 2 | 15 | 17 | 0.104084 |
| 3 | 12 | 11 | 0.101710 |
| 4 | 31 | 22 | 0.101231 |
| 5 | 16 | 0 | 0.095027 |
| 6 | 13 | 4 | 0.091543 |
| 7 | 31 | 4 | 0.087781 |
| 8 | 12 | 30 | 0.086210 |
| 9 | 13 | 2 | 0.081703 |
| 10 | 24 | 23 | 0.079986 |

