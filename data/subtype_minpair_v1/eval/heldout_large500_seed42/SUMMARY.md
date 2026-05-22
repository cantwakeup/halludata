# Large Subtype Minimal-Pair Held-Out Eval

- Raw prediction rows: `157500`

## Best By Subset And Match Type
| eval_subset | vector | match | alpha | accuracy | f1 | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attr_color | g_all_clean | global | 0.1 | 0.6800 | 0.6735 | 0.4800 | 86 | 79 | 165 | 500 |
| attr_color | g_cat_clean | mismatched | 0.1 | 0.7000 | 0.7024 | 0.5080 | 86 | 69 | 155 | 500 |
| attr_color | d_attr_color_g1_s05_clean | subtype_matched | 0.1 | 0.6860 | 0.6879 | 0.5060 | 83 | 73 | 156 | 500 |
| attr_color | g_attr_clean | type_matched | 0.25 | 0.6840 | 0.6749 | 0.4720 | 80 | 71 | 151 | 500 |
| attr_count | g_all_clean | global | 0.25 | 0.5380 | 0.5276 | 0.4780 | 103 | 97 | 200 | 500 |
| attr_count | d_cat_popular_g1_s05_clean | mismatched | 0.5 | 0.5820 | 0.5708 | 0.4740 | 118 | 90 | 208 | 500 |
| attr_count | d_attr_count_g1_s05_clean | subtype_matched | 0.1 | 0.5340 | 0.5115 | 0.4540 | 106 | 102 | 208 | 500 |
| attr_count | d_attr_color_g1_s05_clean | type_matched | 0.1 | 0.5760 | 0.5709 | 0.4880 | 117 | 92 | 209 | 500 |
| cat_hard | g_all_clean | global | 0.1 | 0.8940 | 0.8921 | 0.4820 | 27 | 23 | 50 | 500 |
| cat_hard | d_rel_spatial_g1_s05_clean | mismatched | 0.1 | 0.9120 | 0.9102 | 0.4800 | 34 | 21 | 55 | 500 |
| cat_hard | d_cat_hard_g1_s05_clean | subtype_matched | 0.5 | 0.8760 | 0.8730 | 0.4760 | 30 | 35 | 65 | 500 |
| cat_hard | g_cat_clean | type_matched | 0.1 | 0.9000 | 0.9004 | 0.5040 | 31 | 24 | 55 | 500 |
| cat_popular | g_all_clean | global | 0.1 | 0.9240 | 0.9224 | 0.4800 | 30 | 17 | 47 | 500 |
| cat_popular | d_rel_spatial_g1_s05_clean | mismatched | 0.05 | 0.9360 | 0.9342 | 0.4720 | 32 | 13 | 45 | 500 |
| cat_popular | d_cat_popular_g1_s05_clean | subtype_matched | 0.05 | 0.9200 | 0.9174 | 0.4680 | 31 | 20 | 51 | 500 |
| cat_popular | d_cat_random_g1_s05_clean | type_matched | 0.5 | 0.9280 | 0.9250 | 0.4600 | 29 | 14 | 43 | 500 |
| cat_random | g_all_clean | global | 0.1 | 0.9340 | 0.9322 | 0.4740 | 32 | 17 | 49 | 500 |
| cat_random | d_attr_count_g1_s05_clean | mismatched | 0.25 | 0.9320 | 0.9295 | 0.4640 | 30 | 16 | 46 | 500 |
| cat_random | d_cat_random_g1_s05_clean | subtype_matched | 0.5 | 0.9320 | 0.9300 | 0.4720 | 31 | 17 | 48 | 500 |
| cat_random | g_cat_clean | type_matched | 0.1 | 0.9280 | 0.9256 | 0.4680 | 28 | 16 | 44 | 500 |
| rel_contact | g_all_clean | global | 0.25 | 0.6460 | 0.7055 | 0.7020 | 81 | 66 | 147 | 500 |
| rel_contact | d_cat_random_g1_s05_clean | mismatched | 0.25 | 0.6700 | 0.7245 | 0.6980 | 84 | 57 | 141 | 500 |
| rel_contact | d_rel_contact_g1_s05_clean | subtype_matched | 0.5 | 0.6260 | 0.6919 | 0.7140 | 68 | 63 | 131 | 500 |
| rel_contact | d_rel_spatial_g1_s05_clean | type_matched | 0.25 | 0.6580 | 0.7164 | 0.7060 | 80 | 59 | 139 | 500 |
| rel_spatial | g_all_clean | global | 0.1 | 0.5380 | 0.6473 | 0.8100 | 68 | 58 | 126 | 500 |
| rel_spatial | g_attr_clean | mismatched | 0.05 | 0.5860 | 0.6750 | 0.7740 | 92 | 58 | 150 | 500 |
| rel_spatial | d_rel_spatial_g1_s05_clean | subtype_matched | 0.05 | 0.5460 | 0.6502 | 0.7980 | 74 | 60 | 134 | 500 |
| rel_spatial | g_rel_clean | type_matched | 0.05 | 0.5940 | 0.6710 | 0.7340 | 100 | 62 | 162 | 500 |

## Best By Subset And Vector
| eval_subset | vector | match | alpha | accuracy | f1 | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attr_color | g_cat_clean | mismatched | 0.1 | 0.7000 | 0.7024 | 0.5080 | 86 | 69 | 155 | 500 |
| attr_color | d_rel_spatial_g1_s05_clean | mismatched | 0.5 | 0.6980 | 0.7010 | 0.5100 | 80 | 64 | 144 | 500 |
| attr_color | d_cat_random_g1_s05_clean | mismatched | 0.25 | 0.6960 | 0.6960 | 0.5000 | 94 | 79 | 173 | 500 |
| attr_color | g_rel_clean | mismatched | 0.1 | 0.6960 | 0.6960 | 0.5000 | 88 | 73 | 161 | 500 |
| attr_color | d_cat_hard_g1_s05_clean | mismatched | 0.05 | 0.6980 | 0.6937 | 0.4860 | 86 | 70 | 156 | 500 |
| attr_color | d_cat_popular_g1_s05_clean | mismatched | 0.5 | 0.6880 | 0.6905 | 0.5080 | 85 | 74 | 159 | 500 |
| attr_color | d_attr_color_g1_s05_clean | subtype_matched | 0.1 | 0.6860 | 0.6879 | 0.5060 | 83 | 73 | 156 | 500 |
| attr_color | g_attr_clean | type_matched | 0.25 | 0.6840 | 0.6749 | 0.4720 | 80 | 71 | 151 | 500 |
| attr_color | g_all_clean | global | 0.1 | 0.6800 | 0.6735 | 0.4800 | 86 | 79 | 165 | 500 |
| attr_color | d_attr_count_g1_s05_clean | type_matched | 0.05 | 0.6740 | 0.6733 | 0.4980 | 89 | 85 | 174 | 500 |
| attr_color | d_rel_contact_g1_s05_clean | mismatched | 0.5 | 0.6600 | 0.6627 | 0.5080 | 78 | 81 | 159 | 500 |
| attr_count | d_attr_color_g1_s05_clean | type_matched | 0.1 | 0.5760 | 0.5709 | 0.4880 | 117 | 92 | 209 | 500 |
| attr_count | d_cat_popular_g1_s05_clean | mismatched | 0.5 | 0.5820 | 0.5708 | 0.4740 | 118 | 90 | 208 | 500 |
| attr_count | d_cat_random_g1_s05_clean | mismatched | 0.05 | 0.5780 | 0.5649 | 0.4700 | 109 | 83 | 192 | 500 |
| attr_count | d_cat_hard_g1_s05_clean | mismatched | 0.05 | 0.5840 | 0.5612 | 0.4480 | 115 | 86 | 201 | 500 |
| attr_count | g_cat_clean | mismatched | 0.05 | 0.5640 | 0.5569 | 0.4840 | 112 | 93 | 205 | 500 |
| attr_count | g_rel_clean | mismatched | 0.5 | 0.5660 | 0.5544 | 0.4740 | 110 | 90 | 200 | 500 |
| attr_count | d_rel_spatial_g1_s05_clean | mismatched | 0.5 | 0.5720 | 0.5466 | 0.4440 | 112 | 89 | 201 | 500 |
| attr_count | g_attr_clean | type_matched | 0.05 | 0.5480 | 0.5425 | 0.4880 | 103 | 92 | 195 | 500 |
| attr_count | d_rel_contact_g1_s05_clean | mismatched | 0.05 | 0.5520 | 0.5410 | 0.4760 | 113 | 100 | 213 | 500 |
| attr_count | g_all_clean | global | 0.25 | 0.5380 | 0.5276 | 0.4780 | 103 | 97 | 200 | 500 |
| attr_count | d_attr_count_g1_s05_clean | subtype_matched | 0.1 | 0.5340 | 0.5115 | 0.4540 | 106 | 102 | 208 | 500 |
| cat_hard | d_rel_spatial_g1_s05_clean | mismatched | 0.1 | 0.9120 | 0.9102 | 0.4800 | 34 | 21 | 55 | 500 |
| cat_hard | g_attr_clean | mismatched | 0.05 | 0.9020 | 0.9034 | 0.5140 | 38 | 30 | 68 | 500 |
| cat_hard | d_attr_count_g1_s05_clean | mismatched | 0.5 | 0.9040 | 0.9024 | 0.4840 | 31 | 22 | 53 | 500 |
| cat_hard | g_cat_clean | type_matched | 0.1 | 0.9000 | 0.9004 | 0.5040 | 31 | 24 | 55 | 500 |
| cat_hard | g_rel_clean | mismatched | 0.5 | 0.9020 | 0.8994 | 0.4740 | 32 | 24 | 56 | 500 |
| cat_hard | d_attr_color_g1_s05_clean | mismatched | 0.25 | 0.9000 | 0.8988 | 0.4880 | 33 | 26 | 59 | 500 |
| cat_hard | g_all_clean | global | 0.1 | 0.8940 | 0.8921 | 0.4820 | 27 | 23 | 50 | 500 |
| cat_hard | d_cat_popular_g1_s05_clean | type_matched | 0.05 | 0.8940 | 0.8916 | 0.4780 | 32 | 28 | 60 | 500 |
| cat_hard | d_cat_random_g1_s05_clean | type_matched | 0.25 | 0.8900 | 0.8893 | 0.4940 | 26 | 24 | 50 | 500 |
| cat_hard | d_rel_contact_g1_s05_clean | mismatched | 0.05 | 0.8900 | 0.8893 | 0.4940 | 30 | 28 | 58 | 500 |
| cat_hard | d_cat_hard_g1_s05_clean | subtype_matched | 0.5 | 0.8760 | 0.8730 | 0.4760 | 30 | 35 | 65 | 500 |
| cat_popular | d_rel_spatial_g1_s05_clean | mismatched | 0.05 | 0.9360 | 0.9342 | 0.4720 | 32 | 13 | 45 | 500 |
| cat_popular | d_attr_count_g1_s05_clean | mismatched | 0.05 | 0.9300 | 0.9290 | 0.4860 | 31 | 15 | 46 | 500 |
| cat_popular | d_rel_contact_g1_s05_clean | mismatched | 0.1 | 0.9320 | 0.9289 | 0.4560 | 31 | 14 | 45 | 500 |
| cat_popular | g_rel_clean | mismatched | 0.5 | 0.9300 | 0.9287 | 0.4820 | 31 | 15 | 46 | 500 |
| cat_popular | g_attr_clean | mismatched | 0.05 | 0.9280 | 0.9265 | 0.4800 | 28 | 13 | 41 | 500 |
| cat_popular | d_attr_color_g1_s05_clean | mismatched | 0.1 | 0.9280 | 0.9262 | 0.4760 | 32 | 17 | 49 | 500 |
| cat_popular | d_cat_random_g1_s05_clean | type_matched | 0.5 | 0.9280 | 0.9250 | 0.4600 | 29 | 14 | 43 | 500 |
| cat_popular | g_all_clean | global | 0.1 | 0.9240 | 0.9224 | 0.4800 | 30 | 17 | 47 | 500 |
| cat_popular | d_cat_popular_g1_s05_clean | subtype_matched | 0.05 | 0.9200 | 0.9174 | 0.4680 | 31 | 20 | 51 | 500 |
| cat_popular | d_cat_hard_g1_s05_clean | type_matched | 0.05 | 0.9180 | 0.9165 | 0.4820 | 32 | 22 | 54 | 500 |
| cat_popular | g_cat_clean | type_matched | 0.5 | 0.9140 | 0.9121 | 0.4780 | 31 | 23 | 54 | 500 |
| cat_random | g_all_clean | global | 0.1 | 0.9340 | 0.9322 | 0.4740 | 32 | 17 | 49 | 500 |
| cat_random | d_cat_random_g1_s05_clean | subtype_matched | 0.5 | 0.9320 | 0.9300 | 0.4720 | 31 | 17 | 48 | 500 |
| cat_random | d_attr_count_g1_s05_clean | mismatched | 0.25 | 0.9320 | 0.9295 | 0.4640 | 30 | 16 | 46 | 500 |
| cat_random | g_attr_clean | mismatched | 0.05 | 0.9300 | 0.9290 | 0.4860 | 27 | 14 | 41 | 500 |
| cat_random | g_cat_clean | type_matched | 0.1 | 0.9280 | 0.9256 | 0.4680 | 28 | 16 | 44 | 500 |
| cat_random | d_rel_spatial_g1_s05_clean | mismatched | 0.5 | 0.9240 | 0.9231 | 0.4880 | 31 | 21 | 52 | 500 |
| cat_random | g_rel_clean | mismatched | 0.5 | 0.9240 | 0.9228 | 0.4840 | 30 | 20 | 50 | 500 |
| cat_random | d_attr_color_g1_s05_clean | mismatched | 0.5 | 0.9240 | 0.9208 | 0.4600 | 30 | 20 | 50 | 500 |
| cat_random | d_cat_hard_g1_s05_clean | type_matched | 0.25 | 0.9220 | 0.9182 | 0.4540 | 29 | 20 | 49 | 500 |
| cat_random | d_rel_contact_g1_s05_clean | mismatched | 0.1 | 0.9200 | 0.9167 | 0.4600 | 32 | 24 | 56 | 500 |
| cat_random | d_cat_popular_g1_s05_clean | type_matched | 0.05 | 0.9160 | 0.9132 | 0.4680 | 30 | 24 | 54 | 500 |
| rel_contact | d_cat_random_g1_s05_clean | mismatched | 0.25 | 0.6700 | 0.7245 | 0.6980 | 84 | 57 | 141 | 500 |
| rel_contact | d_attr_count_g1_s05_clean | mismatched | 0.05 | 0.6580 | 0.7201 | 0.7220 | 84 | 63 | 147 | 500 |
| rel_contact | d_cat_hard_g1_s05_clean | mismatched | 0.25 | 0.6580 | 0.7164 | 0.7060 | 84 | 63 | 147 | 500 |
| rel_contact | d_rel_spatial_g1_s05_clean | type_matched | 0.25 | 0.6580 | 0.7164 | 0.7060 | 80 | 59 | 139 | 500 |
| rel_contact | g_rel_clean | type_matched | 0.25 | 0.6540 | 0.7150 | 0.7140 | 86 | 67 | 153 | 500 |
| rel_contact | g_cat_clean | mismatched | 0.1 | 0.6580 | 0.7145 | 0.6980 | 80 | 59 | 139 | 500 |
| rel_contact | g_attr_clean | mismatched | 0.05 | 0.6580 | 0.7126 | 0.6900 | 87 | 66 | 153 | 500 |
| rel_contact | g_all_clean | global | 0.25 | 0.6460 | 0.7055 | 0.7020 | 81 | 66 | 147 | 500 |
| rel_contact | d_cat_popular_g1_s05_clean | mismatched | 0.1 | 0.6320 | 0.6964 | 0.7120 | 79 | 71 | 150 | 500 |
| rel_contact | d_rel_contact_g1_s05_clean | subtype_matched | 0.5 | 0.6260 | 0.6919 | 0.7140 | 68 | 63 | 131 | 500 |
| rel_contact | d_attr_color_g1_s05_clean | mismatched | 0.25 | 0.6240 | 0.6887 | 0.7080 | 75 | 71 | 146 | 500 |
| rel_spatial | g_attr_clean | mismatched | 0.05 | 0.5860 | 0.6750 | 0.7740 | 92 | 58 | 150 | 500 |
| rel_spatial | g_rel_clean | type_matched | 0.05 | 0.5940 | 0.6710 | 0.7340 | 100 | 62 | 162 | 500 |
| rel_spatial | d_attr_color_g1_s05_clean | mismatched | 0.1 | 0.5860 | 0.6667 | 0.7420 | 86 | 52 | 138 | 500 |
| rel_spatial | g_cat_clean | mismatched | 0.5 | 0.5720 | 0.6603 | 0.7600 | 93 | 66 | 159 | 500 |
| rel_spatial | d_rel_contact_g1_s05_clean | type_matched | 0.25 | 0.5580 | 0.6584 | 0.7940 | 81 | 61 | 142 | 500 |
| rel_spatial | d_attr_count_g1_s05_clean | mismatched | 0.25 | 0.5680 | 0.6582 | 0.7640 | 88 | 63 | 151 | 500 |
| rel_spatial | d_cat_random_g1_s05_clean | mismatched | 0.1 | 0.5560 | 0.6531 | 0.7800 | 83 | 64 | 147 | 500 |
| rel_spatial | d_rel_spatial_g1_s05_clean | subtype_matched | 0.05 | 0.5460 | 0.6502 | 0.7980 | 74 | 60 | 134 | 500 |
| rel_spatial | d_cat_hard_g1_s05_clean | mismatched | 0.25 | 0.5600 | 0.6497 | 0.7560 | 83 | 62 | 145 | 500 |
| rel_spatial | d_cat_popular_g1_s05_clean | mismatched | 0.1 | 0.5580 | 0.6475 | 0.7540 | 86 | 66 | 152 | 500 |
| rel_spatial | g_all_clean | global | 0.1 | 0.5380 | 0.6473 | 0.8100 | 68 | 58 | 126 | 500 |

## All Rows
| eval_subset | method | vector | match | alpha | accuracy | precision | recall | f1 | yes_rate | tp | tn | fp | fn | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attr_color | baseline |  |  |  | 0.6660 | 0.6694 | 0.6560 | 0.6626 | 0.4900 | 164 | 169 | 81 | 86 | 0 | 0 | 0 | 500 |
| attr_color | steered | d_attr_color_g1_s05_clean | subtype_matched | 0.05 | 0.6580 | 0.6599 | 0.6520 | 0.6559 | 0.4940 | 163 | 166 | 84 | 87 | 73 | 77 | 150 | 500 |
| attr_color | steered | d_attr_color_g1_s05_clean | subtype_matched | 0.1 | 0.6860 | 0.6838 | 0.6920 | 0.6879 | 0.5060 | 173 | 170 | 80 | 77 | 83 | 73 | 156 | 500 |
| attr_color | steered | d_attr_color_g1_s05_clean | subtype_matched | 0.25 | 0.6880 | 0.6926 | 0.6760 | 0.6842 | 0.4880 | 169 | 175 | 75 | 81 | 93 | 82 | 175 | 500 |
| attr_color | steered | d_attr_color_g1_s05_clean | subtype_matched | 0.5 | 0.6660 | 0.6602 | 0.6840 | 0.6719 | 0.5180 | 171 | 162 | 88 | 79 | 83 | 83 | 166 | 500 |
| attr_color | steered | d_attr_count_g1_s05_clean | type_matched | 0.05 | 0.6740 | 0.6747 | 0.6720 | 0.6733 | 0.4980 | 168 | 169 | 81 | 82 | 89 | 85 | 174 | 500 |
| attr_color | steered | d_attr_count_g1_s05_clean | type_matched | 0.1 | 0.6580 | 0.6612 | 0.6480 | 0.6545 | 0.4900 | 162 | 167 | 83 | 88 | 79 | 83 | 162 | 500 |
| attr_color | steered | d_attr_count_g1_s05_clean | type_matched | 0.25 | 0.6480 | 0.6504 | 0.6400 | 0.6452 | 0.4920 | 160 | 164 | 86 | 90 | 68 | 77 | 145 | 500 |
| attr_color | steered | d_attr_count_g1_s05_clean | type_matched | 0.5 | 0.6720 | 0.6792 | 0.6520 | 0.6653 | 0.4800 | 163 | 173 | 77 | 87 | 88 | 85 | 173 | 500 |
| attr_color | steered | d_cat_hard_g1_s05_clean | mismatched | 0.05 | 0.6980 | 0.7037 | 0.6840 | 0.6937 | 0.4860 | 171 | 178 | 72 | 79 | 86 | 70 | 156 | 500 |
| attr_color | steered | d_cat_hard_g1_s05_clean | mismatched | 0.1 | 0.6580 | 0.6639 | 0.6400 | 0.6517 | 0.4820 | 160 | 169 | 81 | 90 | 77 | 81 | 158 | 500 |
| attr_color | steered | d_cat_hard_g1_s05_clean | mismatched | 0.25 | 0.6960 | 0.7008 | 0.6840 | 0.6923 | 0.4880 | 171 | 177 | 73 | 79 | 89 | 74 | 163 | 500 |
| attr_color | steered | d_cat_hard_g1_s05_clean | mismatched | 0.5 | 0.6440 | 0.6385 | 0.6640 | 0.6510 | 0.5200 | 166 | 156 | 94 | 84 | 74 | 85 | 159 | 500 |
| attr_color | steered | d_cat_popular_g1_s05_clean | mismatched | 0.05 | 0.6860 | 0.6883 | 0.6800 | 0.6841 | 0.4940 | 170 | 173 | 77 | 80 | 86 | 76 | 162 | 500 |
| attr_color | steered | d_cat_popular_g1_s05_clean | mismatched | 0.1 | 0.6700 | 0.6667 | 0.6800 | 0.6733 | 0.5100 | 170 | 165 | 85 | 80 | 80 | 78 | 158 | 500 |
| attr_color | steered | d_cat_popular_g1_s05_clean | mismatched | 0.25 | 0.6620 | 0.6576 | 0.6760 | 0.6667 | 0.5140 | 169 | 162 | 88 | 81 | 71 | 73 | 144 | 500 |
| attr_color | steered | d_cat_popular_g1_s05_clean | mismatched | 0.5 | 0.6880 | 0.6850 | 0.6960 | 0.6905 | 0.5080 | 174 | 170 | 80 | 76 | 85 | 74 | 159 | 500 |
| attr_color | steered | d_cat_random_g1_s05_clean | mismatched | 0.05 | 0.6780 | 0.6787 | 0.6760 | 0.6774 | 0.4980 | 169 | 170 | 80 | 81 | 82 | 76 | 158 | 500 |
| attr_color | steered | d_cat_random_g1_s05_clean | mismatched | 0.1 | 0.7000 | 0.7066 | 0.6840 | 0.6951 | 0.4840 | 171 | 179 | 71 | 79 | 87 | 70 | 157 | 500 |
| attr_color | steered | d_cat_random_g1_s05_clean | mismatched | 0.25 | 0.6960 | 0.6960 | 0.6960 | 0.6960 | 0.5000 | 174 | 174 | 76 | 76 | 94 | 79 | 173 | 500 |
| attr_color | steered | d_cat_random_g1_s05_clean | mismatched | 0.5 | 0.5960 | 0.6026 | 0.5640 | 0.5826 | 0.4680 | 141 | 157 | 93 | 109 | 72 | 107 | 179 | 500 |
| attr_color | steered | d_rel_contact_g1_s05_clean | mismatched | 0.05 | 0.6480 | 0.6412 | 0.6720 | 0.6563 | 0.5240 | 168 | 156 | 94 | 82 | 76 | 85 | 161 | 500 |
| attr_color | steered | d_rel_contact_g1_s05_clean | mismatched | 0.1 | 0.6540 | 0.6571 | 0.6440 | 0.6505 | 0.4900 | 161 | 166 | 84 | 89 | 83 | 89 | 172 | 500 |
| attr_color | steered | d_rel_contact_g1_s05_clean | mismatched | 0.25 | 0.6460 | 0.6431 | 0.6560 | 0.6495 | 0.5100 | 164 | 159 | 91 | 86 | 75 | 85 | 160 | 500 |
| attr_color | steered | d_rel_contact_g1_s05_clean | mismatched | 0.5 | 0.6600 | 0.6575 | 0.6680 | 0.6627 | 0.5080 | 167 | 163 | 87 | 83 | 78 | 81 | 159 | 500 |
| attr_color | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.05 | 0.6900 | 0.6877 | 0.6960 | 0.6918 | 0.5060 | 174 | 171 | 79 | 76 | 90 | 78 | 168 | 500 |
| attr_color | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.1 | 0.6920 | 0.6890 | 0.7000 | 0.6944 | 0.5080 | 175 | 171 | 79 | 75 | 93 | 80 | 173 | 500 |
| attr_color | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.25 | 0.6860 | 0.6795 | 0.7040 | 0.6916 | 0.5180 | 176 | 167 | 83 | 74 | 89 | 79 | 168 | 500 |
| attr_color | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.5 | 0.6980 | 0.6941 | 0.7080 | 0.7010 | 0.5100 | 177 | 172 | 78 | 73 | 80 | 64 | 144 | 500 |
| attr_color | steered | g_all_clean | global | 0.05 | 0.6660 | 0.6694 | 0.6560 | 0.6626 | 0.4900 | 164 | 169 | 81 | 86 | 79 | 79 | 158 | 500 |
| attr_color | steered | g_all_clean | global | 0.1 | 0.6800 | 0.6875 | 0.6600 | 0.6735 | 0.4800 | 165 | 175 | 75 | 85 | 86 | 79 | 165 | 500 |
| attr_color | steered | g_all_clean | global | 0.25 | 0.6540 | 0.6534 | 0.6560 | 0.6547 | 0.5020 | 164 | 163 | 87 | 86 | 70 | 76 | 146 | 500 |
| attr_color | steered | g_all_clean | global | 0.5 | 0.6440 | 0.6364 | 0.6720 | 0.6537 | 0.5280 | 168 | 154 | 96 | 82 | 79 | 90 | 169 | 500 |
| attr_color | steered | g_attr_clean | type_matched | 0.05 | 0.6660 | 0.6602 | 0.6840 | 0.6719 | 0.5180 | 171 | 162 | 88 | 79 | 84 | 84 | 168 | 500 |
| attr_color | steered | g_attr_clean | type_matched | 0.1 | 0.6660 | 0.6751 | 0.6400 | 0.6571 | 0.4740 | 160 | 173 | 77 | 90 | 72 | 72 | 144 | 500 |
| attr_color | steered | g_attr_clean | type_matched | 0.25 | 0.6840 | 0.6949 | 0.6560 | 0.6749 | 0.4720 | 164 | 178 | 72 | 86 | 80 | 71 | 151 | 500 |
| attr_color | steered | g_attr_clean | type_matched | 0.5 | 0.6560 | 0.6639 | 0.6320 | 0.6475 | 0.4760 | 158 | 170 | 80 | 92 | 69 | 74 | 143 | 500 |
| attr_color | steered | g_cat_clean | mismatched | 0.05 | 0.6900 | 0.6806 | 0.7160 | 0.6979 | 0.5260 | 179 | 166 | 84 | 71 | 89 | 77 | 166 | 500 |
| attr_color | steered | g_cat_clean | mismatched | 0.1 | 0.7000 | 0.6969 | 0.7080 | 0.7024 | 0.5080 | 177 | 173 | 77 | 73 | 86 | 69 | 155 | 500 |
| attr_color | steered | g_cat_clean | mismatched | 0.25 | 0.6640 | 0.6640 | 0.6640 | 0.6640 | 0.5000 | 166 | 166 | 84 | 84 | 76 | 77 | 153 | 500 |
| attr_color | steered | g_cat_clean | mismatched | 0.5 | 0.6380 | 0.6386 | 0.6360 | 0.6373 | 0.4980 | 159 | 160 | 90 | 91 | 78 | 92 | 170 | 500 |
| attr_color | steered | g_rel_clean | mismatched | 0.05 | 0.6800 | 0.6844 | 0.6680 | 0.6761 | 0.4880 | 167 | 173 | 77 | 83 | 81 | 74 | 155 | 500 |
| attr_color | steered | g_rel_clean | mismatched | 0.1 | 0.6960 | 0.6960 | 0.6960 | 0.6960 | 0.5000 | 174 | 174 | 76 | 76 | 88 | 73 | 161 | 500 |
| attr_color | steered | g_rel_clean | mismatched | 0.25 | 0.6640 | 0.6602 | 0.6760 | 0.6680 | 0.5120 | 169 | 163 | 87 | 81 | 85 | 86 | 171 | 500 |
| attr_color | steered | g_rel_clean | mismatched | 0.5 | 0.6860 | 0.6782 | 0.7080 | 0.6928 | 0.5220 | 177 | 166 | 84 | 73 | 86 | 76 | 162 | 500 |
| attr_count | baseline |  |  |  | 0.5260 | 0.5281 | 0.4880 | 0.5073 | 0.4620 | 122 | 141 | 109 | 128 | 0 | 0 | 0 | 500 |
| attr_count | steered | d_attr_color_g1_s05_clean | type_matched | 0.05 | 0.5340 | 0.5365 | 0.5000 | 0.5176 | 0.4660 | 125 | 142 | 108 | 125 | 106 | 102 | 208 | 500 |
| attr_count | steered | d_attr_color_g1_s05_clean | type_matched | 0.1 | 0.5760 | 0.5779 | 0.5640 | 0.5709 | 0.4880 | 141 | 147 | 103 | 109 | 117 | 92 | 209 | 500 |
| attr_count | steered | d_attr_color_g1_s05_clean | type_matched | 0.25 | 0.5380 | 0.5446 | 0.4640 | 0.5011 | 0.4260 | 116 | 153 | 97 | 134 | 107 | 101 | 208 | 500 |
| attr_count | steered | d_attr_color_g1_s05_clean | type_matched | 0.5 | 0.5340 | 0.5353 | 0.5160 | 0.5255 | 0.4820 | 129 | 138 | 112 | 121 | 107 | 103 | 210 | 500 |
| attr_count | steered | d_attr_count_g1_s05_clean | subtype_matched | 0.05 | 0.5360 | 0.5405 | 0.4800 | 0.5085 | 0.4440 | 120 | 148 | 102 | 130 | 96 | 91 | 187 | 500 |
| attr_count | steered | d_attr_count_g1_s05_clean | subtype_matched | 0.1 | 0.5340 | 0.5374 | 0.4880 | 0.5115 | 0.4540 | 122 | 145 | 105 | 128 | 106 | 102 | 208 | 500 |
| attr_count | steered | d_attr_count_g1_s05_clean | subtype_matched | 0.25 | 0.5320 | 0.5357 | 0.4800 | 0.5063 | 0.4480 | 120 | 146 | 104 | 130 | 92 | 89 | 181 | 500 |
| attr_count | steered | d_attr_count_g1_s05_clean | subtype_matched | 0.5 | 0.5240 | 0.5275 | 0.4600 | 0.4915 | 0.4360 | 115 | 147 | 103 | 135 | 106 | 107 | 213 | 500 |
| attr_count | steered | d_cat_hard_g1_s05_clean | mismatched | 0.05 | 0.5840 | 0.5938 | 0.5320 | 0.5612 | 0.4480 | 133 | 159 | 91 | 117 | 115 | 86 | 201 | 500 |
| attr_count | steered | d_cat_hard_g1_s05_clean | mismatched | 0.1 | 0.5200 | 0.5231 | 0.4520 | 0.4850 | 0.4320 | 113 | 147 | 103 | 137 | 97 | 100 | 197 | 500 |
| attr_count | steered | d_cat_hard_g1_s05_clean | mismatched | 0.25 | 0.5700 | 0.5785 | 0.5160 | 0.5455 | 0.4460 | 129 | 156 | 94 | 121 | 117 | 95 | 212 | 500 |
| attr_count | steered | d_cat_hard_g1_s05_clean | mismatched | 0.5 | 0.5260 | 0.5257 | 0.5320 | 0.5288 | 0.5060 | 133 | 130 | 120 | 117 | 108 | 108 | 216 | 500 |
| attr_count | steered | d_cat_popular_g1_s05_clean | mismatched | 0.05 | 0.5480 | 0.5522 | 0.5080 | 0.5292 | 0.4600 | 127 | 147 | 103 | 123 | 108 | 97 | 205 | 500 |
| attr_count | steered | d_cat_popular_g1_s05_clean | mismatched | 0.1 | 0.5460 | 0.5469 | 0.5360 | 0.5414 | 0.4900 | 134 | 139 | 111 | 116 | 104 | 94 | 198 | 500 |
| attr_count | steered | d_cat_popular_g1_s05_clean | mismatched | 0.25 | 0.5380 | 0.5404 | 0.5080 | 0.5237 | 0.4700 | 127 | 142 | 108 | 123 | 102 | 96 | 198 | 500 |
| attr_count | steered | d_cat_popular_g1_s05_clean | mismatched | 0.5 | 0.5820 | 0.5865 | 0.5560 | 0.5708 | 0.4740 | 139 | 152 | 98 | 111 | 118 | 90 | 208 | 500 |
| attr_count | steered | d_cat_random_g1_s05_clean | mismatched | 0.05 | 0.5780 | 0.5830 | 0.5480 | 0.5649 | 0.4700 | 137 | 152 | 98 | 113 | 109 | 83 | 192 | 500 |
| attr_count | steered | d_cat_random_g1_s05_clean | mismatched | 0.1 | 0.5660 | 0.5727 | 0.5200 | 0.5451 | 0.4540 | 130 | 153 | 97 | 120 | 114 | 94 | 208 | 500 |
| attr_count | steered | d_cat_random_g1_s05_clean | mismatched | 0.25 | 0.5580 | 0.5639 | 0.5120 | 0.5367 | 0.4540 | 128 | 151 | 99 | 122 | 112 | 96 | 208 | 500 |
| attr_count | steered | d_cat_random_g1_s05_clean | mismatched | 0.5 | 0.4920 | 0.4909 | 0.4320 | 0.4596 | 0.4400 | 108 | 138 | 112 | 142 | 96 | 113 | 209 | 500 |
| attr_count | steered | d_rel_contact_g1_s05_clean | mismatched | 0.05 | 0.5520 | 0.5546 | 0.5280 | 0.5410 | 0.4760 | 132 | 144 | 106 | 118 | 113 | 100 | 213 | 500 |
| attr_count | steered | d_rel_contact_g1_s05_clean | mismatched | 0.1 | 0.5000 | 0.5000 | 0.4760 | 0.4877 | 0.4760 | 119 | 131 | 119 | 131 | 91 | 104 | 195 | 500 |
| attr_count | steered | d_rel_contact_g1_s05_clean | mismatched | 0.25 | 0.5260 | 0.5263 | 0.5200 | 0.5231 | 0.4940 | 130 | 133 | 117 | 120 | 100 | 100 | 200 | 500 |
| attr_count | steered | d_rel_contact_g1_s05_clean | mismatched | 0.5 | 0.5380 | 0.5385 | 0.5320 | 0.5352 | 0.4940 | 133 | 136 | 114 | 117 | 106 | 100 | 206 | 500 |
| attr_count | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.05 | 0.5140 | 0.5158 | 0.4560 | 0.4841 | 0.4420 | 114 | 143 | 107 | 136 | 103 | 109 | 212 | 500 |
| attr_count | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.1 | 0.5200 | 0.5216 | 0.4840 | 0.5021 | 0.4640 | 121 | 139 | 111 | 129 | 111 | 114 | 225 | 500 |
| attr_count | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.25 | 0.5420 | 0.5436 | 0.5240 | 0.5336 | 0.4820 | 131 | 140 | 110 | 119 | 111 | 103 | 214 | 500 |
| attr_count | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.5 | 0.5720 | 0.5811 | 0.5160 | 0.5466 | 0.4440 | 129 | 157 | 93 | 121 | 112 | 89 | 201 | 500 |
| attr_count | steered | g_all_clean | global | 0.05 | 0.5340 | 0.5353 | 0.5160 | 0.5255 | 0.4820 | 129 | 138 | 112 | 121 | 100 | 96 | 196 | 500 |
| attr_count | steered | g_all_clean | global | 0.1 | 0.5380 | 0.5455 | 0.4560 | 0.4967 | 0.4180 | 114 | 155 | 95 | 136 | 106 | 100 | 206 | 500 |
| attr_count | steered | g_all_clean | global | 0.25 | 0.5380 | 0.5397 | 0.5160 | 0.5276 | 0.4780 | 129 | 140 | 110 | 121 | 103 | 97 | 200 | 500 |
| attr_count | steered | g_all_clean | global | 0.5 | 0.5000 | 0.5000 | 0.4800 | 0.4898 | 0.4800 | 120 | 130 | 120 | 130 | 99 | 112 | 211 | 500 |
| attr_count | steered | g_attr_clean | type_matched | 0.05 | 0.5480 | 0.5492 | 0.5360 | 0.5425 | 0.4880 | 134 | 140 | 110 | 116 | 103 | 92 | 195 | 500 |
| attr_count | steered | g_attr_clean | type_matched | 0.1 | 0.5480 | 0.5550 | 0.4840 | 0.5171 | 0.4360 | 121 | 153 | 97 | 129 | 99 | 88 | 187 | 500 |
| attr_count | steered | g_attr_clean | type_matched | 0.25 | 0.5360 | 0.5402 | 0.4840 | 0.5105 | 0.4480 | 121 | 147 | 103 | 129 | 106 | 101 | 207 | 500 |
| attr_count | steered | g_attr_clean | type_matched | 0.5 | 0.5060 | 0.5067 | 0.4560 | 0.4800 | 0.4500 | 114 | 139 | 111 | 136 | 92 | 102 | 194 | 500 |
| attr_count | steered | g_cat_clean | mismatched | 0.05 | 0.5640 | 0.5661 | 0.5480 | 0.5569 | 0.4840 | 137 | 145 | 105 | 113 | 112 | 93 | 205 | 500 |
| attr_count | steered | g_cat_clean | mismatched | 0.1 | 0.5560 | 0.5614 | 0.5120 | 0.5356 | 0.4560 | 128 | 150 | 100 | 122 | 108 | 93 | 201 | 500 |
| attr_count | steered | g_cat_clean | mismatched | 0.25 | 0.5380 | 0.5422 | 0.4880 | 0.5137 | 0.4500 | 122 | 147 | 103 | 128 | 105 | 99 | 204 | 500 |
| attr_count | steered | g_cat_clean | mismatched | 0.5 | 0.4880 | 0.4864 | 0.4280 | 0.4553 | 0.4400 | 107 | 137 | 113 | 143 | 99 | 118 | 217 | 500 |
| attr_count | steered | g_rel_clean | mismatched | 0.05 | 0.5420 | 0.5467 | 0.4920 | 0.5179 | 0.4500 | 123 | 148 | 102 | 127 | 112 | 104 | 216 | 500 |
| attr_count | steered | g_rel_clean | mismatched | 0.1 | 0.5800 | 0.5917 | 0.5160 | 0.5513 | 0.4360 | 129 | 161 | 89 | 121 | 121 | 94 | 215 | 500 |
| attr_count | steered | g_rel_clean | mismatched | 0.25 | 0.5520 | 0.5575 | 0.5040 | 0.5294 | 0.4520 | 126 | 150 | 100 | 124 | 115 | 102 | 217 | 500 |
| attr_count | steered | g_rel_clean | mismatched | 0.5 | 0.5660 | 0.5696 | 0.5400 | 0.5544 | 0.4740 | 135 | 148 | 102 | 115 | 110 | 90 | 200 | 500 |
| cat_hard | baseline |  |  |  | 0.8860 | 0.8939 | 0.8760 | 0.8848 | 0.4900 | 219 | 224 | 26 | 31 | 0 | 0 | 0 | 500 |
| cat_hard | steered | d_attr_color_g1_s05_clean | mismatched | 0.05 | 0.8760 | 0.8852 | 0.8640 | 0.8745 | 0.4880 | 216 | 222 | 28 | 34 | 29 | 34 | 63 | 500 |
| cat_hard | steered | d_attr_color_g1_s05_clean | mismatched | 0.1 | 0.8620 | 0.8755 | 0.8440 | 0.8595 | 0.4820 | 211 | 220 | 30 | 39 | 28 | 40 | 68 | 500 |
| cat_hard | steered | d_attr_color_g1_s05_clean | mismatched | 0.25 | 0.9000 | 0.9098 | 0.8880 | 0.8988 | 0.4880 | 222 | 228 | 22 | 28 | 33 | 26 | 59 | 500 |
| cat_hard | steered | d_attr_color_g1_s05_clean | mismatched | 0.5 | 0.8760 | 0.8821 | 0.8680 | 0.8750 | 0.4920 | 217 | 221 | 29 | 33 | 24 | 29 | 53 | 500 |
| cat_hard | steered | d_attr_count_g1_s05_clean | mismatched | 0.05 | 0.8740 | 0.8816 | 0.8640 | 0.8727 | 0.4900 | 216 | 221 | 29 | 34 | 26 | 32 | 58 | 500 |
| cat_hard | steered | d_attr_count_g1_s05_clean | mismatched | 0.1 | 0.8680 | 0.8680 | 0.8680 | 0.8680 | 0.5000 | 217 | 217 | 33 | 33 | 28 | 37 | 65 | 500 |
| cat_hard | steered | d_attr_count_g1_s05_clean | mismatched | 0.25 | 0.8960 | 0.9091 | 0.8800 | 0.8943 | 0.4840 | 220 | 228 | 22 | 30 | 27 | 22 | 49 | 500 |
| cat_hard | steered | d_attr_count_g1_s05_clean | mismatched | 0.5 | 0.9040 | 0.9174 | 0.8880 | 0.9024 | 0.4840 | 222 | 230 | 20 | 28 | 31 | 22 | 53 | 500 |
| cat_hard | steered | d_cat_hard_g1_s05_clean | subtype_matched | 0.05 | 0.8760 | 0.9017 | 0.8440 | 0.8719 | 0.4680 | 211 | 227 | 23 | 39 | 26 | 31 | 57 | 500 |
| cat_hard | steered | d_cat_hard_g1_s05_clean | subtype_matched | 0.1 | 0.8720 | 0.8811 | 0.8600 | 0.8704 | 0.4880 | 215 | 221 | 29 | 35 | 27 | 34 | 61 | 500 |
| cat_hard | steered | d_cat_hard_g1_s05_clean | subtype_matched | 0.25 | 0.8640 | 0.8730 | 0.8520 | 0.8623 | 0.4880 | 213 | 219 | 31 | 37 | 29 | 40 | 69 | 500 |
| cat_hard | steered | d_cat_hard_g1_s05_clean | subtype_matched | 0.5 | 0.8760 | 0.8950 | 0.8520 | 0.8730 | 0.4760 | 213 | 225 | 25 | 37 | 30 | 35 | 65 | 500 |
| cat_hard | steered | d_cat_popular_g1_s05_clean | type_matched | 0.05 | 0.8940 | 0.9121 | 0.8720 | 0.8916 | 0.4780 | 218 | 229 | 21 | 32 | 32 | 28 | 60 | 500 |
| cat_hard | steered | d_cat_popular_g1_s05_clean | type_matched | 0.1 | 0.8760 | 0.8730 | 0.8800 | 0.8765 | 0.5040 | 220 | 218 | 32 | 30 | 29 | 34 | 63 | 500 |
| cat_hard | steered | d_cat_popular_g1_s05_clean | type_matched | 0.25 | 0.8740 | 0.8979 | 0.8440 | 0.8701 | 0.4700 | 211 | 226 | 24 | 39 | 25 | 31 | 56 | 500 |
| cat_hard | steered | d_cat_popular_g1_s05_clean | type_matched | 0.5 | 0.8860 | 0.8845 | 0.8880 | 0.8862 | 0.5020 | 222 | 221 | 29 | 28 | 34 | 34 | 68 | 500 |
| cat_hard | steered | d_cat_random_g1_s05_clean | type_matched | 0.05 | 0.8860 | 0.8939 | 0.8760 | 0.8848 | 0.4900 | 219 | 224 | 26 | 31 | 32 | 32 | 64 | 500 |
| cat_hard | steered | d_cat_random_g1_s05_clean | type_matched | 0.1 | 0.8900 | 0.8980 | 0.8800 | 0.8889 | 0.4900 | 220 | 225 | 25 | 30 | 28 | 26 | 54 | 500 |
| cat_hard | steered | d_cat_random_g1_s05_clean | type_matched | 0.25 | 0.8900 | 0.8947 | 0.8840 | 0.8893 | 0.4940 | 221 | 224 | 26 | 29 | 26 | 24 | 50 | 500 |
| cat_hard | steered | d_cat_random_g1_s05_clean | type_matched | 0.5 | 0.8600 | 0.8750 | 0.8400 | 0.8571 | 0.4800 | 210 | 220 | 30 | 40 | 24 | 37 | 61 | 500 |
| cat_hard | steered | d_rel_contact_g1_s05_clean | mismatched | 0.05 | 0.8900 | 0.8947 | 0.8840 | 0.8893 | 0.4940 | 221 | 224 | 26 | 29 | 30 | 28 | 58 | 500 |
| cat_hard | steered | d_rel_contact_g1_s05_clean | mismatched | 0.1 | 0.8900 | 0.9012 | 0.8760 | 0.8884 | 0.4860 | 219 | 226 | 24 | 31 | 31 | 29 | 60 | 500 |
| cat_hard | steered | d_rel_contact_g1_s05_clean | mismatched | 0.25 | 0.8860 | 0.8971 | 0.8720 | 0.8844 | 0.4860 | 218 | 225 | 25 | 32 | 31 | 31 | 62 | 500 |
| cat_hard | steered | d_rel_contact_g1_s05_clean | mismatched | 0.5 | 0.8780 | 0.8921 | 0.8600 | 0.8758 | 0.4820 | 215 | 224 | 26 | 35 | 27 | 31 | 58 | 500 |
| cat_hard | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.05 | 0.8960 | 0.9024 | 0.8880 | 0.8952 | 0.4920 | 222 | 226 | 24 | 28 | 30 | 25 | 55 | 500 |
| cat_hard | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.1 | 0.9120 | 0.9292 | 0.8920 | 0.9102 | 0.4800 | 223 | 233 | 17 | 27 | 34 | 21 | 55 | 500 |
| cat_hard | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.25 | 0.8960 | 0.9057 | 0.8840 | 0.8947 | 0.4880 | 221 | 227 | 23 | 29 | 29 | 24 | 53 | 500 |
| cat_hard | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.5 | 0.8920 | 0.8952 | 0.8880 | 0.8916 | 0.4960 | 222 | 224 | 26 | 28 | 30 | 27 | 57 | 500 |
| cat_hard | steered | g_all_clean | global | 0.05 | 0.8800 | 0.8958 | 0.8600 | 0.8776 | 0.4800 | 215 | 225 | 25 | 35 | 26 | 29 | 55 | 500 |
| cat_hard | steered | g_all_clean | global | 0.1 | 0.8940 | 0.9087 | 0.8760 | 0.8921 | 0.4820 | 219 | 228 | 22 | 31 | 27 | 23 | 50 | 500 |
| cat_hard | steered | g_all_clean | global | 0.25 | 0.8680 | 0.8770 | 0.8560 | 0.8664 | 0.4880 | 214 | 220 | 30 | 36 | 24 | 33 | 57 | 500 |
| cat_hard | steered | g_all_clean | global | 0.5 | 0.8740 | 0.8816 | 0.8640 | 0.8727 | 0.4900 | 216 | 221 | 29 | 34 | 29 | 35 | 64 | 500 |
| cat_hard | steered | g_attr_clean | mismatched | 0.05 | 0.9020 | 0.8911 | 0.9160 | 0.9034 | 0.5140 | 229 | 222 | 28 | 21 | 38 | 30 | 68 | 500 |
| cat_hard | steered | g_attr_clean | mismatched | 0.1 | 0.8600 | 0.8629 | 0.8560 | 0.8594 | 0.4960 | 214 | 216 | 34 | 36 | 25 | 38 | 63 | 500 |
| cat_hard | steered | g_attr_clean | mismatched | 0.25 | 0.8880 | 0.9076 | 0.8640 | 0.8852 | 0.4760 | 216 | 228 | 22 | 34 | 29 | 28 | 57 | 500 |
| cat_hard | steered | g_attr_clean | mismatched | 0.5 | 0.8780 | 0.8987 | 0.8520 | 0.8747 | 0.4740 | 213 | 226 | 24 | 37 | 25 | 29 | 54 | 500 |
| cat_hard | steered | g_cat_clean | type_matched | 0.05 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.5000 | 225 | 225 | 25 | 25 | 33 | 26 | 59 | 500 |
| cat_hard | steered | g_cat_clean | type_matched | 0.1 | 0.9000 | 0.8968 | 0.9040 | 0.9004 | 0.5040 | 226 | 224 | 26 | 24 | 31 | 24 | 55 | 500 |
| cat_hard | steered | g_cat_clean | type_matched | 0.25 | 0.8700 | 0.8903 | 0.8440 | 0.8665 | 0.4740 | 211 | 224 | 26 | 39 | 28 | 36 | 64 | 500 |
| cat_hard | steered | g_cat_clean | type_matched | 0.5 | 0.8700 | 0.8807 | 0.8560 | 0.8682 | 0.4860 | 214 | 221 | 29 | 36 | 21 | 29 | 50 | 500 |
| cat_hard | steered | g_rel_clean | mismatched | 0.05 | 0.8700 | 0.8903 | 0.8440 | 0.8665 | 0.4740 | 211 | 224 | 26 | 39 | 27 | 35 | 62 | 500 |
| cat_hard | steered | g_rel_clean | mismatched | 0.1 | 0.8780 | 0.8857 | 0.8680 | 0.8768 | 0.4900 | 217 | 222 | 28 | 33 | 29 | 33 | 62 | 500 |
| cat_hard | steered | g_rel_clean | mismatched | 0.25 | 0.8580 | 0.8745 | 0.8360 | 0.8548 | 0.4780 | 209 | 220 | 30 | 41 | 24 | 38 | 62 | 500 |
| cat_hard | steered | g_rel_clean | mismatched | 0.5 | 0.9020 | 0.9241 | 0.8760 | 0.8994 | 0.4740 | 219 | 232 | 18 | 31 | 32 | 24 | 56 | 500 |
| cat_popular | baseline |  |  |  | 0.8980 | 0.9234 | 0.8680 | 0.8948 | 0.4700 | 217 | 232 | 18 | 33 | 0 | 0 | 0 | 500 |
| cat_popular | steered | d_attr_color_g1_s05_clean | mismatched | 0.05 | 0.9100 | 0.9399 | 0.8760 | 0.9068 | 0.4660 | 219 | 236 | 14 | 31 | 28 | 22 | 50 | 500 |
| cat_popular | steered | d_attr_color_g1_s05_clean | mismatched | 0.1 | 0.9280 | 0.9496 | 0.9040 | 0.9262 | 0.4760 | 226 | 238 | 12 | 24 | 32 | 17 | 49 | 500 |
| cat_popular | steered | d_attr_color_g1_s05_clean | mismatched | 0.25 | 0.9040 | 0.9353 | 0.8680 | 0.9004 | 0.4640 | 217 | 235 | 15 | 33 | 30 | 27 | 57 | 500 |
| cat_popular | steered | d_attr_color_g1_s05_clean | mismatched | 0.5 | 0.9120 | 0.9440 | 0.8760 | 0.9087 | 0.4640 | 219 | 237 | 13 | 31 | 26 | 19 | 45 | 500 |
| cat_popular | steered | d_attr_count_g1_s05_clean | mismatched | 0.05 | 0.9300 | 0.9424 | 0.9160 | 0.9290 | 0.4860 | 229 | 236 | 14 | 21 | 31 | 15 | 46 | 500 |
| cat_popular | steered | d_attr_count_g1_s05_clean | mismatched | 0.1 | 0.9040 | 0.9353 | 0.8680 | 0.9004 | 0.4640 | 217 | 235 | 15 | 33 | 26 | 23 | 49 | 500 |
| cat_popular | steered | d_attr_count_g1_s05_clean | mismatched | 0.25 | 0.9240 | 0.9454 | 0.9000 | 0.9221 | 0.4760 | 225 | 237 | 13 | 25 | 30 | 17 | 47 | 500 |
| cat_popular | steered | d_attr_count_g1_s05_clean | mismatched | 0.5 | 0.9200 | 0.9646 | 0.8720 | 0.9160 | 0.4520 | 218 | 242 | 8 | 32 | 26 | 15 | 41 | 500 |
| cat_popular | steered | d_cat_hard_g1_s05_clean | type_matched | 0.05 | 0.9180 | 0.9336 | 0.9000 | 0.9165 | 0.4820 | 225 | 234 | 16 | 25 | 32 | 22 | 54 | 500 |
| cat_popular | steered | d_cat_hard_g1_s05_clean | type_matched | 0.1 | 0.9120 | 0.9440 | 0.8760 | 0.9087 | 0.4640 | 219 | 237 | 13 | 31 | 26 | 19 | 45 | 500 |
| cat_popular | steered | d_cat_hard_g1_s05_clean | type_matched | 0.25 | 0.9020 | 0.9351 | 0.8640 | 0.8981 | 0.4620 | 216 | 235 | 15 | 34 | 26 | 24 | 50 | 500 |
| cat_popular | steered | d_cat_hard_g1_s05_clean | type_matched | 0.5 | 0.9120 | 0.9440 | 0.8760 | 0.9087 | 0.4640 | 219 | 237 | 13 | 31 | 27 | 20 | 47 | 500 |
| cat_popular | steered | d_cat_popular_g1_s05_clean | subtype_matched | 0.05 | 0.9200 | 0.9487 | 0.8880 | 0.9174 | 0.4680 | 222 | 238 | 12 | 28 | 31 | 20 | 51 | 500 |
| cat_popular | steered | d_cat_popular_g1_s05_clean | subtype_matched | 0.1 | 0.9140 | 0.9481 | 0.8760 | 0.9106 | 0.4620 | 219 | 238 | 12 | 31 | 29 | 21 | 50 | 500 |
| cat_popular | steered | d_cat_popular_g1_s05_clean | subtype_matched | 0.25 | 0.9080 | 0.9359 | 0.8760 | 0.9050 | 0.4680 | 219 | 235 | 15 | 31 | 30 | 25 | 55 | 500 |
| cat_popular | steered | d_cat_popular_g1_s05_clean | subtype_matched | 0.5 | 0.9060 | 0.9356 | 0.8720 | 0.9027 | 0.4660 | 218 | 235 | 15 | 32 | 28 | 24 | 52 | 500 |
| cat_popular | steered | d_cat_random_g1_s05_clean | type_matched | 0.05 | 0.9060 | 0.9247 | 0.8840 | 0.9039 | 0.4780 | 221 | 232 | 18 | 29 | 32 | 28 | 60 | 500 |
| cat_popular | steered | d_cat_random_g1_s05_clean | type_matched | 0.1 | 0.9240 | 0.9492 | 0.8960 | 0.9218 | 0.4720 | 224 | 238 | 12 | 26 | 30 | 17 | 47 | 500 |
| cat_popular | steered | d_cat_random_g1_s05_clean | type_matched | 0.25 | 0.9220 | 0.9648 | 0.8760 | 0.9182 | 0.4540 | 219 | 242 | 8 | 31 | 31 | 19 | 50 | 500 |
| cat_popular | steered | d_cat_random_g1_s05_clean | type_matched | 0.5 | 0.9280 | 0.9652 | 0.8880 | 0.9250 | 0.4600 | 222 | 242 | 8 | 28 | 29 | 14 | 43 | 500 |
| cat_popular | steered | d_rel_contact_g1_s05_clean | mismatched | 0.05 | 0.8980 | 0.9383 | 0.8520 | 0.8931 | 0.4540 | 213 | 236 | 14 | 37 | 25 | 25 | 50 | 500 |
| cat_popular | steered | d_rel_contact_g1_s05_clean | mismatched | 0.1 | 0.9320 | 0.9737 | 0.8880 | 0.9289 | 0.4560 | 222 | 244 | 6 | 28 | 31 | 14 | 45 | 500 |
| cat_popular | steered | d_rel_contact_g1_s05_clean | mismatched | 0.25 | 0.9080 | 0.9435 | 0.8680 | 0.9042 | 0.4600 | 217 | 237 | 13 | 33 | 29 | 24 | 53 | 500 |
| cat_popular | steered | d_rel_contact_g1_s05_clean | mismatched | 0.5 | 0.9260 | 0.9651 | 0.8840 | 0.9228 | 0.4580 | 221 | 242 | 8 | 29 | 26 | 12 | 38 | 500 |
| cat_popular | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.05 | 0.9360 | 0.9619 | 0.9080 | 0.9342 | 0.4720 | 227 | 241 | 9 | 23 | 32 | 13 | 45 | 500 |
| cat_popular | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.1 | 0.9120 | 0.9402 | 0.8800 | 0.9091 | 0.4680 | 220 | 236 | 14 | 30 | 28 | 21 | 49 | 500 |
| cat_popular | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.25 | 0.9160 | 0.9407 | 0.8880 | 0.9136 | 0.4720 | 222 | 236 | 14 | 28 | 29 | 20 | 49 | 500 |
| cat_popular | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.5 | 0.9200 | 0.9449 | 0.8920 | 0.9177 | 0.4720 | 223 | 237 | 13 | 27 | 25 | 14 | 39 | 500 |
| cat_popular | steered | g_all_clean | global | 0.05 | 0.9160 | 0.9483 | 0.8800 | 0.9129 | 0.4640 | 220 | 238 | 12 | 30 | 28 | 19 | 47 | 500 |
| cat_popular | steered | g_all_clean | global | 0.1 | 0.9240 | 0.9417 | 0.9040 | 0.9224 | 0.4800 | 226 | 236 | 14 | 24 | 30 | 17 | 47 | 500 |
| cat_popular | steered | g_all_clean | global | 0.25 | 0.9200 | 0.9565 | 0.8800 | 0.9167 | 0.4600 | 220 | 240 | 10 | 30 | 27 | 16 | 43 | 500 |
| cat_popular | steered | g_all_clean | global | 0.5 | 0.9020 | 0.9136 | 0.8880 | 0.9006 | 0.4860 | 222 | 229 | 21 | 28 | 28 | 26 | 54 | 500 |
| cat_popular | steered | g_attr_clean | mismatched | 0.05 | 0.9280 | 0.9458 | 0.9080 | 0.9265 | 0.4800 | 227 | 237 | 13 | 23 | 28 | 13 | 41 | 500 |
| cat_popular | steered | g_attr_clean | mismatched | 0.1 | 0.9060 | 0.9356 | 0.8720 | 0.9027 | 0.4660 | 218 | 235 | 15 | 32 | 30 | 26 | 56 | 500 |
| cat_popular | steered | g_attr_clean | mismatched | 0.25 | 0.9020 | 0.9467 | 0.8520 | 0.8968 | 0.4500 | 213 | 238 | 12 | 37 | 25 | 23 | 48 | 500 |
| cat_popular | steered | g_attr_clean | mismatched | 0.5 | 0.9080 | 0.9322 | 0.8800 | 0.9053 | 0.4720 | 220 | 234 | 16 | 30 | 30 | 25 | 55 | 500 |
| cat_popular | steered | g_cat_clean | type_matched | 0.05 | 0.9120 | 0.9402 | 0.8800 | 0.9091 | 0.4680 | 220 | 236 | 14 | 30 | 32 | 25 | 57 | 500 |
| cat_popular | steered | g_cat_clean | type_matched | 0.1 | 0.9100 | 0.9362 | 0.8800 | 0.9072 | 0.4700 | 220 | 235 | 15 | 30 | 28 | 22 | 50 | 500 |
| cat_popular | steered | g_cat_clean | type_matched | 0.25 | 0.9020 | 0.9277 | 0.8720 | 0.8990 | 0.4700 | 218 | 233 | 17 | 32 | 29 | 27 | 56 | 500 |
| cat_popular | steered | g_cat_clean | type_matched | 0.5 | 0.9140 | 0.9331 | 0.8920 | 0.9121 | 0.4780 | 223 | 234 | 16 | 27 | 31 | 23 | 54 | 500 |
| cat_popular | steered | g_rel_clean | mismatched | 0.05 | 0.9180 | 0.9372 | 0.8960 | 0.9162 | 0.4780 | 224 | 235 | 15 | 26 | 30 | 20 | 50 | 500 |
| cat_popular | steered | g_rel_clean | mismatched | 0.1 | 0.9180 | 0.9485 | 0.8840 | 0.9151 | 0.4660 | 221 | 238 | 12 | 29 | 29 | 19 | 48 | 500 |
| cat_popular | steered | g_rel_clean | mismatched | 0.25 | 0.9060 | 0.9511 | 0.8560 | 0.9011 | 0.4500 | 214 | 239 | 11 | 36 | 23 | 19 | 42 | 500 |
| cat_popular | steered | g_rel_clean | mismatched | 0.5 | 0.9300 | 0.9461 | 0.9120 | 0.9287 | 0.4820 | 228 | 237 | 13 | 22 | 31 | 15 | 46 | 500 |
| cat_random | baseline |  |  |  | 0.9040 | 0.9391 | 0.8640 | 0.9000 | 0.4600 | 216 | 236 | 14 | 34 | 0 | 0 | 0 | 500 |
| cat_random | steered | d_attr_color_g1_s05_clean | mismatched | 0.05 | 0.9020 | 0.9205 | 0.8800 | 0.8998 | 0.4780 | 220 | 231 | 19 | 30 | 24 | 25 | 49 | 500 |
| cat_random | steered | d_attr_color_g1_s05_clean | mismatched | 0.1 | 0.9160 | 0.9370 | 0.8920 | 0.9139 | 0.4760 | 223 | 235 | 15 | 27 | 29 | 23 | 52 | 500 |
| cat_random | steered | d_attr_color_g1_s05_clean | mismatched | 0.25 | 0.9040 | 0.9280 | 0.8760 | 0.9012 | 0.4720 | 219 | 233 | 17 | 31 | 27 | 27 | 54 | 500 |
| cat_random | steered | d_attr_color_g1_s05_clean | mismatched | 0.5 | 0.9240 | 0.9609 | 0.8840 | 0.9208 | 0.4600 | 221 | 241 | 9 | 29 | 30 | 20 | 50 | 500 |
| cat_random | steered | d_attr_count_g1_s05_clean | mismatched | 0.05 | 0.9280 | 0.9458 | 0.9080 | 0.9265 | 0.4800 | 227 | 237 | 13 | 23 | 31 | 19 | 50 | 500 |
| cat_random | steered | d_attr_count_g1_s05_clean | mismatched | 0.1 | 0.9160 | 0.9333 | 0.8960 | 0.9143 | 0.4800 | 224 | 234 | 16 | 26 | 28 | 22 | 50 | 500 |
| cat_random | steered | d_attr_count_g1_s05_clean | mismatched | 0.25 | 0.9320 | 0.9655 | 0.8960 | 0.9295 | 0.4640 | 224 | 242 | 8 | 26 | 30 | 16 | 46 | 500 |
| cat_random | steered | d_attr_count_g1_s05_clean | mismatched | 0.5 | 0.9160 | 0.9561 | 0.8720 | 0.9121 | 0.4560 | 218 | 240 | 10 | 32 | 26 | 20 | 46 | 500 |
| cat_random | steered | d_cat_hard_g1_s05_clean | type_matched | 0.05 | 0.9100 | 0.9289 | 0.8880 | 0.9080 | 0.4780 | 222 | 233 | 17 | 28 | 26 | 23 | 49 | 500 |
| cat_random | steered | d_cat_hard_g1_s05_clean | type_matched | 0.1 | 0.9200 | 0.9449 | 0.8920 | 0.9177 | 0.4720 | 223 | 237 | 13 | 27 | 25 | 17 | 42 | 500 |
| cat_random | steered | d_cat_hard_g1_s05_clean | type_matched | 0.25 | 0.9220 | 0.9648 | 0.8760 | 0.9182 | 0.4540 | 219 | 242 | 8 | 31 | 29 | 20 | 49 | 500 |
| cat_random | steered | d_cat_hard_g1_s05_clean | type_matched | 0.5 | 0.9120 | 0.9518 | 0.8680 | 0.9079 | 0.4560 | 217 | 239 | 11 | 33 | 27 | 23 | 50 | 500 |
| cat_random | steered | d_cat_popular_g1_s05_clean | type_matched | 0.05 | 0.9160 | 0.9444 | 0.8840 | 0.9132 | 0.4680 | 221 | 237 | 13 | 29 | 30 | 24 | 54 | 500 |
| cat_random | steered | d_cat_popular_g1_s05_clean | type_matched | 0.1 | 0.9020 | 0.9351 | 0.8640 | 0.8981 | 0.4620 | 216 | 235 | 15 | 34 | 26 | 27 | 53 | 500 |
| cat_random | steered | d_cat_popular_g1_s05_clean | type_matched | 0.25 | 0.9120 | 0.9328 | 0.8880 | 0.9098 | 0.4760 | 222 | 234 | 16 | 28 | 29 | 25 | 54 | 500 |
| cat_random | steered | d_cat_popular_g1_s05_clean | type_matched | 0.5 | 0.9000 | 0.9202 | 0.8760 | 0.8975 | 0.4760 | 219 | 231 | 19 | 31 | 27 | 29 | 56 | 500 |
| cat_random | steered | d_cat_random_g1_s05_clean | subtype_matched | 0.05 | 0.9080 | 0.9322 | 0.8800 | 0.9053 | 0.4720 | 220 | 234 | 16 | 30 | 29 | 27 | 56 | 500 |
| cat_random | steered | d_cat_random_g1_s05_clean | subtype_matched | 0.1 | 0.9280 | 0.9496 | 0.9040 | 0.9262 | 0.4760 | 226 | 238 | 12 | 24 | 30 | 18 | 48 | 500 |
| cat_random | steered | d_cat_random_g1_s05_clean | subtype_matched | 0.25 | 0.9200 | 0.9605 | 0.8760 | 0.9163 | 0.4560 | 219 | 241 | 9 | 31 | 29 | 21 | 50 | 500 |
| cat_random | steered | d_cat_random_g1_s05_clean | subtype_matched | 0.5 | 0.9320 | 0.9576 | 0.9040 | 0.9300 | 0.4720 | 226 | 240 | 10 | 24 | 31 | 17 | 48 | 500 |
| cat_random | steered | d_rel_contact_g1_s05_clean | mismatched | 0.05 | 0.9080 | 0.9435 | 0.8680 | 0.9042 | 0.4600 | 217 | 237 | 13 | 33 | 23 | 21 | 44 | 500 |
| cat_random | steered | d_rel_contact_g1_s05_clean | mismatched | 0.1 | 0.9200 | 0.9565 | 0.8800 | 0.9167 | 0.4600 | 220 | 240 | 10 | 30 | 32 | 24 | 56 | 500 |
| cat_random | steered | d_rel_contact_g1_s05_clean | mismatched | 0.25 | 0.9120 | 0.9518 | 0.8680 | 0.9079 | 0.4560 | 217 | 239 | 11 | 33 | 27 | 23 | 50 | 500 |
| cat_random | steered | d_rel_contact_g1_s05_clean | mismatched | 0.5 | 0.9200 | 0.9565 | 0.8800 | 0.9167 | 0.4600 | 220 | 240 | 10 | 30 | 28 | 20 | 48 | 500 |
| cat_random | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.05 | 0.9220 | 0.9271 | 0.9160 | 0.9215 | 0.4940 | 229 | 232 | 18 | 21 | 32 | 23 | 55 | 500 |
| cat_random | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.1 | 0.9220 | 0.9528 | 0.8880 | 0.9193 | 0.4660 | 222 | 239 | 11 | 28 | 31 | 22 | 53 | 500 |
| cat_random | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.25 | 0.9220 | 0.9378 | 0.9040 | 0.9206 | 0.4820 | 226 | 235 | 15 | 24 | 32 | 23 | 55 | 500 |
| cat_random | steered | d_rel_spatial_g1_s05_clean | mismatched | 0.5 | 0.9240 | 0.9344 | 0.9120 | 0.9231 | 0.4880 | 228 | 234 | 16 | 22 | 31 | 21 | 52 | 500 |
| cat_random | steered | g_all_clean | global | 0.05 | 0.9280 | 0.9612 | 0.8920 | 0.9253 | 0.4640 | 223 | 241 | 9 | 27 | 29 | 17 | 46 | 500 |
| cat_random | steered | g_all_clean | global | 0.1 | 0.9340 | 0.9578 | 0.9080 | 0.9322 | 0.4740 | 227 | 240 | 10 | 23 | 32 | 17 | 49 | 500 |
| cat_random | steered | g_all_clean | global | 0.25 | 0.9120 | 0.9402 | 0.8800 | 0.9091 | 0.4680 | 220 | 236 | 14 | 30 | 28 | 24 | 52 | 500 |
| cat_random | steered | g_all_clean | global | 0.5 | 0.9040 | 0.9139 | 0.8920 | 0.9028 | 0.4880 | 223 | 229 | 21 | 27 | 29 | 29 | 58 | 500 |
| cat_random | steered | g_attr_clean | mismatched | 0.05 | 0.9300 | 0.9424 | 0.9160 | 0.9290 | 0.4860 | 229 | 236 | 14 | 21 | 27 | 14 | 41 | 500 |
| cat_random | steered | g_attr_clean | mismatched | 0.1 | 0.9000 | 0.9386 | 0.8560 | 0.8954 | 0.4560 | 214 | 236 | 14 | 36 | 26 | 28 | 54 | 500 |
| cat_random | steered | g_attr_clean | mismatched | 0.25 | 0.9020 | 0.9313 | 0.8680 | 0.8986 | 0.4660 | 217 | 234 | 16 | 33 | 28 | 29 | 57 | 500 |
| cat_random | steered | g_attr_clean | mismatched | 0.5 | 0.9060 | 0.9212 | 0.8880 | 0.9043 | 0.4820 | 222 | 231 | 19 | 28 | 27 | 26 | 53 | 500 |
| cat_random | steered | g_cat_clean | type_matched | 0.05 | 0.9140 | 0.9442 | 0.8800 | 0.9110 | 0.4660 | 220 | 237 | 13 | 30 | 32 | 27 | 59 | 500 |
| cat_random | steered | g_cat_clean | type_matched | 0.1 | 0.9280 | 0.9573 | 0.8960 | 0.9256 | 0.4680 | 224 | 240 | 10 | 26 | 28 | 16 | 44 | 500 |
| cat_random | steered | g_cat_clean | type_matched | 0.25 | 0.9100 | 0.9437 | 0.8720 | 0.9064 | 0.4620 | 218 | 237 | 13 | 32 | 29 | 26 | 55 | 500 |
| cat_random | steered | g_cat_clean | type_matched | 0.5 | 0.9220 | 0.9489 | 0.8920 | 0.9196 | 0.4700 | 223 | 238 | 12 | 27 | 26 | 17 | 43 | 500 |
| cat_random | steered | g_rel_clean | mismatched | 0.05 | 0.9160 | 0.9444 | 0.8840 | 0.9132 | 0.4680 | 221 | 237 | 13 | 29 | 28 | 22 | 50 | 500 |
| cat_random | steered | g_rel_clean | mismatched | 0.1 | 0.9040 | 0.9316 | 0.8720 | 0.9008 | 0.4680 | 218 | 234 | 16 | 32 | 28 | 28 | 56 | 500 |
| cat_random | steered | g_rel_clean | mismatched | 0.25 | 0.9240 | 0.9569 | 0.8880 | 0.9212 | 0.4640 | 222 | 240 | 10 | 28 | 28 | 18 | 46 | 500 |
| cat_random | steered | g_rel_clean | mismatched | 0.5 | 0.9240 | 0.9380 | 0.9080 | 0.9228 | 0.4840 | 227 | 235 | 15 | 23 | 30 | 20 | 50 | 500 |
| rel_contact | baseline |  |  |  | 0.6160 | 0.5819 | 0.8240 | 0.6821 | 0.7080 | 206 | 102 | 148 | 44 | 0 | 0 | 0 | 500 |
| rel_contact | steered | d_attr_color_g1_s05_clean | mismatched | 0.05 | 0.6260 | 0.5935 | 0.8000 | 0.6814 | 0.6740 | 200 | 113 | 137 | 50 | 77 | 72 | 149 | 500 |
| rel_contact | steered | d_attr_color_g1_s05_clean | mismatched | 0.1 | 0.6100 | 0.5779 | 0.8160 | 0.6766 | 0.7060 | 204 | 101 | 149 | 46 | 76 | 79 | 155 | 500 |
| rel_contact | steered | d_attr_color_g1_s05_clean | mismatched | 0.25 | 0.6240 | 0.5876 | 0.8320 | 0.6887 | 0.7080 | 208 | 104 | 146 | 42 | 75 | 71 | 146 | 500 |
| rel_contact | steered | d_attr_color_g1_s05_clean | mismatched | 0.5 | 0.6000 | 0.5694 | 0.8200 | 0.6721 | 0.7200 | 205 | 95 | 155 | 45 | 72 | 80 | 152 | 500 |
| rel_contact | steered | d_attr_count_g1_s05_clean | mismatched | 0.05 | 0.6580 | 0.6094 | 0.8800 | 0.7201 | 0.7220 | 220 | 109 | 141 | 30 | 84 | 63 | 147 | 500 |
| rel_contact | steered | d_attr_count_g1_s05_clean | mismatched | 0.1 | 0.6540 | 0.6149 | 0.8240 | 0.7043 | 0.6700 | 206 | 121 | 129 | 44 | 87 | 68 | 155 | 500 |
| rel_contact | steered | d_attr_count_g1_s05_clean | mismatched | 0.25 | 0.6300 | 0.5900 | 0.8520 | 0.6972 | 0.7220 | 213 | 102 | 148 | 37 | 71 | 64 | 135 | 500 |
| rel_contact | steered | d_attr_count_g1_s05_clean | mismatched | 0.5 | 0.6560 | 0.6140 | 0.8400 | 0.7095 | 0.6840 | 210 | 118 | 132 | 40 | 80 | 60 | 140 | 500 |
| rel_contact | steered | d_cat_hard_g1_s05_clean | mismatched | 0.05 | 0.6260 | 0.5908 | 0.8200 | 0.6868 | 0.6940 | 205 | 108 | 142 | 45 | 83 | 78 | 161 | 500 |
| rel_contact | steered | d_cat_hard_g1_s05_clean | mismatched | 0.1 | 0.6380 | 0.5989 | 0.8360 | 0.6978 | 0.6980 | 209 | 110 | 140 | 41 | 78 | 67 | 145 | 500 |
| rel_contact | steered | d_cat_hard_g1_s05_clean | mismatched | 0.25 | 0.6580 | 0.6119 | 0.8640 | 0.7164 | 0.7060 | 216 | 113 | 137 | 34 | 84 | 63 | 147 | 500 |
| rel_contact | steered | d_cat_hard_g1_s05_clean | mismatched | 0.5 | 0.6600 | 0.6176 | 0.8400 | 0.7119 | 0.6800 | 210 | 120 | 130 | 40 | 89 | 67 | 156 | 500 |
| rel_contact | steered | d_cat_popular_g1_s05_clean | mismatched | 0.05 | 0.6260 | 0.5918 | 0.8120 | 0.6847 | 0.6860 | 203 | 110 | 140 | 47 | 73 | 68 | 141 | 500 |
| rel_contact | steered | d_cat_popular_g1_s05_clean | mismatched | 0.1 | 0.6320 | 0.5927 | 0.8440 | 0.6964 | 0.7120 | 211 | 105 | 145 | 39 | 79 | 71 | 150 | 500 |
| rel_contact | steered | d_cat_popular_g1_s05_clean | mismatched | 0.25 | 0.6300 | 0.5905 | 0.8480 | 0.6962 | 0.7180 | 212 | 103 | 147 | 38 | 69 | 62 | 131 | 500 |
| rel_contact | steered | d_cat_popular_g1_s05_clean | mismatched | 0.5 | 0.6340 | 0.5965 | 0.8280 | 0.6935 | 0.6940 | 207 | 110 | 140 | 43 | 75 | 66 | 141 | 500 |
| rel_contact | steered | d_cat_random_g1_s05_clean | mismatched | 0.05 | 0.6400 | 0.6042 | 0.8120 | 0.6928 | 0.6720 | 203 | 117 | 133 | 47 | 77 | 65 | 142 | 500 |
| rel_contact | steered | d_cat_random_g1_s05_clean | mismatched | 0.1 | 0.6100 | 0.5770 | 0.8240 | 0.6787 | 0.7140 | 206 | 99 | 151 | 44 | 69 | 72 | 141 | 500 |
| rel_contact | steered | d_cat_random_g1_s05_clean | mismatched | 0.25 | 0.6700 | 0.6218 | 0.8680 | 0.7245 | 0.6980 | 217 | 118 | 132 | 33 | 84 | 57 | 141 | 500 |
| rel_contact | steered | d_cat_random_g1_s05_clean | mismatched | 0.5 | 0.6640 | 0.6152 | 0.8760 | 0.7228 | 0.7120 | 219 | 113 | 137 | 31 | 89 | 65 | 154 | 500 |
| rel_contact | steered | d_rel_contact_g1_s05_clean | subtype_matched | 0.05 | 0.6100 | 0.5770 | 0.8240 | 0.6787 | 0.7140 | 206 | 99 | 151 | 44 | 75 | 78 | 153 | 500 |
| rel_contact | steered | d_rel_contact_g1_s05_clean | subtype_matched | 0.1 | 0.6180 | 0.5850 | 0.8120 | 0.6801 | 0.6940 | 203 | 106 | 144 | 47 | 75 | 74 | 149 | 500 |
| rel_contact | steered | d_rel_contact_g1_s05_clean | subtype_matched | 0.25 | 0.6180 | 0.5831 | 0.8280 | 0.6843 | 0.7100 | 207 | 102 | 148 | 43 | 78 | 77 | 155 | 500 |
| rel_contact | steered | d_rel_contact_g1_s05_clean | subtype_matched | 0.5 | 0.6260 | 0.5882 | 0.8400 | 0.6919 | 0.7140 | 210 | 103 | 147 | 40 | 68 | 63 | 131 | 500 |
| rel_contact | steered | d_rel_spatial_g1_s05_clean | type_matched | 0.05 | 0.6340 | 0.5971 | 0.8240 | 0.6924 | 0.6900 | 206 | 111 | 139 | 44 | 80 | 71 | 151 | 500 |
| rel_contact | steered | d_rel_spatial_g1_s05_clean | type_matched | 0.1 | 0.6480 | 0.6016 | 0.8760 | 0.7134 | 0.7280 | 219 | 105 | 145 | 31 | 82 | 66 | 148 | 500 |
| rel_contact | steered | d_rel_spatial_g1_s05_clean | type_matched | 0.25 | 0.6580 | 0.6119 | 0.8640 | 0.7164 | 0.7060 | 216 | 113 | 137 | 34 | 80 | 59 | 139 | 500 |
| rel_contact | steered | d_rel_spatial_g1_s05_clean | type_matched | 0.5 | 0.6480 | 0.6039 | 0.8600 | 0.7096 | 0.7120 | 215 | 109 | 141 | 35 | 77 | 61 | 138 | 500 |
| rel_contact | steered | g_all_clean | global | 0.05 | 0.5960 | 0.5670 | 0.8120 | 0.6678 | 0.7160 | 203 | 95 | 155 | 47 | 63 | 73 | 136 | 500 |
| rel_contact | steered | g_all_clean | global | 0.1 | 0.6060 | 0.5746 | 0.8160 | 0.6744 | 0.7100 | 204 | 99 | 151 | 46 | 67 | 72 | 139 | 500 |
| rel_contact | steered | g_all_clean | global | 0.25 | 0.6460 | 0.6040 | 0.8480 | 0.7055 | 0.7020 | 212 | 111 | 139 | 38 | 81 | 66 | 147 | 500 |
| rel_contact | steered | g_all_clean | global | 0.5 | 0.6400 | 0.6029 | 0.8200 | 0.6949 | 0.6800 | 205 | 115 | 135 | 45 | 76 | 64 | 140 | 500 |
| rel_contact | steered | g_attr_clean | mismatched | 0.05 | 0.6580 | 0.6145 | 0.8480 | 0.7126 | 0.6900 | 212 | 117 | 133 | 38 | 87 | 66 | 153 | 500 |
| rel_contact | steered | g_attr_clean | mismatched | 0.1 | 0.6300 | 0.5905 | 0.8480 | 0.6962 | 0.7180 | 212 | 103 | 147 | 38 | 80 | 73 | 153 | 500 |
| rel_contact | steered | g_attr_clean | mismatched | 0.25 | 0.6220 | 0.5910 | 0.7920 | 0.6769 | 0.6700 | 198 | 113 | 137 | 52 | 81 | 78 | 159 | 500 |
| rel_contact | steered | g_attr_clean | mismatched | 0.5 | 0.6240 | 0.5896 | 0.8160 | 0.6846 | 0.6920 | 204 | 108 | 142 | 46 | 79 | 75 | 154 | 500 |
| rel_contact | steered | g_cat_clean | mismatched | 0.05 | 0.6420 | 0.6035 | 0.8280 | 0.6981 | 0.6860 | 207 | 114 | 136 | 43 | 72 | 59 | 131 | 500 |
| rel_contact | steered | g_cat_clean | mismatched | 0.1 | 0.6580 | 0.6132 | 0.8560 | 0.7145 | 0.6980 | 214 | 115 | 135 | 36 | 80 | 59 | 139 | 500 |
| rel_contact | steered | g_cat_clean | mismatched | 0.25 | 0.6320 | 0.5954 | 0.8240 | 0.6913 | 0.6920 | 206 | 110 | 140 | 44 | 73 | 65 | 138 | 500 |
| rel_contact | steered | g_cat_clean | mismatched | 0.5 | 0.6240 | 0.5901 | 0.8120 | 0.6835 | 0.6880 | 203 | 109 | 141 | 47 | 73 | 69 | 142 | 500 |
| rel_contact | steered | g_rel_clean | type_matched | 0.05 | 0.6120 | 0.5795 | 0.8160 | 0.6777 | 0.7040 | 204 | 102 | 148 | 46 | 73 | 75 | 148 | 500 |
| rel_contact | steered | g_rel_clean | type_matched | 0.1 | 0.6060 | 0.5759 | 0.8040 | 0.6711 | 0.6980 | 201 | 102 | 148 | 49 | 67 | 72 | 139 | 500 |
| rel_contact | steered | g_rel_clean | type_matched | 0.25 | 0.6540 | 0.6078 | 0.8680 | 0.7150 | 0.7140 | 217 | 110 | 140 | 33 | 86 | 67 | 153 | 500 |
| rel_contact | steered | g_rel_clean | type_matched | 0.5 | 0.6260 | 0.5863 | 0.8560 | 0.6959 | 0.7300 | 214 | 99 | 151 | 36 | 73 | 68 | 141 | 500 |
| rel_spatial | baseline |  |  |  | 0.5180 | 0.5116 | 0.7920 | 0.6217 | 0.7740 | 198 | 61 | 189 | 52 | 0 | 0 | 0 | 500 |
| rel_spatial | steered | d_attr_color_g1_s05_clean | mismatched | 0.05 | 0.5360 | 0.5238 | 0.7920 | 0.6306 | 0.7560 | 198 | 70 | 180 | 52 | 79 | 70 | 149 | 500 |
| rel_spatial | steered | d_attr_color_g1_s05_clean | mismatched | 0.1 | 0.5860 | 0.5580 | 0.8280 | 0.6667 | 0.7420 | 207 | 86 | 164 | 43 | 86 | 52 | 138 | 500 |
| rel_spatial | steered | d_attr_color_g1_s05_clean | mismatched | 0.25 | 0.5660 | 0.5422 | 0.8480 | 0.6615 | 0.7820 | 212 | 71 | 179 | 38 | 83 | 59 | 142 | 500 |
| rel_spatial | steered | d_attr_color_g1_s05_clean | mismatched | 0.5 | 0.5420 | 0.5279 | 0.7960 | 0.6348 | 0.7540 | 199 | 72 | 178 | 51 | 86 | 74 | 160 | 500 |
| rel_spatial | steered | d_attr_count_g1_s05_clean | mismatched | 0.05 | 0.5280 | 0.5180 | 0.8040 | 0.6301 | 0.7760 | 201 | 63 | 187 | 49 | 81 | 76 | 157 | 500 |
| rel_spatial | steered | d_attr_count_g1_s05_clean | mismatched | 0.1 | 0.5540 | 0.5370 | 0.7840 | 0.6374 | 0.7300 | 196 | 81 | 169 | 54 | 85 | 67 | 152 | 500 |
| rel_spatial | steered | d_attr_count_g1_s05_clean | mismatched | 0.25 | 0.5680 | 0.5445 | 0.8320 | 0.6582 | 0.7640 | 208 | 76 | 174 | 42 | 88 | 63 | 151 | 500 |
| rel_spatial | steered | d_attr_count_g1_s05_clean | mismatched | 0.5 | 0.5500 | 0.5330 | 0.8080 | 0.6423 | 0.7580 | 202 | 73 | 177 | 48 | 81 | 65 | 146 | 500 |
| rel_spatial | steered | d_cat_hard_g1_s05_clean | mismatched | 0.05 | 0.5520 | 0.5340 | 0.8160 | 0.6456 | 0.7640 | 204 | 72 | 178 | 46 | 83 | 66 | 149 | 500 |
| rel_spatial | steered | d_cat_hard_g1_s05_clean | mismatched | 0.1 | 0.5520 | 0.5339 | 0.8200 | 0.6467 | 0.7680 | 205 | 71 | 179 | 45 | 86 | 69 | 155 | 500 |
| rel_spatial | steered | d_cat_hard_g1_s05_clean | mismatched | 0.25 | 0.5600 | 0.5397 | 0.8160 | 0.6497 | 0.7560 | 204 | 76 | 174 | 46 | 83 | 62 | 145 | 500 |
| rel_spatial | steered | d_cat_hard_g1_s05_clean | mismatched | 0.5 | 0.5500 | 0.5320 | 0.8320 | 0.6490 | 0.7820 | 208 | 67 | 183 | 42 | 81 | 65 | 146 | 500 |
| rel_spatial | steered | d_cat_popular_g1_s05_clean | mismatched | 0.05 | 0.5340 | 0.5222 | 0.8000 | 0.6319 | 0.7660 | 200 | 67 | 183 | 50 | 87 | 79 | 166 | 500 |
| rel_spatial | steered | d_cat_popular_g1_s05_clean | mismatched | 0.1 | 0.5580 | 0.5385 | 0.8120 | 0.6475 | 0.7540 | 203 | 76 | 174 | 47 | 86 | 66 | 152 | 500 |
| rel_spatial | steered | d_cat_popular_g1_s05_clean | mismatched | 0.25 | 0.5300 | 0.5194 | 0.8040 | 0.6311 | 0.7740 | 201 | 64 | 186 | 49 | 73 | 67 | 140 | 500 |
| rel_spatial | steered | d_cat_popular_g1_s05_clean | mismatched | 0.5 | 0.5400 | 0.5260 | 0.8080 | 0.6372 | 0.7680 | 202 | 68 | 182 | 48 | 77 | 66 | 143 | 500 |
| rel_spatial | steered | d_cat_random_g1_s05_clean | mismatched | 0.05 | 0.5520 | 0.5344 | 0.8080 | 0.6433 | 0.7560 | 202 | 74 | 176 | 48 | 86 | 69 | 155 | 500 |
| rel_spatial | steered | d_cat_random_g1_s05_clean | mismatched | 0.1 | 0.5560 | 0.5359 | 0.8360 | 0.6531 | 0.7800 | 209 | 69 | 181 | 41 | 83 | 64 | 147 | 500 |
| rel_spatial | steered | d_cat_random_g1_s05_clean | mismatched | 0.25 | 0.5400 | 0.5259 | 0.8120 | 0.6384 | 0.7720 | 203 | 67 | 183 | 47 | 78 | 67 | 145 | 500 |
| rel_spatial | steered | d_cat_random_g1_s05_clean | mismatched | 0.5 | 0.5560 | 0.5363 | 0.8280 | 0.6509 | 0.7720 | 207 | 71 | 179 | 43 | 77 | 58 | 135 | 500 |
| rel_spatial | steered | d_rel_contact_g1_s05_clean | type_matched | 0.05 | 0.5460 | 0.5300 | 0.8120 | 0.6414 | 0.7660 | 203 | 70 | 180 | 47 | 88 | 74 | 162 | 500 |
| rel_spatial | steered | d_rel_contact_g1_s05_clean | type_matched | 0.1 | 0.5680 | 0.5452 | 0.8200 | 0.6550 | 0.7520 | 205 | 79 | 171 | 45 | 88 | 63 | 151 | 500 |
| rel_spatial | steered | d_rel_contact_g1_s05_clean | type_matched | 0.25 | 0.5580 | 0.5365 | 0.8520 | 0.6584 | 0.7940 | 213 | 66 | 184 | 37 | 81 | 61 | 142 | 500 |
| rel_spatial | steered | d_rel_contact_g1_s05_clean | type_matched | 0.5 | 0.5460 | 0.5308 | 0.7920 | 0.6356 | 0.7460 | 198 | 75 | 175 | 52 | 91 | 77 | 168 | 500 |
| rel_spatial | steered | d_rel_spatial_g1_s05_clean | subtype_matched | 0.05 | 0.5460 | 0.5288 | 0.8440 | 0.6502 | 0.7980 | 211 | 62 | 188 | 39 | 74 | 60 | 134 | 500 |
| rel_spatial | steered | d_rel_spatial_g1_s05_clean | subtype_matched | 0.1 | 0.5420 | 0.5279 | 0.7960 | 0.6348 | 0.7540 | 199 | 72 | 178 | 51 | 88 | 76 | 164 | 500 |
| rel_spatial | steered | d_rel_spatial_g1_s05_clean | subtype_matched | 0.25 | 0.5280 | 0.5184 | 0.7880 | 0.6254 | 0.7600 | 197 | 67 | 183 | 53 | 77 | 72 | 149 | 500 |
| rel_spatial | steered | d_rel_spatial_g1_s05_clean | subtype_matched | 0.5 | 0.5520 | 0.5340 | 0.8160 | 0.6456 | 0.7640 | 204 | 72 | 178 | 46 | 83 | 66 | 149 | 500 |
| rel_spatial | steered | g_all_clean | global | 0.05 | 0.5400 | 0.5259 | 0.8120 | 0.6384 | 0.7720 | 203 | 67 | 183 | 47 | 82 | 71 | 153 | 500 |
| rel_spatial | steered | g_all_clean | global | 0.1 | 0.5380 | 0.5235 | 0.8480 | 0.6473 | 0.8100 | 212 | 57 | 193 | 38 | 68 | 58 | 126 | 500 |
| rel_spatial | steered | g_all_clean | global | 0.25 | 0.5440 | 0.5278 | 0.8360 | 0.6471 | 0.7920 | 209 | 63 | 187 | 41 | 79 | 66 | 145 | 500 |
| rel_spatial | steered | g_all_clean | global | 0.5 | 0.5440 | 0.5289 | 0.8040 | 0.6381 | 0.7600 | 201 | 71 | 179 | 49 | 72 | 59 | 131 | 500 |
| rel_spatial | steered | g_attr_clean | mismatched | 0.05 | 0.5860 | 0.5556 | 0.8600 | 0.6750 | 0.7740 | 215 | 78 | 172 | 35 | 92 | 58 | 150 | 500 |
| rel_spatial | steered | g_attr_clean | mismatched | 0.1 | 0.5220 | 0.5145 | 0.7800 | 0.6200 | 0.7580 | 195 | 66 | 184 | 55 | 76 | 74 | 150 | 500 |
| rel_spatial | steered | g_attr_clean | mismatched | 0.25 | 0.5180 | 0.5115 | 0.8040 | 0.6252 | 0.7860 | 201 | 58 | 192 | 49 | 71 | 71 | 142 | 500 |
| rel_spatial | steered | g_attr_clean | mismatched | 0.5 | 0.5460 | 0.5302 | 0.8080 | 0.6403 | 0.7620 | 202 | 71 | 179 | 48 | 81 | 67 | 148 | 500 |
| rel_spatial | steered | g_cat_clean | mismatched | 0.05 | 0.5280 | 0.5179 | 0.8120 | 0.6324 | 0.7840 | 203 | 61 | 189 | 47 | 71 | 66 | 137 | 500 |
| rel_spatial | steered | g_cat_clean | mismatched | 0.1 | 0.5460 | 0.5299 | 0.8160 | 0.6425 | 0.7700 | 204 | 69 | 181 | 46 | 82 | 68 | 150 | 500 |
| rel_spatial | steered | g_cat_clean | mismatched | 0.25 | 0.5280 | 0.5179 | 0.8080 | 0.6312 | 0.7800 | 202 | 62 | 188 | 48 | 75 | 70 | 145 | 500 |
| rel_spatial | steered | g_cat_clean | mismatched | 0.5 | 0.5720 | 0.5474 | 0.8320 | 0.6603 | 0.7600 | 208 | 78 | 172 | 42 | 93 | 66 | 159 | 500 |
| rel_spatial | steered | g_rel_clean | type_matched | 0.05 | 0.5940 | 0.5640 | 0.8280 | 0.6710 | 0.7340 | 207 | 90 | 160 | 43 | 100 | 62 | 162 | 500 |
| rel_spatial | steered | g_rel_clean | type_matched | 0.1 | 0.5280 | 0.5182 | 0.7960 | 0.6278 | 0.7680 | 199 | 65 | 185 | 51 | 75 | 70 | 145 | 500 |
| rel_spatial | steered | g_rel_clean | type_matched | 0.25 | 0.5240 | 0.5150 | 0.8240 | 0.6338 | 0.8000 | 206 | 56 | 194 | 44 | 70 | 67 | 137 | 500 |
| rel_spatial | steered | g_rel_clean | type_matched | 0.5 | 0.5440 | 0.5270 | 0.8600 | 0.6535 | 0.8160 | 215 | 57 | 193 | 35 | 78 | 65 | 143 | 500 |
