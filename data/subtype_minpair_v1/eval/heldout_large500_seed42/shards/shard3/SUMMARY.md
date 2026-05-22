# Subtype Minimal-Pair Held-Out Eval

## Best Steered Rows By Subset/Vector
| eval_subset | vector | alpha | accuracy | f1 | yes_rate | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rel_contact | d_cat_random_g1_s05_clean | 0.25 | 0.6700 | 0.7245 | 0.6980 | 84 | 57 | 141 | 500 |
| rel_contact | d_attr_count_g1_s05_clean | 0.05 | 0.6580 | 0.7201 | 0.7220 | 84 | 63 | 147 | 500 |
| rel_contact | d_cat_hard_g1_s05_clean | 0.25 | 0.6580 | 0.7164 | 0.7060 | 84 | 63 | 147 | 500 |
| rel_contact | d_rel_spatial_g1_s05_clean | 0.25 | 0.6580 | 0.7164 | 0.7060 | 80 | 59 | 139 | 500 |
| rel_contact | g_rel_clean | 0.25 | 0.6540 | 0.7150 | 0.7140 | 86 | 67 | 153 | 500 |
| rel_contact | g_cat_clean | 0.1 | 0.6580 | 0.7145 | 0.6980 | 80 | 59 | 139 | 500 |
| rel_contact | g_attr_clean | 0.05 | 0.6580 | 0.7126 | 0.6900 | 87 | 66 | 153 | 500 |
| rel_contact | g_all_clean | 0.25 | 0.6460 | 0.7055 | 0.7020 | 81 | 66 | 147 | 500 |
| rel_contact | d_cat_popular_g1_s05_clean | 0.1 | 0.6320 | 0.6964 | 0.7120 | 79 | 71 | 150 | 500 |
| rel_contact | d_rel_contact_g1_s05_clean | 0.5 | 0.6260 | 0.6919 | 0.7140 | 68 | 63 | 131 | 500 |
| rel_contact | d_attr_color_g1_s05_clean | 0.25 | 0.6240 | 0.6887 | 0.7080 | 75 | 71 | 146 | 500 |

## All Rows
| eval_subset | method | vector | alpha | accuracy | precision | recall | f1 | yes_rate | tp | tn | fp | fn | wrong_to_right | right_to_wrong | changed_pred | num_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rel_contact | baseline |  |  | 0.6160 | 0.5819 | 0.8240 | 0.6821 | 0.7080 | 206 | 102 | 148 | 44 | 0 | 0 | 0 | 500 |
| rel_contact | steered | d_attr_color_g1_s05_clean | 0.05 | 0.6260 | 0.5935 | 0.8000 | 0.6814 | 0.6740 | 200 | 113 | 137 | 50 | 77 | 72 | 149 | 500 |
| rel_contact | steered | d_attr_color_g1_s05_clean | 0.1 | 0.6100 | 0.5779 | 0.8160 | 0.6766 | 0.7060 | 204 | 101 | 149 | 46 | 76 | 79 | 155 | 500 |
| rel_contact | steered | d_attr_color_g1_s05_clean | 0.25 | 0.6240 | 0.5876 | 0.8320 | 0.6887 | 0.7080 | 208 | 104 | 146 | 42 | 75 | 71 | 146 | 500 |
| rel_contact | steered | d_attr_color_g1_s05_clean | 0.5 | 0.6000 | 0.5694 | 0.8200 | 0.6721 | 0.7200 | 205 | 95 | 155 | 45 | 72 | 80 | 152 | 500 |
| rel_contact | steered | d_attr_count_g1_s05_clean | 0.05 | 0.6580 | 0.6094 | 0.8800 | 0.7201 | 0.7220 | 220 | 109 | 141 | 30 | 84 | 63 | 147 | 500 |
| rel_contact | steered | d_attr_count_g1_s05_clean | 0.1 | 0.6540 | 0.6149 | 0.8240 | 0.7043 | 0.6700 | 206 | 121 | 129 | 44 | 87 | 68 | 155 | 500 |
| rel_contact | steered | d_attr_count_g1_s05_clean | 0.25 | 0.6300 | 0.5900 | 0.8520 | 0.6972 | 0.7220 | 213 | 102 | 148 | 37 | 71 | 64 | 135 | 500 |
| rel_contact | steered | d_attr_count_g1_s05_clean | 0.5 | 0.6560 | 0.6140 | 0.8400 | 0.7095 | 0.6840 | 210 | 118 | 132 | 40 | 80 | 60 | 140 | 500 |
| rel_contact | steered | d_cat_hard_g1_s05_clean | 0.05 | 0.6260 | 0.5908 | 0.8200 | 0.6868 | 0.6940 | 205 | 108 | 142 | 45 | 83 | 78 | 161 | 500 |
| rel_contact | steered | d_cat_hard_g1_s05_clean | 0.1 | 0.6380 | 0.5989 | 0.8360 | 0.6978 | 0.6980 | 209 | 110 | 140 | 41 | 78 | 67 | 145 | 500 |
| rel_contact | steered | d_cat_hard_g1_s05_clean | 0.25 | 0.6580 | 0.6119 | 0.8640 | 0.7164 | 0.7060 | 216 | 113 | 137 | 34 | 84 | 63 | 147 | 500 |
| rel_contact | steered | d_cat_hard_g1_s05_clean | 0.5 | 0.6600 | 0.6176 | 0.8400 | 0.7119 | 0.6800 | 210 | 120 | 130 | 40 | 89 | 67 | 156 | 500 |
| rel_contact | steered | d_cat_popular_g1_s05_clean | 0.05 | 0.6260 | 0.5918 | 0.8120 | 0.6847 | 0.6860 | 203 | 110 | 140 | 47 | 73 | 68 | 141 | 500 |
| rel_contact | steered | d_cat_popular_g1_s05_clean | 0.1 | 0.6320 | 0.5927 | 0.8440 | 0.6964 | 0.7120 | 211 | 105 | 145 | 39 | 79 | 71 | 150 | 500 |
| rel_contact | steered | d_cat_popular_g1_s05_clean | 0.25 | 0.6300 | 0.5905 | 0.8480 | 0.6962 | 0.7180 | 212 | 103 | 147 | 38 | 69 | 62 | 131 | 500 |
| rel_contact | steered | d_cat_popular_g1_s05_clean | 0.5 | 0.6340 | 0.5965 | 0.8280 | 0.6935 | 0.6940 | 207 | 110 | 140 | 43 | 75 | 66 | 141 | 500 |
| rel_contact | steered | d_cat_random_g1_s05_clean | 0.05 | 0.6400 | 0.6042 | 0.8120 | 0.6928 | 0.6720 | 203 | 117 | 133 | 47 | 77 | 65 | 142 | 500 |
| rel_contact | steered | d_cat_random_g1_s05_clean | 0.1 | 0.6100 | 0.5770 | 0.8240 | 0.6787 | 0.7140 | 206 | 99 | 151 | 44 | 69 | 72 | 141 | 500 |
| rel_contact | steered | d_cat_random_g1_s05_clean | 0.25 | 0.6700 | 0.6218 | 0.8680 | 0.7245 | 0.6980 | 217 | 118 | 132 | 33 | 84 | 57 | 141 | 500 |
| rel_contact | steered | d_cat_random_g1_s05_clean | 0.5 | 0.6640 | 0.6152 | 0.8760 | 0.7228 | 0.7120 | 219 | 113 | 137 | 31 | 89 | 65 | 154 | 500 |
| rel_contact | steered | d_rel_contact_g1_s05_clean | 0.05 | 0.6100 | 0.5770 | 0.8240 | 0.6787 | 0.7140 | 206 | 99 | 151 | 44 | 75 | 78 | 153 | 500 |
| rel_contact | steered | d_rel_contact_g1_s05_clean | 0.1 | 0.6180 | 0.5850 | 0.8120 | 0.6801 | 0.6940 | 203 | 106 | 144 | 47 | 75 | 74 | 149 | 500 |
| rel_contact | steered | d_rel_contact_g1_s05_clean | 0.25 | 0.6180 | 0.5831 | 0.8280 | 0.6843 | 0.7100 | 207 | 102 | 148 | 43 | 78 | 77 | 155 | 500 |
| rel_contact | steered | d_rel_contact_g1_s05_clean | 0.5 | 0.6260 | 0.5882 | 0.8400 | 0.6919 | 0.7140 | 210 | 103 | 147 | 40 | 68 | 63 | 131 | 500 |
| rel_contact | steered | d_rel_spatial_g1_s05_clean | 0.05 | 0.6340 | 0.5971 | 0.8240 | 0.6924 | 0.6900 | 206 | 111 | 139 | 44 | 80 | 71 | 151 | 500 |
| rel_contact | steered | d_rel_spatial_g1_s05_clean | 0.1 | 0.6480 | 0.6016 | 0.8760 | 0.7134 | 0.7280 | 219 | 105 | 145 | 31 | 82 | 66 | 148 | 500 |
| rel_contact | steered | d_rel_spatial_g1_s05_clean | 0.25 | 0.6580 | 0.6119 | 0.8640 | 0.7164 | 0.7060 | 216 | 113 | 137 | 34 | 80 | 59 | 139 | 500 |
| rel_contact | steered | d_rel_spatial_g1_s05_clean | 0.5 | 0.6480 | 0.6039 | 0.8600 | 0.7096 | 0.7120 | 215 | 109 | 141 | 35 | 77 | 61 | 138 | 500 |
| rel_contact | steered | g_all_clean | 0.05 | 0.5960 | 0.5670 | 0.8120 | 0.6678 | 0.7160 | 203 | 95 | 155 | 47 | 63 | 73 | 136 | 500 |
| rel_contact | steered | g_all_clean | 0.1 | 0.6060 | 0.5746 | 0.8160 | 0.6744 | 0.7100 | 204 | 99 | 151 | 46 | 67 | 72 | 139 | 500 |
| rel_contact | steered | g_all_clean | 0.25 | 0.6460 | 0.6040 | 0.8480 | 0.7055 | 0.7020 | 212 | 111 | 139 | 38 | 81 | 66 | 147 | 500 |
| rel_contact | steered | g_all_clean | 0.5 | 0.6400 | 0.6029 | 0.8200 | 0.6949 | 0.6800 | 205 | 115 | 135 | 45 | 76 | 64 | 140 | 500 |
| rel_contact | steered | g_attr_clean | 0.05 | 0.6580 | 0.6145 | 0.8480 | 0.7126 | 0.6900 | 212 | 117 | 133 | 38 | 87 | 66 | 153 | 500 |
| rel_contact | steered | g_attr_clean | 0.1 | 0.6300 | 0.5905 | 0.8480 | 0.6962 | 0.7180 | 212 | 103 | 147 | 38 | 80 | 73 | 153 | 500 |
| rel_contact | steered | g_attr_clean | 0.25 | 0.6220 | 0.5910 | 0.7920 | 0.6769 | 0.6700 | 198 | 113 | 137 | 52 | 81 | 78 | 159 | 500 |
| rel_contact | steered | g_attr_clean | 0.5 | 0.6240 | 0.5896 | 0.8160 | 0.6846 | 0.6920 | 204 | 108 | 142 | 46 | 79 | 75 | 154 | 500 |
| rel_contact | steered | g_cat_clean | 0.05 | 0.6420 | 0.6035 | 0.8280 | 0.6981 | 0.6860 | 207 | 114 | 136 | 43 | 72 | 59 | 131 | 500 |
| rel_contact | steered | g_cat_clean | 0.1 | 0.6580 | 0.6132 | 0.8560 | 0.7145 | 0.6980 | 214 | 115 | 135 | 36 | 80 | 59 | 139 | 500 |
| rel_contact | steered | g_cat_clean | 0.25 | 0.6320 | 0.5954 | 0.8240 | 0.6913 | 0.6920 | 206 | 110 | 140 | 44 | 73 | 65 | 138 | 500 |
| rel_contact | steered | g_cat_clean | 0.5 | 0.6240 | 0.5901 | 0.8120 | 0.6835 | 0.6880 | 203 | 109 | 141 | 47 | 73 | 69 | 142 | 500 |
| rel_contact | steered | g_rel_clean | 0.05 | 0.6120 | 0.5795 | 0.8160 | 0.6777 | 0.7040 | 204 | 102 | 148 | 46 | 73 | 75 | 148 | 500 |
| rel_contact | steered | g_rel_clean | 0.1 | 0.6060 | 0.5759 | 0.8040 | 0.6711 | 0.6980 | 201 | 102 | 148 | 49 | 67 | 72 | 139 | 500 |
| rel_contact | steered | g_rel_clean | 0.25 | 0.6540 | 0.6078 | 0.8680 | 0.7150 | 0.7140 | 217 | 110 | 140 | 33 | 86 | 67 | 153 | 500 |
| rel_contact | steered | g_rel_clean | 0.5 | 0.6260 | 0.5863 | 0.8560 | 0.6959 | 0.7300 | 214 | 99 | 151 | 36 | 73 | 68 | 141 | 500 |
