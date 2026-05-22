# Clean Type Minimal-Pair v2 Report

## Goal

Construct clean minimal-pair data satisfying single-factor intervention, nuisance balance, and condition-key differencing.

## Data Sources

- GQA scene graphs for attribute family and relation gold data.
- Bbox-derived GQA relations only for left/right and above/below with strict margin filtering.

## Attribute Family Summary
| subtype | count | yes | no | source | conditions |
| --- | --- | --- | --- | --- | --- |
| attr_color_clean | 1300 | 650 | 650 | {'gqa': 1300} | 656 |
| attr_count_clean | 1300 | 650 | 650 | {'gqa': 1300} | 160 |
| attr_state_clean | 900 | 450 | 450 | {'gqa': 900} | 72 |
| attr_material_clean | 550 | 275 | 275 | {'gqa': 550} | 66 |
| attr_shape_clean | 540 | 270 | 270 | {'gqa': 540} | 72 |
| attr_action_single_clean | 900 | 450 | 450 | {'gqa': 900} | 98 |

## Relation Gold Summary
| subtype | count | yes | no | source | conditions |
| --- | --- | --- | --- | --- | --- |
| rel_left_right_clean | 720 | 360 | 360 | {'gqa_bbox_derived': 720} | 438 |
| rel_above_below_clean | 720 | 360 | 360 | {'gqa_bbox_derived': 720} | 466 |
| rel_holding_wearing_clean | 600 | 300 | 300 | {'gqa': 600} | 106 |
| rel_sitting_riding_clean | 534 | 267 | 267 | {'gqa': 534} | 82 |

## Audit Artifacts

- `data/clean_type_minpair_v2/INSPECT.md`
- `data/clean_type_minpair_v2/minimal_pairs/DATA_REPORT.md`
- `data/clean_type_minpair_v2/minimal_pairs/CONDITION_REPORT.md`
- `data/clean_type_minpair_v2/minimal_pairs/DATA_AUDIT.md`
- `data/clean_type_minpair_v2/minimal_pairs/examples/`

## Decision

`PASS`

- Core audit checks passed.

Next: run official-LLaVA activation extraction, then build condition-balanced vectors/masks by differencing within condition_key before subtype averaging.