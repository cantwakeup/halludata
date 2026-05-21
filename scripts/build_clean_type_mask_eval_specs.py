#!/usr/bin/env python3
"""Generate held-out eval run specs for clean_type_minpair_v2 mask steering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ATTRIBUTE_SUBTYPES = [
    "attr_color_clean",
    "attr_count_clean",
    "attr_state_clean",
    "attr_material_clean",
    "attr_shape_clean",
    "attr_action_single_clean",
]
RELATION_SUBTYPES = [
    "rel_left_right_clean",
    "rel_above_below_clean",
    "rel_holding_wearing_clean",
    "rel_sitting_riding_clean",
]
SUBTYPES = ATTRIBUTE_SUBTYPES + RELATION_SUBTYPES

MISMATCHES = {
    "attr_color_clean": ["attr_count_clean", "attr_state_clean", "rel_left_right_clean"],
    "attr_count_clean": ["attr_color_clean", "attr_action_single_clean", "rel_holding_wearing_clean"],
    "attr_state_clean": ["attr_material_clean", "attr_action_single_clean", "rel_sitting_riding_clean"],
    "attr_material_clean": ["attr_shape_clean", "attr_color_clean", "rel_holding_wearing_clean"],
    "attr_shape_clean": ["attr_material_clean", "attr_color_clean", "rel_left_right_clean"],
    "attr_action_single_clean": ["attr_state_clean", "rel_sitting_riding_clean", "rel_holding_wearing_clean"],
    "rel_left_right_clean": ["rel_above_below_clean", "attr_color_clean", "attr_action_single_clean"],
    "rel_above_below_clean": ["rel_left_right_clean", "attr_color_clean", "attr_shape_clean"],
    "rel_holding_wearing_clean": ["rel_sitting_riding_clean", "attr_action_single_clean", "attr_count_clean"],
    "rel_sitting_riding_clean": ["rel_holding_wearing_clean", "attr_action_single_clean", "attr_state_clean"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subtypes", default=",".join(SUBTYPES))
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--include-s-direction", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def subtype_to_type(subtype: str) -> str:
    if subtype.startswith("attr_"):
        return "attr"
    if subtype.startswith("rel_"):
        return "rel"
    return subtype.split("_", 1)[0]


def spec(name: str, subtype: str, direction_key: str, mask_key: str, match_type: str) -> dict[str, str]:
    return {
        "name": name,
        "subtype": subtype,
        "direction_key": direction_key,
        "mask_key": mask_key,
        "match_type": match_type,
    }


def build_specs(subtypes: Sequence[str], random_seeds: Sequence[str], include_s_direction: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    subtype_set = set(subtypes)
    for subtype in subtypes:
        typ = subtype_to_type(subtype)
        g_type = f"g_{typ}_clean"
        s_dir = f"s_{subtype}_clean"
        type_mask = f"mask_g_{typ}_norm_top64"
        subtype_mask = f"mask_s_{subtype}_energy_top64"
        rows.append(spec(f"{subtype}__g_all__g_all_norm", subtype, "g_all_clean", "mask_g_all_norm_top64", "g_all_baseline"))
        rows.append(spec(f"{subtype}__{g_type}__{type_mask}", subtype, g_type, type_mask, "g_type_baseline"))
        rows.append(spec(f"{subtype}__{g_type}__{subtype_mask}", subtype, g_type, subtype_mask, "matched_energy"))
        rows.append(spec(f"{subtype}__g_all_clean__{subtype_mask}", subtype, "g_all_clean", subtype_mask, "matched_energy_g_all"))
        if include_s_direction:
            rows.append(spec(f"{subtype}__{s_dir}__{subtype_mask}", subtype, s_dir, subtype_mask, "s_direction_ablation"))
        for other in MISMATCHES.get(subtype, []):
            if other not in subtype_set:
                continue
            mask_key = f"mask_s_{other}_energy_top64"
            rows.append(spec(f"{subtype}__{g_type}__mismatch_{other}", subtype, g_type, mask_key, "mismatched_energy"))
        for seed in random_seeds:
            rows.append(spec(f"{subtype}__{g_type}__random_seed{seed}", subtype, g_type, f"random_mask_top64_seed{seed}", "random_mask"))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    subtypes = split_csv(args.subtypes)
    random_seeds = split_csv(args.random_seeds)
    rows = build_specs(subtypes, random_seeds, bool(args.include_s_direction))
    write_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} clean-v2 mask eval specs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
