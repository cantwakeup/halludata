"""Sampling helpers for per-image cap enforcement before template rendering."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

from expert_data.schemas import FactRecord

CAP_DROP_REASON_BY_SUBTYPE = {
    "cat": "capped_cat_per_image",
    "cnt": "capped_cnt_per_image",
    "col": "capped_col_per_image",
    "rel": "capped_rel_per_image",
}


def _sampling_limits(sampling_cfg: Mapping[str, Any]) -> dict[str, int]:
    """Read subtype-specific per-image caps from configuration."""

    return {
        "cat": int(sampling_cfg.get("max_cat_anchors_per_image", 1_000_000)),
        "cnt": int(sampling_cfg.get("max_cnt_anchors_per_image", 1_000_000)),
        "col": int(sampling_cfg.get("max_col_anchors_per_image", 1_000_000)),
        "rel": int(sampling_cfg.get("max_rel_pairs_per_image", 1_000_000)),
    }


def apply_sampling_caps(
    facts: list[FactRecord],
    sampling_cfg: Mapping[str, Any],
) -> tuple[list[FactRecord], dict[str, int], dict[str, int]]:
    """Apply deterministic per-image caps after filtering and before rendering."""

    limits = _sampling_limits(sampling_cfg)
    counts_by_image_and_subtype: dict[tuple[str, str], int] = defaultdict(int)
    selected_facts: list[FactRecord] = []
    kept_counter: Counter[str] = Counter()
    dropped_counter: Counter[str] = Counter()

    for fact in sorted(facts, key=lambda item: (item.image_id, item.subtype, item.fact_id)):
        key = (fact.image_id, fact.subtype)
        if counts_by_image_and_subtype[key] >= limits.get(fact.subtype, 1_000_000):
            dropped_reason = CAP_DROP_REASON_BY_SUBTYPE.get(fact.subtype)
            if dropped_reason is not None:
                dropped_counter[dropped_reason] += 1
            continue
        counts_by_image_and_subtype[key] += 1
        kept_counter[fact.subtype] += 1
        selected_facts.append(fact)

    return (
        selected_facts,
        {subtype: int(kept_counter.get(subtype, 0)) for subtype in limits},
        {reason: int(dropped_counter.get(reason, 0)) for reason in CAP_DROP_REASON_BY_SUBTYPE.values()},
    )

