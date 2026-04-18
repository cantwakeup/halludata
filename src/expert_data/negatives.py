"""Utilities for building and validating category hard-negative banks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from expert_data.io_utils import read_json
from expert_data.resources.coco80_neg_seed import (
    CATEGORY_SEED_NEGATIVES,
    COCO80_CATEGORY_NAMES,
    COCO80_GROUP_SPECS,
    EXTRA_COMPAT_GROUP_SPECS,
    SEMANTIC_GROUP_FALLBACKS,
    SUPPORTED_NEGATIVE_CATEGORY_NAMES,
)

MIN_NEGATIVE_CANDIDATES = 5
DEFAULT_NEGATIVE_TARGET = 10


def _build_category_metadata() -> dict[str, dict[str, str]]:
    """Build per-category metadata from grouped semantic and supercategory specs."""

    metadata: dict[str, dict[str, str]] = {}
    for group_specs in (COCO80_GROUP_SPECS, EXTRA_COMPAT_GROUP_SPECS):
        for semantic_group, spec in group_specs.items():
            supercategory = str(spec["supercategory"])
            for category in spec["categories"]:
                metadata[str(category)] = {
                    "semantic_group": semantic_group,
                    "supercategory": supercategory,
                }
    return metadata


CATEGORY_METADATA = _build_category_metadata()


def _normalize_manual_candidates(category: str, candidates: list[Any]) -> list[str]:
    """Normalize negative candidates by removing self-references and duplicates."""

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_text = str(candidate).strip()
        if not candidate_text or candidate_text == category or candidate_text in seen:
            continue
        seen.add(candidate_text)
        normalized.append(candidate_text)
    return normalized


def validate_cat_neg_entry(category: str, entry: Mapping[str, Any]) -> None:
    """Validate one category entry in the hard-negative bank."""

    required_fields = {"manual", "semantic_group", "supercategory"}
    missing_fields = required_fields.difference(entry)
    if missing_fields:
        missing_text = ", ".join(sorted(missing_fields))
        raise ValueError(f"Category '{category}' is missing required fields: {missing_text}")
    if not isinstance(entry["manual"], list):
        raise ValueError(f"Category '{category}' manual negatives must be a list")

    manual = [str(candidate) for candidate in entry["manual"]]
    if len(manual) < MIN_NEGATIVE_CANDIDATES:
        raise ValueError(f"Category '{category}' must have at least {MIN_NEGATIVE_CANDIDATES} negatives")
    if len(set(manual)) != len(manual):
        raise ValueError(f"Category '{category}' contains duplicate negatives")
    if category in manual:
        raise ValueError(f"Category '{category}' cannot include itself as a negative")

    for key in ("semantic_group", "supercategory"):
        value = str(entry[key]).strip()
        if not value:
            raise ValueError(f"Category '{category}' must declare a non-empty {key}")


def validate_cat_neg_bank(bank: Mapping[str, Any]) -> None:
    """Validate the full category hard-negative bank."""

    if not isinstance(bank, Mapping) or not bank:
        raise ValueError("Category hard-negative bank must be a non-empty mapping")
    for category, entry in bank.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Category '{category}' entry must be a mapping")
        validate_cat_neg_entry(str(category), entry)


def _categories_by_semantic_group() -> dict[str, list[str]]:
    """Group supported categories by semantic group in declaration order."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for category in SUPPORTED_NEGATIVE_CATEGORY_NAMES:
        grouped[CATEGORY_METADATA[category]["semantic_group"]].append(category)
    return dict(grouped)


def _categories_by_supercategory() -> dict[str, list[str]]:
    """Group supported categories by supercategory in declaration order."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for category in SUPPORTED_NEGATIVE_CATEGORY_NAMES:
        grouped[CATEGORY_METADATA[category]["supercategory"]].append(category)
    return dict(grouped)


def _build_candidate_pool(category: str) -> list[str]:
    """Build an ordered fallback pool for one category using semantic and supercategory neighbors."""

    metadata = CATEGORY_METADATA[category]
    semantic_group = metadata["semantic_group"]
    supercategory = metadata["supercategory"]
    by_semantic = _categories_by_semantic_group()
    by_supercategory = _categories_by_supercategory()

    ordered_candidates: list[str] = []
    ordered_candidates.extend(CATEGORY_SEED_NEGATIVES.get(category, []))
    ordered_candidates.extend(by_semantic.get(semantic_group, []))
    ordered_candidates.extend(by_supercategory.get(supercategory, []))

    for fallback_group in SEMANTIC_GROUP_FALLBACKS.get(semantic_group, []):
        ordered_candidates.extend(by_semantic.get(fallback_group, []))

    ordered_candidates.extend(COCO80_CATEGORY_NAMES)
    ordered_candidates.extend(
        category_name
        for category_name in SUPPORTED_NEGATIVE_CATEGORY_NAMES
        if category_name not in COCO80_CATEGORY_NAMES
    )
    return _normalize_manual_candidates(category, ordered_candidates)


def build_coco_cat_neg_bank(
    min_candidates: int = MIN_NEGATIVE_CANDIDATES,
    target_candidates: int = DEFAULT_NEGATIVE_TARGET,
) -> dict[str, dict[str, Any]]:
    """Build a deterministic COCO-scale hard-negative bank with fallback expansion."""

    bank: dict[str, dict[str, Any]] = {}
    for category in SUPPORTED_NEGATIVE_CATEGORY_NAMES:
        manual_candidates = _build_candidate_pool(category)
        if len(manual_candidates) < min_candidates:
            raise ValueError(
                f"Category '{category}' only has {len(manual_candidates)} negatives after fallback expansion"
            )
        metadata = CATEGORY_METADATA[category]
        bank[category] = {
            "manual": manual_candidates[: max(min_candidates, target_candidates)],
            "semantic_group": metadata["semantic_group"],
            "supercategory": metadata["supercategory"],
        }
    validate_cat_neg_bank(bank)
    return bank


def build_manual_seed_cat_neg_bank() -> dict[str, dict[str, Any]]:
    """Backward-compatible alias that now returns the COCO-scale v2 negative bank."""

    return build_coco_cat_neg_bank()


def load_cat_neg_bank(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate a category hard-negative bank JSON resource."""

    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Category hard-negative bank at {path} must be a mapping")
    bank: dict[str, dict[str, Any]] = {}
    for category, entry in payload.items():
        category_name = str(category)
        if not isinstance(entry, Mapping):
            raise ValueError(f"Category '{category_name}' entry must be a mapping")
        bank[category_name] = {
            "manual": [str(candidate) for candidate in entry.get("manual", [])],
            "semantic_group": str(entry.get("semantic_group", "")),
            "supercategory": str(entry.get("supercategory", "")),
        }
    validate_cat_neg_bank(bank)
    return bank


def get_cat_negative_candidates(
    bank: Mapping[str, Mapping[str, Any]],
    category: str,
) -> list[str]:
    """Return manual hard-negative candidates for a given category."""

    entry = bank.get(category)
    if not isinstance(entry, Mapping):
        return []
    manual_candidates = entry.get("manual", [])
    if not isinstance(manual_candidates, list):
        return []
    return [str(candidate) for candidate in manual_candidates]

