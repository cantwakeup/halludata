"""Utilities for building and validating category hard-negative banks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from expert_data.io_utils import read_json

MIN_NEGATIVE_CANDIDATES = 3

MANUAL_SEED_BANK: dict[str, dict[str, Any]] = {
    "cat": {"manual": ["dog", "rabbit", "fox"], "semantic_group": "animal"},
    "dog": {"manual": ["cat", "rabbit", "fox"], "semantic_group": "animal"},
    "rabbit": {"manual": ["cat", "dog", "fox"], "semantic_group": "animal"},
    "fox": {"manual": ["cat", "dog", "rabbit"], "semantic_group": "animal"},
    "ball": {"manual": ["apple", "orange", "kite"], "semantic_group": "round_object"},
    "mat": {"manual": ["rug", "blanket", "cushion"], "semantic_group": "household"},
}


def _normalize_manual_candidates(category: str, candidates: list[Any]) -> list[str]:
    """Normalize negative candidates by removing self-references and duplicates."""

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_text = str(candidate)
        if not candidate_text or candidate_text == category or candidate_text in seen:
            continue
        seen.add(candidate_text)
        normalized.append(candidate_text)
    return normalized


def validate_cat_neg_entry(category: str, entry: Mapping[str, Any]) -> None:
    """Validate one category entry in the hard-negative bank."""

    if "manual" not in entry or "semantic_group" not in entry:
        raise ValueError(f"Category '{category}' must contain manual and semantic_group fields")
    if not isinstance(entry["manual"], list):
        raise ValueError(f"Category '{category}' manual negatives must be a list")

    manual = [str(candidate) for candidate in entry["manual"]]
    if len(manual) < MIN_NEGATIVE_CANDIDATES:
        raise ValueError(f"Category '{category}' must have at least {MIN_NEGATIVE_CANDIDATES} negatives")
    if len(set(manual)) != len(manual):
        raise ValueError(f"Category '{category}' contains duplicate negatives")
    if category in manual:
        raise ValueError(f"Category '{category}' cannot include itself as a negative")

    semantic_group = str(entry["semantic_group"]).strip()
    if not semantic_group:
        raise ValueError(f"Category '{category}' must declare a semantic_group")


def validate_cat_neg_bank(bank: Mapping[str, Any]) -> None:
    """Validate the full category hard-negative bank."""

    if not isinstance(bank, Mapping) or not bank:
        raise ValueError("Category hard-negative bank must be a non-empty mapping")
    for category, entry in bank.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Category '{category}' entry must be a mapping")
        validate_cat_neg_entry(str(category), entry)


def build_manual_seed_cat_neg_bank() -> dict[str, dict[str, Any]]:
    """Build a deterministic manual seed bank for category hard negatives."""

    bank = deepcopy(MANUAL_SEED_BANK)
    for category, entry in bank.items():
        entry["manual"] = _normalize_manual_candidates(category, list(entry["manual"]))
    validate_cat_neg_bank(bank)
    return bank


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
