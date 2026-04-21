"""Text helpers for count-conditioned noun rendering."""

from __future__ import annotations

from typing import Any

IRREGULAR_PLURALS: dict[str, str] = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "mouse": "mice",
    "tooth": "teeth",
    "foot": "feet",
    "sheep": "sheep",
    "deer": "deer",
}


def normalize_count(count: Any) -> int:
    """Normalize a count-like value into an integer for text rendering."""

    return int(count)


def pluralize_noun(noun: str) -> str:
    """Return the plural form for a noun using a small irregular lexicon first."""

    normalized_noun = str(noun)
    if normalized_noun in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[normalized_noun]
    if normalized_noun.endswith(("s", "x", "z", "ch", "sh")):
        return f"{normalized_noun}es"
    if normalized_noun.endswith("y") and len(normalized_noun) > 1 and normalized_noun[-2].lower() not in "aeiou":
        return f"{normalized_noun[:-1]}ies"
    return f"{normalized_noun}s"


def count_conditioned_noun(noun: str, count: Any) -> str:
    """Return the singular or plural noun form implied by one count value."""

    numeric_count = normalize_count(count)
    if numeric_count == 1:
        return str(noun)
    return pluralize_noun(noun)

