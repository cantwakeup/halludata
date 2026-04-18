"""Utilities for building and validating shell-bank resources."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from string import Formatter
from typing import Any, Mapping

from expert_data.io_utils import read_json

SHELL_PLACEHOLDER_RULES: dict[str, set[str]] = {
    "cat": {"label"},
    "cnt": {"count", "obj_pl"},
    "col": {"obj", "color"},
    "rel": {"obj1", "obj2", "rel"},
}

SHELL_TARGET_SLOT: dict[str, str] = {
    "cat": "label",
    "cnt": "count",
    "col": "color",
    "rel": "rel",
}

DETERMINISTIC_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "cat": [
        {
            "template_id": "cat_tpl_001",
            "text_source": "template",
            "q_template": "What category best describes the object in the image?",
            "r_template": "The object is a {label}.",
        },
        {
            "template_id": "cat_tpl_002",
            "text_source": "template",
            "q_template": "Which label matches the visible object?",
            "r_template": "The correct label is {label}.",
        },
        {
            "template_id": "cat_tpl_003",
            "text_source": "template",
            "q_template": "Identify the category of the salient object.",
            "r_template": "This image contains a {label}.",
        },
        {
            "template_id": "cat_tpl_004",
            "text_source": "template",
            "q_template": "What is the object category shown here?",
            "r_template": "The shown category is {label}.",
        },
    ],
    "cnt": [
        {
            "template_id": "cnt_tpl_001",
            "text_source": "template",
            "q_template": "How many {obj_pl} are visible in the image?",
            "r_template": "The image shows {count} {obj_pl}.",
        },
        {
            "template_id": "cnt_tpl_002",
            "text_source": "template",
            "q_template": "What is the count of {obj_pl} in the scene?",
            "r_template": "There are {count} {obj_pl} in the scene.",
        },
        {
            "template_id": "cnt_tpl_003",
            "text_source": "template",
            "q_template": "Report the number of {obj_pl} present.",
            "r_template": "The visible count is {count} {obj_pl}.",
        },
    ],
    "col": [
        {
            "template_id": "col_tpl_001",
            "text_source": "template",
            "q_template": "What color is the {obj}?",
            "r_template": "The {obj} is {color}.",
        },
        {
            "template_id": "col_tpl_002",
            "text_source": "template",
            "q_template": "Identify the color of the {obj} in the image.",
            "r_template": "The visible {obj} appears {color}.",
        },
        {
            "template_id": "col_tpl_003",
            "text_source": "template",
            "q_template": "Which color best matches the {obj}?",
            "r_template": "The {obj} has a {color} color.",
        },
    ],
    "rel": [
        {
            "template_id": "rel_tpl_001",
            "text_source": "template",
            "q_template": "What is the spatial relation between the {obj1} and the {obj2}?",
            "r_template": "The {obj1} is {rel} the {obj2}.",
        },
        {
            "template_id": "rel_tpl_002",
            "text_source": "template",
            "q_template": "How is the {obj1} positioned relative to the {obj2}?",
            "r_template": "The {obj1} is located {rel} the {obj2}.",
        },
        {
            "template_id": "rel_tpl_003",
            "text_source": "template",
            "q_template": "Describe the relation of the {obj1} to the {obj2}.",
            "r_template": "The relation is that the {obj1} is {rel} the {obj2}.",
        },
    ],
}


def extract_placeholders(template: str) -> set[str]:
    """Extract named `str.format` placeholders from a template string."""

    formatter = Formatter()
    placeholders: set[str] = set()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name:
            placeholders.add(field_name)
    return placeholders


def validate_shell_template(subtype: str, template: Mapping[str, Any]) -> None:
    """Validate one shell template entry for a subtype."""

    required_keys = {"template_id", "text_source", "q_template", "r_template"}
    missing_keys = required_keys - set(template)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Template for subtype '{subtype}' is missing keys: {missing}")

    text_source = str(template["text_source"])
    if text_source not in {"template", "gqa"}:
        raise ValueError(f"Unsupported text_source '{text_source}' for subtype '{subtype}'")

    allowed_placeholders = SHELL_PLACEHOLDER_RULES[subtype]
    q_placeholders = extract_placeholders(str(template["q_template"]))
    r_placeholders = extract_placeholders(str(template["r_template"]))
    combined_placeholders = q_placeholders | r_placeholders
    illegal_placeholders = combined_placeholders - allowed_placeholders
    if illegal_placeholders:
        illegal = ", ".join(sorted(illegal_placeholders))
        raise ValueError(f"Illegal placeholders for subtype '{subtype}': {illegal}")

    missing_placeholders = allowed_placeholders - combined_placeholders
    if missing_placeholders:
        missing = ", ".join(sorted(missing_placeholders))
        raise ValueError(f"Missing placeholders for subtype '{subtype}': {missing}")

    target_slot = SHELL_TARGET_SLOT[subtype]
    if target_slot not in r_placeholders:
        raise ValueError(
            f"Response template for subtype '{subtype}' must include target slot '{target_slot}'"
        )


def validate_shell_bank(shell_bank: Mapping[str, Any]) -> None:
    """Validate the full shell bank structure and all subtype entries."""

    template_ids: set[str] = set()
    for subtype, allowed_placeholders in SHELL_PLACEHOLDER_RULES.items():
        if subtype not in shell_bank:
            raise ValueError(f"Missing subtype '{subtype}' from shell bank")
        entries = shell_bank[subtype]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Shell bank subtype '{subtype}' must be a non-empty list")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(f"Shell bank subtype '{subtype}' contains a non-mapping entry")
            validate_shell_template(subtype, entry)
            template_id = str(entry["template_id"])
            if template_id in template_ids:
                raise ValueError(f"Duplicate template_id detected: {template_id}")
            template_ids.add(template_id)
        if allowed_placeholders and len(entries) < 1:
            raise ValueError(f"Shell bank subtype '{subtype}' has no templates")


def build_deterministic_shell_bank() -> dict[str, list[dict[str, str]]]:
    """Build a deterministic shell bank for all supported mock subtypes."""

    shell_bank = deepcopy(DETERMINISTIC_TEMPLATES)
    validate_shell_bank(shell_bank)
    return shell_bank


def load_shell_bank(path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Load and validate a shell bank JSON resource from disk."""

    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Shell bank at {path} must be a mapping")
    shell_bank = {str(subtype): entries for subtype, entries in payload.items()}
    validate_shell_bank(shell_bank)
    return {
        str(subtype): [
            {
                "template_id": str(entry["template_id"]),
                "text_source": str(entry["text_source"]),
                "q_template": str(entry["q_template"]),
                "r_template": str(entry["r_template"]),
            }
            for entry in entries
        ]
        for subtype, entries in shell_bank.items()
    }


def get_shell_templates(
    shell_bank: Mapping[str, list[dict[str, str]]],
    subtype: str,
) -> list[dict[str, str]]:
    """Return the list of templates available for a subtype."""

    templates = shell_bank.get(subtype)
    if not isinstance(templates, list) or not templates:
        raise KeyError(f"Missing shell templates for subtype '{subtype}'")
    return templates
