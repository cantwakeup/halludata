"""Render atomic fact records into fact-counterfact training pairs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from expert_data.negatives import get_cat_negative_candidates
from expert_data.schemas import FactRecord, PairRecord
from expert_data.shells import get_shell_templates
from expert_data.text_utils import count_conditioned_noun, pluralize_noun

DEFAULT_COLOR_NEGATIVES = ["black", "white", "red", "yellow", "green", "blue", "brown", "orange"]
VALID_REL_PREDICATES = {"left of", "right of", "above", "below"}
RELATION_NEGATIVE_MAP = {
    "left of": "right of",
    "right of": "left of",
    "above": "below",
    "below": "above",
}


def _filter_facts(facts: Iterable[FactRecord], subtype: str) -> list[FactRecord]:
    """Return fact records that belong to one subtype while preserving order."""

    return [fact for fact in facts if fact.subtype == subtype]


def _select_shell_template(
    shell_bank: Mapping[str, list[dict[str, str]]],
    subtype: str,
    index: int,
) -> Mapping[str, str]:
    """Choose a deterministic template for one subtype and fact index."""

    templates = get_shell_templates(shell_bank, subtype)
    return templates[index % len(templates)]


def _render_pair_with_shared_shell(
    fact: FactRecord,
    template: Mapping[str, str],
    shared_context: Mapping[str, Any],
    target_slot: str,
    pos_label: Any,
    neg_label: Any,
    metadata: Mapping[str, Any],
    pos_extra_context: Mapping[str, Any] | None = None,
    neg_extra_context: Mapping[str, Any] | None = None,
) -> PairRecord:
    """Render a pair where positive and negative responses differ at one slot only."""

    question = str(template["q_template"]).format(**shared_context)
    response_template = str(template["r_template"])

    pos_context = dict(shared_context)
    neg_context = dict(shared_context)
    pos_context[target_slot] = pos_label
    neg_context[target_slot] = neg_label
    if pos_extra_context:
        pos_context.update(dict(pos_extra_context))
    if neg_extra_context:
        neg_context.update(dict(neg_extra_context))

    return PairRecord(
        pair_id=f"{fact.fact_id}_{template['template_id']}_pair",
        fact_id=fact.fact_id,
        image_id=fact.image_id,
        subtype=fact.subtype,
        question=question,
        response_pos=response_template.format(**pos_context),
        response_neg=response_template.format(**neg_context),
        pos_label=pos_label,
        neg_label=neg_label,
        metadata={
            "template_id": str(template["template_id"]),
            "text_source": str(template["text_source"]),
            "target_slot": target_slot,
            "subject_category": fact.subject.category,
            "source_meta": dict(fact.meta),
            **dict(metadata),
        },
    )


def select_cat_negative(
    fact: FactRecord,
    cat_neg_bank: Mapping[str, Mapping[str, Any]],
    image_categories: Sequence[str],
) -> str | None:
    """Select a category negative that avoids labels already visible in the same image."""

    manual_candidates = get_cat_negative_candidates(cat_neg_bank, fact.subject.category)
    if not manual_candidates:
        return None

    blocked_categories = set(image_categories)
    blocked_categories.add(fact.subject.category)
    for candidate in manual_candidates:
        if candidate not in blocked_categories:
            return candidate

    for candidate in manual_candidates:
        if candidate != fact.subject.category:
            return candidate
    return None


def generate_count_negative(count: Any) -> int | None:
    """Generate a nearby incorrect count following the current deterministic policy."""

    try:
        numeric_count = int(count)
    except (TypeError, ValueError):
        return None
    if numeric_count < 1 or numeric_count > 5:
        return None
    if numeric_count == 1:
        return 2
    if numeric_count == 5:
        return 4
    return numeric_count + 1


def generate_color_negative(color: Any, palette: Sequence[str] | None = None) -> str | None:
    """Select a foil color different from the positive color."""

    if color is None:
        return None
    normalized_color = str(color)
    candidates = list(palette or DEFAULT_COLOR_NEGATIVES)
    for candidate in candidates:
        if candidate != normalized_color:
            return candidate
    return None


def generate_relation_negative(predicate: Any) -> str | None:
    """Invert one supported spatial predicate into its counterfactual pair."""

    if predicate is None:
        return None
    return RELATION_NEGATIVE_MAP.get(str(predicate))


def render_cat_pairs(
    facts: Iterable[FactRecord],
    shell_bank: Mapping[str, list[dict[str, str]]],
    cat_neg_bank: Mapping[str, Mapping[str, Any]],
    image_categories_by_id: Mapping[str, Sequence[str]],
    pair_limit: int | None = None,
) -> list[PairRecord]:
    """Render category pairs from atomic category facts and image context."""

    pairs: list[PairRecord] = []
    for index, fact in enumerate(_filter_facts(facts, "cat")):
        negative_label = select_cat_negative(
            fact=fact,
            cat_neg_bank=cat_neg_bank,
            image_categories=image_categories_by_id.get(fact.image_id, []),
        )
        if negative_label is None:
            continue
        template = _select_shell_template(shell_bank, "cat", index)
        pairs.append(
            _render_pair_with_shared_shell(
                fact=fact,
                template=template,
                shared_context={},
                target_slot="label",
                pos_label=fact.positive_value,
                neg_label=negative_label,
                metadata={
                    "anchor_label": fact.subject.category,
                    "image_categories": list(image_categories_by_id.get(fact.image_id, [])),
                },
            )
        )
        if pair_limit is not None and len(pairs) >= pair_limit:
            break
    return pairs


def render_cnt_pairs(
    facts: Iterable[FactRecord],
    shell_bank: Mapping[str, list[dict[str, str]]],
    pair_limit: int | None = None,
) -> list[PairRecord]:
    """Render count pairs from atomic count facts."""

    pairs: list[PairRecord] = []
    for index, fact in enumerate(_filter_facts(facts, "cnt")):
        negative_count = generate_count_negative(fact.positive_value)
        if negative_count is None:
            continue
        template = _select_shell_template(shell_bank, "cnt", index)
        pairs.append(
            _render_pair_with_shared_shell(
                fact=fact,
                template=template,
                shared_context={"obj_pl": pluralize_noun(fact.subject.category)},
                target_slot="count",
                pos_label=int(fact.positive_value),
                neg_label=negative_count,
                metadata={"anchor_label": fact.subject.category},
                pos_extra_context={"obj_count": count_conditioned_noun(fact.subject.category, fact.positive_value)},
                neg_extra_context={"obj_count": count_conditioned_noun(fact.subject.category, negative_count)},
            )
        )
        if pair_limit is not None and len(pairs) >= pair_limit:
            break
    return pairs


def render_col_pairs(
    facts: Iterable[FactRecord],
    shell_bank: Mapping[str, list[dict[str, str]]],
    pair_limit: int | None = None,
) -> list[PairRecord]:
    """Render color pairs from atomic color facts."""

    pairs: list[PairRecord] = []
    for index, fact in enumerate(_filter_facts(facts, "col")):
        negative_color = generate_color_negative(fact.positive_value)
        if negative_color is None:
            continue
        template = _select_shell_template(shell_bank, "col", index)
        pairs.append(
            _render_pair_with_shared_shell(
                fact=fact,
                template=template,
                shared_context={"obj": fact.subject.category},
                target_slot="color",
                pos_label=fact.positive_value,
                neg_label=negative_color,
                metadata={"anchor_label": fact.subject.category},
            )
        )
        if pair_limit is not None and len(pairs) >= pair_limit:
            break
    return pairs


def render_rel_pairs(
    facts: Iterable[FactRecord],
    shell_bank: Mapping[str, list[dict[str, str]]],
    pair_limit: int | None = None,
) -> list[PairRecord]:
    """Render relation pairs from atomic relation facts."""

    pairs: list[PairRecord] = []
    for index, fact in enumerate(_filter_facts(facts, "rel")):
        if fact.object is None:
            continue
        negative_relation = generate_relation_negative(fact.positive_value)
        if negative_relation is None:
            continue
        template = _select_shell_template(shell_bank, "rel", index)
        rel_pattern = f"{fact.subject.category}|{fact.positive_value}|{fact.object.category}"
        pairs.append(
            _render_pair_with_shared_shell(
                fact=fact,
                template=template,
                shared_context={"obj1": fact.subject.category, "obj2": fact.object.category},
                target_slot="rel",
                pos_label=fact.positive_value,
                neg_label=negative_relation,
                metadata={
                    "anchor_label": fact.subject.category,
                    "object_category": fact.object.category,
                    "rel_pattern": rel_pattern,
                },
            )
        )
        if pair_limit is not None and len(pairs) >= pair_limit:
            break
    return pairs
