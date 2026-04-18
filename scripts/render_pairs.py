"""Render fact-counterfact pairs from COCO-backed fact outputs with local hardening."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.io_utils import load_fact_records, read_json, read_jsonl, read_yaml, write_json, write_jsonl
from expert_data.negatives import load_cat_neg_bank
from expert_data.renderers import (
    VALID_REL_PREDICATES,
    generate_color_negative,
    generate_count_negative,
    generate_relation_negative,
    render_cat_pairs,
    render_cnt_pairs,
    render_col_pairs,
    render_rel_pairs,
    select_cat_negative,
)
from expert_data.schemas import FactRecord, PairRecord
from expert_data.shells import load_shell_bank

DROPPED_REASONS = (
    "ambiguous_cat_anchor",
    "missing_color",
    "ambiguous_rel_anchor",
    "same_category_relation",
    "count_out_of_range",
    "no_valid_negative",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pair rendering."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v0_mini.yaml", help="Path to the YAML config file.")
    parser.add_argument("--shell-bank", default=None, help="Optional shell bank JSON override.")
    parser.add_argument("--cat-neg-bank", default=None, help="Optional category negative-bank JSON override.")
    parser.add_argument("--fact-index", default=None, help="Optional fact-index JSONL override.")
    parser.add_argument("--atomic-facts", default=None, help="Optional atomic-facts JSONL override.")
    parser.add_argument("--output-unbalanced", default=None, help="Optional unbalanced pairs JSONL override.")
    parser.add_argument("--output-balanced", default=None, help="Optional balanced pairs JSONL override.")
    parser.add_argument("--output-stats", default=None, help="Optional pair-stats JSON override.")
    parser.add_argument("--dry-run", action="store_true", help="Compute counts only without writing pair JSONL files.")
    parser.add_argument("--stats-only", action="store_true", help="Rebuild stats from existing outputs only.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _validate_subtype_list(subtypes: Iterable[Any]) -> list[str]:
    """Normalize subtype names from configuration."""

    return [str(subtype) for subtype in subtypes]


def _counter_to_dict(counter: Counter[str], subtypes: Iterable[str]) -> dict[str, int]:
    """Convert a counter into a deterministic subtype-count mapping."""

    return {subtype: int(counter.get(subtype, 0)) for subtype in subtypes}


def build_fact_index_context(fact_index_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build lookup tables from the aggregated fact-index rows."""

    rows_by_image: dict[str, dict[str, Any]] = {}
    counts_by_image: dict[str, dict[str, int]] = {}
    image_categories_by_id: dict[str, list[str]] = {}
    objects_by_image_and_id: dict[str, dict[str, dict[str, Any]]] = {}

    for row in fact_index_rows:
        image_id = str(row["image_id"])
        rows_by_image[image_id] = row
        counts_by_image[image_id] = {
            str(category): int(count)
            for category, count in dict(row.get("counts", {})).items()
        }

        ordered_categories: list[str] = []
        object_lookup: dict[str, dict[str, Any]] = {}
        seen_categories: set[str] = set()
        for object_entry in row.get("objects", []):
            object_id = str(object_entry.get("object_id", ""))
            category = str(object_entry.get("category", ""))
            object_lookup[object_id] = dict(object_entry)
            if category and category not in seen_categories:
                seen_categories.add(category)
                ordered_categories.append(category)
        image_categories_by_id[image_id] = ordered_categories
        objects_by_image_and_id[image_id] = object_lookup

    return {
        "rows_by_image": rows_by_image,
        "counts_by_image": counts_by_image,
        "image_categories_by_id": image_categories_by_id,
        "objects_by_image_and_id": objects_by_image_and_id,
    }


def _fact_has_valid_cat_anchor(
    fact: FactRecord,
    fact_index_context: Mapping[str, Any],
    min_area_ratio: float,
) -> bool:
    """Check whether a category fact references a unique, sufficiently large object anchor."""

    image_counts = fact_index_context["counts_by_image"].get(fact.image_id, {})
    if int(image_counts.get(fact.subject.category, 0)) != 1:
        return False
    return float(fact.meta.get("area_ratio", 0.0)) >= min_area_ratio


def filter_atomic_facts(
    facts: list[FactRecord],
    fact_index_context: Mapping[str, Any],
    enabled_subtypes: list[str],
    filters_cfg: Mapping[str, Any],
    cat_neg_bank: Mapping[str, Mapping[str, Any]],
) -> tuple[list[FactRecord], dict[str, int], dict[str, int], dict[str, int]]:
    """Apply subtype purity filters and track why facts are dropped."""

    min_area_ratio = float(filters_cfg.get("object_min_area_ratio", 0.01))
    filtered_facts: list[FactRecord] = []
    before_counter: Counter[str] = Counter()
    after_counter: Counter[str] = Counter()
    dropped_counter: Counter[str] = Counter()

    for fact in facts:
        if fact.subtype not in enabled_subtypes:
            continue
        before_counter[fact.subtype] += 1
        image_counts = fact_index_context["counts_by_image"].get(fact.image_id, {})
        image_categories = fact_index_context["image_categories_by_id"].get(fact.image_id, [])

        drop_reason: str | None = None
        if fact.subtype == "cat":
            if not _fact_has_valid_cat_anchor(fact, fact_index_context, min_area_ratio):
                drop_reason = "ambiguous_cat_anchor"
            elif select_cat_negative(fact, cat_neg_bank, image_categories) is None:
                drop_reason = "no_valid_negative"
        elif fact.subtype == "cnt":
            negative_count = generate_count_negative(fact.positive_value)
            if negative_count is None:
                drop_reason = "count_out_of_range"
        elif fact.subtype == "col":
            if int(image_counts.get(fact.subject.category, 0)) != 1:
                drop_reason = "ambiguous_cat_anchor"
            elif fact.positive_value in {None, ""}:
                drop_reason = "missing_color"
            elif generate_color_negative(fact.positive_value) is None:
                drop_reason = "no_valid_negative"
        elif fact.subtype == "rel":
            if fact.object is None:
                drop_reason = "no_valid_negative"
            elif fact.subject.category == fact.object.category:
                drop_reason = "same_category_relation"
            elif (
                int(image_counts.get(fact.subject.category, 0)) != 1
                or int(image_counts.get(fact.object.category, 0)) != 1
            ):
                drop_reason = "ambiguous_rel_anchor"
            elif str(fact.positive_value) not in VALID_REL_PREDICATES:
                drop_reason = "no_valid_negative"
            elif generate_relation_negative(fact.positive_value) is None:
                drop_reason = "no_valid_negative"

        if drop_reason is not None:
            dropped_counter[drop_reason] += 1
            continue

        filtered_facts.append(fact)
        after_counter[fact.subtype] += 1

    return (
        filtered_facts,
        _counter_to_dict(before_counter, enabled_subtypes),
        _counter_to_dict(after_counter, enabled_subtypes),
        {reason: int(dropped_counter.get(reason, 0)) for reason in DROPPED_REASONS},
    )


def apply_sampling_caps(
    facts: list[FactRecord],
    sampling_cfg: Mapping[str, Any],
) -> list[FactRecord]:
    """Apply per-image anchor caps before rendering unbalanced pairs."""

    limits = {
        "cat": int(sampling_cfg.get("max_cat_anchors_per_image", 1_000_000)),
        "cnt": int(sampling_cfg.get("max_cnt_anchors_per_image", 1_000_000)),
        "col": int(sampling_cfg.get("max_col_anchors_per_image", 1_000_000)),
        "rel": int(sampling_cfg.get("max_rel_pairs_per_image", 1_000_000)),
    }
    counts_by_key: dict[tuple[str, str], int] = defaultdict(int)
    selected_facts: list[FactRecord] = []

    for fact in facts:
        key = (fact.subtype, fact.image_id)
        if counts_by_key[key] >= limits.get(fact.subtype, 1_000_000):
            continue
        counts_by_key[key] += 1
        selected_facts.append(fact)
    return selected_facts


def render_unbalanced_pairs(
    facts: list[FactRecord],
    shell_bank: Mapping[str, list[dict[str, str]]],
    cat_neg_bank: Mapping[str, Mapping[str, Any]],
    fact_index_context: Mapping[str, Any],
    enabled_subtypes: list[str],
    pair_limit: int | None,
) -> list[PairRecord]:
    """Render all unbalanced subtype pairs after filtering and per-image sampling."""

    image_categories_by_id = fact_index_context["image_categories_by_id"]
    renderer_map = {
        "cat": lambda subset: render_cat_pairs(
            subset,
            shell_bank=shell_bank,
            cat_neg_bank=cat_neg_bank,
            image_categories_by_id=image_categories_by_id,
            pair_limit=pair_limit,
        ),
        "cnt": lambda subset: render_cnt_pairs(subset, shell_bank=shell_bank, pair_limit=pair_limit),
        "col": lambda subset: render_col_pairs(subset, shell_bank=shell_bank, pair_limit=pair_limit),
        "rel": lambda subset: render_rel_pairs(subset, shell_bank=shell_bank, pair_limit=pair_limit),
    }

    unbalanced_pairs: list[PairRecord] = []
    for subtype in enabled_subtypes:
        renderer = renderer_map.get(subtype)
        if renderer is None:
            continue
        subtype_facts = [fact for fact in facts if fact.subtype == subtype]
        unbalanced_pairs.extend(renderer(subtype_facts))
    return unbalanced_pairs


def balance_pairs(
    pairs: list[PairRecord],
    enabled_subtypes: list[str],
    targets: Mapping[str, Any],
    caps: Mapping[str, Any],
) -> list[PairRecord]:
    """Apply label and relation-pattern caps, then keep pairs up to subtype targets."""

    per_label_cap = int(caps.get("per_label_cap", 1_000_000))
    per_rel_pattern_cap = int(caps.get("per_rel_pattern_cap", 1_000_000))

    pairs_by_subtype: dict[str, list[PairRecord]] = {subtype: [] for subtype in enabled_subtypes}
    for pair in pairs:
        pairs_by_subtype.setdefault(pair.subtype, []).append(pair)

    balanced_pairs: list[PairRecord] = []
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rel_pattern_counts: Counter[str] = Counter()

    for subtype in enabled_subtypes:
        subtype_target = int(targets.get(subtype, len(pairs_by_subtype.get(subtype, []))))
        kept = 0
        for pair in pairs_by_subtype.get(subtype, []):
            if kept >= subtype_target:
                break
            if subtype in {"cat", "cnt", "col"}:
                anchor_label = str(pair.metadata.get("anchor_label", ""))
                if anchor_label and label_counts[subtype][anchor_label] >= per_label_cap:
                    continue
                if anchor_label:
                    label_counts[subtype][anchor_label] += 1
            elif subtype == "rel":
                rel_pattern = str(pair.metadata.get("rel_pattern", ""))
                if rel_pattern and rel_pattern_counts[rel_pattern] >= per_rel_pattern_cap:
                    continue
                if rel_pattern:
                    rel_pattern_counts[rel_pattern] += 1
            balanced_pairs.append(pair)
            kept += 1

    return balanced_pairs


def _count_pairs_by_subtype(pairs: list[PairRecord], enabled_subtypes: list[str]) -> dict[str, int]:
    """Count rendered pairs for each enabled subtype."""

    return _counter_to_dict(Counter(pair.subtype for pair in pairs), enabled_subtypes)


def _template_usage_by_subtype(pairs: list[PairRecord], enabled_subtypes: list[str]) -> dict[str, dict[str, int]]:
    """Count template usage for each subtype."""

    usage: dict[str, Counter[str]] = {subtype: Counter() for subtype in enabled_subtypes}
    for pair in pairs:
        template_id = str(pair.metadata.get("template_id", "unknown"))
        usage.setdefault(pair.subtype, Counter())[template_id] += 1
    return {subtype: dict(sorted(counter.items())) for subtype, counter in usage.items()}


def _label_key_for_pair(pair: PairRecord) -> str:
    """Choose the key used for label-usage statistics."""

    if pair.subtype == "rel":
        return str(pair.metadata.get("rel_pattern", ""))
    return str(pair.metadata.get("anchor_label", pair.pos_label))


def _label_usage_topk(
    pairs: list[PairRecord],
    enabled_subtypes: list[str],
    topk: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Summarize the most frequent labels or relation patterns for each subtype."""

    usage: dict[str, Counter[str]] = {subtype: Counter() for subtype in enabled_subtypes}
    for pair in pairs:
        label_key = _label_key_for_pair(pair)
        if label_key:
            usage.setdefault(pair.subtype, Counter())[label_key] += 1
    return {
        subtype: [{"label": label, "count": count} for label, count in counter.most_common(topk)]
        for subtype, counter in usage.items()
    }


def _per_image_pair_counts(pairs: list[PairRecord]) -> dict[str, int]:
    """Count pairs per image for one rendered output split."""

    counter: Counter[str] = Counter(pair.image_id for pair in pairs)
    return {image_id: int(counter[image_id]) for image_id in sorted(counter)}


def _subtype_counts_by_image_sample(
    pairs: list[PairRecord],
    enabled_subtypes: list[str],
    sample_size: int = 5,
) -> list[dict[str, Any]]:
    """Summarize subtype counts for the first few image ids in a split."""

    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in pairs:
        grouped[pair.image_id][pair.subtype] += 1
    sample_rows: list[dict[str, Any]] = []
    for image_id in sorted(grouped)[:sample_size]:
        sample_rows.append(
            {
                "image_id": image_id,
                "counts": _counter_to_dict(grouped[image_id], enabled_subtypes),
            }
        )
    return sample_rows


def build_pair_stats(
    enabled_subtypes: list[str],
    counts_before_filter: dict[str, int],
    counts_after_filter: dict[str, int],
    dropped_by_reason: Mapping[str, int],
    unbalanced_pairs: list[PairRecord],
    balanced_pairs: list[PairRecord],
) -> dict[str, Any]:
    """Build the pair statistics report written alongside the pair outputs."""

    return {
        "counts_before_filter": counts_before_filter,
        "counts_after_filter": counts_after_filter,
        "counts_unbalanced": _count_pairs_by_subtype(unbalanced_pairs, enabled_subtypes),
        "counts_balanced": _count_pairs_by_subtype(balanced_pairs, enabled_subtypes),
        "dropped_by_reason": {reason: int(dropped_by_reason.get(reason, 0)) for reason in DROPPED_REASONS},
        "template_usage": {
            "unbalanced": _template_usage_by_subtype(unbalanced_pairs, enabled_subtypes),
            "balanced": _template_usage_by_subtype(balanced_pairs, enabled_subtypes),
        },
        "label_usage_topk": {
            "unbalanced": _label_usage_topk(unbalanced_pairs, enabled_subtypes),
            "balanced": _label_usage_topk(balanced_pairs, enabled_subtypes),
        },
        "per_image_pair_counts": {
            "unbalanced": _per_image_pair_counts(unbalanced_pairs),
            "balanced": _per_image_pair_counts(balanced_pairs),
        },
        "subtype_counts_by_image_sample": {
            "unbalanced": _subtype_counts_by_image_sample(unbalanced_pairs, enabled_subtypes),
            "balanced": _subtype_counts_by_image_sample(balanced_pairs, enabled_subtypes),
        },
    }


def summarize_pairs(pairs: list[PairRecord]) -> str:
    """Build a human-readable subtype summary string for CLI output."""

    counts = Counter(pair.subtype for pair in pairs)
    return ", ".join(f"{subtype}={counts.get(subtype, 0)}" for subtype in sorted(counts)) or "none"


def _load_pair_rows(path: Path) -> list[PairRecord]:
    """Load pair rows from JSONL into typed pair records."""

    rows = read_jsonl(path)
    pair_rows: list[PairRecord] = []
    for row in rows:
        pair_rows.append(
            PairRecord(
                pair_id=str(row["pair_id"]),
                fact_id=str(row["fact_id"]),
                image_id=str(row["image_id"]),
                subtype=str(row["subtype"]),
                question=str(row["question"]),
                response_pos=str(row["response_pos"]),
                response_neg=str(row["response_neg"]),
                pos_label=row.get("pos_label"),
                neg_label=row.get("neg_label"),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return pair_rows


def render_pairs_from_config(
    config_path: str | Path,
    cli_args: argparse.Namespace | None = None,
) -> tuple[list[PairRecord], list[PairRecord], dict[str, Any], dict[str, Path]]:
    """Render unbalanced and balanced pairs plus stats from the configured resources."""

    config = read_yaml(config_path)
    input_paths = dict(config.get("input_paths", {}))
    output_paths = dict(config.get("output_paths", {}))
    enabled_subtypes = _validate_subtype_list(config.get("enabled_subtypes", ["cat", "cnt", "col", "rel"]))

    shell_bank_path = resolve_project_path(getattr(cli_args, "shell_bank", None) or input_paths["shell_bank"])
    cat_neg_bank_path = resolve_project_path(getattr(cli_args, "cat_neg_bank", None) or input_paths["cat_neg_bank"])
    fact_index_path = resolve_project_path(getattr(cli_args, "fact_index", None) or input_paths["fact_index"])
    atomic_facts_path = resolve_project_path(getattr(cli_args, "atomic_facts", None) or input_paths["atomic_facts"])
    output_unbalanced_path = resolve_project_path(
        getattr(cli_args, "output_unbalanced", None)
        or output_paths.get("pairs_unbalanced", "data/outputs/pairs_unbalanced_v0.jsonl")
    )
    output_balanced_path = resolve_project_path(
        getattr(cli_args, "output_balanced", None)
        or output_paths.get("pairs_balanced", config.get("output_path", "data/outputs/pairs_balanced_v0.jsonl"))
    )
    output_stats_path = resolve_project_path(
        getattr(cli_args, "output_stats", None)
        or output_paths.get("pair_stats", "data/outputs/pair_stats_v0.json")
    )

    shell_bank = load_shell_bank(shell_bank_path)
    cat_neg_bank = load_cat_neg_bank(cat_neg_bank_path)
    fact_index_rows = read_jsonl(fact_index_path)
    atomic_facts = load_fact_records(atomic_facts_path)
    filters_cfg = dict(config.get("filters", {}))
    sampling_cfg = dict(config.get("sampling", {}))
    targets = dict(config.get("targets", {}))
    caps = dict(config.get("caps", {}))
    pair_limit = config.get("pair_limits", {}).get("per_subtype")

    fact_index_context = build_fact_index_context(fact_index_rows)
    filtered_facts, counts_before_filter, counts_after_filter, dropped_by_reason = filter_atomic_facts(
        atomic_facts,
        fact_index_context=fact_index_context,
        enabled_subtypes=enabled_subtypes,
        filters_cfg=filters_cfg,
        cat_neg_bank=cat_neg_bank,
    )

    if getattr(cli_args, "stats_only", False):
        unbalanced_pairs = _load_pair_rows(output_unbalanced_path) if output_unbalanced_path.exists() else []
        balanced_pairs = _load_pair_rows(output_balanced_path) if output_balanced_path.exists() else []
    else:
        sampled_facts = apply_sampling_caps(filtered_facts, sampling_cfg=sampling_cfg)
        unbalanced_pairs = render_unbalanced_pairs(
            sampled_facts,
            shell_bank=shell_bank,
            cat_neg_bank=cat_neg_bank,
            fact_index_context=fact_index_context,
            enabled_subtypes=enabled_subtypes,
            pair_limit=int(pair_limit) if pair_limit is not None else None,
        )
        balanced_pairs = balance_pairs(
            unbalanced_pairs,
            enabled_subtypes=enabled_subtypes,
            targets=targets,
            caps=caps,
        )

    pair_stats = build_pair_stats(
        enabled_subtypes=enabled_subtypes,
        counts_before_filter=counts_before_filter,
        counts_after_filter=counts_after_filter,
        dropped_by_reason=dropped_by_reason,
        unbalanced_pairs=unbalanced_pairs,
        balanced_pairs=balanced_pairs,
    )

    if not getattr(cli_args, "dry_run", False) and not getattr(cli_args, "stats_only", False):
        write_jsonl(output_unbalanced_path, unbalanced_pairs)
        write_jsonl(output_balanced_path, balanced_pairs)
        legacy_output_path = config.get("output_path")
        if legacy_output_path:
            write_jsonl(resolve_project_path(legacy_output_path), balanced_pairs)
    write_json(output_stats_path, pair_stats)

    return unbalanced_pairs, balanced_pairs, pair_stats, {
        "pairs_unbalanced": output_unbalanced_path,
        "pairs_balanced": output_balanced_path,
        "pair_stats": output_stats_path,
    }


def main() -> int:
    """Run the CLI entry point for hardened pair rendering."""

    args = parse_args()
    config_path = resolve_project_path(args.config)
    unbalanced_pairs, balanced_pairs, pair_stats, output_paths = render_pairs_from_config(
        config_path=config_path,
        cli_args=args,
    )

    if args.stats_only:
        print(f"Rebuilt stats only at {output_paths['pair_stats']}")
    elif args.dry_run:
        print("Dry run complete; pair files were not written.")
        print(f"Would produce {len(unbalanced_pairs)} unbalanced pairs and {len(balanced_pairs)} balanced pairs.")
        print(f"Stats written to {output_paths['pair_stats']}")
    else:
        print(f"Wrote {len(unbalanced_pairs)} unbalanced pairs to {output_paths['pairs_unbalanced']}")
        print(f"Wrote {len(balanced_pairs)} balanced pairs to {output_paths['pairs_balanced']}")
        print(f"Wrote pair stats to {output_paths['pair_stats']}")

    print(f"Unbalanced subtype summary: {summarize_pairs(unbalanced_pairs)}")
    print(f"Balanced subtype summary: {summarize_pairs(balanced_pairs)}")
    print(
        "Dropped by reason: "
        + ", ".join(f"{reason}={count}" for reason, count in pair_stats["dropped_by_reason"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
