"""Build image-disjoint AFTER-template v2 pair banks.

V2 keeps the v1 image-disjoint idea but moves closer to AFTER FAS:

- visual side: image + benchmark-style question
- trusted side: text-only image-level factual description + same question
- cat/attr: COCO-derived object/count/color facts
- rel: preferred external relation annotations, with COCO bbox fallback

The output is isolated under data/after_template_disjoint_v2 by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.coco_io import load_coco_instances
from expert_data.io_utils import write_json, write_jsonl
from expert_data.text_utils import count_conditioned_noun, pluralize_noun

EXPERT_TYPES = ("cat", "attr", "rel")
SOURCE_NAME = "after_template_disjoint_v2"
CURRENT_SOURCE_NAME = SOURCE_NAME

RELATION_PHRASES = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "above": "above",
    "below": "below",
    "in": "in",
    "inside": "inside",
    "on": "on",
    "under": "under",
    "beside": "beside",
    "close_to": "close to",
    "next_to": "next to",
    "near": "near",
    "in_front_of": "in front of",
    "behind": "behind",
    "carrying": "carrying",
    "eating": "eating",
    "drinking": "drinking",
    "looking_at": "looking at",
    "watching": "watching",
    "playing_with": "playing with",
    "using": "using",
    "holding": "holding",
    "riding": "riding",
    "wearing": "wearing",
    "touching": "touching",
    "direct_contact": "in direct contact with",
}
OPPOSITE_RELATIONS = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "on": "under",
    "under": "on",
    "in_front_of": "behind",
    "behind": "in_front_of",
}
REL_SUBTYPE = {
    "left_of": "rel_position_horizontal",
    "right_of": "rel_position_horizontal",
    "above": "rel_position_vertical",
    "below": "rel_position_vertical",
    "direct_contact": "rel_contact",
    "touching": "rel_contact",
    "in": "rel_contact",
    "inside": "rel_contact",
    "beside": "rel_contact",
    "close_to": "rel_contact",
    "holding": "rel_interaction",
    "riding": "rel_interaction",
    "wearing": "rel_interaction",
    "carrying": "rel_interaction",
    "eating": "rel_interaction",
    "drinking": "rel_interaction",
    "looking_at": "rel_interaction",
    "watching": "rel_interaction",
    "playing_with": "rel_interaction",
    "using": "rel_interaction",
    "on": "rel_contact",
    "under": "rel_position_vertical",
    "next_to": "rel_contact",
    "near": "rel_contact",
    "in_front_of": "rel_position_depth",
    "behind": "rel_position_depth",
}
REL_BUCKETS = {
    "left_of": "horizontal",
    "right_of": "horizontal",
    "above": "vertical",
    "below": "vertical",
    "under": "vertical",
    "in_front_of": "depth",
    "behind": "depth",
    "on": "contact",
    "in": "contact",
    "inside": "contact",
    "next_to": "contact",
    "near": "contact",
    "beside": "contact",
    "close_to": "contact",
    "touching": "contact",
    "direct_contact": "contact",
    "holding": "interaction",
    "riding": "interaction",
    "wearing": "interaction",
    "carrying": "interaction",
    "eating": "interaction",
    "drinking": "interaction",
    "looking_at": "interaction",
    "watching": "interaction",
    "playing_with": "interaction",
    "using": "interaction",
}
REL_BUCKET_ORDER = ("horizontal", "vertical", "depth", "contact", "interaction", "semantic")
DEFAULT_RELATION_BUCKET_RATIO = "horizontal=0.5,vertical=0.1,depth=0.15,contact=0.15,interaction=0.1,semantic=0.0"
VAGUE_RELATIONS = {
    "",
    "of",
    "with",
    "by",
    "at",
    "around",
    "surrounding",
    "surrounded_by",
}


def _load_script_module(module_name: str, script_name: str) -> Any:
    """Load a sibling script without relying on import package names."""

    helper_path = Path(__file__).resolve().with_name(script_name)
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STYLE = _load_script_module("_halludata_after_style_pairs_v2", "build_after_style_pairs.py")
RELATION_V2 = _load_script_module("_halludata_after_template_relation_v2_for_disjoint_v2", "build_after_template_relation_v2.py")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-instances", required=True, help="COCO instances JSON for cat/attr and optional rel fallback.")
    parser.add_argument("--image-root", required=True, help="COCO image root for cat/attr and optional rel fallback.")
    parser.add_argument("--output-dir", default="data/after_template_disjoint_v2/pairs")
    parser.add_argument("--source-name", default=SOURCE_NAME, help="Source tag stored in every output row.")
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--type-image-ratio", default="cat=0.3,attr=0.3,rel=0.4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", default="0.6,0.2,0.2")
    parser.add_argument("--max-cat-pairs-per-image", type=int, default=2)
    parser.add_argument("--max-attr-pairs-per-image", type=int, default=2)
    parser.add_argument("--max-rel-pairs-per-image", type=int, default=4)
    parser.add_argument("--relation-source", default="", help="Optional VG/GQA/AMBER-style relation JSON/JSONL.")
    parser.add_argument("--relation-image-root", default="", help="Image root for --relation-source. Defaults to --image-root.")
    parser.add_argument("--relation-fallback", choices=["coco_bbox", "none"], default="coco_bbox")
    parser.add_argument(
        "--relation-bucket-ratio",
        default=DEFAULT_RELATION_BUCKET_RATIO,
        help="Per-image external relation sampling mix across horizontal/vertical/depth/contact/interaction/semantic buckets.",
    )
    parser.add_argument("--progress-every", type=int, default=50000, help="Progress interval while parsing external relation rows.")
    parser.add_argument(
        "--rel-template-variant",
        choices=["basic", "inverse", "contrastive_inverse"],
        default="contrastive_inverse",
        help="Template for COCO bbox fallback rows.",
    )
    parser.add_argument(
        "--negative-relation-trusted-text",
        choices=["contrastive", "positive_only"],
        default="contrastive",
        help="For generated no relation queries, use yes+not text or only the positive true relation fact.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def optional_path(raw_path: str | Path) -> Path | None:
    """Resolve a path if non-empty."""

    text = str(raw_path).strip()
    if not text:
        return None
    return resolve_project_path(text)


def parse_type_ratio(raw_ratio: str) -> dict[str, float]:
    """Parse cat/attr/rel image allocation ratios."""

    text = str(raw_ratio).strip()
    if "=" in text:
        ratios: dict[str, float] = {}
        for piece in text.split(","):
            if not piece.strip():
                continue
            key, value = piece.split("=", 1)
            ratios[key.strip()] = float(value.strip())
    else:
        values = [float(piece.strip()) for piece in text.split(",") if piece.strip()]
        if len(values) != len(EXPERT_TYPES):
            raise ValueError("--type-image-ratio must provide cat,attr,rel")
        ratios = dict(zip(EXPERT_TYPES, values))
    missing = [expert for expert in EXPERT_TYPES if expert not in ratios]
    if missing:
        raise ValueError(f"--type-image-ratio missing types: {missing}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--type-image-ratio must sum to 1.0, got {total}")
    return ratios


def allocation_counts(num_images: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Convert ratios to integer counts with largest remainders."""

    exact = {expert: float(num_images) * float(ratios[expert]) for expert in EXPERT_TYPES}
    counts = {expert: int(math.floor(value)) for expert, value in exact.items()}
    remaining = int(num_images) - sum(counts.values())
    order = sorted(EXPERT_TYPES, key=lambda expert: (exact[expert] - counts[expert], expert), reverse=True)
    for expert in order[:remaining]:
        counts[expert] += 1
    return counts


def parse_named_ratio(raw_ratio: str, allowed_keys: Iterable[str], *, name: str) -> dict[str, float]:
    """Parse a named ratio string like ``a=0.5,b=0.5``."""

    allowed = tuple(allowed_keys)
    ratios = {key: 0.0 for key in allowed}
    for piece in str(raw_ratio).split(","):
        if not piece.strip():
            continue
        if "=" not in piece:
            raise ValueError(f"{name} entries must use key=value, got {piece!r}")
        key, value = piece.split("=", 1)
        key = key.strip()
        if key not in ratios:
            raise ValueError(f"{name} has unsupported key {key!r}; allowed keys are {allowed}")
        ratios[key] = float(value.strip())
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError(f"{name} must have positive total weight")
    return {key: value / total for key, value in ratios.items()}


def article_for(noun: str) -> str:
    """Return a simple indefinite article."""

    return STYLE.article_for(noun)


def render_visual_prompt(question: str) -> str:
    """Render image-query prompt text."""

    return f"Question: {question}\nPlease answer the question based on the image."


def render_trusted_prompt(trusted_factual_text: str, question: str) -> str:
    """Render trusted text-only prompt text in AFTER FAS style."""

    return (
        f"The given image depicts the following scene: {trusted_factual_text}\n"
        "Please directly answer the following question from the image description, "
        f"without guessing or reasoning. Question: {question}"
    )


def count_sentence(category: str, count: int) -> str:
    """Render a factual count sentence."""

    noun = count_conditioned_noun(category, count)
    verb = "is" if int(count) == 1 else "are"
    return f"There {verb} {int(count)} {noun} in the image."


def object_scene_sentence(annotations: Iterable[Mapping[str, Any]], max_categories: int = 8) -> str:
    """Summarize visible object counts as a compact image-level description."""

    counts = Counter(str(row["category_name"]) for row in annotations)
    if not counts:
        return "The image contains visible objects."
    pieces = [
        count_sentence(category, int(count)).rstrip(".")
        for category, count in counts.most_common(max_categories)
    ]
    if len(counts) > max_categories:
        pieces.append("There are additional objects in the image")
    return ". ".join(pieces) + "."


def make_row(
    *,
    row_id: str,
    image: str,
    image_id: str | int,
    question: str,
    trusted_factual_text: str,
    hallucination_type: str,
    subtype: str,
    objects: list[str],
    factual_fact: str,
    label: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one v2 AFTER-template row."""

    item = {
        "id": row_id,
        "pair_id": row_id,
        "image": image,
        "image_id": image_id,
        "question": question,
        "visual_prompt": render_visual_prompt(question),
        "trusted_factual_text": trusted_factual_text,
        "trusted_prompt": render_trusted_prompt(trusted_factual_text, question),
        "hallucination_type": hallucination_type,
        "subtype": subtype,
        "objects": objects,
        "factual_fact": factual_fact,
        "label": label,
        "source": CURRENT_SOURCE_NAME,
        "prompt_style": "after_fas_complete_scene_v2",
    }
    if extra:
        item.update(dict(extra))
    return item


def valid_annotations_for(
    image_id: int,
    annotations_by_image: Mapping[int, list[dict[str, Any]]],
    categories_by_id: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Return valid COCO annotations for one image."""

    return STYLE.valid_annotations(annotations_by_image.get(image_id, []), categories_by_id)


def build_cat_rows(
    *,
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    all_categories: list[str],
    max_pairs: int,
    rng: random.Random,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    """Build category rows with image-level trusted descriptions."""

    rows: list[dict[str, Any]] = []
    image_id = int(image["id"])
    image_name = str(image.get("file_name") or f"{image_id:012d}.jpg")
    present_categories = sorted({str(row["category_name"]) for row in annotations})
    base_scene = object_scene_sentence(annotations)
    if present_categories:
        category = rng.choice(present_categories)
        article = article_for(category)
        fact = f"There is {article} {category} in the image."
        rows.append(
            make_row(
                row_id=f"{CURRENT_SOURCE_NAME}_cat_present_{image_id}_{category.replace(' ', '_')}",
                image=image_name,
                image_id=image_id,
                question=f"Is there {article} {category} in the image?",
                trusted_factual_text=f"{base_scene} {fact}",
                hallucination_type="cat",
                subtype="cat_present",
                objects=[category],
                factual_fact=fact,
                label="yes",
            )
        )
    else:
        skipped["cat_no_present_category"] += 1

    absent_categories = sorted(set(all_categories) - set(present_categories))
    if absent_categories:
        category = rng.choice(absent_categories)
        article = article_for(category)
        fact = f"There is no {category} in the image."
        rows.append(
            make_row(
                row_id=f"{CURRENT_SOURCE_NAME}_cat_absent_{image_id}_{category.replace(' ', '_')}",
                image=image_name,
                image_id=image_id,
                question=f"Is there {article} {category} in the image?",
                trusted_factual_text=f"{base_scene} {fact}",
                hallucination_type="cat",
                subtype="cat_absent",
                objects=[category],
                factual_fact=fact,
                label="no",
            )
        )
    else:
        skipped["cat_no_absent_category"] += 1
    return rows[: max(0, int(max_pairs))]


def build_attr_rows(
    *,
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    image_root: Path,
    max_pairs: int,
    rng: random.Random,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    """Build count/color rows with image-level trusted descriptions."""

    rows: list[dict[str, Any]] = []
    image_id = int(image["id"])
    image_name = str(image.get("file_name") or f"{image_id:012d}.jpg")
    base_scene = object_scene_sentence(annotations)

    count_pair = STYLE.build_count_pair(image, annotations, rng)
    if count_pair is not None:
        fact = str(count_pair["factual_answer"]).strip()
        rows.append(
            make_row(
                row_id=f"{CURRENT_SOURCE_NAME}_attr_count_{image_id}_{len(rows)}",
                image=image_name,
                image_id=image_id,
                question=str(count_pair["question"]),
                trusted_factual_text=f"{base_scene} {fact}",
                hallucination_type="attr",
                subtype="attr_count",
                objects=list(count_pair.get("objects", [])),
                factual_fact=fact,
            )
        )
    else:
        skipped["attr_no_count"] += 1

    if len(rows) < int(max_pairs):
        color_pair = STYLE.build_color_pair(image, annotations, image_root, rng, skipped)
        if color_pair is not None:
            fact = str(color_pair["factual_answer"]).strip()
            rows.append(
                make_row(
                    row_id=f"{CURRENT_SOURCE_NAME}_attr_color_{image_id}_{len(rows)}",
                    image=image_name,
                    image_id=image_id,
                    question=str(color_pair["question"]),
                    trusted_factual_text=f"{base_scene} {fact}",
                    hallucination_type="attr",
                    subtype="attr_color",
                    objects=list(color_pair.get("objects", [])),
                    factual_fact=fact,
                )
            )
    if not rows:
        skipped["attr_no_pairs_for_image"] += 1
    return rows[: max(0, int(max_pairs))]


def read_records(path: Path) -> list[Any]:
    """Read JSON or JSONL records."""

    return list(iter_records(path))


def iter_records(path: Path, rng: random.Random | None = None) -> Iterable[Any]:
    """Iterate JSON or JSONL records without building avoidable intermediate lists."""

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yield json.loads(text)
        return
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        if rng is not None:
            rng.shuffle(payload)
        for item in payload:
            yield item
        return
    if isinstance(payload, dict):
        for key in ("data", "annotations", "relationships", "samples", "questions"):
            value = payload.get(key)
            if isinstance(value, list):
                if rng is not None:
                    rng.shuffle(value)
                for item in value:
                    yield item
                return
        if payload and all(isinstance(value, Mapping) for value in payload.values()):
            keys = list(payload)
            if rng is not None:
                rng.shuffle(keys)
            for key in keys:
                value = payload[key]
                item = dict(value)
                item.setdefault("image_id", key)
                yield item
            return
        yield payload


def flatten_relation_records(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Flatten common VG/GQA/AMBER-style relation records into dictionaries."""

    return list(iter_flatten_relation_records(records))


def iter_flatten_relation_records(records: Iterable[Any]) -> Iterable[dict[str, Any]]:
    """Yield common VG/GQA/AMBER-style relation records as flat dictionaries."""

    for record in records:
        if not isinstance(record, Mapping):
            continue
        objects = record.get("objects")
        if isinstance(objects, Mapping):
            image_meta = {
                key: record[key]
                for key in ("image", "image_id", "img_id", "file_name")
                if key in record
            }
            image_identifier = str(image_meta.get("image") or image_meta.get("file_name") or image_meta.get("image_id") or image_meta.get("img_id") or "")
            if image_identifier and "image" not in image_meta and "file_name" not in image_meta:
                image_meta["image"] = f"{image_identifier}.jpg"
            object_names: dict[str, str] = {}
            for object_id, obj in objects.items():
                if isinstance(obj, Mapping):
                    name = string_field(obj, "name", "names", "category", "label")
                    if name:
                        object_names[str(object_id)] = name
            for object_id, obj in objects.items():
                if not isinstance(obj, Mapping):
                    continue
                subject = object_names.get(str(object_id)) or string_field(obj, "name", "names", "category", "label")
                relations = obj.get("relations") or obj.get("relationships")
                if not isinstance(relations, list):
                    continue
                for relation in relations:
                    if not isinstance(relation, Mapping):
                        continue
                    merged = dict(image_meta)
                    merged["subject"] = subject
                    merged["predicate"] = string_field(relation, "name", "predicate", "relation", "rel")
                    object_ref = string_field(relation, "object", "object_id", "obj", "to")
                    merged["object"] = object_names.get(object_ref, object_ref)
                    yield merged
            continue
        relations = record.get("relationships") or record.get("relations")
        if isinstance(relations, list):
            for relation in relations:
                if isinstance(relation, Mapping):
                    merged = dict(relation)
                    for key in ("image", "image_id", "img_id", "file_name"):
                        if key in record and key not in merged:
                            merged[key] = record[key]
                    yield merged
            continue
        yield dict(record)


def string_field(row: Mapping[str, Any], *keys: str) -> str:
    """Read a string field from a row or nested name field."""

    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if isinstance(value, Mapping):
            for name_key in ("name", "names", "category", "label"):
                nested = value.get(name_key)
                if isinstance(nested, list) and nested:
                    return str(nested[0]).strip()
                if nested not in (None, ""):
                    return str(nested).strip()
        if isinstance(value, list) and value:
            return str(value[0]).strip()
        if value not in (None, ""):
            return str(value).strip()
    return ""


def normalize_relation(raw_relation: str) -> str:
    """Normalize a predicate/relation string into a compact key."""

    text = str(raw_relation).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    aliases = {
        "left": "left_of",
        "left of": "left_of",
        "to the left of": "left_of",
        "right": "right_of",
        "right of": "right_of",
        "to the right of": "right_of",
        "top": "above",
        "on top of": "above",
        "above": "above",
        "below": "below",
        "bottom": "below",
        "in": "in",
        "inside": "inside",
        "under": "under",
        "underneath": "under",
        "beside": "beside",
        "next": "next_to",
        "next to": "next_to",
        "near": "near",
        "close to": "close_to",
        "in front of": "in_front_of",
        "front of": "in_front_of",
        "behind": "behind",
        "carry": "carrying",
        "carrying": "carrying",
        "eat": "eating",
        "eating": "eating",
        "drink": "drinking",
        "drinking": "drinking",
        "look at": "looking_at",
        "looking at": "looking_at",
        "watch": "watching",
        "watching": "watching",
        "play with": "playing_with",
        "playing with": "playing_with",
        "use": "using",
        "using": "using",
        "holding": "holding",
        "hold": "holding",
        "riding": "riding",
        "ride": "riding",
        "wearing": "wearing",
        "wear": "wearing",
        "touching": "touching",
        "touch": "touching",
        "direct contact": "direct_contact",
        "contact": "direct_contact",
        "on": "on",
    }
    return aliases.get(text, text.replace(" ", "_"))


def clean_entity_name(raw_name: str) -> str:
    """Normalize object names for readable templates and equality checks."""

    name = str(raw_name).strip().lower()
    name = re.sub(r"[^a-z0-9\s_-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    for article in ("the ", "a ", "an "):
        if name.startswith(article):
            name = name[len(article) :].strip()
    return name


def comparable_entity_name(raw_name: str) -> str:
    """Return a compact object name used for duplicate-subject filtering."""

    name = clean_entity_name(raw_name)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    if name.endswith("ies") and len(name) > 4:
        name = name[:-3] + "y"
    elif name.endswith("s") and not name.endswith("ss") and len(name) > 3:
        name = name[:-1]
    return name


def looks_plural(noun: str) -> bool:
    """Small grammar helper for relation templates."""

    text = clean_entity_name(noun)
    if not text:
        return False
    head = text.split()[-1]
    if head in {"people", "men", "women", "children", "teeth", "feet"}:
        return True
    if head.endswith(("ss", "us")):
        return False
    return head.endswith("s")


def relation_be(noun: str) -> str:
    """Return ``is`` or ``are`` for simple relation facts/questions."""

    return "are" if looks_plural(noun) else "is"


def relation_bucket(relation: str) -> str:
    """Map a relation key to a coarse sampling bucket."""

    return REL_BUCKETS.get(str(relation), "semantic")


def relation_is_usable(subject: str, relation: str, obj: str) -> bool:
    """Filter vague or self-referential relation rows."""

    if normalize_relation(relation) in VAGUE_RELATIONS:
        return False
    if comparable_entity_name(subject) == comparable_entity_name(obj):
        return False
    return bool(clean_entity_name(subject) and clean_entity_name(obj))


def parse_amber_relation_query(question: str) -> tuple[str, str, str] | None:
    """Extract simple AMBER relation objects from yes/no relation questions."""

    text = str(question).strip()
    match = re.search(r"direct contact between the (.+?) and (.+?)\?", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), "direct_contact", match.group(2).strip()
    match = re.search(r"is the (.+?) (to the left of|to the right of|above|below|on|under|next to|near|behind|in front of|touching) the (.+?) in", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), normalize_relation(match.group(2)), match.group(3).strip()
    return None


def image_from_relation_row(row: Mapping[str, Any]) -> str:
    """Resolve an image filename/path field from a relation row."""

    image = string_field(row, "image", "file_name", "filename", "img", "image_path", "img_path")
    if image:
        return image
    image_id = string_field(row, "image_id", "img_id", "id")
    return image_id


def relation_scene_text(subject: str, relation: str, obj: str, label: str = "yes") -> str:
    """Render image-level trusted text for one relation."""

    phrase = RELATION_PHRASES.get(relation, relation.replace("_", " "))
    if label == "no":
        return f"The {subject} {relation_be(subject)} not {phrase} the {obj} in the image."
    return f"The {subject} {relation_be(subject)} {phrase} the {obj} in the image."


def absolutize_image_path(image: str, image_root: Path) -> str:
    """Make external relation images absolute so mixed roots work during extraction."""

    path = Path(str(image))
    if path.is_absolute():
        return str(path)
    return str(image_root / path)


def relation_question(subject: str, relation: str, obj: str, label: str = "yes") -> str:
    """Render a yes/no relation question."""

    phrase = RELATION_PHRASES.get(relation, relation.replace("_", " "))
    if relation == "direct_contact":
        return f"Is there direct contact between the {subject} and the {obj}?"
    aux = "Are" if looks_plural(subject) else "Is"
    return f"{aux} the {subject} {phrase} the {obj} in the image?"


def relation_bucket_quotas(max_pairs: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Convert bucket ratios to small per-image quotas."""

    max_pairs = max(1, int(max_pairs))
    exact = {bucket: max_pairs * float(ratios.get(bucket, 0.0)) for bucket in REL_BUCKET_ORDER}
    quotas = {bucket: int(math.floor(value)) for bucket, value in exact.items()}
    remaining = max_pairs - sum(quotas.values())
    order = sorted(REL_BUCKET_ORDER, key=lambda bucket: (exact[bucket] - quotas[bucket], ratios.get(bucket, 0.0)), reverse=True)
    for bucket in order[:remaining]:
        quotas[bucket] += 1
    return quotas


def select_balanced_relation_rows(
    rows: list[dict[str, Any]],
    max_pairs_per_image: int,
    bucket_ratios: Mapping[str, float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Select relation rows per image while preventing left/right domination."""

    max_pairs = max(1, int(max_pairs_per_image))

    by_bucket: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in REL_BUCKET_ORDER}
    for row in rows:
        bucket = relation_bucket(str(row.get("true_relation") or row.get("queried_relation") or ""))
        by_bucket.setdefault(bucket, []).append(row)
    for bucket_rows in by_bucket.values():
        rng.shuffle(bucket_rows)

    quotas = relation_bucket_quotas(max_pairs, bucket_ratios)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def take_from(bucket: str, count: int) -> None:
        if count <= 0:
            return
        for row in by_bucket.get(bucket, []):
            if len(selected) >= max_pairs:
                return
            row_id = str(row.get("id"))
            if row_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row_id)
            count -= 1
            if count <= 0:
                return

    for bucket in REL_BUCKET_ORDER:
        take_from(bucket, quotas.get(bucket, 0))

    # Fill remaining slots without letting horizontal rows exceed their quota.
    fill_order = tuple(bucket for bucket in REL_BUCKET_ORDER if bucket != "horizontal")
    while len(selected) < max_pairs:
        before = len(selected)
        for bucket in fill_order:
            take_from(bucket, 1)
            if len(selected) >= max_pairs:
                break
        if len(selected) == before:
            break
    return selected


def external_relation_rows(
    path: Path,
    max_pairs_per_image: int,
    skipped: Counter[str],
    *,
    rng: random.Random,
    desired_images: int,
    progress_every: int,
    negative_trusted_text: str,
) -> dict[str, list[dict[str, Any]]]:
    """Load external relation annotations into row groups keyed by image string."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    print(f"[relation-source] loading {path}", file=sys.stderr, flush=True)
    records = iter_records(path, rng=rng)
    print("[relation-source] parsing relation rows", file=sys.stderr, flush=True)
    for index, row in enumerate(iter_flatten_relation_records(records), start=1):
        if progress_every > 0 and index % progress_every == 0:
            print(
                f"[relation-source] parsed {index} relation rows, "
                f"valid_images={len(groups)}/{desired_images}",
                file=sys.stderr,
                flush=True,
            )
        question = string_field(row, "question", "query", "text")
        label = string_field(row, "label", "answer", "truth", "ground_truth").lower()
        if label in {"true", "1"}:
            label = "yes"
        elif label in {"false", "0"}:
            label = "no"
        elif label not in {"yes", "no"}:
            label = "yes"

        subject = string_field(row, "subject", "subj", "object_a", "entity1", "from")
        obj = string_field(row, "object", "obj", "object_b", "entity2", "to")
        relation = normalize_relation(string_field(row, "predicate", "relation", "rel", "true_relation"))
        if (not subject or not obj or not relation) and question:
            parsed = parse_amber_relation_query(question)
            if parsed is not None:
                subject, relation, obj = parsed
        subject = clean_entity_name(subject)
        obj = clean_entity_name(obj)
        relation = normalize_relation(relation)
        if not subject or not obj or not relation:
            skipped["external_rel_missing_fields"] += 1
            continue
        if not relation_is_usable(subject, relation, obj):
            skipped[f"external_rel_filtered_{relation_bucket(relation)}"] += 1
            continue

        image = image_from_relation_row(row)
        if not image:
            skipped["external_rel_missing_image"] += 1
            continue

        yes_text = relation_scene_text(subject, relation, obj, "yes")
        no_text = relation_scene_text(subject, relation, obj, "no")
        factual_text = yes_text if label == "yes" or negative_trusted_text == "positive_only" else no_text
        row_question = question or relation_question(subject, relation, obj, label)
        row_id_base = f"{CURRENT_SOURCE_NAME}_external_rel_{index}_{subject}_{relation}_{obj}".replace(" ", "_")
        rows = [
            make_row(
                row_id=row_id_base,
                image=image,
                image_id=string_field(row, "image_id", "img_id", "id") or image,
                question=row_question,
                trusted_factual_text=factual_text,
                hallucination_type="rel",
                subtype=REL_SUBTYPE.get(relation, "rel_semantic"),
                objects=[subject, obj],
                factual_fact=factual_text,
                label=label,
                extra={
                    "object_a": subject,
                    "object_b": obj,
                    "true_relation": relation,
                    "queried_relation": relation,
                    "relation_bucket": relation_bucket(relation),
                    "relation_source": str(path),
                },
            )
        ]

        opposite = OPPOSITE_RELATIONS.get(relation)
        if label == "yes" and opposite and max_pairs_per_image > 1:
            opposite_text = relation_scene_text(subject, opposite, obj, "no")
            trusted_negative_text = yes_text if negative_trusted_text == "positive_only" else f"{yes_text} {opposite_text}"
            rows.append(
                make_row(
                    row_id=f"{row_id_base}_opposite_no",
                    image=image,
                    image_id=string_field(row, "image_id", "img_id", "id") or image,
                    question=relation_question(subject, opposite, obj, "no"),
                    trusted_factual_text=trusted_negative_text,
                    hallucination_type="rel",
                    subtype=REL_SUBTYPE.get(opposite, "rel_semantic"),
                    objects=[subject, obj],
                    factual_fact=yes_text,
                    label="no",
                    extra={
                        "object_a": subject,
                        "object_b": obj,
                        "true_relation": relation,
                        "queried_relation": opposite,
                        "relation_bucket": relation_bucket(opposite),
                        "relation_source": str(path),
                    },
                )
            )
        groups[image].extend(rows[: max(1, int(max_pairs_per_image))])
        if desired_images > 0 and len(groups) >= desired_images:
            skipped["external_rel_stopped_after_target_images"] += 1
            break
    print(
        f"[relation-source] ready valid_images={len(groups)}, "
        f"parsed_rows={index if 'index' in locals() else 0}",
        file=sys.stderr,
        flush=True,
    )
    return groups


def coco_relation_rows(
    *,
    image: Mapping[str, Any],
    annotations: list[dict[str, Any]],
    categories_by_id: Mapping[int, str],
    max_pairs: int,
    rng: random.Random,
    skipped: Counter[str],
    template_variant: str,
) -> list[dict[str, Any]]:
    """Build relation rows from COCO bbox fallback, retagging as v2."""

    width = float(image.get("width", 0.0) or 0.0)
    height = float(image.get("height", 0.0) or 0.0)
    clean = RELATION_V2.clean_annotations(annotations, categories_by_id, width, height, skipped)
    if len(clean) < 2:
        skipped["rel_fallback_no_valid_objects"] += 1
        return []
    rows = RELATION_V2.build_pairs_for_image(
        image=image,
        annotations=clean,
        max_pairs_per_image=max_pairs,
        rng=rng,
        skipped=skipped,
        template_variant=template_variant,
    )
    retagged: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        old_id = str(item.get("id") or item.get("pair_id"))
        item["id"] = f"{CURRENT_SOURCE_NAME}_coco_bbox_{old_id}"
        item["pair_id"] = item["id"]
        item["source"] = CURRENT_SOURCE_NAME
        item["relation_source"] = "coco_bbox_fallback"
        item["relation_bucket"] = relation_bucket(str(item.get("true_relation") or item.get("queried_relation") or ""))
        item["prompt_style"] = "after_fas_complete_scene_v2"
        retagged.append(item)
    return retagged


def split_type_images(image_ids_by_type: Mapping[str, list[str]], ratios: tuple[float, float, float]) -> dict[str, set[str]]:
    """Split image keys per type, then merge split sets."""

    merged = {"train": set(), "val": set(), "test": set()}
    for image_keys in image_ids_by_type.values():
        keys = list(image_keys)
        train_end = int(round(len(keys) * ratios[0]))
        val_end = train_end + int(round(len(keys) * ratios[1]))
        split_ids = {
            "train": set(keys[:train_end]),
            "val": set(keys[train_end:val_end]),
            "test": set(keys[val_end:]),
        }
        for split, ids in split_ids.items():
            merged[split].update(ids)
    return merged


def summarize(
    *,
    rows_by_split: Mapping[str, list[dict[str, Any]]],
    image_ids_by_type: Mapping[str, list[str]],
    target_counts: Mapping[str, int],
    skipped: Counter[str],
    relation_source_path: Path | None,
    relation_fallback_used: int,
) -> dict[str, Any]:
    """Summarize v2 rows."""

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    image_sets = {expert: set(ids) for expert, ids in image_ids_by_type.items()}
    overlaps = {
        f"{left}_{right}": len(image_sets[left] & image_sets[right])
        for index, left in enumerate(EXPERT_TYPES)
        for right in EXPERT_TYPES[index + 1 :]
    }
    return {
        "total_pairs": len(all_rows),
        "train_pairs": len(rows_by_split.get("train", [])),
        "val_pairs": len(rows_by_split.get("val", [])),
        "test_pairs": len(rows_by_split.get("test", [])),
        "num_images": len({str(row["image_id"]) for row in all_rows}),
        "target_type_image_counts": dict(target_counts),
        "actual_type_image_counts": {expert: len(ids) for expert, ids in image_ids_by_type.items()},
        "cross_type_image_overlap": overlaps,
        "type_counts": dict(Counter(str(row["hallucination_type"]) for row in all_rows)),
        "subtype_counts": dict(Counter(str(row["subtype"]) for row in all_rows)),
        "label_counts": dict(Counter(str(row.get("label") or "unknown") for row in all_rows)),
        "label_counts_by_type": {
            expert: dict(Counter(str(row.get("label") or "unknown") for row in all_rows if str(row["hallucination_type"]) == expert))
            for expert in EXPERT_TYPES
        },
        "relation_counts": dict(Counter(str(row.get("true_relation", "")) for row in all_rows if str(row["hallucination_type"]) == "rel")),
        "queried_relation_counts": dict(Counter(str(row.get("queried_relation", "")) for row in all_rows if str(row["hallucination_type"]) == "rel")),
        "relation_bucket_counts": dict(
            Counter(str(row.get("relation_bucket") or relation_bucket(str(row.get("true_relation", "")))) for row in all_rows if str(row["hallucination_type"]) == "rel")
        ),
        "relation_source": str(relation_source_path) if relation_source_path else "",
        "relation_source_counts": dict(
            Counter(str(row.get("relation_source") or "unknown") for row in all_rows if str(row["hallucination_type"]) == "rel")
        ),
        "relation_fallback_used_images": int(relation_fallback_used),
        "skipped": dict(skipped),
        "prompt_style": "after_fas_complete_scene_v2",
    }


def main() -> int:
    """Build v2 pair splits."""

    global CURRENT_SOURCE_NAME
    args = parse_args()
    try:
        CURRENT_SOURCE_NAME = str(args.source_name)
        output_dir = resolve_project_path(args.output_dir)
        output_paths = {split: output_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
        stats_path = output_dir / "stats.json"
        assignments_path = output_dir / "image_assignments.json"
        if not args.overwrite:
            existing = [path for path in [*output_paths.values(), stats_path, assignments_path] if path.exists()]
            if existing:
                raise FileExistsError(f"Output already exists: {existing[0]}. Pass --overwrite to replace.")

        rng = random.Random(int(args.seed))
        split_ratio = STYLE.parse_split_ratio(args.split_ratio)
        type_ratios = parse_type_ratio(args.type_image_ratio)
        relation_bucket_ratios = parse_named_ratio(
            args.relation_bucket_ratio,
            REL_BUCKET_ORDER,
            name="--relation-bucket-ratio",
        )
        target_counts = allocation_counts(int(args.num_images), type_ratios)
        image_root = resolve_project_path(args.image_root)
        relation_source_path = optional_path(args.relation_source)
        relation_image_root = optional_path(args.relation_image_root) or image_root
        images_by_id, categories_by_id, annotations_by_image = load_coco_instances(resolve_project_path(args.coco_instances))
        all_categories = sorted(categories_by_id.values())
        skipped: Counter[str] = Counter()

        external_rel_groups: dict[str, list[dict[str, Any]]] = {}
        if relation_source_path is not None:
            external_rel_groups = external_relation_rows(
                relation_source_path,
                int(args.max_rel_pairs_per_image),
                skipped,
                rng=rng,
                desired_images=target_counts["rel"],
                progress_every=int(args.progress_every),
                negative_trusted_text=str(args.negative_relation_trusted_text),
            )

        def annotations_for(image_id: int) -> list[dict[str, Any]]:
            return valid_annotations_for(image_id, annotations_by_image, categories_by_id)

        builders: dict[str, Callable[[str], list[dict[str, Any]]]] = {}

        def cat_builder(image_key: str) -> list[dict[str, Any]]:
            image_id = int(image_key)
            annotations = annotations_for(image_id)
            if not annotations:
                skipped["cat_no_valid_objects"] += 1
                return []
            return build_cat_rows(
                image=images_by_id[image_id],
                annotations=annotations,
                all_categories=all_categories,
                max_pairs=int(args.max_cat_pairs_per_image),
                rng=rng,
                skipped=skipped,
            )

        def attr_builder(image_key: str) -> list[dict[str, Any]]:
            image_id = int(image_key)
            annotations = annotations_for(image_id)
            if not annotations:
                skipped["attr_no_valid_objects"] += 1
                return []
            return build_attr_rows(
                image=images_by_id[image_id],
                annotations=annotations,
                image_root=image_root,
                max_pairs=int(args.max_attr_pairs_per_image),
                rng=rng,
                skipped=skipped,
            )

        relation_fallback_used = 0

        def rel_builder(image_key: str) -> list[dict[str, Any]]:
            nonlocal relation_fallback_used
            if image_key in external_rel_groups:
                return select_balanced_relation_rows(
                    external_rel_groups[image_key],
                    int(args.max_rel_pairs_per_image),
                    relation_bucket_ratios,
                    rng,
                )
            if args.relation_fallback == "none":
                skipped["rel_no_external_rows"] += 1
                return []
            try:
                image_id = int(image_key)
            except ValueError:
                skipped["rel_external_key_not_coco_fallback"] += 1
                return []
            if image_id not in images_by_id:
                skipped["rel_fallback_image_not_in_coco"] += 1
                return []
            rows = coco_relation_rows(
                image=images_by_id[image_id],
                annotations=annotations_by_image.get(image_id, []),
                categories_by_id=categories_by_id,
                max_pairs=int(args.max_rel_pairs_per_image),
                rng=rng,
                skipped=skipped,
                template_variant=str(args.rel_template_variant),
            )
            if rows:
                relation_fallback_used += 1
            return rows

        builders.update({"cat": cat_builder, "attr": attr_builder, "rel": rel_builder})

        coco_candidate_keys = [str(image_id) for image_id in images_by_id]
        rng.shuffle(coco_candidate_keys)
        rel_candidate_keys = list(external_rel_groups) if external_rel_groups else list(coco_candidate_keys)
        rng.shuffle(rel_candidate_keys)
        candidate_keys = {"cat": coco_candidate_keys, "attr": coco_candidate_keys, "rel": rel_candidate_keys}

        assigned_keys: set[str] = set()
        rows_by_image_key: dict[str, list[dict[str, Any]]] = {}
        image_ids_by_type: dict[str, list[str]] = {expert: [] for expert in EXPERT_TYPES}

        for expert in ("rel", "attr", "cat"):
            for image_key in candidate_keys[expert]:
                if image_key in assigned_keys:
                    continue
                if len(image_ids_by_type[expert]) >= target_counts[expert]:
                    break
                rows = builders[expert](image_key)
                if not rows:
                    continue
                if expert == "rel" and external_rel_groups and image_key in external_rel_groups:
                    for row in rows:
                        row["image"] = absolutize_image_path(str(row.get("image", "")), relation_image_root)
                        row.setdefault("image_root_hint", str(relation_image_root))
                rows_by_image_key[image_key] = rows
                image_ids_by_type[expert].append(image_key)
                assigned_keys.add(image_key)

        selected_keys = [image_key for expert in EXPERT_TYPES for image_key in image_ids_by_type[expert]]
        split_ids = split_type_images(image_ids_by_type, split_ratio)
        rows_by_split = {
            split: [row for image_key in selected_keys if image_key in image_keys for row in rows_by_image_key[image_key]]
            for split, image_keys in split_ids.items()
        }

        for split, rows in rows_by_split.items():
            write_jsonl(output_paths[split], rows)
        stats = {
            "source": CURRENT_SOURCE_NAME,
            "coco_instances": str(resolve_project_path(args.coco_instances)),
            "image_root": str(image_root),
            "relation_image_root": str(relation_image_root),
            "num_requested_images": int(args.num_images),
            "num_selected_images": len(selected_keys),
            "type_image_ratio": type_ratios,
            "split_ratio": list(split_ratio),
            "max_pairs_per_image": {
                "cat": int(args.max_cat_pairs_per_image),
                "attr": int(args.max_attr_pairs_per_image),
                "rel": int(args.max_rel_pairs_per_image),
            },
            "relation_fallback": str(args.relation_fallback),
            "relation_bucket_ratio": relation_bucket_ratios,
            "rel_template_variant": str(args.rel_template_variant),
            "negative_relation_trusted_text": str(args.negative_relation_trusted_text),
            "seed": int(args.seed),
            "outputs": {split: str(path) for split, path in output_paths.items()},
            **summarize(
                rows_by_split=rows_by_split,
                image_ids_by_type=image_ids_by_type,
                target_counts=target_counts,
                skipped=skipped,
                relation_source_path=relation_source_path,
                relation_fallback_used=relation_fallback_used,
            ),
        }
        write_json(stats_path, stats)
        write_json(assignments_path, {expert: image_ids_by_type[expert] for expert in EXPERT_TYPES})
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote image-disjoint AFTER-template v2 pairs to {output_dir}")
    print(
        "Summary: "
        f"total_pairs={stats['total_pairs']}, "
        f"selected_images={stats['num_selected_images']}, "
        f"type_images={stats['actual_type_image_counts']}, "
        f"relation_fallback_used_images={stats['relation_fallback_used_images']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
