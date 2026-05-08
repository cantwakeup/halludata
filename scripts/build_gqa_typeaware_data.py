"""Build GQA type-aware diagnostic data for cat/attr/rel experts.

The output rows intentionally keep two compatible schemas:

- GQA diagnostic fields: type, trusted_text, answer, subject/object/relation.
- Existing AFTER-template fields: hallucination_type, visual_prompt,
  trusted_factual_text, trusted_prompt, label.

This lets the rows feed the current activation extraction and benchmark runners
without adding another model-inference framework.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import write_json, write_jsonl

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional.
    def tqdm(items: Iterable[Any], **_: Any) -> Iterable[Any]:
        return items


SOURCE_PREFIX = "gqa_scene_graph"
NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
}

BAD_OBJECT_NAMES = {
    "object",
    "objects",
    "thing",
    "things",
    "stuff",
    "area",
    "part",
    "item",
    "items",
    "something",
    "anything",
    "other",
    "others",
    "unknown",
    "none",
}

BAD_ATTRIBUTES = {
    "very",
    "other",
    "same",
    "own",
    "possible",
    "visible",
    "unknown",
    "different",
    "many",
    "some",
    "several",
    "few",
}

COLOR_ATTRIBUTES = {
    "black",
    "blue",
    "brown",
    "gray",
    "grey",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "tan",
    "white",
    "yellow",
    "silver",
    "gold",
    "golden",
}

SIZE_ATTRIBUTES = {
    "big",
    "small",
    "large",
    "little",
    "tiny",
    "tall",
    "short",
    "long",
    "wide",
    "narrow",
}

MATERIAL_ATTRIBUTES = {
    "wooden",
    "wood",
    "metal",
    "metallic",
    "plastic",
    "glass",
    "brick",
    "concrete",
    "leather",
    "cloth",
    "paper",
    "stone",
}

STATE_ATTRIBUTES = {
    "open",
    "closed",
    "full",
    "empty",
    "clean",
    "dirty",
    "wet",
    "dry",
    "standing",
    "sitting",
    "lying",
    "parked",
    "cut",
    "broken",
}

VAGUE_RELATIONS = {
    "of",
    "with",
    "by",
    "at",
    "from",
    "for",
    "has",
    "have",
    "had",
    "belonging to",
}

RELATION_ALIASES = {
    "to the left of": "left of",
    "left of": "left of",
    "to the right of": "right of",
    "right of": "right of",
    "in the front of": "in front of",
    "in front of": "in front of",
    "in back of": "behind",
    "at the back of": "behind",
    "on top of": "on",
    "inside of": "inside",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gqa-root", default="/home/huiwei/sy/sy_data/GQA")
    parser.add_argument("--split", default="train", help="GQA split, usually train or val.")
    parser.add_argument("--out-root", default="data/gqa_typeaware_v1")
    parser.add_argument("--max-cat", type=int, default=3000)
    parser.add_argument("--max-attr", type=int, default=3000)
    parser.add_argument("--max-rel", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--attr-count-fraction",
        type=float,
        default=0.25,
        help="Approximate fraction of attr examples reserved for scene-graph count questions.",
    )
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def clean_phrase(value: Any) -> str:
    text = normalize_spaces(str(value).replace("_", " ").lower())
    text = re.sub(r"[^a-z0-9\-\s]", "", text)
    return normalize_spaces(text)


def is_clean_object_name(name: str) -> bool:
    if not name or name in BAD_OBJECT_NAMES:
        return False
    if len(name) > 40:
        return False
    if len(name.split()) > 5:
        return False
    if not re.search(r"[a-z]", name):
        return False
    return True


def is_clean_attribute(attr: str) -> bool:
    if not attr or attr in BAD_ATTRIBUTES:
        return False
    if len(attr) > 30 or len(attr.split()) > 3:
        return False
    if not re.search(r"[a-z]", attr):
        return False
    return True


def normalize_relation(value: Any) -> str:
    relation = clean_phrase(value)
    relation = RELATION_ALIASES.get(relation, relation)
    return relation


def is_clean_relation(relation: str) -> bool:
    if not relation or relation in VAGUE_RELATIONS:
        return False
    if len(relation) > 45 or len(relation.split()) > 5:
        return False
    if not re.search(r"[a-z]", relation):
        return False
    return True


def relation_display(relation: str) -> str:
    if relation == "left of":
        return "to the left of"
    if relation == "right of":
        return "to the right of"
    return relation


def relation_bucket(relation: str) -> str:
    relation = normalize_relation(relation)
    if "left" in relation or "right" in relation:
        return "horizontal"
    if relation in {"above", "below", "under", "over", "beneath"}:
        return "vertical"
    if "front of" in relation or relation == "behind":
        return "depth"
    if relation in {
        "on",
        "in",
        "inside",
        "attached to",
        "hanging from",
        "covering",
        "covered by",
        "sitting on",
        "standing on",
    }:
        return "contact"
    if any(token in relation for token in ("holding", "wearing", "riding", "carrying", "eating", "drinking")):
        return "interaction"
    if "playing" in relation or "looking at" in relation:
        return "interaction"
    return "semantic"


def attribute_bucket(attr: str) -> str:
    if attr in COLOR_ATTRIBUTES:
        return "color"
    if attr in SIZE_ATTRIBUTES:
        return "size"
    if attr in MATERIAL_ATTRIBUTES:
        return "material"
    if attr in STATE_ATTRIBUTES:
        return "state"
    return "general"


def pluralize(noun: str, count: int) -> str:
    if count == 1:
        return noun
    if noun.endswith("s"):
        return noun
    return f"{noun}s"


def resolve_gqa_file(root: Path, split: str, kind: str, *, required: bool = True) -> Path | None:
    split = split.lower()
    checked: list[Path] = []
    if kind == "scene_graph":
        exact = [
            root / "raw" / "sceneGraphs" / f"{split}_sceneGraphs.json",
            root / "raw" / "sceneGraphs" / f"{split}_scene_graphs.json",
            root / "sceneGraphs" / f"{split}_sceneGraphs.json",
            root / f"{split}_sceneGraphs.json",
            root / "raw" / "sceneGraphs" / "scene_graphs.json",
            root / "raw" / "sceneGraphs" / "sceneGraphs.json",
        ]
        search_dirs = [root / "raw" / "sceneGraphs", root / "sceneGraphs", root]
        name_needles = ("scenegraph", "scene_graph")
    elif kind == "questions":
        exact = [
            root / "raw" / "questions" / f"{split}_all_questions.json",
            root / "raw" / "questions" / f"{split}_balanced_questions.json",
            root / "questions" / f"{split}_all_questions.json",
            root / "questions" / f"{split}_balanced_questions.json",
        ]
        search_dirs = [root / "raw" / "questions", root / "questions", root]
        name_needles = ("question",)
    else:
        raise ValueError(f"Unknown GQA file kind: {kind}")

    for path in exact:
        checked.append(path)
        if path.exists():
            return path

    candidates: list[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*.json"):
            lower = str(path).lower()
            if any(needle in lower for needle in name_needles) and split in path.name.lower():
                candidates.append(path)
    if candidates:
        return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]

    if not required:
        return None
    found = sorted({str(path) for directory in search_dirs if directory.exists() for path in directory.rglob("*.json")})
    checked_text = "\n".join(f"  - {path}" for path in checked)
    found_text = "\n".join(f"  - {path}" for path in found[:80]) or "  - <none>"
    raise FileNotFoundError(
        f"Could not find GQA {kind} JSON for split={split} under {root}.\n"
        f"Checked preferred paths:\n{checked_text}\n"
        f"JSON files discovered:\n{found_text}"
    )


def discover_image_roots(root: Path) -> list[Path]:
    candidates = [
        root / "raw" / "images" / "images",
        root / "raw" / "images",
        root / "images",
        root / "image",
        root / "sample" / "image",
    ]
    roots: list[Path] = []
    for path in candidates:
        if path.exists() and path.is_dir():
            roots.append(path)
    if not roots:
        raise FileNotFoundError(
            f"Could not find a GQA image directory under {root}. "
            "Expected raw/images/images, raw/images, images, image, or sample/image."
        )
    return roots


def resolve_image_path(image_id: str, image_roots: list[Path]) -> Path | None:
    for root in image_roots:
        for suffix in (".jpg", ".jpeg", ".png"):
            path = root / f"{image_id}{suffix}"
            if path.exists():
                return path.resolve()
    return None


def load_gqa_scene_graphs(scene_graph_path: Path) -> dict[str, Any]:
    with scene_graph_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict of image_id -> scene graph in {scene_graph_path}")
    return payload


def load_optional_questions(question_path: Path | None) -> dict[str, Any] | list[Any] | None:
    if question_path is None:
        return None
    with question_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_records(scene_graphs: Mapping[str, Any], image_roots: list[Path]) -> tuple[list[dict[str, Any]], Counter]:
    skipped: Counter = Counter()
    records: list[dict[str, Any]] = []
    for image_id, scene in tqdm(scene_graphs.items(), desc="prepare scene graphs"):
        image_path = resolve_image_path(str(image_id), image_roots)
        if image_path is None:
            skipped["missing_image"] += 1
            continue
        raw_objects = scene.get("objects", {}) if isinstance(scene, Mapping) else {}
        objects: dict[str, dict[str, Any]] = {}
        for object_id, raw_object in raw_objects.items():
            if not isinstance(raw_object, Mapping):
                skipped["bad_object_record"] += 1
                continue
            name = clean_phrase(raw_object.get("name", ""))
            if not is_clean_object_name(name):
                skipped["bad_object_name"] += 1
                continue
            attrs = [
                clean_phrase(attr)
                for attr in raw_object.get("attributes", [])
                if is_clean_attribute(clean_phrase(attr))
            ]
            relations: list[dict[str, str]] = []
            for relation in raw_object.get("relations", []):
                if not isinstance(relation, Mapping):
                    continue
                target_id = str(relation.get("object", ""))
                rel = normalize_relation(relation.get("name", ""))
                if not target_id or not is_clean_relation(rel):
                    skipped["bad_relation"] += 1
                    continue
                relations.append({"object_id": target_id, "relation": rel})
            objects[str(object_id)] = {
                "object_id": str(object_id),
                "name": name,
                "attributes": sorted(set(attrs)),
                "relations": relations,
            }
        if not objects:
            skipped["no_clean_objects"] += 1
            continue
        records.append(
            {
                "image_id": str(image_id),
                "image_path": str(image_path),
                "objects": objects,
                "width": scene.get("width") if isinstance(scene, Mapping) else None,
                "height": scene.get("height") if isinstance(scene, Mapping) else None,
            }
        )
    return records, skipped


def build_object_vocab(records: list[dict[str, Any]], min_count: int = 2) -> list[str]:
    counts = Counter(obj["name"] for record in records for obj in record["objects"].values())
    return sorted(name for name, count in counts.items() if count >= min_count and is_clean_object_name(name))


def build_attribute_vocab(records: list[dict[str, Any]], min_count: int = 2) -> list[str]:
    counts = Counter(attr for record in records for obj in record["objects"].values() for attr in obj["attributes"])
    return sorted(attr for attr, count in counts.items() if count >= min_count and is_clean_attribute(attr))


def build_relation_vocab(records: list[dict[str, Any]], min_count: int = 2) -> list[str]:
    counts: Counter = Counter()
    for record in records:
        objects = record["objects"]
        for obj in objects.values():
            for rel in obj["relations"]:
                target = objects.get(rel["object_id"])
                if target and target["name"] != obj["name"] and is_clean_relation(rel["relation"]):
                    counts[rel["relation"]] += 1
    return sorted(rel for rel, count in counts.items() if count >= min_count and is_clean_relation(rel))


def visual_prompt(question: str) -> str:
    return f"Question: {question}\nPlease answer the question based on the image."


def trusted_prompt(trusted_text: str, question: str) -> str:
    return (
        f"The given image depicts the following scene: {trusted_text}\n"
        "Please directly answer the following question from the image description, "
        f"without guessing or reasoning. Question: {question}"
    )


def make_row(
    *,
    row_id: str,
    record: Mapping[str, Any],
    expert_type: str,
    subtype: str,
    question: str,
    answer: str,
    trusted_text: str,
    subject: str,
    obj: str = "",
    attribute: str = "",
    relation: str = "",
    source: str,
) -> dict[str, Any]:
    if answer not in {"yes", "no"}:
        raise ValueError(f"Invalid yes/no answer for {row_id}: {answer}")
    if not question or not trusted_text:
        raise ValueError(f"Empty question/trusted_text for {row_id}")
    bucket = relation_bucket(relation) if expert_type == "rel" and relation else ""
    row = {
        "id": row_id,
        "image_id": str(record["image_id"]),
        "image_path": str(record["image_path"]),
        "image": str(record["image_path"]),
        "type": expert_type,
        "hallucination_type": expert_type,
        "subtype": subtype,
        "question": question,
        "answer": answer,
        "label": answer,
        "trusted_text": trusted_text,
        "trusted_factual_text": trusted_text,
        "visual_prompt": visual_prompt(question),
        "trusted_prompt": trusted_prompt(trusted_text, question),
        "subject": subject,
        "object": obj,
        "attribute": attribute,
        "relation": relation,
        "objects": [value for value in (subject, obj) if value],
        "source": source,
        "template_variant": "gqa_typeaware_v1",
    }
    if expert_type == "rel":
        row.update(
            {
                "object_a": subject,
                "object_b": obj,
                "true_relation": relation,
                "queried_relation": relation,
                "relation_bucket": bucket,
            }
        )
    return row


def balanced_targets(max_examples: int) -> tuple[int, int]:
    yes_target = int(max_examples) // 2
    no_target = int(max_examples) - yes_target
    return yes_target, no_target


def build_cat_examples(
    records: list[dict[str, Any]],
    object_vocab: list[str],
    max_examples: int,
    rng: random.Random,
    source: str,
) -> tuple[list[dict[str, Any]], Counter]:
    skipped: Counter = Counter()
    yes_target, no_target = balanced_targets(max_examples)
    yes_rows: list[dict[str, Any]] = []
    no_rows: list[dict[str, Any]] = []
    shuffled = list(records)
    rng.shuffle(shuffled)
    for record in tqdm(shuffled, desc="build cat"):
        present = sorted({obj["name"] for obj in record["objects"].values()})
        absent = [name for name in object_vocab if name not in present]
        rng.shuffle(present)
        if len(yes_rows) < yes_target and present:
            subject = present[0]
            question = f"Is there a {subject} in the image?"
            trusted = f"There is a {subject} in the image."
            yes_rows.append(
                make_row(
                    row_id=f"gqa_{record['image_id']}_cat_{subject.replace(' ', '_')}_yes",
                    record=record,
                    expert_type="cat",
                    subtype="cat_present",
                    question=question,
                    answer="yes",
                    trusted_text=trusted,
                    subject=subject,
                    source=source,
                )
            )
        if len(no_rows) < no_target and absent:
            subject = rng.choice(absent)
            question = f"Is there a {subject} in the image?"
            trusted = f"There is no {subject} in the image."
            no_rows.append(
                make_row(
                    row_id=f"gqa_{record['image_id']}_cat_{subject.replace(' ', '_')}_no",
                    record=record,
                    expert_type="cat",
                    subtype="cat_absent",
                    question=question,
                    answer="no",
                    trusted_text=trusted,
                    subject=subject,
                    source=source,
                )
            )
        if len(yes_rows) >= yes_target and len(no_rows) >= no_target:
            break
    if len(yes_rows) < yes_target:
        skipped["cat_yes_under_target"] = yes_target - len(yes_rows)
    if len(no_rows) < no_target:
        skipped["cat_no_under_target"] = no_target - len(no_rows)
    rows = yes_rows + no_rows
    rng.shuffle(rows)
    return rows[:max_examples], skipped


def choose_wrong_attribute(
    attrs_for_object: set[str],
    positive_attr: str,
    attribute_vocab: list[str],
    rng: random.Random,
) -> str | None:
    bucket = attribute_bucket(positive_attr)
    same_bucket = [
        attr for attr in attribute_vocab
        if attr not in attrs_for_object and attr != positive_attr and attribute_bucket(attr) == bucket
    ]
    candidates = same_bucket or [attr for attr in attribute_vocab if attr not in attrs_for_object and attr != positive_attr]
    return rng.choice(candidates) if candidates else None


def build_count_row(
    record: Mapping[str, Any],
    object_name: str,
    true_count: int,
    answer: str,
    rng: random.Random,
    source: str,
) -> dict[str, Any] | None:
    if true_count < 1 or true_count > 6:
        return None
    if answer == "yes":
        query_count = true_count
        trusted = f"There are {true_count} {pluralize(object_name, true_count)} in the image."
    else:
        wrong_counts = [value for value in range(1, 7) if value != true_count]
        query_count = rng.choice(wrong_counts)
        trusted = (
            f"There are {true_count} {pluralize(object_name, true_count)} in the image, "
            f"not {query_count}."
        )
    question = f"Are there {NUMBER_WORDS[query_count]} {pluralize(object_name, query_count)} in the image?"
    return make_row(
        row_id=f"gqa_{record['image_id']}_attr_count_{object_name.replace(' ', '_')}_{query_count}_{answer}",
        record=record,
        expert_type="attr",
        subtype="attr_count",
        question=question,
        answer=answer,
        trusted_text=trusted,
        subject=object_name,
        attribute=f"count={query_count}",
        source=source,
    )


def build_attr_examples(
    records: list[dict[str, Any]],
    attribute_vocab: list[str],
    max_examples: int,
    rng: random.Random,
    source: str,
    attr_count_fraction: float,
) -> tuple[list[dict[str, Any]], Counter]:
    skipped: Counter = Counter()
    yes_target, no_target = balanced_targets(max_examples)
    count_yes_target = int(yes_target * max(0.0, min(1.0, attr_count_fraction)))
    count_no_target = int(no_target * max(0.0, min(1.0, attr_count_fraction)))
    attr_yes_target = yes_target - count_yes_target
    attr_no_target = no_target - count_no_target
    yes_rows: list[dict[str, Any]] = []
    no_rows: list[dict[str, Any]] = []
    count_yes_rows: list[dict[str, Any]] = []
    count_no_rows: list[dict[str, Any]] = []
    shuffled = list(records)
    rng.shuffle(shuffled)
    for record in tqdm(shuffled, desc="build attr"):
        objects = list(record["objects"].values())
        rng.shuffle(objects)
        for obj in objects:
            attrs = sorted(set(obj["attributes"]))
            if not attrs:
                continue
            positive_attr = rng.choice(attrs)
            wrong_attr = choose_wrong_attribute(set(attrs), positive_attr, attribute_vocab, rng)
            subject = obj["name"]
            if len(yes_rows) < attr_yes_target:
                question = f"Is the {subject} {positive_attr} in the image?"
                trusted = f"The {subject} is {positive_attr} in the image."
                yes_rows.append(
                    make_row(
                        row_id=f"gqa_{record['image_id']}_attr_{subject.replace(' ', '_')}_{positive_attr.replace(' ', '_')}_yes",
                        record=record,
                        expert_type="attr",
                        subtype=f"attr_{attribute_bucket(positive_attr)}",
                        question=question,
                        answer="yes",
                        trusted_text=trusted,
                        subject=subject,
                        attribute=positive_attr,
                        source=source,
                    )
                )
            if len(no_rows) < attr_no_target and wrong_attr:
                question = f"Is the {subject} {wrong_attr} in the image?"
                trusted = f"The {subject} is not {wrong_attr} in the image."
                no_rows.append(
                    make_row(
                        row_id=f"gqa_{record['image_id']}_attr_{subject.replace(' ', '_')}_{wrong_attr.replace(' ', '_')}_no",
                        record=record,
                        expert_type="attr",
                        subtype=f"attr_{attribute_bucket(wrong_attr)}",
                        question=question,
                        answer="no",
                        trusted_text=trusted,
                        subject=subject,
                        attribute=wrong_attr,
                        source=source,
                    )
                )
            if len(yes_rows) >= attr_yes_target and len(no_rows) >= attr_no_target:
                break

        name_counts = Counter(obj["name"] for obj in record["objects"].values())
        count_candidates = [name for name, count in name_counts.items() if 1 <= count <= 6]
        rng.shuffle(count_candidates)
        if count_candidates:
            name = count_candidates[0]
            true_count = name_counts[name]
            if len(count_yes_rows) < count_yes_target:
                row = build_count_row(record, name, true_count, "yes", rng, source)
                if row:
                    count_yes_rows.append(row)
            if len(count_no_rows) < count_no_target:
                row = build_count_row(record, name, true_count, "no", rng, source)
                if row:
                    count_no_rows.append(row)

        if (
            len(yes_rows) >= attr_yes_target
            and len(no_rows) >= attr_no_target
            and len(count_yes_rows) >= count_yes_target
            and len(count_no_rows) >= count_no_target
        ):
            break

    rows = yes_rows + no_rows + count_yes_rows + count_no_rows
    if len(rows) < max_examples:
        skipped["attr_under_target"] = max_examples - len(rows)
    rng.shuffle(rows)
    return rows[:max_examples], skipped


def image_relation_triples(record: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    objects = record["objects"]
    for subject in objects.values():
        for rel in subject["relations"]:
            target = objects.get(rel["object_id"])
            if not target:
                continue
            subject_name = subject["name"]
            object_name = target["name"]
            relation = rel["relation"]
            if subject_name == object_name:
                continue
            if not is_clean_relation(relation):
                continue
            triples.append((subject_name, relation, object_name))
    return triples


def choose_negative_relation(
    subject: str,
    relation: str,
    obj: str,
    image_objects: list[str],
    existing: set[tuple[str, str, str]],
    relation_vocab: list[str],
    rng: random.Random,
) -> tuple[str, str] | None:
    bucket = relation_bucket(relation)
    same_bucket = [
        rel for rel in relation_vocab
        if rel != relation and relation_bucket(rel) == bucket and (subject, rel, obj) not in existing
    ]
    relation_candidates = same_bucket or [
        rel for rel in relation_vocab
        if rel != relation and (subject, rel, obj) not in existing
    ]
    if relation_candidates:
        return rng.choice(relation_candidates), obj
    object_candidates = [
        candidate for candidate in image_objects
        if candidate not in {subject, obj} and (subject, relation, candidate) not in existing
    ]
    if object_candidates:
        return relation, rng.choice(object_candidates)
    return None


def build_rel_examples(
    records: list[dict[str, Any]],
    relation_vocab: list[str],
    max_examples: int,
    rng: random.Random,
    source: str,
) -> tuple[list[dict[str, Any]], Counter]:
    skipped: Counter = Counter()
    yes_target, no_target = balanced_targets(max_examples)
    yes_rows: list[dict[str, Any]] = []
    no_rows: list[dict[str, Any]] = []
    shuffled = list(records)
    rng.shuffle(shuffled)
    for record in tqdm(shuffled, desc="build rel"):
        triples = image_relation_triples(record)
        if not triples:
            skipped["rel_no_clean_triples"] += 1
            continue
        rng.shuffle(triples)
        existing = set(triples)
        image_objects = sorted({obj["name"] for obj in record["objects"].values()})
        for subject, relation, obj in triples:
            rel_text = relation_display(relation)
            bucket = relation_bucket(relation)
            if len(yes_rows) < yes_target:
                question = f"Is the {subject} {rel_text} the {obj} in the image?"
                trusted = f"The {subject} is {rel_text} the {obj} in the image."
                yes_rows.append(
                    make_row(
                        row_id=f"gqa_{record['image_id']}_rel_{subject.replace(' ', '_')}_{relation.replace(' ', '_')}_{obj.replace(' ', '_')}_yes",
                        record=record,
                        expert_type="rel",
                        subtype=f"rel_{bucket}",
                        question=question,
                        answer="yes",
                        trusted_text=trusted,
                        subject=subject,
                        obj=obj,
                        relation=relation,
                        source=source,
                    )
                )
            if len(no_rows) < no_target:
                negative = choose_negative_relation(subject, relation, obj, image_objects, existing, relation_vocab, rng)
                if negative is None:
                    skipped["rel_no_negative_candidate"] += 1
                    continue
                wrong_relation, wrong_object = negative
                wrong_text = relation_display(wrong_relation)
                wrong_bucket = relation_bucket(wrong_relation)
                question = f"Is the {subject} {wrong_text} the {wrong_object} in the image?"
                trusted = f"The {subject} is not {wrong_text} the {wrong_object} in the image."
                no_rows.append(
                    make_row(
                        row_id=f"gqa_{record['image_id']}_rel_{subject.replace(' ', '_')}_{wrong_relation.replace(' ', '_')}_{wrong_object.replace(' ', '_')}_no",
                        record=record,
                        expert_type="rel",
                        subtype=f"rel_{wrong_bucket}",
                        question=question,
                        answer="no",
                        trusted_text=trusted,
                        subject=subject,
                        obj=wrong_object,
                        relation=wrong_relation,
                        source=source,
                    )
                )
            if len(yes_rows) >= yes_target and len(no_rows) >= no_target:
                break
        if len(yes_rows) >= yes_target and len(no_rows) >= no_target:
            break
    if len(yes_rows) < yes_target:
        skipped["rel_yes_under_target"] = yes_target - len(yes_rows)
    if len(no_rows) < no_target:
        skipped["rel_no_under_target"] = no_target - len(no_rows)
    rows = yes_rows + no_rows
    rng.shuffle(rows)
    return rows[:max_examples], skipped


def split_output_paths(out_root: Path, split: str) -> dict[str, Path]:
    split = split.lower()
    if split == "train":
        base = out_root / "train_vector"
        return {
            "base": base,
            "cat": base / "cat.jsonl",
            "attr": base / "attr.jsonl",
            "rel": base / "rel.jsonl",
            "all": base / "all.jsonl",
            "stats": base / "stats.json",
        }
    base = out_root / f"{split}_eval"
    return {
        "base": base,
        "cat": base / f"gqa_cat_{split}.jsonl",
        "attr": base / f"gqa_attr_{split}.jsonl",
        "rel": base / f"gqa_rel_{split}.jsonl",
        "all": base / f"gqa_all_{split}.jsonl",
        "stats": base / "stats.json",
    }


def assert_can_write(paths: Mapping[str, Path], overwrite: bool) -> None:
    existing = [path for key, path in paths.items() if key != "base" and path.exists()]
    if existing and not overwrite:
        text = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(f"Output already exists; pass --overwrite to replace:\n{text}")


def top_counter(counter: Counter, n: int = 30) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common(n)}


def stats_for_examples(
    *,
    split: str,
    gqa_root: Path,
    scene_graph_path: Path,
    question_path: Path | None,
    image_roots: list[Path],
    records: list[dict[str, Any]],
    rows_by_type: Mapping[str, list[dict[str, Any]]],
    skipped: Counter,
    out_root: Path,
) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_type.values() for row in rows]
    image_ids = sorted({str(row["image_id"]) for row in all_rows})
    counts_by_type = {key: len(rows) for key, rows in rows_by_type.items()}
    answer_counts_by_type = {
        key: dict(Counter(row["answer"] for row in rows))
        for key, rows in rows_by_type.items()
    }
    subtype_counts = Counter(row["subtype"] for row in all_rows)
    type_subtype_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for row in all_rows:
        type_subtype_counts[row["type"]][row["subtype"]] = type_subtype_counts[row["type"]].get(row["subtype"], 0) + 1

    object_counts = Counter(row["subject"] for row in all_rows if row.get("subject"))
    object_counts.update(row["object"] for row in all_rows if row.get("object"))
    attr_counts = Counter(row["attribute"] for row in all_rows if row.get("attribute"))
    rel_counts = Counter(row["relation"] for row in all_rows if row.get("relation"))
    relation_bucket_counts = Counter(row.get("relation_bucket", "") for row in all_rows if row.get("relation_bucket"))
    image_count_by_type = {
        key: len({str(row["image_id"]) for row in rows})
        for key, rows in rows_by_type.items()
    }

    stats = {
        "source": f"{SOURCE_PREFIX}_{split}",
        "split": split,
        "gqa_root": str(gqa_root),
        "scene_graph_path": str(scene_graph_path),
        "question_path": str(question_path) if question_path else "",
        "image_roots": [str(path) for path in image_roots],
        "num_scene_graph_images_loaded": len(records),
        "total_examples": len(all_rows),
        "image_count": len(image_ids),
        "image_count_by_type": image_count_by_type,
        "type_counts": counts_by_type,
        "answer_counts_by_type": answer_counts_by_type,
        "subtype_counts": dict(subtype_counts),
        "type_subtype_counts": {key: dict(value) for key, value in type_subtype_counts.items()},
        "relation_bucket_counts": dict(relation_bucket_counts),
        "top_objects": top_counter(object_counts),
        "top_attributes": top_counter(attr_counts),
        "top_relations": top_counter(rel_counts),
        "skipped": dict(skipped),
        "image_ids": image_ids,
    }
    other_split = "val" if split == "train" else "train"
    other_stats_path = split_output_paths(out_root, other_split)["stats"]
    if other_stats_path.exists():
        with other_stats_path.open("r", encoding="utf-8") as handle:
            other_stats = json.load(handle)
        overlap = sorted(set(image_ids) & set(map(str, other_stats.get("image_ids", []))))
        stats["other_split_checked"] = other_split
        stats["train_val_image_overlap"] = len(overlap)
        stats["overlap_image_ids_preview"] = overlap[:20]
        if overlap:
            raise RuntimeError(
                f"Image-level leakage detected between split={split} and {other_split}: "
                f"{len(overlap)} overlapping image ids. First examples: {overlap[:10]}"
            )
    else:
        stats["other_split_checked"] = ""
        stats["train_val_image_overlap"] = None
    return stats


def validate_rows(rows: list[dict[str, Any]]) -> None:
    required = ("id", "image_id", "image_path", "type", "subtype", "question", "answer", "trusted_text")
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        missing = [field for field in required if row.get(field) in (None, "")]
        if missing:
            raise ValueError(f"row {index} missing required fields: {missing}")
        if row["answer"] not in {"yes", "no"}:
            raise ValueError(f"row {index} has non yes/no answer: {row['answer']}")
        if not Path(row["image_path"]).exists():
            raise FileNotFoundError(f"row {index} image_path does not exist: {row['image_path']}")
        if row["id"] in seen_ids:
            raise ValueError(f"duplicate row id: {row['id']}")
        seen_ids.add(row["id"])


def dedupe_and_uniquify_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    """Remove exact duplicate examples and make remaining row ids unique.

    GQA may contain multiple same-named objects in one image, for example two
    brown goats. Those produce identical diagnostic questions, so we keep one
    copy and avoid failing the whole build on duplicate ids.
    """

    skipped: Counter = Counter()
    seen_signatures: set[tuple[Any, ...]] = set()
    seen_ids: Counter = Counter()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        signature = (
            row.get("image_id"),
            row.get("type"),
            row.get("subtype"),
            row.get("question"),
            row.get("answer"),
            row.get("trusted_text"),
            row.get("subject"),
            row.get("object"),
            row.get("attribute"),
            row.get("relation"),
        )
        if signature in seen_signatures:
            skipped["exact_duplicate_examples"] += 1
            continue
        seen_signatures.add(signature)

        row = dict(row)
        base_id = str(row["id"])
        seen_ids[base_id] += 1
        if seen_ids[base_id] > 1:
            skipped["duplicate_ids_renamed"] += 1
            row["id"] = f"{base_id}_dup{seen_ids[base_id]}"
        deduped.append(row)
    return deduped, skipped


def print_balance_warning(name: str, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["answer"] for row in rows)
    total = sum(counts.values())
    if not total:
        print(f"[warn] {name}: no rows generated", file=sys.stderr)
        return
    ratio = min(counts.values() or [0]) / max(counts.values() or [1]) if len(counts) > 1 else 0.0
    if ratio < 0.8:
        print(f"[warn] {name}: yes/no imbalance {dict(counts)}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    split = str(args.split).lower()
    gqa_root = Path(args.gqa_root).expanduser()
    out_root = Path(args.out_root)
    paths = split_output_paths(out_root, split)
    try:
        assert_can_write(paths, bool(args.overwrite))
        scene_graph_path = resolve_gqa_file(gqa_root, split, "scene_graph", required=True)
        question_path = resolve_gqa_file(gqa_root, split, "questions", required=False)
        image_roots = discover_image_roots(gqa_root)
        print(f"[gqa] scene graph: {scene_graph_path}")
        print(f"[gqa] questions: {question_path or '<not used/not found>'}")
        print(f"[gqa] image roots: {', '.join(str(path) for path in image_roots)}")

        scene_graphs = load_gqa_scene_graphs(scene_graph_path)
        _ = load_optional_questions(question_path)
        records, prepare_skipped = prepare_records(scene_graphs, image_roots)
        if not records:
            raise RuntimeError("No usable GQA scene-graph records after image/object filtering.")

        object_vocab = build_object_vocab(records)
        attribute_vocab = build_attribute_vocab(records)
        relation_vocab = build_relation_vocab(records)
        if not object_vocab:
            raise RuntimeError("Object vocabulary is empty after filtering.")
        if not attribute_vocab:
            raise RuntimeError("Attribute vocabulary is empty after filtering.")
        if not relation_vocab:
            raise RuntimeError("Relation vocabulary is empty after filtering.")

        source = f"{SOURCE_PREFIX}_{split}"
        cat_rows, cat_skipped = build_cat_examples(records, object_vocab, int(args.max_cat), rng, source)
        attr_rows, attr_skipped = build_attr_examples(
            records,
            attribute_vocab,
            int(args.max_attr),
            rng,
            source,
            float(args.attr_count_fraction),
        )
        rel_rows, rel_skipped = build_rel_examples(records, relation_vocab, int(args.max_rel), rng, source)
        cat_rows, cat_dedupe = dedupe_and_uniquify_rows(cat_rows)
        attr_rows, attr_dedupe = dedupe_and_uniquify_rows(attr_rows)
        rel_rows, rel_dedupe = dedupe_and_uniquify_rows(rel_rows)
        rows_by_type = {"cat": cat_rows, "attr": attr_rows, "rel": rel_rows}
        all_rows = cat_rows + attr_rows + rel_rows
        validate_rows(all_rows)
        for key, rows in rows_by_type.items():
            print_balance_warning(key, rows)

        skipped = Counter()
        skipped.update(prepare_skipped)
        skipped.update({f"cat_{key}": value for key, value in cat_skipped.items()})
        skipped.update({f"attr_{key}": value for key, value in attr_skipped.items()})
        skipped.update({f"rel_{key}": value for key, value in rel_skipped.items()})
        skipped.update({f"cat_{key}": value for key, value in cat_dedupe.items()})
        skipped.update({f"attr_{key}": value for key, value in attr_dedupe.items()})
        skipped.update({f"rel_{key}": value for key, value in rel_dedupe.items()})
        stats = stats_for_examples(
            split=split,
            gqa_root=gqa_root,
            scene_graph_path=scene_graph_path,
            question_path=question_path,
            image_roots=image_roots,
            records=records,
            rows_by_type=rows_by_type,
            skipped=skipped,
            out_root=out_root,
        )

        paths["base"].mkdir(parents=True, exist_ok=True)
        write_jsonl(paths["cat"], cat_rows)
        write_jsonl(paths["attr"], attr_rows)
        write_jsonl(paths["rel"], rel_rows)
        write_jsonl(paths["all"], all_rows)
        write_json(paths["stats"], stats)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote GQA type-aware data to {paths['base'].resolve()}")
    print(json.dumps({
        "split": split,
        "cat": len(cat_rows),
        "attr": len(attr_rows),
        "rel": len(rel_rows),
        "total": len(all_rows),
        "stats": str(paths["stats"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
