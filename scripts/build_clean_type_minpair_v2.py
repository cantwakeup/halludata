#!/usr/bin/env python3
"""Build clean attribute-family and relation-gold minimal-pair data.

This builder is intentionally conservative:

* each pair changes exactly one target factor;
* yes/no rows are constructed symmetrically from the same fact/counterfact;
* every row carries a condition_key for later condition-wise differencing;
* selection happens at pair level so yes/no balance is preserved.

It does not extract activations. The output is a data/audit package under
data/clean_type_minpair_v2 by default.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

DEFAULT_TARGETS = {
    "train": {
        "attr_color_clean": 1000,
        "attr_count_clean": 1000,
        "attr_state_clean": 700,
        "attr_material_clean": 400,
        "attr_shape_clean": 400,
        "attr_action_single_clean": 700,
        "rel_left_right_clean": 600,
        "rel_above_below_clean": 600,
        "rel_holding_wearing_clean": 600,
        "rel_sitting_riding_clean": 500,
    },
    "val": {
        "attr_color_clean": 300,
        "attr_count_clean": 300,
        "attr_state_clean": 200,
        "attr_material_clean": 150,
        "attr_shape_clean": 150,
        "attr_action_single_clean": 200,
        "rel_left_right_clean": 200,
        "rel_above_below_clean": 200,
        "rel_holding_wearing_clean": 200,
        "rel_sitting_riding_clean": 180,
    },
}

COLORS = ["black", "white", "red", "blue", "green", "yellow", "brown", "gray", "orange", "pink", "purple"]
COLOR_ALIASES = {"grey": "gray"}
COLOR_OBJECTS = {
    "car",
    "bus",
    "truck",
    "train",
    "boat",
    "shirt",
    "hat",
    "dress",
    "jacket",
    "pants",
    "shoes",
    "shoe",
    "umbrella",
    "bag",
    "backpack",
    "chair",
    "table",
    "bench",
    "sofa",
    "bed",
    "bottle",
    "cup",
    "sign",
    "building",
}
COUNT_OBJECTS = {
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "dog",
    "cat",
    "horse",
    "cow",
    "sheep",
    "bird",
    "chair",
    "cup",
    "bottle",
    "book",
    "umbrella",
    "bag",
}
REL_OBJECTS = {
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "dog",
    "cat",
    "horse",
    "cow",
    "sheep",
    "bird",
    "chair",
    "table",
    "bench",
    "sofa",
    "bed",
    "cup",
    "bottle",
    "plate",
    "bowl",
    "book",
    "bag",
    "umbrella",
}
PART_STUFF_BLACKLIST = {
    "leg",
    "legs",
    "arm",
    "arms",
    "hand",
    "hands",
    "foot",
    "feet",
    "hair",
    "ear",
    "ears",
    "beak",
    "leaf",
    "leaves",
    "sky",
    "grass",
    "ground",
    "wall",
    "road",
    "floor",
    "post",
    "pole",
    "letters",
    "letter",
    "shadow",
    "water",
    "seafood",
    "shrimp",
    "clouds",
    "cloud",
    "window",
    "windows",
    "tree",
    "trees",
}

STATE_PAIRS = {
    "open": "closed",
    "closed": "open",
    "full": "empty",
    "empty": "full",
    "clean": "dirty",
    "dirty": "clean",
    "wet": "dry",
    "dry": "wet",
    "on": "off",
    "off": "on",
    "broken": "intact",
    "intact": "broken",
}
STATE_OBJECTS = {"door", "window", "umbrella", "bottle", "cup", "plate", "light", "lamp", "screen", "sign", "bag"}
MATERIAL_ALIASES = {
    "wood": "wooden",
    "wooden": "wooden",
    "metal": "metal",
    "metallic": "metal",
    "glass": "glass",
    "plastic": "plastic",
    "cloth": "cloth",
    "leather": "leather",
    "stone": "stone",
    "paper": "paper",
}
MATERIAL_PAIRS = {
    "wooden": "metal",
    "metal": "wooden",
    "glass": "plastic",
    "plastic": "glass",
    "cloth": "leather",
    "leather": "cloth",
    "stone": "wooden",
    "paper": "plastic",
}
MATERIAL_OBJECTS = {"table", "chair", "bench", "bottle", "cup", "bag", "jacket", "sofa", "sign", "box"}
SHAPE_PAIRS = {
    "round": "square",
    "square": "round",
    "circular": "rectangular",
    "rectangular": "circular",
    "long": "short",
    "short": "long",
    "flat": "curved",
    "curved": "flat",
}
SHAPE_OBJECTS = {"sign", "plate", "table", "window", "mirror", "clock", "board", "box", "cake"}
ACTION_PAIRS = {
    "standing": "sitting",
    "sitting": "standing",
    "running": "standing",
    "walking": "standing",
    "sleeping": "awake",
    "awake": "sleeping",
    "flying": "standing",
    "lying": "standing",
}
ACTION_OBJECTS = {"person", "man", "woman", "boy", "girl", "dog", "cat", "bird", "horse"}

PERSON_SUBJECTS = {"person", "man", "woman", "boy", "girl"}
WEARING_OBJECTS = {"shirt", "hat", "jacket", "coat", "pants", "shoes", "shoe", "helmet", "backpack", "dress", "glasses"}
HOLDING_OBJECTS = {"umbrella", "phone", "cup", "bottle", "bat", "racket", "bag", "book", "knife", "frisbee", "hat", "helmet", "backpack", "glasses"}
WEAR_HOLD_DUAL = {"hat", "helmet", "backpack", "glasses", "bag"}
SITTING_OBJECTS = {"chair", "bench", "sofa", "bed"}
RIDING_OBJECTS = {"horse", "bicycle", "motorcycle", "skateboard"}
SIT_RIDE_SUBJECTS = PERSON_SUBJECTS | {"dog", "cat"}

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
GRAMMAR_PATTERNS = [
    "Are there one",
    "There are one",
    "watchs",
    "womans",
    "feets",
    "shoes is",
    "pants is",
    "There is a pants",
    "There is a shoes",
]


@dataclass
class PairCandidate:
    subtype: str
    split: str
    rows: List[Dict[str, Any]]
    balance: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gqa-train-scene-graph", default="/home/huiwei/sy/sy_data/GQA/raw/sceneGraphs/train_sceneGraphs.json")
    parser.add_argument("--gqa-val-scene-graph", default="/home/huiwei/sy/sy_data/GQA/raw/sceneGraphs/val_sceneGraphs.json")
    parser.add_argument("--gqa-image-root", default="/home/huiwei/sy/sy_data/GQA/raw/images/images")
    parser.add_argument("--output-root", default="data/clean_type_minpair_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-per-subtype", type=int, default=0, help="0 uses subtype defaults.")
    parser.add_argument("--max-val-per-subtype", type=int, default=0, help="0 uses subtype defaults.")
    parser.add_argument("--dry-run", action="store_true", help="Build reports/examples but do not write train.jsonl/val.jsonl.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_text(value: Any) -> str:
    text = str(value or "").replace("_", " ").lower()
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_color(value: str) -> str:
    value = clean_text(value)
    return COLOR_ALIASES.get(value, value)


def normalize_relation(value: Any) -> str:
    text = clean_text(value)
    aliases = {
        "left": "left of",
        "to the left of": "left of",
        "right": "right of",
        "to the right of": "right of",
        "over": "above",
        "on top of": "above",
        "under": "below",
        "beneath": "below",
        "holding": "holding",
        "hold": "holding",
        "wearing": "wearing",
        "wear": "wearing",
        "sitting": "sitting on",
        "sitting on": "sitting on",
        "riding": "riding",
        "ride": "riding",
        "standing next to": "standing beside",
        "standing beside": "standing beside",
        "next to": "standing beside",
    }
    return aliases.get(text, text)


def article(noun: str) -> str:
    return "an" if clean_text(noun)[:1] in {"a", "e", "i", "o", "u"} else "a"


IRREGULAR_PLURALS = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "foot": "feet",
    "tooth": "teeth",
    "mouse": "mice",
    "sheep": "sheep",
    "fish": "fish",
}


def singular(noun: str) -> str:
    noun = clean_text(noun)
    reverse = {v: k for k, v in IRREGULAR_PLURALS.items()}
    if noun in reverse:
        return reverse[noun]
    if noun.endswith("ies") and len(noun) > 4:
        return noun[:-3] + "y"
    if noun.endswith("ches") or noun.endswith("shes") or noun.endswith("xes"):
        return noun[:-2]
    if noun.endswith("s") and not noun.endswith("ss") and len(noun) > 3:
        return noun[:-1]
    return noun


def plural(noun: str, count: int | None = None) -> str:
    noun = singular(noun)
    if count == 1:
        return noun
    if noun in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[noun]
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def number_word(count: int) -> str:
    return NUMBER_WORDS.get(int(count), str(count))


def be_for_subject(subject: str) -> str:
    return "are" if singular(subject) != clean_text(subject) else "is"


def base_scene_one(obj: str) -> str:
    return f"There is {article(obj)} {singular(obj)} in the image."


def base_scene_count(obj: str) -> str:
    return f"The image contains {plural(obj)}."


def base_scene_two(subject: str, obj: str) -> str:
    if clean_text(subject) == clean_text(obj):
        return f"There are {plural(subject)} in the image."
    return f"There are {article(subject)} {singular(subject)} and {article(obj)} {singular(obj)} in the image."


def render_visual_prompt(question: str) -> str:
    return f"Question: {question}\nPlease answer the question based on the image."


def render_trusted_prompt(text: str, question: str) -> str:
    return (
        f"The given image depicts the following scene: {text}\n"
        "Please directly answer the following question from the image description, without guessing or reasoning.\n"
        f"Question: {question}"
    )


def make_row(
    *,
    row_id: str,
    split: str,
    source: str,
    expert_type: str,
    subtype: str,
    condition_key: Sequence[Any],
    image_id: str,
    image_path: str,
    question: str,
    gt_answer: str,
    base_scene: str,
    target_fact: str,
    target_counterfact: str,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    fact_text = f"{base_scene} {target_fact}".strip()
    counterfact_text = f"{base_scene} {target_counterfact}".strip()
    return {
        "id": row_id,
        "split": split,
        "source": source,
        "expert_type": expert_type,
        "hallucination_type": expert_type,
        "subtype": subtype,
        "condition_key": list(condition_key),
        "image_id": str(image_id),
        "image_path": str(image_path),
        "question": question,
        "gt_answer": gt_answer,
        "label": gt_answer,
        "base_scene": base_scene,
        "target_fact": target_fact,
        "target_counterfact": target_counterfact,
        "fact_text": fact_text,
        "counterfact_text": counterfact_text,
        "trusted_factual_text": fact_text,
        "visual_prompt": render_visual_prompt(question),
        "trusted_prompt_fact": render_trusted_prompt(fact_text, question),
        "trusted_prompt_counterfact": render_trusted_prompt(counterfact_text, question),
        "metadata": dict(metadata),
    }


def pair_rows(
    *,
    split: str,
    source: str,
    expert_type: str,
    subtype: str,
    pair_id: str,
    image_id: str,
    image_path: str,
    base_scene: str,
    target_fact: str,
    target_counterfact: str,
    true_question: str,
    false_question: str,
    true_condition: Sequence[Any],
    false_condition: Sequence[Any],
    metadata: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    return [
        make_row(
            row_id=f"{pair_id}_yes",
            split=split,
            source=source,
            expert_type=expert_type,
            subtype=subtype,
            condition_key=true_condition,
            image_id=image_id,
            image_path=image_path,
            question=true_question,
            gt_answer="yes",
            base_scene=base_scene,
            target_fact=target_fact,
            target_counterfact=target_counterfact,
            metadata={**metadata, "pair_id": pair_id, "query_is_fact": True},
        ),
        make_row(
            row_id=f"{pair_id}_no",
            split=split,
            source=source,
            expert_type=expert_type,
            subtype=subtype,
            condition_key=false_condition,
            image_id=image_id,
            image_path=image_path,
            question=false_question,
            gt_answer="no",
            base_scene=base_scene,
            target_fact=target_fact,
            target_counterfact=target_counterfact,
            metadata={**metadata, "pair_id": pair_id, "query_is_fact": False},
        ),
    ]


def resolve_gqa_image(image_id: str, roots: Sequence[Path]) -> str:
    for root in roots:
        for suffix in (".jpg", ".jpeg", ".png"):
            p = root / f"{image_id}{suffix}"
            if p.exists():
                return str(p)
    return ""


def discover_image_roots(explicit: Path) -> List[Path]:
    roots = [explicit]
    roots.extend([
        explicit / "images",
        explicit.parent,
        explicit.parent / "images",
        explicit.parent / "images" / "images",
    ])
    seen = []
    for root in roots:
        if root.exists() and root.is_dir() and root not in seen:
            seen.append(root)
    return seen


def object_bbox(raw: Mapping[str, Any]) -> Dict[str, float]:
    x = float(raw.get("x", raw.get("left", 0.0)) or 0.0)
    y = float(raw.get("y", raw.get("top", 0.0)) or 0.0)
    w = float(raw.get("w", raw.get("width", 0.0)) or 0.0)
    h = float(raw.get("h", raw.get("height", 0.0)) or 0.0)
    return {"x": x, "y": y, "w": w, "h": h}


def load_gqa_records(scene_graph_path: Path, image_roots: Sequence[Path], audit: Counter[str]) -> List[Dict[str, Any]]:
    payload = read_json(scene_graph_path)
    records: List[Dict[str, Any]] = []
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected dict image_id -> scene graph: {scene_graph_path}")
    for image_id, scene in payload.items():
        if not isinstance(scene, Mapping):
            audit["bad_scene_record"] += 1
            continue
        image_path = resolve_gqa_image(str(image_id), image_roots)
        if not image_path:
            audit["missing_image"] += 1
            continue
        raw_objects = scene.get("objects", {})
        if not isinstance(raw_objects, Mapping):
            audit["missing_objects"] += 1
            continue
        objects: Dict[str, Dict[str, Any]] = {}
        for object_id, raw in raw_objects.items():
            if not isinstance(raw, Mapping):
                audit["bad_object_record"] += 1
                continue
            name = clean_text(raw.get("name", ""))
            if not name or name in {"object", "thing", "stuff", "area", "item"}:
                audit["bad_object_name"] += 1
                continue
            attrs = []
            for attr in raw.get("attributes", []) or []:
                attr_clean = clean_text(attr)
                if attr_clean:
                    attrs.append(attr_clean)
            rels = []
            for rel in raw.get("relations", []) or []:
                if not isinstance(rel, Mapping):
                    continue
                target_id = str(rel.get("object", ""))
                rel_name = normalize_relation(rel.get("name", ""))
                if target_id and rel_name:
                    rels.append({"object_id": target_id, "relation": rel_name})
            objects[str(object_id)] = {
                "object_id": str(object_id),
                "name": name,
                "attributes": sorted(set(attrs)),
                "relations": rels,
                "bbox": object_bbox(raw),
            }
        if not objects:
            audit["empty_objects_after_filter"] += 1
            continue
        max_x = max((obj["bbox"]["x"] + obj["bbox"]["w"] for obj in objects.values()), default=1.0)
        max_y = max((obj["bbox"]["y"] + obj["bbox"]["h"] for obj in objects.values()), default=1.0)
        width = float(scene.get("width", scene.get("image_width", max_x)) or max_x or 1.0)
        height = float(scene.get("height", scene.get("image_height", max_y)) or max_y or 1.0)
        records.append(
            {
                "image_id": str(image_id),
                "image_path": image_path,
                "width": max(width, 1.0),
                "height": max(height, 1.0),
                "objects": objects,
                "raw_scene_keys": list(scene.keys())[:20],
            }
        )
    return records


def object_allowed(name: str, allowed: set[str]) -> bool:
    name = clean_text(name)
    return name in allowed and name not in PART_STUFF_BLACKLIST


def material_display(value: str) -> str:
    return "wood" if value == "wooden" else value


def attr_value_from_attrs(attrs: Sequence[str], mapping: Mapping[str, str] | None, allowed_values: set[str]) -> List[str]:
    values = []
    for attr in attrs:
        value = mapping.get(attr, attr) if mapping else attr
        if value in allowed_values:
            values.append(value)
    return sorted(set(values))


def build_attr_color_pairs(records: Sequence[Mapping[str, Any]], split: str, rng: random.Random) -> List[PairCandidate]:
    color_by_object: Dict[str, Counter[str]] = defaultdict(Counter)
    raw = []
    for rec in records:
        for obj in rec["objects"].values():
            name = obj["name"]
            if not object_allowed(name, COLOR_OBJECTS):
                continue
            colors = [normalize_color(attr) for attr in obj["attributes"] if normalize_color(attr) in COLORS]
            for color in colors:
                color_by_object[name][color] += 1
                raw.append((rec, obj, color))
    rng.shuffle(raw)
    pairs = []
    for rec, obj, color in raw:
        counter_options = [c for c, _ in color_by_object[obj["name"]].most_common() if c != color]
        strategy = "object_common_color"
        if not counter_options:
            counter_options = [c for c in COLORS if c != color]
            strategy = "global_color"
        counter = rng.choice(counter_options[:4]) if counter_options else ""
        if not counter:
            continue
        name = obj["name"]
        base = base_scene_one(name)
        fact = f"The {singular(name)} is {color}."
        cf = f"The {singular(name)} is {counter}."
        pair_id = f"{split}_attr_color_clean_{rec['image_id']}_{obj['object_id']}_{color}_{counter}"
        rows = pair_rows(
            split=split,
            source="gqa",
            expert_type="attr",
            subtype="attr_color_clean",
            pair_id=pair_id,
            image_id=rec["image_id"],
            image_path=rec["image_path"],
            base_scene=base,
            target_fact=fact,
            target_counterfact=cf,
            true_question=f"Is the {singular(name)} {color}?",
            false_question=f"Is the {singular(name)} {counter}?",
            true_condition=["attr_color_clean", name, color, counter, color],
            false_condition=["attr_color_clean", name, color, counter, counter],
            metadata={
                "object": name,
                "true_value": color,
                "counter_value": counter,
                "counter_strategy": strategy,
                "bbox": obj["bbox"],
                "filters_passed": ["object_whitelist", "color_whitelist"],
            },
        )
        pairs.append(PairCandidate("attr_color_clean", split, rows, {"object": name, "value": color, "value_pair": f"{color}->{counter}"}))
    return pairs


def count_sentence(count: int, obj: str) -> str:
    if count == 1:
        return f"There is one {singular(obj)} in the image."
    return f"There are {number_word(count)} {plural(obj, count)} in the image."


def count_question(count: int, obj: str) -> str:
    if count == 1:
        return f"Is there one {singular(obj)} in the image?"
    return f"Are there {number_word(count)} {plural(obj, count)} in the image?"


def build_attr_count_pairs(records: Sequence[Mapping[str, Any]], split: str, rng: random.Random) -> List[PairCandidate]:
    pairs = []
    for rec in records:
        counts = Counter(obj["name"] for obj in rec["objects"].values() if object_allowed(obj["name"], COUNT_OBJECTS))
        for name, count in counts.items():
            if int(count) not in {1, 2, 3, 4}:
                continue
            counter = count + 1 if count < 4 else 3
            base = base_scene_count(name)
            pair_id = f"{split}_attr_count_clean_{rec['image_id']}_{name.replace(' ', '_')}_{count}_{counter}"
            rows = pair_rows(
                split=split,
                source="gqa",
                expert_type="attr",
                subtype="attr_count_clean",
                pair_id=pair_id,
                image_id=rec["image_id"],
                image_path=rec["image_path"],
                base_scene=base,
                target_fact=count_sentence(count, name),
                target_counterfact=count_sentence(counter, name),
                true_question=count_question(count, name),
                false_question=count_question(counter, name),
                true_condition=["attr_count_clean", name, int(count), int(counter), int(count)],
                false_condition=["attr_count_clean", name, int(count), int(counter), int(counter)],
                metadata={
                    "object": name,
                    "true_value": int(count),
                    "counter_value": int(counter),
                    "filters_passed": ["object_whitelist", "count_1_to_4"],
                },
            )
            pairs.append(PairCandidate("attr_count_clean", split, rows, {"object": name, "value": str(count), "value_pair": f"{count}->{counter}"}))
    rng.shuffle(pairs)
    return pairs


def build_simple_attr_pairs(
    records: Sequence[Mapping[str, Any]],
    split: str,
    rng: random.Random,
    *,
    subtype: str,
    value_pairs: Mapping[str, str],
    object_whitelist: set[str],
    expert_label: str,
    fact_template: str,
    question_template: str,
    value_aliases: Mapping[str, str] | None = None,
    display_fn=lambda x: x,
) -> List[PairCandidate]:
    allowed_values = set(value_pairs)
    pairs = []
    for rec in records:
        for obj in rec["objects"].values():
            name = obj["name"]
            if not object_allowed(name, object_whitelist):
                continue
            values = attr_value_from_attrs(obj["attributes"], value_aliases, allowed_values)
            for value in values:
                counter = value_pairs.get(value)
                if not counter:
                    continue
                value_text = display_fn(value)
                counter_text = display_fn(counter)
                base = base_scene_one(name)
                fact = fact_template.format(object=singular(name), value=value_text)
                cf = fact_template.format(object=singular(name), value=counter_text)
                pair_id = f"{split}_{subtype}_{rec['image_id']}_{obj['object_id']}_{value}_{counter}"
                rows = pair_rows(
                    split=split,
                    source="gqa",
                    expert_type="attr",
                    subtype=subtype,
                    pair_id=pair_id,
                    image_id=rec["image_id"],
                    image_path=rec["image_path"],
                    base_scene=base,
                    target_fact=fact,
                    target_counterfact=cf,
                    true_question=question_template.format(object=singular(name), value=value_text),
                    false_question=question_template.format(object=singular(name), value=counter_text),
                    true_condition=[subtype, name, value, counter, value],
                    false_condition=[subtype, name, value, counter, counter],
                    metadata={
                        "object": name,
                        "true_value": value,
                        "counter_value": counter,
                        "bbox": obj["bbox"],
                        "filters_passed": ["object_whitelist", f"{expert_label}_value_pair"],
                    },
                )
                pairs.append(PairCandidate(subtype, split, rows, {"object": name, "value": value, "value_pair": f"{value}->{counter}"}))
    rng.shuffle(pairs)
    return pairs


def center_norm(obj: Mapping[str, Any], rec: Mapping[str, Any]) -> Tuple[float, float]:
    bbox = obj["bbox"]
    cx = (float(bbox["x"]) + float(bbox["w"]) / 2.0) / float(rec["width"])
    cy = (float(bbox["y"]) + float(bbox["h"]) / 2.0) / float(rec["height"])
    return cx, cy


def build_bbox_relation_pairs(
    records: Sequence[Mapping[str, Any]],
    split: str,
    rng: random.Random,
    *,
    subtype: str,
    axis: str,
) -> Tuple[List[PairCandidate], Counter[str], List[float]]:
    pairs: List[PairCandidate] = []
    audit = Counter()
    margins: List[float] = []
    for rec in records:
        objs = [obj for obj in rec["objects"].values() if object_allowed(obj["name"], REL_OBJECTS)]
        for subj in objs:
            for obj in objs:
                if subj["object_id"] == obj["object_id"] or subj["name"] == obj["name"]:
                    continue
                sx, sy = center_norm(subj, rec)
                ox, oy = center_norm(obj, rec)
                dx = sx - ox
                dy = sy - oy
                if axis == "horizontal":
                    if abs(dx) <= 0.15 or abs(dx) <= 1.5 * abs(dy):
                        audit["filtered_ambiguous_horizontal"] += 1
                        continue
                    true_rel, counter_rel = ("right of", "left of") if dx > 0 else ("left of", "right of")
                    q_true = f"Is the {singular(subj['name'])} to the {true_rel.split()[0]} of the {singular(obj['name'])}?"
                    q_false = f"Is the {singular(subj['name'])} to the {counter_rel.split()[0]} of the {singular(obj['name'])}?"
                    fact = f"The {singular(subj['name'])} is to the {true_rel.split()[0]} of the {singular(obj['name'])}."
                    cf = f"The {singular(subj['name'])} is to the {counter_rel.split()[0]} of the {singular(obj['name'])}."
                    margin = abs(dx)
                else:
                    if abs(dy) <= 0.15 or abs(dy) <= 1.5 * abs(dx):
                        audit["filtered_ambiguous_vertical"] += 1
                        continue
                    true_rel, counter_rel = ("below", "above") if dy > 0 else ("above", "below")
                    q_true = f"Is the {singular(subj['name'])} {true_rel} the {singular(obj['name'])}?"
                    q_false = f"Is the {singular(subj['name'])} {counter_rel} the {singular(obj['name'])}?"
                    fact = f"The {singular(subj['name'])} is {true_rel} the {singular(obj['name'])}."
                    cf = f"The {singular(subj['name'])} is {counter_rel} the {singular(obj['name'])}."
                    margin = abs(dy)
                margins.append(float(margin))
                base = base_scene_two(subj["name"], obj["name"])
                pair_id = f"{split}_{subtype}_{rec['image_id']}_{subj['object_id']}_{obj['object_id']}_{true_rel.replace(' ', '_')}"
                rows = pair_rows(
                    split=split,
                    source="gqa_bbox_derived",
                    expert_type="rel",
                    subtype=subtype,
                    pair_id=pair_id,
                    image_id=rec["image_id"],
                    image_path=rec["image_path"],
                    base_scene=base,
                    target_fact=fact,
                    target_counterfact=cf,
                    true_question=q_true,
                    false_question=q_false,
                    true_condition=[subtype, subj["name"], obj["name"], true_rel, counter_rel, true_rel],
                    false_condition=[subtype, subj["name"], obj["name"], true_rel, counter_rel, counter_rel],
                    metadata={
                        "subject": subj["name"],
                        "object2": obj["name"],
                        "predicate": true_rel,
                        "counter_predicate": counter_rel,
                        "bbox": {"subject": subj["bbox"], "object": obj["bbox"]},
                        "bbox_margin": float(margin),
                        "filters_passed": ["object_whitelist", f"{axis}_bbox_margin"],
                    },
                )
                pairs.append(PairCandidate(subtype, split, rows, {"subject": subj["name"], "object": obj["name"], "predicate": true_rel}))
    rng.shuffle(pairs)
    return pairs, audit, margins


def relation_fact(subject: str, predicate: str, obj: str) -> str:
    return f"The {singular(subject)} is {predicate} the {singular(obj)}."


def relation_question(subject: str, predicate: str, obj: str) -> str:
    return f"Is the {singular(subject)} {predicate} the {singular(obj)}?"


def build_relation_scene_pairs(records: Sequence[Mapping[str, Any]], split: str, rng: random.Random) -> Tuple[List[PairCandidate], List[PairCandidate], Counter[str]]:
    holding_wearing: List[PairCandidate] = []
    sitting_riding: List[PairCandidate] = []
    audit = Counter()
    for rec in records:
        objects = rec["objects"]
        for subj in objects.values():
            subject = subj["name"]
            for rel in subj["relations"]:
                target = objects.get(str(rel["object_id"]))
                if target is None:
                    audit["missing_relation_target"] += 1
                    continue
                obj = target["name"]
                pred = normalize_relation(rel["relation"])
                if subject not in PERSON_SUBJECTS and subject not in SIT_RIDE_SUBJECTS:
                    continue
                if pred == "wearing" and subject in PERSON_SUBJECTS and obj in WEAR_HOLD_DUAL:
                    counter = "holding"
                    subtype = "rel_holding_wearing_clean"
                    target_list = holding_wearing
                elif pred == "holding" and subject in PERSON_SUBJECTS and obj in HOLDING_OBJECTS:
                    counter = "standing beside"
                    subtype = "rel_holding_wearing_clean"
                    target_list = holding_wearing
                elif pred == "sitting on" and subject in SIT_RIDE_SUBJECTS and obj in SITTING_OBJECTS:
                    counter = "standing beside"
                    subtype = "rel_sitting_riding_clean"
                    target_list = sitting_riding
                elif pred == "riding" and subject in PERSON_SUBJECTS and obj in RIDING_OBJECTS:
                    counter = "standing beside"
                    subtype = "rel_sitting_riding_clean"
                    target_list = sitting_riding
                else:
                    continue
                text = f"{pred} {obj} {counter} {obj}"
                if any(bad in text for bad in ("wearing umbrella", "eating hat", "riding shirt", "holding ground")):
                    audit["filtered_unnatural_counterfact"] += 1
                    continue
                base = base_scene_two(subject, obj)
                pair_id = f"{split}_{subtype}_{rec['image_id']}_{subj['object_id']}_{target['object_id']}_{pred.replace(' ', '_')}"
                rows = pair_rows(
                    split=split,
                    source="gqa",
                    expert_type="rel",
                    subtype=subtype,
                    pair_id=pair_id,
                    image_id=rec["image_id"],
                    image_path=rec["image_path"],
                    base_scene=base,
                    target_fact=relation_fact(subject, pred, obj),
                    target_counterfact=relation_fact(subject, counter, obj),
                    true_question=relation_question(subject, pred, obj),
                    false_question=relation_question(subject, counter, obj),
                    true_condition=[subtype, subject, obj, pred, counter, pred],
                    false_condition=[subtype, subject, obj, pred, counter, counter],
                    metadata={
                        "subject": subject,
                        "object2": obj,
                        "predicate": pred,
                        "counter_predicate": counter,
                        "bbox": {"subject": subj["bbox"], "object": target["bbox"]},
                        "filters_passed": ["subject_whitelist", "object_predicate_whitelist", "natural_counterfact"],
                    },
                )
                target_list.append(PairCandidate(subtype, split, rows, {"subject": subject, "object": obj, "predicate": pred}))
    rng.shuffle(holding_wearing)
    rng.shuffle(sitting_riding)
    return holding_wearing, sitting_riding, audit


def target_for(split: str, subtype: str, args: argparse.Namespace) -> int:
    target = DEFAULT_TARGETS[split][subtype]
    cap = int(args.max_train_per_subtype if split == "train" else args.max_val_per_subtype)
    return min(target, cap) if cap > 0 else target


def select_balanced_pairs(
    pairs: Sequence[PairCandidate],
    *,
    target_rows: int,
    rng: random.Random,
    object_cap: float = 0.15,
    value_cap: float = 0.30,
    predicate_cap: float = 0.45,
) -> List[PairCandidate]:
    target_pairs = max(1, target_rows // 2)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    caps = {
        "object": max(1, math.ceil(target_pairs * object_cap)),
        "subject": max(1, math.ceil(target_pairs * object_cap)),
        "value": max(1, math.ceil(target_pairs * value_cap)),
        "predicate": max(1, math.ceil(target_pairs * predicate_cap)),
    }
    counters = {key: Counter() for key in caps}
    selected: List[PairCandidate] = []
    for pair in shuffled:
        if len(selected) >= target_pairs:
            break
        ok = True
        for key, cap in caps.items():
            val = pair.balance.get(key)
            if val and counters[key][val] >= cap:
                ok = False
                break
        if not ok:
            continue
        selected.append(pair)
        for key in caps:
            val = pair.balance.get(key)
            if val:
                counters[key][val] += 1
    return selected


def flatten_pairs(pairs: Sequence[PairCandidate]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        rows.extend(pair.rows)
    return rows


def counts_by(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def condition_text(row: Mapping[str, Any]) -> str:
    return json.dumps(row.get("condition_key", []), ensure_ascii=False)


def dist_from_metadata(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    c = Counter()
    for row in rows:
        meta = row.get("metadata", {})
        if isinstance(meta, Mapping):
            value = meta.get(key)
            if value not in (None, ""):
                c[str(value)] += 1
    return dict(c)


def top_rows(counter: Mapping[str, int], n: int = 20) -> List[Dict[str, Any]]:
    return [{"item": k, "count": v} for k, v in Counter(counter).most_common(n)]


def md_table(headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def grammar_errors(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    out = Counter()
    for row in rows:
        text = " ".join(
            str(row.get(key, ""))
            for key in ["question", "base_scene", "target_fact", "target_counterfact", "fact_text", "counterfact_text"]
        )
        for pattern in GRAMMAR_PATTERNS:
            if pattern.lower() in text.lower():
                out[pattern] += 1
    return out


def image_overlap(rows: Sequence[Mapping[str, Any]], key: str = "subtype") -> List[Dict[str, Any]]:
    by_key: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_key[str(row.get(key, ""))].add(str(row.get("image_id", "")))
    labels = sorted(k for k in by_key if k)
    out = []
    for a in labels:
        for b in labels:
            out.append({"a": a, "b": b, "overlap": len(by_key[a] & by_key[b])})
    return out


def sample_examples(rows: Sequence[Mapping[str, Any]], output_dir: Path, rng: random.Random) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_subtype: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subtype[str(row.get("subtype", ""))].append(row)
    for subtype, subtype_rows in sorted(by_subtype.items()):
        chosen = list(subtype_rows)
        rng.shuffle(chosen)
        lines = [f"# Examples: {subtype}", ""]
        for row in chosen[:20]:
            lines.append(f"## {row.get('id', '')}")
            lines.append("")
            lines.append(f"- image_id: `{row.get('image_id', '')}`")
            lines.append(f"- image_path: `{row.get('image_path', '')}`")
            lines.append(f"- question: {row.get('question', '')}")
            lines.append(f"- gt_answer: `{row.get('gt_answer', '')}`")
            lines.append(f"- fact_text: {row.get('fact_text', '')}")
            lines.append(f"- counterfact_text: {row.get('counterfact_text', '')}")
            lines.append(f"- condition_key: `{json.dumps(row.get('condition_key', []), ensure_ascii=False)}`")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(row.get("metadata", {}), indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
        (output_dir / f"{subtype}.md").write_text("\n".join(lines), encoding="utf-8")


def subtype_summary(rows: Sequence[Mapping[str, Any]], subtypes: Sequence[str] = SUBTYPES) -> List[Dict[str, Any]]:
    out = []
    for subtype in subtypes:
        subset = [row for row in rows if row.get("subtype") == subtype]
        yn = Counter(row.get("gt_answer") for row in subset)
        out.append(
            {
                "subtype": subtype,
                "count": len(subset),
                "yes": yn.get("yes", 0),
                "no": yn.get("no", 0),
                "source": dict(Counter(row.get("source", "") for row in subset)),
                "conditions": len(set(condition_text(row) for row in subset)),
            }
        )
    return out


def write_inspect(
    output: Path,
    *,
    args: argparse.Namespace,
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
    image_roots: Sequence[Path],
    audit: Mapping[str, int],
) -> None:
    sample_obj: Mapping[str, Any] = {}
    sample_rel: Mapping[str, Any] = {}
    for rec in list(train_records) + list(val_records):
        for obj in rec["objects"].values():
            sample_obj = obj
            if obj.get("relations"):
                sample_rel = obj["relations"][0]
                break
        if sample_obj:
            break
    lines = ["# Clean Type Minimal-Pair v2 Inspection", ""]
    lines.append("## Data Sources")
    lines.append("")
    lines.append(f"- GQA train scene graph: `{args.gqa_train_scene_graph}`")
    lines.append(f"- GQA val scene graph: `{args.gqa_val_scene_graph}`")
    lines.append(f"- GQA image root argument: `{args.gqa_image_root}`")
    lines.append(f"- Resolved image roots: `{[str(p) for p in image_roots]}`")
    lines.append(f"- Train records with images: `{len(train_records)}`")
    lines.append(f"- Val records with images: `{len(val_records)}`")
    lines.append("")
    lines.append("## GQA Object Field Example")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(sample_obj, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## GQA Attribute Field Example")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(sample_obj.get("attributes", []), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## GQA Relation Field Example")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(sample_rel, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## BBox Format")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(sample_obj.get("bbox", {}), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Reusable Scripts")
    lines.append("")
    for path in sorted((PROJECT_ROOT / "scripts").glob("*.py")):
        name = path.name
        if any(token in name for token in ["subtype", "gqa", "coco", "audit", "report", "minpair"]):
            lines.append(f"- `{path.relative_to(PROJECT_ROOT)}`")
    lines.append("")
    lines.append("## Data Quality Risks")
    lines.append("")
    lines.append("- GQA attributes can be sparse or ambiguous for material/shape/state.")
    lines.append("- Bbox-derived relations need strict margin filtering; ambiguous pairs are filtered.")
    lines.append("- Interaction counterfacts are conservative to avoid unnatural pairs such as wearing umbrella.")
    lines.append("- Count questions rely on object instance names; pluralization is explicitly audited.")
    lines.append("")
    lines.append("## Loader Audit Counters")
    lines.append("")
    lines.append(md_table(["item", "count"], top_rows(audit, 30)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def condition_report(output: Path, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(train_rows) + list(val_rows)
    lines = ["# Condition Balance Report", ""]
    for subtype in SUBTYPES:
        subset = [row for row in rows if row.get("subtype") == subtype]
        if not subset:
            lines.append(f"## {subtype}")
            lines.append("")
            lines.append("No rows.")
            lines.append("")
            continue
        conditions = Counter(condition_text(row) for row in subset)
        max_condition = max(conditions.values()) if conditions else 0
        lines.append(f"## {subtype}")
        lines.append("")
        lines.append(f"- rows: `{len(subset)}`")
        lines.append(f"- unique conditions: `{len(conditions)}`")
        lines.append(f"- max condition share: `{max_condition / len(subset):.4f}`")
        lines.append("")
        lines.append("### Yes/No")
        lines.append(md_table(["item", "count"], top_rows(counts_by(subset, "gt_answer"), 10)))
        lines.append("")
        lines.append("### Top Conditions")
        lines.append(md_table(["item", "count"], top_rows(conditions, 20)))
        lines.append("")
        lines.append("### Object / Value / Predicate Distribution")
        dist_rows = []
        for key in ["object", "subject", "object2", "true_value", "counter_value", "predicate", "counter_predicate"]:
            c = dist_from_metadata(subset, key)
            if c:
                top = Counter(c).most_common(1)[0]
                dist_rows.append({"field": key, "unique": len(c), "top": top[0], "top_count": top[1], "top_share": top[1] / len(subset)})
        lines.append(md_table(["field", "unique", "top", "top_count", "top_share"], dist_rows))
        lines.append("")
    lines.append("## Train/Val Image Overlap")
    train_images = {str(row.get("image_id", "")) for row in train_rows}
    val_images = {str(row.get("image_id", "")) for row in val_rows}
    lines.append(f"- overlap: `{len(train_images & val_images)}`")
    lines.append("")
    lines.append("## Subtype Image Overlap")
    lines.append(md_table(["a", "b", "overlap"], image_overlap(rows, "subtype")))
    output.write_text("\n".join(lines), encoding="utf-8")


def data_audit_report(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    builder_audit: Mapping[str, int],
    bbox_audit: Mapping[str, int],
    relation_audit: Mapping[str, int],
    bbox_margins: Mapping[str, Sequence[float]],
) -> None:
    lines = ["# Data Audit", ""]
    lines.append("## Grammar Audit")
    lines.append("")
    lines.append(md_table(["item", "count"], top_rows(grammar_errors(rows), 30)))
    lines.append("")
    lines.append("## Attribute Audit")
    for subtype in ATTRIBUTE_SUBTYPES:
        subset = [row for row in rows if row.get("subtype") == subtype]
        lines.append(f"### {subtype}")
        lines.append("")
        lines.append(f"- rows: `{len(subset)}`")
        lines.append("")
        lines.append("Object distribution:")
        lines.append(md_table(["item", "count"], top_rows(dist_from_metadata(subset, "object"), 20)))
        lines.append("")
        lines.append("True value distribution:")
        lines.append(md_table(["item", "count"], top_rows(dist_from_metadata(subset, "true_value"), 20)))
        lines.append("")
    lines.append("## Relation Audit")
    for subtype in RELATION_SUBTYPES:
        subset = [row for row in rows if row.get("subtype") == subtype]
        lines.append(f"### {subtype}")
        lines.append("")
        lines.append(f"- rows: `{len(subset)}`")
        lines.append("")
        lines.append("Subject distribution:")
        lines.append(md_table(["item", "count"], top_rows(dist_from_metadata(subset, "subject"), 20)))
        lines.append("")
        lines.append("Object distribution:")
        lines.append(md_table(["item", "count"], top_rows(dist_from_metadata(subset, "object2"), 20)))
        lines.append("")
        lines.append("Predicate distribution:")
        lines.append(md_table(["item", "count"], top_rows(dist_from_metadata(subset, "predicate"), 20)))
        if subtype in bbox_margins and bbox_margins[subtype]:
            vals = list(bbox_margins[subtype])
            lines.append("")
            lines.append(f"- bbox margin min/mean/max: `{min(vals):.4f}` / `{sum(vals)/len(vals):.4f}` / `{max(vals):.4f}`")
        lines.append("")
    lines.append("## Loader / Builder Audit Counters")
    lines.append("")
    merged = Counter()
    merged.update(builder_audit)
    merged.update(bbox_audit)
    merged.update(relation_audit)
    lines.append(md_table(["item", "count"], top_rows(merged, 50)))
    output.write_text("\n".join(lines), encoding="utf-8")


def data_report(output: Path, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], targets: Mapping[str, Mapping[str, int]], decision: str) -> None:
    lines = ["# Clean Type Minimal-Pair v2 Data Report", ""]
    lines.append(f"- Decision: `{decision}`")
    lines.append("")
    lines.append("## Train Summary")
    lines.append(md_table(["subtype", "count", "yes", "no", "source", "conditions"], subtype_summary(train_rows)))
    lines.append("")
    lines.append("## Val Summary")
    lines.append(md_table(["subtype", "count", "yes", "no", "source", "conditions"], subtype_summary(val_rows)))
    lines.append("")
    lines.append("## Target Coverage")
    rows = []
    for split, split_rows in [("train", train_rows), ("val", val_rows)]:
        by_subtype = Counter(row["subtype"] for row in split_rows)
        for subtype in SUBTYPES:
            target = int(targets[split][subtype])
            actual = int(by_subtype.get(subtype, 0))
            rows.append({"split": split, "subtype": subtype, "actual": actual, "target": target, "coverage": actual / target if target else 0.0})
    lines.append(md_table(["split", "subtype", "actual", "target", "coverage"], rows))
    output.write_text("\n".join(lines), encoding="utf-8")


def final_report(output: Path, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], decision: str, reasons: Sequence[str]) -> None:
    lines = ["# Clean Type Minimal-Pair v2 Report", ""]
    lines.append("## Goal")
    lines.append("")
    lines.append("Construct clean minimal-pair data satisfying single-factor intervention, nuisance balance, and condition-key differencing.")
    lines.append("")
    lines.append("## Data Sources")
    lines.append("")
    lines.append("- GQA scene graphs for attribute family and relation gold data.")
    lines.append("- Bbox-derived GQA relations only for left/right and above/below with strict margin filtering.")
    lines.append("")
    lines.append("## Attribute Family Summary")
    all_rows = list(train_rows) + list(val_rows)
    lines.append(md_table(["subtype", "count", "yes", "no", "source", "conditions"], subtype_summary(all_rows, ATTRIBUTE_SUBTYPES)))
    lines.append("")
    lines.append("## Relation Gold Summary")
    rel_rows = [row for row in all_rows if str(row.get("subtype", "")).startswith("rel_")]
    lines.append(md_table(["subtype", "count", "yes", "no", "source", "conditions"], subtype_summary(rel_rows, RELATION_SUBTYPES)))
    lines.append("")
    lines.append("## Audit Artifacts")
    lines.append("")
    lines.append("- `data/clean_type_minpair_v2/INSPECT.md`")
    lines.append("- `data/clean_type_minpair_v2/minimal_pairs/DATA_REPORT.md`")
    lines.append("- `data/clean_type_minpair_v2/minimal_pairs/CONDITION_REPORT.md`")
    lines.append("- `data/clean_type_minpair_v2/minimal_pairs/DATA_AUDIT.md`")
    lines.append("- `data/clean_type_minpair_v2/minimal_pairs/examples/`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"`{decision}`")
    lines.append("")
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.append("")
    if decision == "PASS":
        lines.append("Next: run official-LLaVA activation extraction, then build condition-balanced vectors/masks by differencing within condition_key before subtype averaging.")
    elif decision == "WARN":
        lines.append("Next: extract only passing subtypes first; repair shortage/noisy subtypes before broad experiments.")
    else:
        lines.append("Next: do not extract activations yet; fix the listed data quality blockers.")
    output.write_text("\n".join(lines), encoding="utf-8")


def make_targets(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    return {split: {subtype: target_for(split, subtype, args) for subtype in SUBTYPES} for split in ["train", "val"]}


def build_all_pairs(records: Sequence[Mapping[str, Any]], split: str, rng: random.Random) -> Tuple[Dict[str, List[PairCandidate]], Counter[str], Dict[str, List[float]]]:
    builder_audit = Counter()
    bbox_margins: Dict[str, List[float]] = defaultdict(list)
    by_subtype: Dict[str, List[PairCandidate]] = defaultdict(list)
    by_subtype["attr_color_clean"] = build_attr_color_pairs(records, split, rng)
    by_subtype["attr_count_clean"] = build_attr_count_pairs(records, split, rng)
    by_subtype["attr_state_clean"] = build_simple_attr_pairs(
        records,
        split,
        rng,
        subtype="attr_state_clean",
        value_pairs=STATE_PAIRS,
        object_whitelist=STATE_OBJECTS,
        expert_label="state",
        fact_template="The {object} is {value}.",
        question_template="Is the {object} {value}?",
    )
    by_subtype["attr_material_clean"] = build_simple_attr_pairs(
        records,
        split,
        rng,
        subtype="attr_material_clean",
        value_pairs=MATERIAL_PAIRS,
        object_whitelist=MATERIAL_OBJECTS,
        expert_label="material",
        fact_template="The {object} is made of {value}.",
        question_template="Is the {object} made of {value}?",
        value_aliases=MATERIAL_ALIASES,
        display_fn=material_display,
    )
    by_subtype["attr_shape_clean"] = build_simple_attr_pairs(
        records,
        split,
        rng,
        subtype="attr_shape_clean",
        value_pairs=SHAPE_PAIRS,
        object_whitelist=SHAPE_OBJECTS,
        expert_label="shape",
        fact_template="The {object} is {value}.",
        question_template="Is the {object} {value}?",
    )
    by_subtype["attr_action_single_clean"] = build_simple_attr_pairs(
        records,
        split,
        rng,
        subtype="attr_action_single_clean",
        value_pairs=ACTION_PAIRS,
        object_whitelist=ACTION_OBJECTS,
        expert_label="action",
        fact_template="The {object} is {value}.",
        question_template="Is the {object} {value}?",
    )
    horizontal, h_audit, h_margins = build_bbox_relation_pairs(records, split, rng, subtype="rel_left_right_clean", axis="horizontal")
    vertical, v_audit, v_margins = build_bbox_relation_pairs(records, split, rng, subtype="rel_above_below_clean", axis="vertical")
    hold_wear, sit_ride, rel_audit = build_relation_scene_pairs(records, split, rng)
    by_subtype["rel_left_right_clean"] = horizontal
    by_subtype["rel_above_below_clean"] = vertical
    by_subtype["rel_holding_wearing_clean"] = hold_wear
    by_subtype["rel_sitting_riding_clean"] = sit_ride
    builder_audit.update(h_audit)
    builder_audit.update(v_audit)
    builder_audit.update(rel_audit)
    bbox_margins["rel_left_right_clean"].extend(h_margins)
    bbox_margins["rel_above_below_clean"].extend(v_margins)
    return by_subtype, builder_audit, bbox_margins


def decision_status(train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], targets: Mapping[str, Mapping[str, int]]) -> Tuple[str, List[str]]:
    rows = list(train_rows) + list(val_rows)
    by_split_subtype = defaultdict(int)
    for row in train_rows:
        by_split_subtype[("train", row["subtype"])] += 1
    for row in val_rows:
        by_split_subtype[("val", row["subtype"])] += 1
    reasons = []
    fail = False
    warn = False
    for subtype in ["attr_color_clean", "attr_count_clean"]:
        coverage = by_split_subtype[("train", subtype)] / max(targets["train"][subtype], 1)
        if coverage < 0.70:
            fail = True
            reasons.append(f"{subtype} train coverage below 70% ({coverage:.2%}).")
    if by_split_subtype[("train", "attr_state_clean")] <= 0:
        fail = True
        reasons.append("attr_state_clean has no train samples.")
    for subtype in ["rel_left_right_clean", "rel_holding_wearing_clean"]:
        if by_split_subtype[("train", subtype)] <= 0:
            fail = True
            reasons.append(f"{subtype} has no train samples.")
    grammar = grammar_errors(rows)
    if sum(grammar.values()) > 0:
        fail = True
        reasons.append(f"Grammar audit found serious patterns: {dict(grammar)}.")
    for subtype in SUBTYPES:
        train_count = by_split_subtype[("train", subtype)]
        target = targets["train"][subtype]
        if train_count and train_count / max(target, 1) < 0.50:
            warn = True
            reasons.append(f"{subtype} is available but below 50% target ({train_count}/{target}).")
        subset = [row for row in rows if row.get("subtype") == subtype]
        if subset:
            yn = Counter(row["gt_answer"] for row in subset)
            ratio = yn.get("yes", 0) / max(len(subset), 1)
            if not (0.45 <= ratio <= 0.55):
                warn = True
                reasons.append(f"{subtype} yes/no ratio is not close to 1:1 ({ratio:.2f}).")
    if fail:
        return "FAIL", reasons
    if warn:
        return "WARN", reasons
    return "PASS", ["Core audit checks passed."]


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    output_root = resolve(args.output_root)
    minpair_dir = output_root / "minimal_pairs"
    if output_root.exists() and any(output_root.iterdir()) and not (args.overwrite or args.dry_run):
        raise FileExistsError(f"Output root is not empty: {output_root}. Pass --overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)
    minpair_dir.mkdir(parents=True, exist_ok=True)

    image_roots = discover_image_roots(resolve(args.gqa_image_root))
    loader_audit = Counter()
    train_records = load_gqa_records(resolve(args.gqa_train_scene_graph), image_roots, loader_audit)
    val_records = load_gqa_records(resolve(args.gqa_val_scene_graph), image_roots, loader_audit)

    write_inspect(
        output_root / "INSPECT.md",
        args=args,
        train_records=train_records,
        val_records=val_records,
        image_roots=image_roots,
        audit=loader_audit,
    )

    targets = make_targets(args)
    all_builder_audit = Counter()
    all_bbox_margins: Dict[str, List[float]] = defaultdict(list)
    selected_rows_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}

    for split, records in [("train", train_records), ("val", val_records)]:
        by_subtype, builder_audit, bbox_margins = build_all_pairs(records, split, rng)
        all_builder_audit.update(builder_audit)
        for subtype, values in bbox_margins.items():
            all_bbox_margins[subtype].extend(values)
        for subtype in SUBTYPES:
            selected_pairs = select_balanced_pairs(
                by_subtype.get(subtype, []),
                target_rows=targets[split][subtype],
                rng=rng,
                object_cap=0.15,
                value_cap=0.30,
                predicate_cap=0.45,
            )
            selected_rows_by_split[split].extend(flatten_pairs(selected_pairs))

    train_rows = selected_rows_by_split["train"]
    val_rows = selected_rows_by_split["val"]
    decision, reasons = decision_status(train_rows, val_rows, targets)

    if not args.dry_run:
        write_jsonl(minpair_dir / "train.jsonl", train_rows)
        write_jsonl(minpair_dir / "val.jsonl", val_rows)
    write_json(minpair_dir / "targets.json", targets)
    sample_examples(train_rows + val_rows, minpair_dir / "examples", rng)
    data_report(minpair_dir / "DATA_REPORT.md", train_rows, val_rows, targets, decision)
    condition_report(minpair_dir / "CONDITION_REPORT.md", train_rows, val_rows)
    data_audit_report(
        minpair_dir / "DATA_AUDIT.md",
        train_rows + val_rows,
        builder_audit=loader_audit,
        bbox_audit=all_builder_audit,
        relation_audit=Counter(),
        bbox_margins=all_bbox_margins,
    )
    final_report(output_root / "REPORT.md", train_rows, val_rows, decision, reasons)

    print(f"Wrote clean type minimal-pair reports to {output_root}")
    if not args.dry_run:
        print(f"Wrote train rows: {len(train_rows)} -> {minpair_dir / 'train.jsonl'}")
        print(f"Wrote val rows: {len(val_rows)} -> {minpair_dir / 'val.jsonl'}")
    print(f"Decision: {decision}")
    for reason in reasons:
        print(f"- {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
