#!/usr/bin/env python3
"""Build subtype-aware symmetric minimal-pair data for activation editing.

The output intentionally mirrors the AFTER prompt schema while adding
fact/counterfact text for the new shared-private subtype experiments.
All rows are yes/no image-question examples; the trusted branches are text-only.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLOR_WORDS = ("red", "blue", "green", "yellow", "black", "white", "brown", "gray", "orange", "pink", "purple")
CONTACT_RELATIONS = {
    "holding": "wearing",
    "wearing": "holding",
    "riding": "standing beside",
    "sitting on": "standing next to",
    "standing on": "sitting next to",
    "carrying": "touching",
    "eating": "holding",
    "touching": "carrying",
    "lying on": "standing next to",
    "leaning on": "standing beside",
}
WEAK_RELATIONS = {"of", "with", "at", "near", "by", "in", "on"}
SPATIAL_INVERSE = {
    "left of": "right of",
    "right of": "left of",
    "above": "below",
    "below": "above",
}
SUBTYPES = (
    "cat_random",
    "cat_popular",
    "cat_hard",
    "attr_color",
    "attr_count",
    "rel_spatial",
    "rel_contact",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/subtype_minpair_v1/minimal_pairs")
    parser.add_argument("--coco-train-json", default="/home/huiwei/sy/sy_data/COCO2014/annotations/instances_train2014.json")
    parser.add_argument("--coco-val-json", default="/home/huiwei/sy/sy_data/COCO2014/annotations/instances_val2014.json")
    parser.add_argument("--coco-train-image-root", default="/home/huiwei/sy/sy_data/COCO2014/train2014")
    parser.add_argument("--coco-val-image-root", default="/home/huiwei/sy/sy_data/COCO2014/val2014")
    parser.add_argument("--gqa-root", default="/home/huiwei/sy/sy_data/GQA")
    parser.add_argument("--gqa-train-scene-graph", default="")
    parser.add_argument("--gqa-val-scene-graph", default="")
    parser.add_argument("--gqa-image-root", default="")
    parser.add_argument("--train-cat-per-subtype", type=int, default=600)
    parser.add_argument("--val-cat-per-subtype", type=int, default=200)
    parser.add_argument("--train-attr-per-subtype", type=int, default=500)
    parser.add_argument("--val-attr-per-subtype", type=int, default=200)
    parser.add_argument("--train-rel-per-subtype", type=int, default=500)
    parser.add_argument("--val-rel-per-subtype", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def article(noun: str) -> str:
    return "an" if str(noun).strip().lower()[:1] in {"a", "e", "i", "o", "u"} else "a"


def plural(noun: str, count: int | None = None) -> str:
    if count == 1:
        return noun
    if noun.endswith("s"):
        return noun
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def count_word(count: int) -> str:
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    return words.get(int(count), str(count))


def render_visual_prompt(question: str) -> str:
    return f"Question: {question}\nPlease answer the question based on the image."


def render_trusted_prompt(text: str, question: str) -> str:
    return (
        f"The given image depicts the following scene: {text}\n"
        "Please directly answer the following question from the image description, without guessing or reasoning.\n"
        f"Question: {question}"
    )


def base_scene_from_counts(counts: Mapping[str, int], exclude: set[str], max_items: int = 8) -> str:
    pieces = []
    for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if name in exclude:
            continue
        pieces.append(f"There {'is' if count == 1 else 'are'} {count_word(count)} {plural(name, count)} in the image")
        if len(pieces) >= max_items:
            break
    return ". ".join(pieces) + "." if pieces else "The image contains visible objects."


def object_exists_scene(objects: Iterable[str]) -> str:
    unique = sorted({obj for obj in objects if obj})
    if not unique:
        return "The image contains visible objects."
    if len(unique) == 1:
        return f"There is {article(unique[0])} {unique[0]} in the image."
    return f"There are {', '.join(article(obj) + ' ' + obj for obj in unique[:-1])}, and {article(unique[-1])} {unique[-1]} in the image."


def make_row(
    *,
    row_id: str,
    split: str,
    source: str,
    expert_type: str,
    subtype: str,
    image_id: str,
    image_path: str,
    question: str,
    gt_answer: str,
    base_scene: str,
    target_fact: str,
    target_counterfact: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    fact_text = f"{base_scene} {target_fact}".strip()
    counterfact_text = f"{base_scene} {target_counterfact}".strip()
    return {
        "id": row_id,
        "split": split,
        "source": source,
        "expert_type": expert_type,
        "hallucination_type": expert_type,
        "subtype": subtype,
        "image_id": str(image_id),
        "image_path": str(image_path),
        "question": question,
        "gt_answer": gt_answer,
        "label": gt_answer,
        "fact_text": fact_text,
        "counterfact_text": counterfact_text,
        "trusted_factual_text": fact_text,
        "base_scene": base_scene,
        "target_fact": target_fact,
        "target_counterfact": target_counterfact,
        "visual_prompt": render_visual_prompt(question),
        "trusted_prompt": render_trusted_prompt(fact_text, question),
        "trusted_prompt_fact": render_trusted_prompt(fact_text, question),
        "trusted_prompt_counterfact": render_trusted_prompt(counterfact_text, question),
        "metadata": dict(metadata),
    }


def load_coco_records(instances_json: Path, image_root: Path) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Counter[str]]]:
    payload = read_json(instances_json)
    categories = {int(cat["id"]): str(cat["name"]) for cat in payload.get("categories", [])}
    image_by_id = {int(image["id"]): dict(image) for image in payload.get("images", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        cat = categories.get(int(ann.get("category_id", -1)))
        if not cat:
            continue
        anns_by_image[int(ann["image_id"])].append({**ann, "category_name": cat})
    records = []
    category_counts: Counter[str] = Counter()
    cooc: dict[str, Counter[str]] = defaultdict(Counter)
    for image_id, image in image_by_id.items():
        anns = anns_by_image.get(image_id, [])
        counts = Counter(str(ann["category_name"]) for ann in anns)
        if not counts:
            continue
        names = sorted(counts)
        category_counts.update(counts)
        for name in names:
            for other in names:
                if other != name:
                    cooc[name][other] += 1
        image_path = image_root / str(image.get("file_name", f"{image_id:012d}.jpg"))
        records.append(
            {
                "image_id": str(image_id),
                "image_path": str(image_path),
                "file_name": str(image.get("file_name", "")),
                "counts": dict(counts),
            }
        )
    return records, category_counts, cooc


def choose_absent(
    subtype: str,
    present: set[str],
    all_categories: list[str],
    popular: list[str],
    cooc: Mapping[str, Counter[str]],
    rng: random.Random,
) -> tuple[str | None, str]:
    absent = [name for name in all_categories if name not in present]
    if not absent:
        return None, "none"
    if subtype == "cat_random":
        return rng.choice(absent), "random_absent"
    if subtype == "cat_popular":
        for name in popular:
            if name in absent:
                return name, "popular_absent"
        return rng.choice(absent), "popular_fallback_random"
    scores: Counter[str] = Counter()
    for present_name in present:
        for absent_name in absent:
            scores[absent_name] += int(cooc.get(present_name, Counter()).get(absent_name, 0))
    if scores:
        best_score = max(scores.values())
        if best_score > 0:
            best = sorted([name for name, score in scores.items() if score == best_score])
            return rng.choice(best), "cooccurrence_hard_absent"
    return rng.choice(absent), "hard_fallback_random"


def build_cat_rows(
    records: list[dict[str, Any]],
    *,
    split: str,
    target_per_subtype: int,
    all_categories: list[str],
    popular: list[str],
    cooc: Mapping[str, Counter[str]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_subtype: dict[str, list[dict[str, Any]]] = {key: [] for key in ("cat_random", "cat_popular", "cat_hard")}
    shuffled = list(records)
    rng.shuffle(shuffled)
    for subtype in by_subtype:
        for record in shuffled:
            if len(by_subtype[subtype]) >= target_per_subtype:
                break
            counts = {str(k): int(v) for k, v in dict(record["counts"]).items()}
            present = set(counts)
            if not present:
                continue
            pos = rng.choice(sorted(present))
            neg, strategy = choose_absent(subtype, present, all_categories, popular, cooc, rng)
            if not neg:
                continue
            pos_base = base_scene_from_counts(counts, {pos})
            neg_base = base_scene_from_counts(counts, {neg})
            pos_article = article(pos)
            neg_article = article(neg)
            pair_id = f"{split}_coco_{record['image_id']}_{subtype}_{pos.replace(' ', '_')}_{neg.replace(' ', '_')}"
            by_subtype[subtype].append(
                make_row(
                    row_id=f"{pair_id}_yes",
                    split=split,
                    source="coco",
                    expert_type="cat",
                    subtype=subtype,
                    image_id=record["image_id"],
                    image_path=record["image_path"],
                    question=f"Is there {pos_article} {pos} in the image?",
                    gt_answer="yes",
                    base_scene=pos_base,
                    target_fact=f"There is {pos_article} {pos} in the image.",
                    target_counterfact=f"There is no {pos} in the image.",
                    metadata={"positive_object": pos, "negative_object": neg, "negative_strategy": strategy, "pair_id": pair_id},
                )
            )
            if len(by_subtype[subtype]) >= target_per_subtype:
                break
            by_subtype[subtype].append(
                make_row(
                    row_id=f"{pair_id}_no",
                    split=split,
                    source="coco",
                    expert_type="cat",
                    subtype=subtype,
                    image_id=record["image_id"],
                    image_path=record["image_path"],
                    question=f"Is there {neg_article} {neg} in the image?",
                    gt_answer="no",
                    base_scene=neg_base,
                    target_fact=f"There is no {neg} in the image.",
                    target_counterfact=f"There is {neg_article} {neg} in the image.",
                    metadata={"positive_object": pos, "negative_object": neg, "negative_strategy": strategy, "pair_id": pair_id},
                )
            )
        rows.extend(by_subtype[subtype][:target_per_subtype])
    return rows


def discover_gqa_scene_graph(gqa_root: Path, split: str) -> Path | None:
    candidates = [
        gqa_root / "raw" / "sceneGraphs" / f"{split}_sceneGraphs.json",
        gqa_root / "raw" / "sceneGraphs" / f"{split}_scene_graphs.json",
        gqa_root / "sceneGraphs" / f"{split}_sceneGraphs.json",
        gqa_root / f"{split}_sceneGraphs.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in gqa_root.rglob("*.json") if gqa_root.exists() else []:
        lower = path.name.lower()
        if split.lower() in lower and ("scenegraph" in lower or "scene_graph" in lower):
            return path
    return None


def discover_gqa_image_roots(gqa_root: Path, explicit: str) -> list[Path]:
    roots = []
    if explicit:
        roots.append(resolve(explicit))
    roots.extend(
        [
            gqa_root / "raw" / "images" / "images",
            gqa_root / "raw" / "images",
            gqa_root / "images",
        ]
    )
    return [path for path in roots if path.exists()]


def resolve_gqa_image(image_id: str, roots: list[Path]) -> str:
    for root in roots:
        for suffix in (".jpg", ".jpeg", ".png"):
            path = root / f"{image_id}{suffix}"
            if path.exists():
                return str(path)
    return ""


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_relation(value: Any) -> str:
    text = clean_text(value)
    aliases = {
        "to the left of": "left of",
        "left": "left of",
        "to the right of": "right of",
        "right": "right of",
        "over": "above",
        "under": "below",
        "beneath": "below",
    }
    return aliases.get(text, text)


def load_gqa_records(scene_graph_path: Path, image_roots: list[Path]) -> list[dict[str, Any]]:
    payload = read_json(scene_graph_path)
    records = []
    for image_id, scene in payload.items():
        if not isinstance(scene, Mapping):
            continue
        image_path = resolve_gqa_image(str(image_id), image_roots)
        if not image_path:
            continue
        objects = {}
        for object_id, raw in dict(scene.get("objects", {})).items():
            if not isinstance(raw, Mapping):
                continue
            name = clean_text(raw.get("name", ""))
            if not name or name in {"object", "thing", "things", "stuff"}:
                continue
            attrs = sorted({clean_text(attr) for attr in raw.get("attributes", []) if clean_text(attr)})
            rels = []
            for rel in raw.get("relations", []):
                if not isinstance(rel, Mapping):
                    continue
                target_id = str(rel.get("object", ""))
                rel_name = normalize_relation(rel.get("name", ""))
                if target_id and rel_name:
                    rels.append({"object_id": target_id, "relation": rel_name})
            objects[str(object_id)] = {
                "object_id": str(object_id),
                "name": name,
                "attributes": attrs,
                "relations": rels,
                "x": float(raw.get("x", 0.0) or 0.0),
                "y": float(raw.get("y", 0.0) or 0.0),
                "w": float(raw.get("w", raw.get("width", 0.0)) or 0.0),
                "h": float(raw.get("h", raw.get("height", 0.0)) or 0.0),
            }
        if objects:
            records.append({"image_id": str(image_id), "image_path": image_path, "objects": objects})
    return records


def split_gqa_pools(records: list[dict[str, Any]], rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    return shuffled[:midpoint], shuffled[midpoint:]


def choose_counter_color(obj_name: str, true_color: str, color_by_object: Mapping[str, Counter[str]], rng: random.Random) -> str:
    common = [color for color, _count in color_by_object.get(obj_name, Counter()).most_common() if color != true_color]
    if common:
        return common[0]
    return rng.choice([color for color in COLOR_WORDS if color != true_color])


def build_attr_color_rows(records: list[dict[str, Any]], *, split: str, target: int, rng: random.Random) -> list[dict[str, Any]]:
    color_by_object: dict[str, Counter[str]] = defaultdict(Counter)
    candidates = []
    for record in records:
        for obj in record["objects"].values():
            colors = [attr for attr in obj["attributes"] if attr in COLOR_WORDS]
            for color in colors:
                color_by_object[obj["name"]][color] += 1
                candidates.append((record, obj, color))
    rng.shuffle(candidates)
    rows = []
    for record, obj, color in candidates:
        if len(rows) >= target:
            break
        counter = choose_counter_color(obj["name"], color, color_by_object, rng)
        base = object_exists_scene([obj["name"]])
        pair_id = f"{split}_gqa_attr_color_{record['image_id']}_{obj['object_id']}_{color}_{counter}"
        rows.append(
            make_row(
                row_id=f"{pair_id}_yes",
                split=split,
                source="gqa",
                expert_type="attr",
                subtype="attr_color",
                image_id=record["image_id"],
                image_path=record["image_path"],
                question=f"Is the {obj['name']} {color}?",
                gt_answer="yes",
                base_scene=base,
                target_fact=f"The {obj['name']} is {color}.",
                target_counterfact=f"The {obj['name']} is {counter}.",
                metadata={"object": obj["name"], "attribute": color, "counter_attribute": counter, "pair_id": pair_id},
            )
        )
        if len(rows) >= target:
            break
        rows.append(
            make_row(
                row_id=f"{pair_id}_no",
                split=split,
                source="gqa",
                expert_type="attr",
                subtype="attr_color",
                image_id=record["image_id"],
                image_path=record["image_path"],
                question=f"Is the {obj['name']} {counter}?",
                gt_answer="no",
                base_scene=base,
                target_fact=f"The {obj['name']} is {color}.",
                target_counterfact=f"The {obj['name']} is {counter}.",
                metadata={"object": obj["name"], "attribute": color, "counter_attribute": counter, "pair_id": pair_id},
            )
        )
    return rows[:target]


def build_attr_count_rows(records: list[dict[str, Any]], *, split: str, target: int, rng: random.Random) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        counts = Counter(obj["name"] for obj in record["objects"].values())
        for name, count in counts.items():
            if 1 <= int(count) <= 5:
                candidates.append((record, name, int(count)))
    rng.shuffle(candidates)
    rows = []
    for record, name, count in candidates:
        if len(rows) >= target:
            break
        counter = count + 1 if count < 5 else count - 1
        base = object_exists_scene([name])
        pair_id = f"{split}_gqa_attr_count_{record['image_id']}_{name.replace(' ', '_')}_{count}_{counter}"
        rows.append(
            make_row(
                row_id=f"{pair_id}_yes",
                split=split,
                source="gqa",
                expert_type="attr",
                subtype="attr_count",
                image_id=record["image_id"],
                image_path=record["image_path"],
                question=f"Are there {count_word(count)} {plural(name, count)} in the image?",
                gt_answer="yes",
                base_scene=base,
                target_fact=f"There {'is' if count == 1 else 'are'} {count_word(count)} {plural(name, count)} in the image.",
                target_counterfact=f"There {'is' if counter == 1 else 'are'} {count_word(counter)} {plural(name, counter)} in the image.",
                metadata={"object": name, "count": count, "counter_count": counter, "pair_id": pair_id},
            )
        )
        if len(rows) >= target:
            break
        rows.append(
            make_row(
                row_id=f"{pair_id}_no",
                split=split,
                source="gqa",
                expert_type="attr",
                subtype="attr_count",
                image_id=record["image_id"],
                image_path=record["image_path"],
                question=f"Are there {count_word(counter)} {plural(name, counter)} in the image?",
                gt_answer="no",
                base_scene=base,
                target_fact=f"There {'is' if count == 1 else 'are'} {count_word(count)} {plural(name, count)} in the image.",
                target_counterfact=f"There {'is' if counter == 1 else 'are'} {count_word(counter)} {plural(name, counter)} in the image.",
                metadata={"object": name, "count": count, "counter_count": counter, "pair_id": pair_id},
            )
        )
    return rows[:target]


def bbox_relation(a: Mapping[str, Any], b: Mapping[str, Any], *, min_offset: float, ratio: float) -> tuple[str, str] | None:
    ax = float(a["x"]) + float(a["w"]) / 2.0
    ay = float(a["y"]) + float(a["h"]) / 2.0
    bx = float(b["x"]) + float(b["w"]) / 2.0
    by = float(b["y"]) + float(b["h"]) / 2.0
    dx = ax - bx
    dy = ay - by
    if abs(dx) >= min_offset and abs(dx) >= abs(dy) * ratio:
        return ("right of", "left of") if dx > 0 else ("left of", "right of")
    if abs(dy) >= min_offset and abs(dy) >= abs(dx) * ratio:
        return ("below", "above") if dy > 0 else ("above", "below")
    return None


def build_rel_spatial_rows(records: list[dict[str, Any]], *, split: str, target: int, rng: random.Random) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        objs = list(record["objects"].values())
        for i, a in enumerate(objs):
            for b in objs[i + 1 :]:
                rel = bbox_relation(a, b, min_offset=35.0, ratio=1.5)
                if rel is not None and a["name"] != b["name"]:
                    candidates.append((record, a, b, rel[0], rel[1]))
    rng.shuffle(candidates)
    rows = []
    for record, a, b, rel, inverse in candidates:
        if len(rows) >= target:
            break
        base = object_exists_scene([a["name"], b["name"]])
        pair_id = f"{split}_gqa_rel_spatial_{record['image_id']}_{a['object_id']}_{rel.replace(' ', '_')}_{b['object_id']}"
        rows.append(
            make_row(
                row_id=f"{pair_id}_yes",
                split=split,
                source="gqa_bbox_derived",
                expert_type="rel",
                subtype="rel_spatial",
                image_id=record["image_id"],
                image_path=record["image_path"],
                question=f"Is the {a['name']} {rel} the {b['name']} in the image?",
                gt_answer="yes",
                base_scene=base,
                target_fact=f"The {a['name']} is {rel} the {b['name']} in the image.",
                target_counterfact=f"The {a['name']} is {inverse} the {b['name']} in the image.",
                metadata={"object_a": a["name"], "object_b": b["name"], "relation": rel, "counter_relation": inverse, "derived_from": "bbox_center", "pair_id": pair_id},
            )
        )
        if len(rows) >= target:
            break
        rows.append(
            make_row(
                row_id=f"{pair_id}_no",
                split=split,
                source="gqa_bbox_derived",
                expert_type="rel",
                subtype="rel_spatial",
                image_id=record["image_id"],
                image_path=record["image_path"],
                question=f"Is the {a['name']} {inverse} the {b['name']} in the image?",
                gt_answer="no",
                base_scene=base,
                target_fact=f"The {a['name']} is {rel} the {b['name']} in the image.",
                target_counterfact=f"The {a['name']} is {inverse} the {b['name']} in the image.",
                metadata={"object_a": a["name"], "object_b": b["name"], "relation": rel, "counter_relation": inverse, "derived_from": "bbox_center", "pair_id": pair_id},
            )
        )
    return rows[:target]


def build_rel_contact_rows(records: list[dict[str, Any]], *, split: str, target: int, rng: random.Random) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        objects = record["objects"]
        for obj in objects.values():
            for raw_rel in obj["relations"]:
                relation = normalize_relation(raw_rel["relation"])
                if relation in WEAK_RELATIONS or relation not in CONTACT_RELATIONS:
                    continue
                target_obj = objects.get(str(raw_rel["object_id"]))
                if not target_obj or target_obj["name"] == obj["name"]:
                    continue
                candidates.append((record, obj, target_obj, relation, CONTACT_RELATIONS[relation]))
    rng.shuffle(candidates)
    rows = []
    for record, a, b, rel, counter_rel in candidates:
        if len(rows) >= target:
            break
        base = object_exists_scene([a["name"], b["name"]])
        pair_id = f"{split}_gqa_rel_contact_{record['image_id']}_{a['object_id']}_{rel.replace(' ', '_')}_{b['object_id']}"
        rows.append(
            make_row(
                row_id=f"{pair_id}_yes",
                split=split,
                source="gqa",
                expert_type="rel",
                subtype="rel_contact",
                image_id=record["image_id"],
                image_path=record["image_path"],
                question=f"Is the {a['name']} {rel} the {b['name']} in the image?",
                gt_answer="yes",
                base_scene=base,
                target_fact=f"The {a['name']} is {rel} the {b['name']} in the image.",
                target_counterfact=f"The {a['name']} is {counter_rel} the {b['name']} in the image.",
                metadata={"object_a": a["name"], "object_b": b["name"], "relation": rel, "counter_relation": counter_rel, "pair_id": pair_id},
            )
        )
        if len(rows) >= target:
            break
        rows.append(
            make_row(
                row_id=f"{pair_id}_no",
                split=split,
                source="gqa",
                expert_type="rel",
                subtype="rel_contact",
                image_id=record["image_id"],
                image_path=record["image_path"],
                question=f"Is the {a['name']} {counter_rel} the {b['name']} in the image?",
                gt_answer="no",
                base_scene=base,
                target_fact=f"The {a['name']} is {rel} the {b['name']} in the image.",
                target_counterfact=f"The {a['name']} is {counter_rel} the {b['name']} in the image.",
                metadata={"object_a": a["name"], "object_b": b["name"], "relation": rel, "counter_relation": counter_rel, "pair_id": pair_id},
            )
        )
    return rows[:target]


def counts_by(rows: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def nested_count(rows: list[Mapping[str, Any]], a: str, b: str) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        out[str(row.get(a, ""))][str(row.get(b, ""))] += 1
    return {key: dict(value) for key, value in out.items()}


def image_overlap(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    by_type: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_type[str(row["expert_type"])].add(str(row["image_id"]))
    keys = ["cat", "attr", "rel"]
    return {a: {b: len(by_type[a] & by_type[b]) for b in keys} for a in keys}


def top_metadata(rows: list[Mapping[str, Any]], field: str, n: int = 20) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        meta = row.get("metadata", {})
        if isinstance(meta, Mapping) and meta.get(field):
            counter[str(meta[field])] += 1
    return dict(counter.most_common(n))


def sample_examples(rows: list[Mapping[str, Any]], rng: random.Random, n: int = 5) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for subtype in SUBTYPES:
        subset = [row for row in rows if row.get("subtype") == subtype]
        rng.shuffle(subset)
        out[subtype] = [
            {
                "id": row["id"],
                "gt_answer": row["gt_answer"],
                "question": row["question"],
                "visual_prompt": row["visual_prompt"],
                "trusted_prompt_fact": row["trusted_prompt_fact"],
                "trusted_prompt_counterfact": row["trusted_prompt_counterfact"],
            }
            for row in subset[:n]
        ]
    return out


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, rows: list[dict[str, Any]], stats: Mapping[str, Any], examples: Mapping[str, Any]) -> None:
    lines = ["# Subtype Minimal-Pair Data Report", ""]
    lines.append("## Counts")
    table_rows = []
    for split in ("train", "val"):
        for subtype in SUBTYPES:
            subset = [row for row in rows if row["split"] == split and row["subtype"] == subtype]
            table_rows.append([split, subtype, len(subset), counts_by(subset, "gt_answer").get("yes", 0), counts_by(subset, "gt_answer").get("no", 0), counts_by(subset, "source")])
    lines.append(markdown_table(["split", "subtype", "n", "yes", "no", "sources"], table_rows))
    lines.append("")
    lines.append("## Image Overlap")
    lines.append("```json")
    lines.append(json.dumps(stats["image_overlap"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Label Distributions")
    lines.append("```json")
    lines.append(json.dumps(stats["label_distributions"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Warnings")
    if stats["warnings"]:
        for warning in stats["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Examples")
    for subtype, subtype_examples in examples.items():
        lines.append(f"### {subtype}")
        lines.append("```json")
        lines.append(json.dumps(subtype_examples, indent=2, ensure_ascii=False))
        lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    out_dir = resolve(args.output_dir)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    all_path = out_dir / "all.jsonl"
    report_path = out_dir / "DATA_REPORT.md"
    stats_path = out_dir / "stats.json"
    if any(path.exists() for path in (train_path, val_path, all_path, report_path)) and not args.overwrite:
        raise FileExistsError(f"Output already exists under {out_dir}. Pass --overwrite.")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    source_status: dict[str, Any] = {}

    coco_train_json = resolve(args.coco_train_json)
    coco_val_json = resolve(args.coco_val_json)
    if coco_train_json.exists() and coco_val_json.exists():
        train_coco, cat_counts, cooc = load_coco_records(coco_train_json, resolve(args.coco_train_image_root))
        val_coco, _val_counts, _val_cooc = load_coco_records(coco_val_json, resolve(args.coco_val_image_root))
        all_categories = sorted(cat_counts)
        popular = [name for name, _ in cat_counts.most_common()]
        rows.extend(
            build_cat_rows(
                train_coco,
                split="train",
                target_per_subtype=int(args.train_cat_per_subtype),
                all_categories=all_categories,
                popular=popular,
                cooc=cooc,
                rng=rng,
            )
        )
        rows.extend(
            build_cat_rows(
                val_coco,
                split="val",
                target_per_subtype=int(args.val_cat_per_subtype),
                all_categories=all_categories,
                popular=popular,
                cooc=cooc,
                rng=rng,
            )
        )
        source_status["coco"] = {"train_records": len(train_coco), "val_records": len(val_coco), "category_count": len(all_categories)}
    else:
        warnings.append(f"COCO instances missing: {coco_train_json} / {coco_val_json}; category subtypes skipped.")
        source_status["coco"] = {"available": False, "train_json": str(coco_train_json), "val_json": str(coco_val_json)}

    gqa_root = resolve(args.gqa_root)
    gqa_image_roots = discover_gqa_image_roots(gqa_root, args.gqa_image_root)
    train_sg = resolve(args.gqa_train_scene_graph) if args.gqa_train_scene_graph else discover_gqa_scene_graph(gqa_root, "train")
    val_sg = resolve(args.gqa_val_scene_graph) if args.gqa_val_scene_graph else discover_gqa_scene_graph(gqa_root, "val")
    if train_sg and val_sg and gqa_image_roots:
        train_records = load_gqa_records(train_sg, gqa_image_roots)
        val_records = load_gqa_records(val_sg, gqa_image_roots)
        train_attr_pool, train_rel_pool = split_gqa_pools(train_records, rng)
        val_attr_pool, val_rel_pool = split_gqa_pools(val_records, rng)
        rows.extend(build_attr_color_rows(train_attr_pool, split="train", target=int(args.train_attr_per_subtype), rng=rng))
        rows.extend(build_attr_count_rows(train_attr_pool, split="train", target=int(args.train_attr_per_subtype), rng=rng))
        rows.extend(build_rel_spatial_rows(train_rel_pool, split="train", target=int(args.train_rel_per_subtype), rng=rng))
        rows.extend(build_rel_contact_rows(train_rel_pool, split="train", target=int(args.train_rel_per_subtype), rng=rng))
        rows.extend(build_attr_color_rows(val_attr_pool, split="val", target=int(args.val_attr_per_subtype), rng=rng))
        rows.extend(build_attr_count_rows(val_attr_pool, split="val", target=int(args.val_attr_per_subtype), rng=rng))
        rows.extend(build_rel_spatial_rows(val_rel_pool, split="val", target=int(args.val_rel_per_subtype), rng=rng))
        rows.extend(build_rel_contact_rows(val_rel_pool, split="val", target=int(args.val_rel_per_subtype), rng=rng))
        source_status["gqa"] = {
            "train_scene_graph": str(train_sg),
            "val_scene_graph": str(val_sg),
            "image_roots": [str(path) for path in gqa_image_roots],
            "train_records": len(train_records),
            "val_records": len(val_records),
            "train_attr_pool": len(train_attr_pool),
            "train_rel_pool": len(train_rel_pool),
            "val_attr_pool": len(val_attr_pool),
            "val_rel_pool": len(val_rel_pool),
        }
        for split, attr_pool, rel_pool in (("train", train_attr_pool, train_rel_pool), ("val", val_attr_pool, val_rel_pool)):
            overlap = {record["image_id"] for record in attr_pool} & {record["image_id"] for record in rel_pool}
            if overlap:
                warnings.append(f"GQA attr/rel image overlap for {split}: {len(overlap)}")
    else:
        warnings.append("GQA scene graph or image roots missing; attr/rel subtypes skipped.")
        source_status["gqa"] = {"available": False, "train_scene_graph": str(train_sg), "val_scene_graph": str(val_sg), "image_roots": [str(path) for path in gqa_image_roots]}

    for split in ("train", "val"):
        for subtype in SUBTYPES:
            n = sum(1 for row in rows if row["split"] == split and row["subtype"] == subtype)
            target = {
                ("train", "cat"): int(args.train_cat_per_subtype),
                ("val", "cat"): int(args.val_cat_per_subtype),
                ("train", "attr"): int(args.train_attr_per_subtype),
                ("val", "attr"): int(args.val_attr_per_subtype),
                ("train", "rel"): int(args.train_rel_per_subtype),
                ("val", "rel"): int(args.val_rel_per_subtype),
            }[(split, subtype.split("_", 1)[0])]
            if n < target:
                warnings.append(f"{split}/{subtype} has {n} rows, below target {target}.")

    rows.sort(key=lambda row: (row["split"], row["expert_type"], row["subtype"], row["id"]))
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    stats = {
        "source_status": source_status,
        "counts_by_split_subtype": nested_count(rows, "split", "subtype"),
        "yes_no_by_subtype": {subtype: counts_by([row for row in rows if row["subtype"] == subtype], "gt_answer") for subtype in SUBTYPES},
        "source_by_subtype": {subtype: counts_by([row for row in rows if row["subtype"] == subtype], "source") for subtype in SUBTYPES},
        "image_overlap": image_overlap(rows),
        "label_distributions": {
            "objects": top_metadata(rows, "object") or top_metadata(rows, "positive_object"),
            "colors": top_metadata(rows, "attribute"),
            "counts": top_metadata(rows, "count"),
            "relations": top_metadata(rows, "relation"),
            "negative_strategy": top_metadata(rows, "negative_strategy"),
        },
        "warnings": warnings,
    }
    examples = sample_examples(rows, rng)
    if args.dry_run:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(all_path, rows)
    write_json(stats_path, stats)
    write_report(report_path, rows, stats, examples)
    print(f"Wrote train rows: {train_path} ({len(train_rows)})")
    print(f"Wrote val rows: {val_path} ({len(val_rows)})")
    print(f"Wrote report: {report_path}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
