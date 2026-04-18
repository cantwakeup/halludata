"""End-to-end tests for the COCO fact-index scaffold."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.facts import build_relations_for_image
from expert_data.schemas import ObjectInfo

FACT_INDEX_PATH = PROJECT_ROOT / "data" / "outputs" / "fact_index_v0.jsonl"
ATOMIC_FACTS_PATH = PROJECT_ROOT / "data" / "outputs" / "atomic_facts_v0.jsonl"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load JSONL rows into a list of dictionaries for assertions."""

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


class CocoFactIndexTest(unittest.TestCase):
    """Verify the mock COCO loader can build a usable fact index."""

    def test_build_coco_fact_index_with_mock_data(self) -> None:
        """Build both COCO outputs and validate their semantics separately."""

        result = subprocess.run(
            [sys.executable, "scripts/build_coco_fact_index.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"build_coco_fact_index.py failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertTrue(FACT_INDEX_PATH.exists(), msg=f"Missing output file: {FACT_INDEX_PATH}")
        self.assertTrue(ATOMIC_FACTS_PATH.exists(), msg=f"Missing output file: {ATOMIC_FACTS_PATH}")

        fact_index_records = load_jsonl(FACT_INDEX_PATH)
        self.assertEqual(len(fact_index_records), 2)
        self.assertTrue(
            all("objects" in record and "counts" in record and "meta" in record for record in fact_index_records)
        )
        self.assertTrue(
            all(
                {"object_id", "annotation_id", "category", "bbox", "area_ratio", "dominant_color"}.issubset(
                    set(record["objects"][0].keys())
                )
                for record in fact_index_records
                if record["objects"]
            )
        )
        self.assertTrue(
            all(record["meta"].get("source") == "coco_instance" for record in fact_index_records),
            msg="Every fact-index row should retain its COCO source metadata",
        )
        self.assertTrue(
            any(record["relations"] for record in fact_index_records),
            msg="Expected at least one image-level fact-index row with non-empty relations",
        )
        self.assertTrue(
            any(
                {"subject_category", "object_category", "dx", "dy", "iou"}.issubset(set(relation.keys()))
                for record in fact_index_records
                for relation in record["relations"]
            ),
            msg="Expected relation rows to include normalized geometry metadata",
        )

        atomic_records = load_jsonl(ATOMIC_FACTS_PATH)
        subtypes = {str(record["subtype"]) for record in atomic_records}
        self.assertTrue({"cat", "cnt", "rel"}.issubset(subtypes))
        self.assertTrue(
            any(
                record["image_id"] == "101"
                and record["subtype"] == "cnt"
                and record["subject"]["category"] == "cat"
                and record["positive_value"] == 2
                for record in atomic_records
            ),
            msg="Expected a count fact showing two cats in image 101",
        )
        self.assertTrue(
            any(
                record["subtype"] == "rel"
                and record["positive_value"] in {"left of", "right of", "above", "below"}
                for record in atomic_records
            ),
            msg="Expected at least one valid spatial relation fact",
        )

    def test_build_relations_for_image_extracts_a_primary_relation(self) -> None:
        """Directly verify relation extraction from a small synthetic object set."""

        objects = [
            {
                "bbox": [10, 10, 10, 10],
                "object_info": ObjectInfo(
                    object_id="obj_cat",
                    name="cat",
                    category="cat",
                    color=None,
                    aliases=[],
                ),
            },
            {
                "bbox": [35, 10, 10, 10],
                "object_info": ObjectInfo(
                    object_id="obj_dog",
                    name="dog",
                    category="dog",
                    color=None,
                    aliases=[],
                ),
            },
            {
                "bbox": [10, 35, 10, 10],
                "object_info": ObjectInfo(
                    object_id="obj_ball",
                    name="ball",
                    category="ball",
                    color=None,
                    aliases=[],
                ),
            },
        ]
        relations = build_relations_for_image(
            objects,
            {
                "rel_min_abs_dx": 0.75,
                "rel_max_abs_dy": 1.0,
                "rel_max_iou": 0.1,
            },
        )
        relation_tuples = {
            (relation.subject_id, relation.predicate, relation.object_id)
            for relation in relations
        }
        self.assertTrue(
            ("obj_cat", "left of", "obj_dog") in relation_tuples
            or ("obj_cat", "above", "obj_ball") in relation_tuples,
            msg=f"Expected a left-of or above relation, got {sorted(relation_tuples)}",
        )


if __name__ == "__main__":
    unittest.main()
