"""Build subtype-level prototype vectors from balanced pairs via an activation adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.io_utils import read_jsonl, write_json
from expert_data.model_adapter import flatten_layer_head_vectors, load_activation_adapter
from expert_data.prototypes import aggregate_prototypes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prototype extraction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/outputs/pairs_balanced_v0.jsonl", help="Balanced pair JSONL input.")
    parser.add_argument("--adapter", default="mock", help="Activation adapter name, such as mock or custom.")
    parser.add_argument("--subtypes", default="cat,cnt,col,rel", help="Comma-separated subtype allowlist.")
    parser.add_argument("--output", default="data/outputs/prototypes_v0.json", help="Prototype JSON output path.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _parse_subtypes(raw_subtypes: str) -> list[str]:
    """Parse a comma-separated subtype list into a stable, non-empty sequence."""

    return [part.strip() for part in str(raw_subtypes).split(",") if part.strip()]


def main() -> int:
    """Run the prototype-building scaffold using the requested activation adapter."""

    args = parse_args()
    adapter = load_activation_adapter(args.adapter)
    allowed_subtypes = set(_parse_subtypes(args.subtypes))
    rows = read_jsonl(resolve_project_path(args.pairs))

    features_by_subtype: dict[str, dict[str, list[list[float]]]] = {}
    for row in rows:
        subtype = str(row.get("subtype", ""))
        if subtype not in allowed_subtypes:
            continue
        pair_id = str(row["pair_id"])
        image_id = str(row["image_id"])
        question = str(row["question"])
        pos_activation = adapter.encode_pair(
            image_id=image_id,
            question=question,
            response=str(row["response_pos"]),
            pair_id=pair_id,
            subtype=subtype,
            branch="pos",
        )
        neg_activation = adapter.encode_pair(
            image_id=image_id,
            question=question,
            response=str(row["response_neg"]),
            pair_id=pair_id,
            subtype=subtype,
            branch="neg",
        )
        features_by_subtype.setdefault(subtype, {"pos": [], "neg": []})
        features_by_subtype[subtype]["pos"].append(flatten_layer_head_vectors(pos_activation))
        features_by_subtype[subtype]["neg"].append(flatten_layer_head_vectors(neg_activation))

    prototypes = aggregate_prototypes(features_by_subtype)
    output_payload: dict[str, object] = {
        "_meta": {
            "adapter": args.adapter,
            "pairs_path": str(resolve_project_path(args.pairs)),
            "subtypes": sorted(prototypes),
        }
    }
    output_payload.update(prototypes)
    output_path = resolve_project_path(args.output)
    write_json(output_path, output_payload)
    print(f"Wrote prototypes for {len(prototypes)} subtypes to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

