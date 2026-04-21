"""Run a lightweight offline head-ranking pilot from balanced pair data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.head_ranking import compute_head_ranking
from expert_data.io_utils import read_jsonl, write_json
from expert_data.model_adapter import load_activation_adapter


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the head-ranking pilot."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/outputs/pairs_balanced_v0.jsonl", help="Balanced pair JSONL input.")
    parser.add_argument("--adapter", default="mock", help="Activation adapter name, such as mock or custom.")
    parser.add_argument("--top-k", type=int, default=64, help="How many heads to keep in the ranked output.")
    parser.add_argument("--output", default="data/outputs/head_ranking_v0.json", help="Head-ranking JSON output path.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    """Run the offline head-ranking scaffold using the requested activation adapter."""

    args = parse_args()
    adapter = load_activation_adapter(args.adapter)
    rows = read_jsonl(resolve_project_path(args.pairs))

    features_by_subtype: dict[str, dict[str, dict[str, list[list[float]]]]] = {}
    for row in rows:
        subtype = str(row.get("subtype", ""))
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
        subtype_features = features_by_subtype.setdefault(subtype, {})
        for head_key, vector in dict(pos_activation.get("layer_head_vectors", {})).items():
            subtype_features.setdefault(str(head_key), {"pos": [], "neg": []})
            subtype_features[str(head_key)]["pos"].append([float(value) for value in vector])
        for head_key, vector in dict(neg_activation.get("layer_head_vectors", {})).items():
            subtype_features.setdefault(str(head_key), {"pos": [], "neg": []})
            subtype_features[str(head_key)]["neg"].append([float(value) for value in vector])

    ranking = compute_head_ranking(features_by_subtype, top_k=int(args.top_k))
    output_payload: dict[str, object] = {
        "_meta": {
            "adapter": args.adapter,
            "pairs_path": str(resolve_project_path(args.pairs)),
            "top_k": int(args.top_k),
            "subtypes": sorted(ranking),
        }
    }
    output_payload.update(ranking)
    output_path = resolve_project_path(args.output)
    write_json(output_path, output_payload)
    print(f"Wrote head-ranking pilot for {len(ranking)} subtypes to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

