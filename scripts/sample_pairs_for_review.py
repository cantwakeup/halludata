"""Sample balanced pairs by subtype for lightweight manual review."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for review-sample generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/outputs/pairs_balanced_v0.jsonl", help="Path to the source pair JSONL.")
    parser.add_argument("--per-subtype", type=int, default=30, help="How many rows to sample per subtype.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for subtype sampling.")
    parser.add_argument(
        "--output",
        default="data/outputs/pair_review_sample_v0.jsonl",
        help="Path to the sampled review JSONL output.",
    )
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    """Sample review-ready pair rows and write them back out as JSONL."""

    args = parse_args()
    rows = read_jsonl(resolve_project_path(args.input))
    rng = random.Random(int(args.seed))

    rows_by_subtype: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        subtype = str(row.get("subtype", "unknown"))
        rows_by_subtype.setdefault(subtype, []).append(row)

    sampled_rows: list[dict[str, object]] = []
    for subtype in sorted(rows_by_subtype):
        subtype_rows = list(rows_by_subtype[subtype])
        rng.shuffle(subtype_rows)
        for row in subtype_rows[: max(int(args.per_subtype), 0)]:
            sampled_rows.append(
                {
                    "pair_id": row.get("pair_id"),
                    "subtype": row.get("subtype"),
                    "image_id": row.get("image_id"),
                    "question": row.get("question"),
                    "response_pos": row.get("response_pos"),
                    "response_neg": row.get("response_neg"),
                    "pos_label": row.get("pos_label"),
                    "neg_label": row.get("neg_label"),
                    "metadata": dict(row.get("metadata", {})),
                }
            )

    output_path = resolve_project_path(args.output)
    write_jsonl(output_path, sampled_rows)
    print(f"Wrote {len(sampled_rows)} review samples to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

