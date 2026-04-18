"""Build the deterministic category hard-negative resource used by the mock pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.io_utils import write_json
from expert_data.negatives import build_manual_seed_cat_neg_bank


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for hard-negative bank construction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/outputs/cat_neg_bank_v1.json",
        help="Path to the category negative bank JSON output.",
    )
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths for CLI inputs and outputs."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    """Build and write the deterministic category hard-negative bank."""

    args = parse_args()
    negative_bank = build_manual_seed_cat_neg_bank()
    output_path = write_json(resolve_project_path(args.output), negative_bank)
    print(f"Wrote category hard-negative bank to {output_path}")
    print(f"Covered categories: {', '.join(sorted(negative_bank))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
