"""Create deterministic image-level train/val/test splits for pair banks."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.io_utils import read_jsonl, write_json, write_jsonl

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SUBTYPE_ORDER = ("cat", "cnt", "col", "rel")
REQUIRED_PAIR_FIELDS = ("pair_id", "image_id", "subtype", "question", "response_pos", "response_neg")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for image-level pair-bank splitting."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/outputs/pairs_balanced_v0.jsonl", help="Input pair JSONL path.")
    parser.add_argument(
        "--out-dir",
        default="data/outputs/splits/v0_mini_seed42",
        help="Directory where split files and metadata will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Target train pair ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Target validation pair ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Target test pair ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic image ordering.")
    parser.add_argument("--split-name", default="v0_mini_seed42", help="Human-readable split name for the manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def sha256_file(path: str | Path) -> str:
    """Compute a SHA-256 digest for a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    """Validate split ratios and return them in canonical split-name order."""

    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    if any(value <= 0.0 for value in ratios.values()):
        raise ValueError("Split ratios must all be positive.")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(
            "Split ratios must sum to 1.0; "
            f"got train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )
    return ratios


def extract_template_id(pair: Mapping[str, Any]) -> str:
    """Extract a template identifier from known pair fields, falling back to a sentinel."""

    if pair.get("template_id"):
        return str(pair["template_id"])
    metadata = pair.get("metadata")
    if isinstance(metadata, Mapping):
        if metadata.get("template_id"):
            return str(metadata["template_id"])
        if metadata.get("shell_id"):
            return str(metadata["shell_id"])
    return "__missing_template__"


def validate_pair_rows(pairs: list[dict[str, Any]]) -> None:
    """Ensure every pair row has the minimum fields needed for leakage-safe splitting."""

    for row_number, pair in enumerate(pairs, start=1):
        missing_fields = [field for field in REQUIRED_PAIR_FIELDS if field not in pair or pair[field] in {None, ""}]
        if missing_fields:
            raise ValueError(
                f"Pair row {row_number} is missing required field(s): {', '.join(sorted(missing_fields))}"
            )


def _subtype_keys(global_counts: Mapping[str, int]) -> list[str]:
    """Return subtype keys with the project-standard order first and extras sorted afterward."""

    extras = sorted(subtype for subtype in global_counts if subtype not in DEFAULT_SUBTYPE_ORDER)
    return [subtype for subtype in DEFAULT_SUBTYPE_ORDER if subtype in global_counts] + extras


def group_pairs_by_image(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group pair rows by image_id while preserving original rows inside each image group."""

    groups_by_image: dict[str, dict[str, Any]] = {}
    for input_index, pair in enumerate(pairs):
        image_id = str(pair["image_id"])
        group = groups_by_image.setdefault(
            image_id,
            {
                "image_id": image_id,
                "first_index": input_index,
                "pairs": [],
                "subtype_counts": Counter(),
            },
        )
        group["pairs"].append(pair)
        group["subtype_counts"][str(pair["subtype"])] += 1

    groups: list[dict[str, Any]] = []
    for group in groups_by_image.values():
        subtype_counts = dict(sorted(group["subtype_counts"].items()))
        groups.append(
            {
                "image_id": group["image_id"],
                "first_index": group["first_index"],
                "pairs": list(group["pairs"]),
                "num_pairs": len(group["pairs"]),
                "subtype_counts": subtype_counts,
            }
        )
    return groups


def _placement_penalty(
    split_name: str,
    group: Mapping[str, Any],
    split_state: Mapping[str, Mapping[str, Any]],
    target_pairs: Mapping[str, float],
    target_subtypes: Mapping[str, Mapping[str, float]],
    subtype_keys: list[str],
) -> float:
    """Score how far a candidate placement would move one split from its targets."""

    candidate_pairs = int(split_state[split_name]["num_pairs"]) + int(group["num_pairs"])
    pair_target = max(float(target_pairs[split_name]), 1.0)
    penalty = abs(candidate_pairs - pair_target) / pair_target
    if candidate_pairs > pair_target:
        penalty += 2.0 * (candidate_pairs - pair_target) / pair_target

    subtype_penalty = 0.0
    current_subtypes = split_state[split_name]["subtype_counts"]
    group_subtypes = group["subtype_counts"]
    for subtype in subtype_keys:
        candidate_count = int(current_subtypes.get(subtype, 0)) + int(group_subtypes.get(subtype, 0))
        subtype_target = max(float(target_subtypes[split_name].get(subtype, 0.0)), 1.0)
        subtype_penalty += abs(candidate_count - subtype_target) / subtype_target
        if candidate_count > subtype_target:
            subtype_penalty += (candidate_count - subtype_target) / subtype_target
    if subtype_keys:
        penalty += subtype_penalty / len(subtype_keys)

    return penalty


def assign_image_splits(
    image_groups: list[dict[str, Any]],
    ratios: Mapping[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    """Assign image groups to train/val/test with deterministic greedy stratification."""

    total_pairs = sum(int(group["num_pairs"]) for group in image_groups)
    global_subtype_counts: Counter[str] = Counter()
    for group in image_groups:
        global_subtype_counts.update(group["subtype_counts"])
    subtype_keys = _subtype_keys(global_subtype_counts)

    target_pairs = {split: total_pairs * ratios[split] for split in SPLIT_NAMES}
    target_subtypes = {
        split: {subtype: global_subtype_counts[subtype] * ratios[split] for subtype in subtype_keys}
        for split in SPLIT_NAMES
    }

    rng = random.Random(int(seed))
    shuffled_groups = list(image_groups)
    rng.shuffle(shuffled_groups)
    for shuffle_rank, group in enumerate(shuffled_groups):
        group["_shuffle_rank"] = shuffle_rank
    shuffled_groups.sort(key=lambda group: (-int(group["num_pairs"]), int(group["_shuffle_rank"])))

    split_state: dict[str, dict[str, Any]] = {
        split: {"num_pairs": 0, "subtype_counts": Counter(), "num_images": 0}
        for split in SPLIT_NAMES
    }
    assignments: list[dict[str, Any]] = []
    for group in shuffled_groups:
        penalties = [
            (
                _placement_penalty(
                    split,
                    group,
                    split_state,
                    target_pairs,
                    target_subtypes,
                    subtype_keys,
                ),
                split_state[split]["num_images"],
                split_state[split]["num_pairs"],
                split,
            )
            for split in SPLIT_NAMES
        ]
        _, _, _, chosen_split = min(penalties)
        split_state[chosen_split]["num_pairs"] += int(group["num_pairs"])
        split_state[chosen_split]["num_images"] += 1
        split_state[chosen_split]["subtype_counts"].update(group["subtype_counts"])
        assignments.append(
            {
                "image_id": str(group["image_id"]),
                "split": chosen_split,
                "num_pairs": int(group["num_pairs"]),
                "subtype_counts": {subtype: int(group["subtype_counts"].get(subtype, 0)) for subtype in subtype_keys},
            }
        )

    return sorted(assignments, key=lambda item: str(item["image_id"]))


def _empty_split_stats(subtype_keys: list[str]) -> dict[str, Any]:
    """Create an empty split-stat accumulator with stable subtype keys."""

    return {
        "num_pairs": 0,
        "num_images": 0,
        "subtype_counts": {subtype: 0 for subtype in subtype_keys},
        "template_counts": {},
        "unique_image_ids": 0,
    }


def _count_subtypes(pairs: list[dict[str, Any]]) -> Counter[str]:
    """Count pair subtypes using string-normalized subtype values."""

    return Counter(str(pair["subtype"]) for pair in pairs)


def build_split_stats(
    pairs: list[dict[str, Any]],
    split_rows: Mapping[str, list[dict[str, Any]]],
    assignments: list[dict[str, Any]],
    input_pairs_path: str | Path,
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, Any]:
    """Build split statistics and leakage checks for a completed image-level split."""

    global_subtype_counts = _count_subtypes(pairs)
    subtype_keys = _subtype_keys(global_subtype_counts)
    assignment_by_image = {str(row["image_id"]): str(row["split"]) for row in assignments}

    split_stats = {split: _empty_split_stats(subtype_keys) for split in SPLIT_NAMES}
    split_image_ids: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    all_output_pair_ids: list[str] = []

    for split in SPLIT_NAMES:
        rows = list(split_rows[split])
        image_ids = {str(row["image_id"]) for row in rows}
        split_image_ids[split] = image_ids
        split_stats[split]["num_pairs"] = len(rows)
        split_stats[split]["num_images"] = len(image_ids)
        split_stats[split]["unique_image_ids"] = len(image_ids)
        subtype_counts = _count_subtypes(rows)
        split_stats[split]["subtype_counts"] = {subtype: int(subtype_counts.get(subtype, 0)) for subtype in subtype_keys}
        split_stats[split]["template_counts"] = dict(
            sorted(Counter(extract_template_id(row) for row in rows).items())
        )
        all_output_pair_ids.extend(str(row["pair_id"]) for row in rows)

    input_pair_ids = [str(row["pair_id"]) for row in pairs]
    output_pair_id_counts = Counter(all_output_pair_ids)
    input_pair_id_counts = Counter(input_pair_ids)
    missing_pair_ids = [
        pair_id
        for pair_id, count in input_pair_id_counts.items()
        if output_pair_id_counts.get(pair_id, 0) < count
    ]
    duplicate_pair_count = sum(max(count - 1, 0) for count in output_pair_id_counts.values())

    leakage_checks = {
        "image_overlap_train_val": len(split_image_ids["train"] & split_image_ids["val"]),
        "image_overlap_train_test": len(split_image_ids["train"] & split_image_ids["test"]),
        "image_overlap_val_test": len(split_image_ids["val"] & split_image_ids["test"]),
        "num_duplicate_pair_ids": duplicate_pair_count,
        "num_missing_pair_ids": len(missing_pair_ids),
    }

    return {
        "input_pairs": str(input_pairs_path),
        "seed": int(seed),
        "ratios": {split: float(ratios[split]) for split in SPLIT_NAMES},
        "total_pairs": len(pairs),
        "total_images": len(assignment_by_image),
        "subtype_counts": {subtype: int(global_subtype_counts.get(subtype, 0)) for subtype in subtype_keys},
        "splits": split_stats,
        "leakage_checks": leakage_checks,
    }


def build_dataset_manifest(
    split_name: str,
    input_pairs_path: str | Path,
    input_pairs_sha256: str,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a reproducibility manifest for a completed pair-bank split."""

    return {
        "split_name": str(split_name),
        "input_pairs_path": str(input_pairs_path),
        "input_pairs_sha256": input_pairs_sha256,
        "num_pairs": int(stats["total_pairs"]),
        "num_images": int(stats["total_images"]),
        "subtype_counts": dict(stats.get("subtype_counts", {})),
        "split_files": {
            "train": "pairs_train.jsonl",
            "val": "pairs_val.jsonl",
            "test": "pairs_test.jsonl",
        },
        "assignment_file": "split_assignments.jsonl",
        "stats_file": "split_stats.json",
        "notes": [
            "image-level split",
            "same image_id never crosses splits",
            "designed for real LVLM activation extraction",
        ],
    }


def make_pair_splits(
    pairs_path: str | Path,
    out_dir: str | Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    split_name: str = "v0_mini_seed42",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create image-level split files, stats, and a manifest from a pair JSONL file."""

    resolved_pairs_path = resolve_project_path(pairs_path)
    resolved_out_dir = resolve_project_path(out_dir)
    ratios = validate_ratios(train_ratio, val_ratio, test_ratio)

    if resolved_out_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {resolved_out_dir}. Pass --overwrite to replace files.")
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_jsonl(resolved_pairs_path)
    validate_pair_rows(pairs)
    image_groups = group_pairs_by_image(pairs)
    assignments = assign_image_splits(image_groups=image_groups, ratios=ratios, seed=seed)
    split_by_image = {str(row["image_id"]): str(row["split"]) for row in assignments}
    split_rows = {
        split: [row for row in pairs if split_by_image[str(row["image_id"])] == split]
        for split in SPLIT_NAMES
    }

    split_stats = build_split_stats(
        pairs=pairs,
        split_rows=split_rows,
        assignments=assignments,
        input_pairs_path=resolved_pairs_path,
        ratios=ratios,
        seed=seed,
    )
    manifest = build_dataset_manifest(
        split_name=split_name,
        input_pairs_path=resolved_pairs_path,
        input_pairs_sha256=sha256_file(resolved_pairs_path),
        stats=split_stats,
    )

    output_paths = {
        "train": resolved_out_dir / "pairs_train.jsonl",
        "val": resolved_out_dir / "pairs_val.jsonl",
        "test": resolved_out_dir / "pairs_test.jsonl",
        "assignments": resolved_out_dir / "split_assignments.jsonl",
        "stats": resolved_out_dir / "split_stats.json",
        "manifest": resolved_out_dir / "dataset_manifest.json",
    }
    write_jsonl(output_paths["train"], split_rows["train"])
    write_jsonl(output_paths["val"], split_rows["val"])
    write_jsonl(output_paths["test"], split_rows["test"])
    write_jsonl(output_paths["assignments"], assignments)
    write_json(output_paths["stats"], split_stats)
    write_json(output_paths["manifest"], manifest)

    return {
        "output_dir": resolved_out_dir,
        "output_paths": output_paths,
        "assignments": assignments,
        "stats": split_stats,
        "manifest": manifest,
    }


def main() -> int:
    """Run image-level pair split creation from command-line arguments."""

    args = parse_args()
    try:
        result = make_pair_splits(
            pairs_path=args.pairs,
            out_dir=args.out_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            split_name=args.split_name,
            overwrite=bool(args.overwrite),
        )
    except (FileExistsError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    stats = result["stats"]
    print(f"Wrote image-level pair splits to {result['output_dir']}")
    for split in SPLIT_NAMES:
        split_stats = stats["splits"][split]
        subtype_counts = ", ".join(
            f"{subtype}={count}" for subtype, count in split_stats["subtype_counts"].items()
        )
        print(
            f"{split}: num_pairs={split_stats['num_pairs']}, "
            f"num_images={split_stats['num_images']}, {subtype_counts}"
        )
    leakage_summary = ", ".join(f"{key}={value}" for key, value in stats["leakage_checks"].items())
    print(f"Leakage checks: {leakage_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
