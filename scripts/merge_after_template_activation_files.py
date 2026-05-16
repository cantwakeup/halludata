"""Merge AFTER-template activation .pt files into one cache.

This merger is for direct cache files produced by
``extract_after_template_activations.py`` or
``extract_after_template_activations_official_llava.py``. It preserves the
standard downstream schema: z_text/z_visual plus z_pos/z_neg aliases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import tensor_shape, utc_now_iso, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-files", nargs="+", required=True, help="Shard .pt files to merge.")
    parser.add_argument("--output", required=True, help="Merged output .pt path.")
    parser.add_argument("--metadata-output", default="", help="Merged metadata JSONL path.")
    parser.add_argument("--manifest-output", default="", help="Merged manifest JSON path.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_torch() -> Any:
    """Import torch lazily."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("merge_after_template_activation_files requires torch.") from exc


def load_cache(path: Path) -> dict[str, Any]:
    """Load one torch cache file."""

    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return payload


def row_item(batch_value: Any, row_position: int) -> Any:
    """Return one row from a tensor-like or list-backed batch."""

    return batch_value[row_position]


def stack_rows(items: list[Any]) -> Any:
    """Stack row tensors into a batch."""

    if not items:
        return []
    torch = require_torch()
    if isinstance(items[0], torch.Tensor):
        return torch.stack(items, dim=0).cpu()
    return torch.tensor(items)


def merge_files(activation_files: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Merge shards and return cache, metadata rows, and manifest fields."""

    merged_rows: list[dict[str, Any]] = []
    seen_row_indices: set[int] = set()
    seen_pair_ids: set[str] = set()
    expected_lhd: list[int] | None = None
    source_manifests: list[str] = []

    for shard_path in activation_files:
        cache = load_cache(shard_path)
        metadata_rows = list(cache.get("metadata") or [])
        z_text = cache.get("z_text", cache.get("z_pos"))
        z_visual = cache.get("z_visual", cache.get("z_neg"))
        if z_text is None or z_visual is None:
            raise ValueError(f"{shard_path} must contain z_text/z_visual or z_pos/z_neg")
        shape_text = tensor_shape(z_text)
        shape_visual = tensor_shape(z_visual)
        if len(shape_text) != 4 or len(shape_visual) != 4:
            raise ValueError(f"{shard_path} must contain [N,L,H,D] activations, got {shape_text} and {shape_visual}")
        if shape_text != shape_visual:
            raise ValueError(f"{shard_path} has mismatched shapes: {shape_text} vs {shape_visual}")
        if len(metadata_rows) != int(shape_text[0]):
            raise ValueError(f"{shard_path} metadata row count does not match activation rows")
        if expected_lhd is None:
            expected_lhd = shape_text[1:]
        elif shape_text[1:] != expected_lhd:
            raise ValueError(f"{shard_path} has L/H/D {shape_text[1:]}, expected {expected_lhd}")

        manifest_path = shard_path.with_suffix(".manifest.json")
        if manifest_path.exists():
            source_manifests.append(str(manifest_path))

        for row_position, metadata in enumerate(metadata_rows):
            row_index = int(metadata["row_index"])
            pair_id = str(metadata["pair_id"])
            if row_index in seen_row_indices:
                raise ValueError(f"Duplicate row_index while merging: {row_index}")
            if pair_id in seen_pair_ids:
                raise ValueError(f"Duplicate pair_id while merging: {pair_id}")
            seen_row_indices.add(row_index)
            seen_pair_ids.add(pair_id)
            merged_rows.append(
                {
                    "row_index": row_index,
                    "metadata": dict(metadata),
                    "z_text": row_item(z_text, row_position),
                    "z_visual": row_item(z_visual, row_position),
                }
            )

    merged_rows.sort(key=lambda row: int(row["row_index"]))
    metadata_rows = [dict(row["metadata"]) for row in merged_rows]
    z_text = stack_rows([row["z_text"] for row in merged_rows])
    z_visual = stack_rows([row["z_visual"] for row in merged_rows])
    cache = {
        "pair_ids": [str(row["pair_id"]) for row in metadata_rows],
        "row_indices": [int(row["row_index"]) for row in metadata_rows],
        "image_ids": [str(row["image_id"]) for row in metadata_rows],
        "hallucination_types": [str(row["hallucination_type"]) for row in metadata_rows],
        "subtypes": [str(row["subtype"]) for row in metadata_rows],
        "layers": list(range(int(expected_lhd[0]))) if expected_lhd else [],
        "metadata": metadata_rows,
        "z_text": z_text,
        "z_visual": z_visual,
        "z_pos": z_text,
        "z_neg": z_visual,
    }
    manifest = {
        "source": "merged_after_template_activation_files",
        "activation_files": [str(path) for path in activation_files],
        "source_manifests": source_manifests,
        "num_pairs": len(metadata_rows),
        "shape": tensor_shape(z_text),
        "created_at": utc_now_iso(),
        "notes": [
            "merged direct AFTER-template activation .pt files",
            "sorted by original metadata row_index",
            "z_pos/z_neg aliases mirror z_text/z_visual",
        ],
    }
    return cache, metadata_rows, manifest


def main() -> int:
    """Merge activation files from the command line."""

    args = parse_args()
    try:
        torch = require_torch()
        activation_files = [resolve_project_path(path) for path in args.activation_files]
        output_path = resolve_project_path(args.output)
        metadata_path = (
            resolve_project_path(args.metadata_output)
            if str(args.metadata_output).strip()
            else output_path.with_suffix(".meta.jsonl")
        )
        manifest_path = (
            resolve_project_path(args.manifest_output)
            if str(args.manifest_output).strip()
            else output_path.with_suffix(".manifest.json")
        )
        for path in activation_files:
            if not path.exists():
                raise FileNotFoundError(f"Missing activation shard: {path}")
        if (output_path.exists() or metadata_path.exists() or manifest_path.exists()) and not args.overwrite:
            raise FileExistsError("Merged outputs exist. Pass --overwrite to replace them.")

        cache, metadata_rows, manifest = merge_files(activation_files)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, output_path)
        write_jsonl(metadata_path, metadata_rows)
        manifest.update(
            {
                "output": str(output_path),
                "metadata_output": str(metadata_path),
                "manifest_output": str(manifest_path),
            }
        )
        write_json(manifest_path, manifest)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote merged activation cache to {output_path}")
    print(f"Wrote merged metadata to {metadata_path}")
    print(f"Wrote merged manifest to {manifest_path}")
    print(f"num_pairs={manifest['num_pairs']} shape={manifest['shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
