"""Merge activation-cache shards into one row-index ordered cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import load_activation_cache, save_activation_cache, tensor_shape, utc_now_iso


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for shard merging."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", nargs="+", required=True, help="Shard cache directories to merge.")
    parser.add_argument("--out-dir", required=True, help="Merged activation-cache output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty out-dir.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Create or validate the merged output directory."""

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite to replace cache files.")
    out_dir.mkdir(parents=True, exist_ok=True)


def _import_torch_or_none() -> Any | None:
    """Import torch lazily for tensor concatenation when available."""

    try:
        import torch

        return torch
    except ImportError:
        return None


def _stack_rows(items: list[Any]) -> Any:
    """Stack sorted per-row tensors or nested lists into [N, L, H, D]."""

    if not items:
        return []
    torch = _import_torch_or_none()
    if torch is not None and isinstance(items[0], torch.Tensor):
        return torch.stack(items, dim=0).cpu()
    return items


def _row_item(batch_value: Any, row_position: int) -> Any:
    """Return one row from a tensor-like or list-backed [N, L, H, D] batch."""

    return batch_value[row_position]


def merge_shards(shard_dirs: list[str | Path], out_dir: str | Path, overwrite: bool = False) -> dict[str, Any]:
    """Merge activation shards, checking duplicate IDs and shape consistency."""

    resolved_shards = [resolve_project_path(path) for path in shard_dirs]
    resolved_out_dir = resolve_project_path(out_dir)
    ensure_output_dir(resolved_out_dir, overwrite)

    merged_rows: list[dict[str, Any]] = []
    seen_row_indices: set[int] = set()
    seen_pair_ids: set[str] = set()
    expected_lhd: list[int] | None = None
    first_manifest: dict[str, Any] = {}

    for shard_dir in resolved_shards:
        payload = load_activation_cache(shard_dir)
        activations = payload["activations"]
        metadata_rows = payload["metadata"]
        manifest = payload["manifest"]
        if not first_manifest:
            first_manifest = dict(manifest)

        z_pos = activations["z_pos"]
        z_neg = activations["z_neg"]
        shape_pos = tensor_shape(z_pos)
        shape_neg = tensor_shape(z_neg)
        if len(shape_pos) != 4 or len(shape_neg) != 4:
            raise ValueError(f"Shard {shard_dir} must contain [N,L,H,D] z_pos/z_neg tensors.")
        if shape_pos[1:] != shape_neg[1:]:
            raise ValueError(f"Shard {shard_dir} has mismatched z_pos/z_neg shapes: {shape_pos} vs {shape_neg}")
        if expected_lhd is None:
            expected_lhd = shape_pos[1:]
        elif shape_pos[1:] != expected_lhd:
            raise ValueError(f"Shard {shard_dir} has shape {shape_pos[1:]}, expected {expected_lhd}")
        if shape_pos[0] != len(metadata_rows):
            raise ValueError(f"Shard {shard_dir} metadata rows do not match activation rows.")

        for row_position, metadata in enumerate(metadata_rows):
            row_index = int(metadata["row_index"])
            pair_id = str(metadata["pair_id"])
            if row_index in seen_row_indices:
                raise ValueError(f"Duplicate row_index while merging shards: {row_index}")
            if pair_id in seen_pair_ids:
                raise ValueError(f"Duplicate pair_id while merging shards: {pair_id}")
            seen_row_indices.add(row_index)
            seen_pair_ids.add(pair_id)
            merged_rows.append(
                {
                    "row_index": row_index,
                    "metadata": metadata,
                    "z_pos": _row_item(z_pos, row_position),
                    "z_neg": _row_item(z_neg, row_position),
                }
            )

    merged_rows.sort(key=lambda row: int(row["row_index"]))
    metadata_rows = [dict(row["metadata"]) for row in merged_rows]
    z_pos = _stack_rows([row["z_pos"] for row in merged_rows])
    z_neg = _stack_rows([row["z_neg"] for row in merged_rows])
    cache_dict = {
        "pair_ids": [str(row["pair_id"]) for row in metadata_rows],
        "row_indices": [int(row["row_index"]) for row in metadata_rows],
        "image_ids": [str(row["image_id"]) for row in metadata_rows],
        "subtypes": [str(row["subtype"]) for row in metadata_rows],
        "z_pos": z_pos,
        "z_neg": z_neg,
    }

    num_layers, num_heads, head_dim = expected_lhd or [0, 0, 0]
    manifest = {
        "adapter": first_manifest.get("adapter"),
        "model_id": first_manifest.get("model_id"),
        "pairs_path": first_manifest.get("pairs_path"),
        "pairs_sha256": first_manifest.get("pairs_sha256"),
        "image_root": first_manifest.get("image_root"),
        "instances_json": first_manifest.get("instances_json"),
        "split": first_manifest.get("split"),
        "num_pairs": len(metadata_rows),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "dtype": first_manifest.get("dtype"),
        "image_mode": first_manifest.get("image_mode"),
        "shard_index": None,
        "num_shards": None,
        "source_shards": [str(path) for path in resolved_shards],
        "created_at": utc_now_iso(),
        "notes": [
            "merged activation shards",
            "teacher-forced forward",
            "last answer token",
            "o_proj pre-hook activations",
        ],
    }
    output_paths = save_activation_cache(resolved_out_dir, cache_dict, metadata_rows, manifest)
    return {"out_dir": resolved_out_dir, "output_paths": output_paths, "manifest": manifest}


def main() -> int:
    """Run shard merging from command-line arguments."""

    args = parse_args()
    try:
        result = merge_shards(args.shard_dirs, args.out_dir, overwrite=bool(args.overwrite))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    manifest = result["manifest"]
    print(f"Wrote merged activation cache to {result['out_dir']}")
    print(
        "Summary: "
        f"num_pairs={manifest['num_pairs']}, num_layers={manifest['num_layers']}, "
        f"num_heads={manifest['num_heads']}, head_dim={manifest['head_dim']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
