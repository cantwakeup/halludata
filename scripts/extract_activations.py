"""Extract pos/neg activation caches from pair JSONL files."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import save_activation_cache, sha256_file, tensor_shape, utc_now_iso, write_jsonl
from expert_data.image_resolver import CocoImageResolver
from expert_data.io_utils import read_jsonl
from expert_data.model_adapter import LlavaActivationAdapter, MockActivationAdapter

VALID_SUBTYPES = {"cat", "cnt", "col", "rel"}
REQUIRED_PAIR_FIELDS = ("pair_id", "image_id", "subtype", "question", "response_pos", "response_neg")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for activation extraction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/outputs/pairs_balanced_v0.jsonl", help="Input pair JSONL path.")
    parser.add_argument("--out-dir", default="data/outputs/activations/llava_v15_7b/default", help="Output cache dir.")
    parser.add_argument("--adapter", choices=["mock", "llava"], default="mock", help="Activation adapter to use.")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="Hugging Face model ID.")
    parser.add_argument("--image-root", default="", help="COCO image root; required for real LLaVA extraction.")
    parser.add_argument("--instances-json", default="", help="Optional COCO instances JSON used to resolve file_name.")
    parser.add_argument("--split", default=None, help="Optional split label written into metadata.")
    parser.add_argument("--device", default="cuda:0", help="Torch device used by the real adapter.")
    parser.add_argument(
        "--compute-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Model compute dtype for real extraction.",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Activation storage dtype.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved; first version only supports batch size 1.")
    parser.add_argument("--max-samples", type=int, default=0, help="Process at most this many selected rows; 0 means all.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of extraction shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index in [0, num_shards).")
    parser.add_argument("--image-mode", choices=["real", "blank"], default="real", help="Image mode recorded in manifest.")
    parser.add_argument("--shuffle-image-seed", type=int, default=None, help="Reserved image-shuffle control seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output dir.")
    parser.add_argument("--dry-run", action="store_true", help="Print extraction plan without loading a model.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N processed rows.")
    parser.add_argument("--continue-on-error", action="store_true", help="Log row errors and continue instead of failing.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_optional_project_path(raw_path: str | Path | None) -> Path | None:
    """Resolve an optional path, treating empty strings as absent."""

    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    return resolve_project_path(text)


def validate_pair_row(row: Mapping[str, Any], row_index: int) -> None:
    """Validate one pair row before activation extraction."""

    missing_fields = [field for field in REQUIRED_PAIR_FIELDS if field not in row or row[field] in {None, ""}]
    if missing_fields:
        raise ValueError(f"row_index={row_index} missing required field(s): {', '.join(sorted(missing_fields))}")
    subtype = str(row["subtype"])
    if subtype not in VALID_SUBTYPES:
        raise ValueError(f"row_index={row_index} has unsupported subtype '{subtype}'.")


def select_rows_for_shard(
    rows: list[dict[str, Any]],
    num_shards: int,
    shard_index: int,
    max_samples: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Select original row indices for one modulo shard and optional smoke limit."""

    if int(num_shards) < 1:
        raise ValueError("--num-shards must be >= 1.")
    if int(shard_index) < 0 or int(shard_index) >= int(num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
    selected = [
        (row_index, row)
        for row_index, row in enumerate(rows)
        if row_index % int(num_shards) == int(shard_index)
    ]
    if int(max_samples) > 0:
        return selected[: int(max_samples)]
    return selected


def _import_torch_or_none() -> Any | None:
    """Import torch lazily for stacking tensors when available."""

    try:
        import torch

        return torch
    except ImportError:
        return None


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    """Resolve a dtype string to torch dtype."""

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(dtype_name)]


def mock_activation_to_grid(activation: Mapping[str, Any], adapter: MockActivationAdapter) -> list[list[list[float]]]:
    """Convert mock layer-head vectors into [layers, heads, head_dim] nested lists."""

    vectors = dict(activation.get("layer_head_vectors", {}))
    return [
        [
            [float(value) for value in vectors[f"l{layer}_h{head}"]]
            for head in range(adapter.num_heads)
        ]
        for layer in range(adapter.num_layers)
    ]


def stack_activation_items(items: list[Any], storage_dtype: str) -> Any:
    """Stack per-row activation grids into [N, L, H, D], using torch when possible."""

    if not items:
        return []
    torch = _import_torch_or_none()
    if torch is not None:
        if isinstance(items[0], torch.Tensor):
            return torch.stack(items, dim=0).to(_torch_dtype(torch, storage_dtype)).cpu()
        return torch.tensor(items, dtype=_torch_dtype(torch, storage_dtype))
    return items


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Create or validate the output directory before writing cache files."""

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite to replace cache files.")
    out_dir.mkdir(parents=True, exist_ok=True)


def build_mock_adapter_result(
    adapter: MockActivationAdapter,
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    """Encode one pair with the mock adapter while returning cache-compatible shapes."""

    pair_id = str(pair["pair_id"])
    image_id = str(pair["image_id"])
    subtype = str(pair["subtype"])
    question = str(pair["question"])
    pos_activation = adapter.encode_pair(
        image_id=image_id,
        question=question,
        response=str(pair["response_pos"]),
        pair_id=pair_id,
        subtype=subtype,
        branch="pos",
    )
    neg_activation = adapter.encode_pair(
        image_id=image_id,
        question=question,
        response=str(pair["response_neg"]),
        pair_id=pair_id,
        subtype=subtype,
        branch="neg",
    )
    return {
        "z_pos": mock_activation_to_grid(pos_activation, adapter),
        "z_neg": mock_activation_to_grid(neg_activation, adapter),
        "meta": {
            "num_layers": int(adapter.num_layers),
            "num_heads": int(adapter.num_heads),
            "head_dim": int(adapter.vector_dim),
            "target_token_index_pos": -1,
            "target_token_index_neg": -1,
        },
    }


def _metadata_row(
    row_index: int,
    pair: Mapping[str, Any],
    image_path: str,
    split: str | None,
    adapter_name: str,
    model_id: str,
    branch_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one metadata row for a pair activation cache."""

    return {
        "row_index": int(row_index),
        "pair_id": str(pair["pair_id"]),
        "image_id": str(pair["image_id"]),
        "subtype": str(pair["subtype"]),
        "split": split,
        "image_path": image_path,
        "question": str(pair["question"]),
        "response_pos": str(pair["response_pos"]),
        "response_neg": str(pair["response_neg"]),
        "target_token_index_pos": int(branch_meta["target_token_index_pos"]),
        "target_token_index_neg": int(branch_meta["target_token_index_neg"]),
        "num_layers": int(branch_meta["num_layers"]),
        "num_heads": int(branch_meta["num_heads"]),
        "head_dim": int(branch_meta["head_dim"]),
        "adapter": adapter_name,
        "model_id": model_id,
    }


def _shape_meta(z_pos: Any) -> tuple[int, int, int]:
    """Extract [layers, heads, head_dim] metadata from one per-row activation."""

    shape = tensor_shape(z_pos)
    if len(shape) != 3:
        raise RuntimeError(f"Expected per-row activation shape [L,H,D], got {shape}.")
    return int(shape[0]), int(shape[1]), int(shape[2])


def extract_activations(args: argparse.Namespace) -> dict[str, Any]:
    """Run activation extraction and save one cache directory."""

    if int(args.batch_size) != 1:
        raise ValueError("This first extraction pipeline only supports --batch-size 1.")
    pairs_path = resolve_project_path(args.pairs)
    out_dir = resolve_project_path(args.out_dir)
    rows = read_jsonl(pairs_path)
    for row_index, row in enumerate(rows):
        validate_pair_row(row, row_index)
    selected_rows = select_rows_for_shard(rows, args.num_shards, args.shard_index, args.max_samples)

    subtype_counts = Counter(str(row["subtype"]) for _, row in selected_rows)
    if args.dry_run:
        return {
            "pairs_path": pairs_path,
            "out_dir": out_dir,
            "selected_rows": selected_rows,
            "subtype_counts": subtype_counts,
        }

    ensure_output_dir(out_dir, bool(args.overwrite))

    if args.adapter == "mock":
        adapter: Any = MockActivationAdapter()
        resolver = None
    else:
        if args.image_mode != "real":
            raise ValueError("--image-mode blank is reserved for future controls; use --image-mode real for LLaVA.")
        image_root = resolve_optional_project_path(args.image_root)
        if image_root is None:
            raise ValueError("--image-root is required when --adapter llava.")
        instances_json = resolve_optional_project_path(args.instances_json)
        resolver = CocoImageResolver(image_root=image_root, instances_json=instances_json)
        adapter = LlavaActivationAdapter(
            model_id=args.model_id,
            device=args.device,
            compute_dtype=args.compute_dtype,
            storage_dtype=args.storage_dtype,
        )

    z_pos_items: list[Any] = []
    z_neg_items: list[Any] = []
    metadata_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    first_shape: tuple[int, int, int] | None = None

    for processed_count, (row_index, pair) in enumerate(selected_rows, start=1):
        try:
            if args.adapter == "mock":
                image_path = ""
                result = build_mock_adapter_result(adapter, pair)
            else:
                image_path = resolver.resolve(pair["image_id"])
                result = adapter.encode_pair(
                    image_path=image_path,
                    question=str(pair["question"]),
                    response_pos=str(pair["response_pos"]),
                    response_neg=str(pair["response_neg"]),
                    pair_id=str(pair["pair_id"]),
                    subtype=str(pair["subtype"]),
                )
            shape = _shape_meta(result["z_pos"])
            if first_shape is None:
                first_shape = shape
            elif shape != first_shape:
                raise RuntimeError(f"Inconsistent activation shape at row_index={row_index}: {shape} != {first_shape}")
            z_pos_items.append(result["z_pos"])
            z_neg_items.append(result["z_neg"])
            metadata_rows.append(
                _metadata_row(
                    row_index=row_index,
                    pair=pair,
                    image_path=image_path,
                    split=args.split,
                    adapter_name=args.adapter,
                    model_id=args.model_id if args.adapter == "llava" else "mock",
                    branch_meta=result["meta"],
                )
            )
            if processed_count % max(int(args.progress_every), 1) == 0:
                print(f"[extract_activations] processed {processed_count}/{len(selected_rows)} selected rows")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            errors.append({"row_index": int(row_index), "pair_id": pair.get("pair_id"), "error": str(exc)})

    if errors:
        write_jsonl(out_dir / "error_log.jsonl", errors)

    z_pos = stack_activation_items(z_pos_items, args.storage_dtype)
    z_neg = stack_activation_items(z_neg_items, args.storage_dtype)
    cache_dict = {
        "pair_ids": [row["pair_id"] for row in metadata_rows],
        "row_indices": [row["row_index"] for row in metadata_rows],
        "image_ids": [row["image_id"] for row in metadata_rows],
        "subtypes": [row["subtype"] for row in metadata_rows],
        "z_pos": z_pos,
        "z_neg": z_neg,
    }

    num_layers, num_heads, head_dim = first_shape if first_shape is not None else (0, 0, 0)
    manifest = {
        "adapter": args.adapter,
        "model_id": args.model_id if args.adapter == "llava" else "mock",
        "pairs_path": str(pairs_path),
        "pairs_sha256": sha256_file(pairs_path),
        "image_root": str(resolve_optional_project_path(args.image_root) or ""),
        "instances_json": str(resolve_optional_project_path(args.instances_json) or ""),
        "split": args.split,
        "num_pairs": len(metadata_rows),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "dtype": str(args.storage_dtype),
        "image_mode": str(args.image_mode),
        "shard_index": int(args.shard_index) if int(args.num_shards) > 1 else None,
        "num_shards": int(args.num_shards) if int(args.num_shards) > 1 else None,
        "created_at": utc_now_iso(),
        "shuffle_image_seed": args.shuffle_image_seed,
        "notes": [
            "teacher-forced forward",
            "last answer token",
            "o_proj pre-hook activations",
        ],
    }
    output_paths = save_activation_cache(out_dir, cache_dict, metadata_rows, manifest)
    return {
        "out_dir": out_dir,
        "output_paths": output_paths,
        "manifest": manifest,
        "metadata_rows": metadata_rows,
        "errors": errors,
    }


def main() -> int:
    """Run activation extraction from command-line arguments."""

    args = parse_args()
    try:
        result = extract_activations(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        selected_rows = result["selected_rows"]
        subtype_counts = ", ".join(f"{key}={value}" for key, value in sorted(result["subtype_counts"].items()))
        print(f"Dry run: selected_rows={len(selected_rows)} from {result['pairs_path']}")
        print(f"Would write to {result['out_dir']}")
        print(f"Subtype counts: {subtype_counts}")
        return 0

    manifest = result["manifest"]
    print(f"Wrote activation cache to {result['out_dir']}")
    print(
        "Summary: "
        f"adapter={manifest['adapter']}, num_pairs={manifest['num_pairs']}, "
        f"num_layers={manifest['num_layers']}, num_heads={manifest['num_heads']}, "
        f"head_dim={manifest['head_dim']}, dtype={manifest['dtype']}"
    )
    if result["errors"]:
        print(f"Logged {len(result['errors'])} row errors to error_log.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
