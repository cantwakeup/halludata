"""Extract teacher-forced activations for cat truthfulness pair banks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import sha256_file, tensor_shape, utc_now_iso, write_json, write_jsonl
from expert_data.image_resolver import CocoImageResolver
from expert_data.io_utils import read_jsonl
from expert_data.model_adapter import LlavaActivationAdapter, MockActivationAdapter


REQUIRED_FIELDS = ("pair_id", "image_id", "subtype", "question", "factual_answer", "counterfactual_answer")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for cat truthfulness activation extraction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="Cat truthfulness pair JSONL.")
    parser.add_argument("--output", required=True, help="Output .pt activation cache path.")
    parser.add_argument("--adapter", choices=["mock", "llava"], default="llava")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--instances-json", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--storage-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_optional_project_path(raw_path: str | Path | None) -> Path | None:
    """Resolve optional project-relative paths and treat empty strings as absent."""

    text = "" if raw_path is None else str(raw_path).strip()
    if not text:
        return None
    return resolve_project_path(text)


def require_torch() -> Any:
    """Import torch lazily for direct .pt cache writing."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("cat truthfulness activation extraction requires torch.") from exc


def torch_dtype(torch: Any, dtype_name: str) -> Any:
    """Resolve storage dtype name to a torch dtype."""

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(dtype_name)]


def mock_activation_to_grid(activation: Mapping[str, Any], adapter: MockActivationAdapter) -> list[list[list[float]]]:
    """Convert mock layer-head vectors into a [layers, heads, head_dim] grid.

    This helper intentionally lives in this script instead of importing from
    ``scripts.extract_activations`` because some cloud environments install an
    unrelated top-level ``scripts`` package, for example ROS tooling.
    """

    vectors = dict(activation.get("layer_head_vectors", {}))
    return [
        [
            [float(value) for value in vectors[f"l{layer}_h{head}"]]
            for head in range(adapter.num_heads)
        ]
        for layer in range(adapter.num_layers)
    ]


def stack_activation_items(items: list[Any], storage_dtype: str) -> Any:
    """Stack per-row activation grids into a [N, L, H, D] torch tensor."""

    if not items:
        return []
    torch = require_torch()
    dtype = torch_dtype(torch, storage_dtype)
    if isinstance(items[0], torch.Tensor):
        return torch.stack(items, dim=0).to(dtype).cpu()
    return torch.tensor(items, dtype=dtype)


def validate_row(row: Mapping[str, Any], row_index: int) -> None:
    """Validate one cat truthfulness pair row."""

    missing = [field for field in REQUIRED_FIELDS if row.get(field) in (None, "")]
    if missing:
        raise ValueError(f"row_index={row_index} missing field(s): {', '.join(missing)}")
    if str(row["subtype"]) not in {"cat_truth_present", "cat_truth_absent"}:
        raise ValueError(f"row_index={row_index} has unsupported subtype {row['subtype']}")


def resolve_image_path(row: Mapping[str, Any], image_root: Path | None, resolver: CocoImageResolver | None) -> str:
    """Resolve the image path for one pair row."""

    image = str(row.get("image") or row.get("image_path") or "").strip()
    if image:
        path = Path(image)
        if not path.is_absolute() and image_root is not None:
            path = image_root / path
        if path.exists():
            return str(path)
    if resolver is None:
        raise FileNotFoundError(f"Could not resolve image for pair_id={row.get('pair_id')}")
    return resolver.resolve(row["image_id"])


def metadata_row(
    row_index: int,
    row: Mapping[str, Any],
    image_path: str,
    split: str,
    branch_meta: Mapping[str, Any],
    adapter: str,
    model_id: str,
) -> dict[str, Any]:
    """Build one metadata row for a cat truthfulness activation."""

    return {
        "row_index": int(row_index),
        "pair_id": str(row["pair_id"]),
        "image_id": str(row["image_id"]),
        "image": str(row.get("image", "")),
        "image_path": image_path,
        "split": split or None,
        "subtype": str(row["subtype"]),
        "label": str(row.get("label", "")),
        "object": str(row.get("object", "")),
        "question": str(row["question"]),
        "factual_answer": str(row["factual_answer"]),
        "counterfactual_answer": str(row["counterfactual_answer"]),
        "target_token_index_factual": int(branch_meta["target_token_index_pos"]),
        "target_token_index_counterfactual": int(branch_meta["target_token_index_neg"]),
        "num_layers": int(branch_meta["num_layers"]),
        "num_heads": int(branch_meta["num_heads"]),
        "head_dim": int(branch_meta["head_dim"]),
        "adapter": adapter,
        "model_id": model_id,
    }


def mock_result(adapter: MockActivationAdapter, row: Mapping[str, Any]) -> dict[str, Any]:
    """Encode one row with the mock adapter using factual/counterfactual branches."""

    pos = adapter.encode_pair(
        image_id=str(row["image_id"]),
        question=str(row["question"]),
        response=str(row["factual_answer"]),
        pair_id=str(row["pair_id"]),
        subtype="cat",
        branch="pos",
    )
    neg = adapter.encode_pair(
        image_id=str(row["image_id"]),
        question=str(row["question"]),
        response=str(row["counterfactual_answer"]),
        pair_id=str(row["pair_id"]),
        subtype="cat",
        branch="neg",
    )
    return {
        "z_pos": mock_activation_to_grid(pos, adapter),
        "z_neg": mock_activation_to_grid(neg, adapter),
        "meta": {
            "num_layers": int(adapter.num_layers),
            "num_heads": int(adapter.num_heads),
            "head_dim": int(adapter.vector_dim),
            "target_token_index_pos": -1,
            "target_token_index_neg": -1,
        },
    }


def main() -> int:
    """Extract and save cat truthfulness activations."""

    args = parse_args()
    try:
        torch = require_torch()
        pairs_path = resolve_project_path(args.pairs)
        output_path = resolve_project_path(args.output)
        meta_path = output_path.with_suffix(".meta.jsonl")
        manifest_path = output_path.with_suffix(".manifest.json")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace.")
        rows = read_jsonl(pairs_path)
        if int(args.max_samples) > 0:
            rows = rows[: int(args.max_samples)]
        for row_index, row in enumerate(rows):
            validate_row(row, row_index)

        image_root = resolve_optional_project_path(args.image_root)
        resolver = None
        adapter: Any
        if args.adapter == "mock":
            adapter = MockActivationAdapter()
        else:
            if image_root is None:
                raise ValueError("--image-root is required for --adapter llava")
            instances_json = resolve_optional_project_path(args.instances_json)
            resolver = CocoImageResolver(image_root=image_root, instances_json=instances_json) if instances_json else None
            adapter = LlavaActivationAdapter(
                model_id=args.model_id,
                device=args.device,
                compute_dtype=args.compute_dtype,
                storage_dtype=args.storage_dtype,
            )

        z_factual_items: list[Any] = []
        z_counter_items: list[Any] = []
        metadata_rows: list[dict[str, Any]] = []
        first_shape: list[int] | None = None
        for row_index, row in enumerate(rows):
            image_path = "" if args.adapter == "mock" else resolve_image_path(row, image_root, resolver)
            if args.adapter == "mock":
                result = mock_result(adapter, row)
            else:
                result = adapter.encode_pair(
                    image_path=image_path,
                    question=str(row["question"]),
                    response_pos=str(row["factual_answer"]),
                    response_neg=str(row["counterfactual_answer"]),
                    pair_id=str(row["pair_id"]),
                    subtype=str(row["subtype"]),
                )
            shape = tensor_shape(result["z_pos"])
            if first_shape is None:
                first_shape = shape
            elif shape != first_shape:
                raise RuntimeError(f"Inconsistent activation shape: {shape} != {first_shape}")
            z_factual_items.append(result["z_pos"])
            z_counter_items.append(result["z_neg"])
            metadata_rows.append(metadata_row(row_index, row, image_path, args.split, result["meta"], args.adapter, args.model_id))
            if int(args.progress_every) > 0 and (row_index + 1) % int(args.progress_every) == 0:
                print(f"[extract_cat_truthfulness] processed {row_index + 1}/{len(rows)} rows")

        z_factual = stack_activation_items(z_factual_items, args.storage_dtype)
        z_counter = stack_activation_items(z_counter_items, args.storage_dtype)
        cache = {
            "pair_ids": [row["pair_id"] for row in metadata_rows],
            "row_indices": [row["row_index"] for row in metadata_rows],
            "image_ids": [row["image_id"] for row in metadata_rows],
            "labels": [row["label"] for row in metadata_rows],
            "objects": [row["object"] for row in metadata_rows],
            "subtypes": [row["subtype"] for row in metadata_rows],
            "z_factual": z_factual,
            "z_counterfactual": z_counter,
            "z_pos": z_factual,
            "z_neg": z_counter,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, output_path)
        write_jsonl(meta_path, metadata_rows)
        manifest = {
            "pairs_path": str(pairs_path),
            "pairs_sha256": sha256_file(pairs_path),
            "output": str(output_path),
            "metadata": str(meta_path),
            "adapter": args.adapter,
            "model_id": args.model_id if args.adapter == "llava" else "mock",
            "split": args.split or None,
            "num_pairs": len(metadata_rows),
            "shape": [len(metadata_rows), *(first_shape or [0, 0, 0])],
            "dtype": str(args.storage_dtype),
            "created_at": utc_now_iso(),
            "notes": [
                "teacher-forced factual vs counterfactual answers",
                "answer last token extraction",
                "z_factual - z_counterfactual is the cat truthfulness direction source",
            ],
        }
        write_json(manifest_path, manifest)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote cat truthfulness activations to {output_path}")
    print(f"Wrote metadata to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
