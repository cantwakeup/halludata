"""Extract AFTER-template visual/text activations.

Each row is encoded as:

- z_visual: image + visual_prompt
- z_text: trusted_prompt, text-only by default

The steering direction built downstream is z_text - z_visual.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import sha256_file, tensor_shape, utc_now_iso, write_json, write_jsonl
from expert_data.image_resolver import CocoImageResolver
from expert_data.io_utils import read_jsonl
from expert_data.model_adapter import LlavaActivationAdapter, MockActivationAdapter


REQUIRED_FIELDS = (
    "id",
    "image_id",
    "question",
    "visual_prompt",
    "trusted_factual_text",
    "trusted_prompt",
    "hallucination_type",
    "subtype",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for AFTER-template activation extraction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="", help="Alias for --model-id.")
    parser.add_argument("--model-id", default="", help="HF model id or local path.")
    parser.add_argument("--pair-file", required=True, help="AFTER-template pair JSONL.")
    parser.add_argument("--image-root", default="", help="Image directory for relative image filenames.")
    parser.add_argument("--instances-json", default="", help="Optional instances json for image id resolution.")
    parser.add_argument("--output", required=True, help="Output .pt cache path.")
    parser.add_argument("--metadata-output", default="", help="Output metadata JSONL path.")
    parser.add_argument("--adapter", choices=["mock", "llava"], default="llava")
    parser.add_argument("--layers", default="all", help="Currently records all model layers; accepted for provenance.")
    parser.add_argument("--position-mode", choices=["last_token"], default="last_token")
    parser.add_argument("--trusted-input-mode", choices=["text_only", "image_with_fact"], default="text_only")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved; v1 extraction runs one pair at a time.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--storage-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--split", default="")
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
    """Resolve optional project-relative paths, treating empty values as absent."""

    text = "" if raw_path is None else str(raw_path).strip()
    if not text:
        return None
    return resolve_project_path(text)


def require_torch() -> Any:
    """Import torch lazily for cache writing."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("extract_after_template_activations requires torch.") from exc


def torch_dtype(torch: Any, dtype_name: str) -> Any:
    """Resolve a storage dtype string to a torch dtype."""

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(dtype_name)]


def mock_activation_to_grid(activation: Mapping[str, Any], adapter: MockActivationAdapter) -> list[list[list[float]]]:
    """Convert mock layer-head vectors into a [layers, heads, head_dim] grid."""

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
    """Validate one AFTER-template pair row."""

    missing = [field for field in REQUIRED_FIELDS if row.get(field) in (None, "")]
    if missing:
        raise ValueError(f"row_index={row_index} missing field(s): {', '.join(missing)}")
    if str(row["hallucination_type"]) not in {"cat", "attr", "rel"}:
        raise ValueError(f"row_index={row_index} has unsupported hallucination_type={row['hallucination_type']}")


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
        raise FileNotFoundError(f"Could not resolve image for id={row.get('id')}")
    return resolver.resolve(row["image_id"])


def metadata_row(
    row_index: int,
    row: Mapping[str, Any],
    image_path: str,
    split: str,
    branch_meta: Mapping[str, Any],
    adapter: str,
    model_id: str,
    trusted_input_mode: str,
) -> dict[str, Any]:
    """Build one metadata row for an AFTER-template activation."""

    identifier = str(row.get("id") or row.get("pair_id"))
    return {
        "row_index": int(row_index),
        "id": identifier,
        "pair_id": identifier,
        "image_id": str(row["image_id"]),
        "image": str(row.get("image", "")),
        "image_path": image_path,
        "split": split or None,
        "hallucination_type": str(row["hallucination_type"]),
        "subtype": str(row["subtype"]),
        "objects": list(row.get("objects", [])),
        "question": str(row["question"]),
        "visual_prompt": str(row["visual_prompt"]),
        "trusted_factual_text": str(row["trusted_factual_text"]),
        "trusted_prompt": str(row["trusted_prompt"]),
        "factual_fact": str(row.get("factual_fact", "")),
        "source": str(row.get("source", "after_template_v1")),
        "label": str(row.get("label", "")),
        "object_a": str(row.get("object_a", "")),
        "object_b": str(row.get("object_b", "")),
        "bbox_a": list(row.get("bbox_a", [])),
        "bbox_b": list(row.get("bbox_b", [])),
        "true_relation": str(row.get("true_relation", "")),
        "queried_relation": str(row.get("queried_relation", "")),
        "template_variant": str(row.get("template_variant", "")),
        "target_token_index_visual": int(branch_meta["target_token_index_visual"]),
        "target_token_index_text": int(branch_meta["target_token_index_text"]),
        "num_layers": int(branch_meta["num_layers"]),
        "num_heads": int(branch_meta["num_heads"]),
        "head_dim": int(branch_meta["head_dim"]),
        "adapter": adapter,
        "model_id": model_id,
        "trusted_input_mode": trusted_input_mode,
    }


def mock_result(adapter: MockActivationAdapter, row: Mapping[str, Any]) -> dict[str, Any]:
    """Encode one row with the mock adapter."""

    identifier = str(row.get("id") or row.get("pair_id"))
    subtype = str(row["hallucination_type"])
    visual = adapter.encode_pair(
        image_id=str(row["image_id"]),
        question=str(row["visual_prompt"]),
        response="",
        pair_id=identifier,
        subtype=subtype,
        branch="neg",
    )
    text = adapter.encode_pair(
        image_id=str(row["image_id"]),
        question=str(row["trusted_prompt"]),
        response=str(row["trusted_factual_text"]),
        pair_id=identifier,
        subtype=subtype,
        branch="pos",
    )
    return {
        "z_visual": mock_activation_to_grid(visual, adapter),
        "z_text": mock_activation_to_grid(text, adapter),
        "meta": {
            "num_layers": int(adapter.num_layers),
            "num_heads": int(adapter.num_heads),
            "head_dim": int(adapter.vector_dim),
            "target_token_index_visual": -1,
            "target_token_index_text": -1,
        },
    }


def main() -> int:
    """Extract and save AFTER-template activations."""

    args = parse_args()
    try:
        torch = require_torch()
        if int(args.batch_size) != 1:
            raise ValueError("AFTER-template v1 extraction currently supports --batch-size 1 only")
        pair_path = resolve_project_path(args.pair_file)
        output_path = resolve_project_path(args.output)
        metadata_path = resolve_project_path(args.metadata_output) if str(args.metadata_output).strip() else output_path.with_suffix(".meta.jsonl")
        manifest_path = output_path.with_suffix(".manifest.json")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace.")

        rows = read_jsonl(pair_path)
        if int(args.max_samples) > 0:
            rows = rows[: int(args.max_samples)]
        for row_index, row in enumerate(rows):
            validate_row(row, row_index)

        image_root = resolve_optional_project_path(args.image_root)
        model_id = str(args.model_path or args.model_id or "llava-hf/llava-1.5-7b-hf")
        resolver = None
        adapter: Any
        if args.adapter == "mock":
            adapter = MockActivationAdapter()
            model_id = "mock"
        else:
            if image_root is None:
                raise ValueError("--image-root is required for --adapter llava")
            instances_json = resolve_optional_project_path(args.instances_json)
            resolver = CocoImageResolver(image_root=image_root, instances_json=instances_json) if instances_json else None
            adapter = LlavaActivationAdapter(
                model_id=model_id,
                device=args.device,
                compute_dtype=args.compute_dtype,
                storage_dtype=args.storage_dtype,
            )

        z_text_items: list[Any] = []
        z_visual_items: list[Any] = []
        metadata_rows: list[dict[str, Any]] = []
        first_shape: list[int] | None = None
        for row_index, row in enumerate(rows):
            image_path = "" if args.adapter == "mock" else resolve_image_path(row, image_root, resolver)
            result = mock_result(adapter, row) if args.adapter == "mock" else adapter.encode_prompt_pair(
                image_path=image_path,
                visual_prompt=str(row["visual_prompt"]),
                trusted_prompt=str(row["trusted_prompt"]),
                trusted_input_mode=str(args.trusted_input_mode),
                pair_id=str(row.get("id") or row.get("pair_id")),
                subtype=str(row["hallucination_type"]),
            )
            shape = tensor_shape(result["z_text"])
            if first_shape is None:
                first_shape = shape
            elif shape != first_shape:
                raise RuntimeError(f"Inconsistent activation shape: {shape} != {first_shape}")
            if tensor_shape(result["z_visual"]) != first_shape:
                raise RuntimeError(f"Visual/text activation shapes differ at row_index={row_index}")
            z_text_items.append(result["z_text"])
            z_visual_items.append(result["z_visual"])
            metadata_rows.append(
                metadata_row(
                    row_index,
                    row,
                    image_path,
                    args.split,
                    result["meta"],
                    args.adapter,
                    model_id,
                    str(args.trusted_input_mode),
                )
            )
            if int(args.progress_every) > 0 and (row_index + 1) % int(args.progress_every) == 0:
                print(f"[extract_after_template] processed {row_index + 1}/{len(rows)} rows")

        z_text = stack_activation_items(z_text_items, args.storage_dtype)
        z_visual = stack_activation_items(z_visual_items, args.storage_dtype)
        cache = {
            "pair_ids": [row["pair_id"] for row in metadata_rows],
            "row_indices": [row["row_index"] for row in metadata_rows],
            "image_ids": [row["image_id"] for row in metadata_rows],
            "hallucination_types": [row["hallucination_type"] for row in metadata_rows],
            "subtypes": [row["subtype"] for row in metadata_rows],
            "layers": list(range(int(first_shape[0]))) if first_shape else [],
            "metadata": metadata_rows,
            "z_text": z_text,
            "z_visual": z_visual,
            "z_pos": z_text,
            "z_neg": z_visual,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, output_path)
        write_jsonl(metadata_path, metadata_rows)
        manifest = {
            "source": "after_template_v1",
            "pair_file": str(pair_path),
            "pair_file_sha256": sha256_file(pair_path),
            "output": str(output_path),
            "metadata_output": str(metadata_path),
            "adapter": args.adapter,
            "model_id": model_id,
            "split": args.split or None,
            "layers": args.layers,
            "position_mode": args.position_mode,
            "trusted_input_mode": str(args.trusted_input_mode),
            "num_pairs": len(metadata_rows),
            "shape": [len(metadata_rows), *(first_shape or [0, 0, 0])],
            "dtype": str(args.storage_dtype),
            "created_at": utc_now_iso(),
            "notes": [
                "AFTER-template prompt-only extraction",
                "z_visual is image + visual_prompt last-token activation",
                "z_text is trusted_prompt last-token activation",
                "downstream direction is z_text - z_visual",
            ],
        }
        write_json(manifest_path, manifest)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote AFTER-template activations to {output_path}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
