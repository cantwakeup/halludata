#!/usr/bin/env python3
"""Extract official-LLaVA activations for subtype minimal-pair rows.

Each row is encoded through three branches:

- z_visual: image + visual_prompt
- z_fact_text: trusted_prompt_fact, text-only
- z_counterfact_text: trusted_prompt_counterfact, text-only

The output keeps all decoder layers and preserves the [layer, head, head_dim]
shape expected by the existing steering controller.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from extract_after_template_activations_official_llava import (  # noqa: E402
    OfficialLlavaActivationExtractor,
    build_conv_prompt,
    import_official_llava,
    load_official_model,
    require_torch,
)


REQUIRED_FIELDS = (
    "id",
    "split",
    "expert_type",
    "subtype",
    "image_path",
    "visual_prompt",
    "trusted_prompt_fact",
    "trusted_prompt_counterfact",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default="")
    parser.add_argument("--yesno-output", default="")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--llava-repo", "--llava-repo-path", dest="llava_repo_path", required=True)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1, help="Kept for CLI compatibility; extraction is sequential.")
    parser.add_argument("--dtype", "--storage-dtype", dest="storage_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-samples-per-subtype", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compat-new-transformers", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_row(row: Mapping[str, Any], index: int) -> None:
    missing = [key for key in REQUIRED_FIELDS if row.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Row {index} is missing required fields: {missing}")


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    selected: list[tuple[int, dict[str, Any]]] = []
    counts: Counter[str] = Counter()
    eligible_ordinal = 0
    for index, row in enumerate(rows):
        subtype = str(row.get("subtype", ""))
        if int(args.max_samples_per_subtype) > 0 and counts[subtype] >= int(args.max_samples_per_subtype):
            continue
        counts[subtype] += 1
        if eligible_ordinal % int(args.num_shards) == int(args.shard_index):
            selected.append((index, row))
        eligible_ordinal += 1
    return selected


def build_text_prompt(user_text: str, *, model: Any, conv_mode: str, llava: Any) -> str:
    return build_conv_prompt(str(user_text), include_image=False, model=model, conv_mode=conv_mode, llava=llava)


def build_assistant_answer_prompt(question: str, answer: str, *, conv_mode: str, llava: Any) -> str:
    conv = llava.conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], str(question))
    conv.append_message(conv.roles[1], str(answer))
    return conv.get_prompt()


def encode_three_branches(
    extractor: OfficialLlavaActivationExtractor,
    row: Mapping[str, Any],
    *,
    model: Any,
    conv_mode: str,
    llava: Any,
) -> dict[str, Any]:
    image_path = str(row["image_path"])
    visual_full_prompt = build_conv_prompt(
        str(row["visual_prompt"]),
        include_image=True,
        model=model,
        conv_mode=conv_mode,
        llava=llava,
    )
    fact_full_prompt = build_text_prompt(
        str(row["trusted_prompt_fact"]),
        model=model,
        conv_mode=conv_mode,
        llava=llava,
    )
    counter_full_prompt = build_text_prompt(
        str(row["trusted_prompt_counterfact"]),
        model=model,
        conv_mode=conv_mode,
        llava=llava,
    )
    z_visual, token_visual, actual_visual, seq_visual = extractor._run_prompt(
        visual_full_prompt,
        image_path=image_path,
        include_image=True,
    )
    z_fact, token_fact, actual_fact, seq_fact = extractor._run_prompt(
        fact_full_prompt,
        image_path=None,
        include_image=False,
    )
    z_counter, token_counter, actual_counter, seq_counter = extractor._run_prompt(
        counter_full_prompt,
        image_path=None,
        include_image=False,
    )
    return {
        "z_visual": z_visual,
        "z_fact_text": z_fact,
        "z_counterfact_text": z_counter,
        "meta": {
            "visual_full_prompt": visual_full_prompt,
            "fact_full_prompt": fact_full_prompt,
            "counterfact_full_prompt": counter_full_prompt,
            "target_token_index_visual": int(actual_visual),
            "target_token_index_fact": int(actual_fact),
            "target_token_index_counterfact": int(actual_counter),
            "tokenized_target_index_visual": int(token_visual),
            "tokenized_target_index_fact": int(token_fact),
            "tokenized_target_index_counterfact": int(token_counter),
            "hidden_sequence_len_visual_approx": int(seq_visual),
            "hidden_sequence_len_fact_approx": int(seq_fact),
            "hidden_sequence_len_counterfact_approx": int(seq_counter),
        },
    }


def tensor_shape(tensor: Any) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def stack_or_empty(torch: Any, items: list[Any], storage_dtype: str) -> Any:
    if not items:
        raise ValueError("No activation tensors were collected.")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(storage_dtype)]
    return torch.stack([item.to("cpu").to(dtype) for item in items], dim=0)


def metadata_for_row(index: int, row: Mapping[str, Any], branch_meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_index": int(index),
        "id": str(row.get("id", "")),
        "split": str(row.get("split", "")),
        "source": str(row.get("source", "")),
        "expert_type": str(row.get("expert_type", "")),
        "hallucination_type": str(row.get("hallucination_type", row.get("expert_type", ""))),
        "subtype": str(row.get("subtype", "")),
        "image_id": str(row.get("image_id", "")),
        "image_path": str(row.get("image_path", "")),
        "question": str(row.get("question", "")),
        "gt_answer": str(row.get("gt_answer", row.get("label", ""))).lower(),
        "label": str(row.get("label", row.get("gt_answer", ""))).lower(),
        "fact_text": str(row.get("fact_text", "")),
        "counterfact_text": str(row.get("counterfact_text", "")),
        "base_scene": str(row.get("base_scene", "")),
        "target_fact": str(row.get("target_fact", "")),
        "target_counterfact": str(row.get("target_counterfact", "")),
        "metadata": dict(row.get("metadata", {})) if isinstance(row.get("metadata", {}), Mapping) else row.get("metadata", {}),
        "branch_meta": dict(branch_meta),
    }


def load_existing(torch: Any, output_path: Path, metadata_path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not output_path.exists():
        return None, []
    try:
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(output_path, map_location="cpu")
    metadata = list(payload.get("metadata", []))
    if metadata_path.exists():
        metadata = read_jsonl(metadata_path)
    return payload, metadata


def extract_yesno_direction(
    extractor: OfficialLlavaActivationExtractor,
    *,
    conv_mode: str,
    llava: Any,
) -> dict[str, Any]:
    prompts = [
        "Question: Is there an object in the image?",
        "Question: Is the object red?",
        "Question: Is the object to the left of another object?",
        "Question: Are there two objects in the image?",
    ]
    yes_items = []
    no_items = []
    pairs = []
    for prompt in prompts:
        yes_full = build_assistant_answer_prompt(prompt, "yes.", conv_mode=conv_mode, llava=llava)
        no_full = build_assistant_answer_prompt(prompt, "no.", conv_mode=conv_mode, llava=llava)
        z_yes, *_ = extractor._run_prompt(yes_full, image_path=None, include_image=False)
        z_no, *_ = extractor._run_prompt(no_full, image_path=None, include_image=False)
        yes_items.append(z_yes.float())
        no_items.append(z_no.float())
        pairs.append({"prompt": prompt, "yes_full_prompt": yes_full, "no_full_prompt": no_full})
    torch = extractor.torch
    direction = torch.stack([yes - no for yes, no in zip(yes_items, no_items)], dim=0).mean(dim=0)
    return {
        "yesno_direction": direction.to("cpu").float(),
        "prompts": pairs,
        "schema": {
            "mode": "answer_token",
            "direction": "mean(z_yes_answer_prompt - z_no_answer_prompt)",
            "shape": tensor_shape(direction),
            "dtype": "float32",
        },
    }


def main() -> int:
    args = parse_args()
    torch = require_torch()
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    input_path = resolve(args.input_jsonl)
    output_path = resolve(args.output)
    metadata_path = resolve(args.metadata_output) if str(args.metadata_output).strip() else output_path.with_suffix(".meta.jsonl")
    yesno_path = resolve(args.yesno_output) if str(args.yesno_output).strip() else output_path.with_name(output_path.stem + ".yesno.pt")
    if output_path.exists() and not args.overwrite and not args.resume:
        payload, metadata = load_existing(torch, output_path, metadata_path)
        if payload and metadata:
            print(f"Output already exists with {len(metadata)} rows: {output_path}")
            return 0
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite or --resume.")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {input_path}")
    rows = read_jsonl(input_path)
    selected = select_rows(rows, args)
    if not selected:
        raise ValueError("No rows selected for extraction.")
    for index, row in selected:
        validate_row(row, index)

    existing_payload: dict[str, Any] | None = None
    existing_metadata: list[dict[str, Any]] = []
    processed_ids: set[str] = set()
    if args.resume and output_path.exists():
        existing_payload, existing_metadata = load_existing(torch, output_path, metadata_path)
        processed_ids = {str(row.get("id", "")) for row in existing_metadata}
        selected = [(idx, row) for idx, row in selected if str(row.get("id", "")) not in processed_ids]
        print(f"Resume mode: found {len(processed_ids)} existing rows; remaining {len(selected)}.")
        if not selected:
            print(f"Nothing to do: {output_path}")
            return 0

    llava = import_official_llava(str(args.llava_repo_path))
    llava_args = argparse.Namespace(**vars(args))
    tokenizer, model, image_processor, context_len, model_name = load_official_model(llava_args, llava)
    if bool(args.compat_new_transformers):
        try:
            from extract_after_template_activations_official_llava import maybe_apply_new_transformers_compat

            maybe_apply_new_transformers_compat(model)
        except Exception:
            pass
    extractor = OfficialLlavaActivationExtractor(
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        llava=llava,
        conv_mode=str(args.conv_mode),
        storage_dtype=str(args.storage_dtype),
    )
    z_visual_items: list[Any] = []
    z_fact_items: list[Any] = []
    z_counter_items: list[Any] = []
    metadata_rows: list[dict[str, Any]] = []
    first_shape: list[int] | None = None
    try:
        for processed_index, (source_index, row) in enumerate(selected, start=1):
            result = encode_three_branches(
                extractor,
                row,
                model=model,
                conv_mode=str(args.conv_mode),
                llava=llava,
            )
            shape = tensor_shape(result["z_visual"])
            if first_shape is None:
                first_shape = shape
            if shape != first_shape:
                raise RuntimeError(f"Inconsistent visual activation shape at row {source_index}: {shape} != {first_shape}")
            for key in ("z_fact_text", "z_counterfact_text"):
                if tensor_shape(result[key]) != first_shape:
                    raise RuntimeError(f"Inconsistent {key} shape at row {source_index}: {tensor_shape(result[key])} != {first_shape}")
            z_visual_items.append(result["z_visual"])
            z_fact_items.append(result["z_fact_text"])
            z_counter_items.append(result["z_counterfact_text"])
            metadata_rows.append(metadata_for_row(source_index, row, result["meta"]))
            if int(args.progress_every) > 0 and processed_index % int(args.progress_every) == 0:
                print(f"[subtype-extract] processed {processed_index}/{len(selected)} rows (source row {source_index})")

        yesno_payload = extract_yesno_direction(extractor, conv_mode=str(args.conv_mode), llava=llava)
    finally:
        extractor.close()

    new_payload = {
        "z_visual": stack_or_empty(torch, z_visual_items, str(args.storage_dtype)),
        "z_fact_text": stack_or_empty(torch, z_fact_items, str(args.storage_dtype)),
        "z_counterfact_text": stack_or_empty(torch, z_counter_items, str(args.storage_dtype)),
        "metadata": metadata_rows,
    }
    if existing_payload is not None and existing_metadata:
        new_payload = {
            "z_visual": torch.cat([existing_payload["z_visual"], new_payload["z_visual"]], dim=0),
            "z_fact_text": torch.cat([existing_payload["z_fact_text"], new_payload["z_fact_text"]], dim=0),
            "z_counterfact_text": torch.cat([existing_payload["z_counterfact_text"], new_payload["z_counterfact_text"]], dim=0),
            "metadata": existing_metadata + metadata_rows,
        }
    counts = Counter(str(row.get("subtype", "")) for row in new_payload["metadata"])
    schema = {
        "script": "scripts/extract_subtype_minpair_activations.py",
        "model_path": str(args.model_path),
        "model_name": str(model_name),
        "context_len": int(context_len),
        "llava_repo_path": str(args.llava_repo_path),
        "conv_mode": str(args.conv_mode),
        "storage_dtype": str(args.storage_dtype),
        "num_layers": int(new_payload["z_visual"].shape[1]),
        "num_heads": int(new_payload["z_visual"].shape[2]),
        "head_dim": int(new_payload["z_visual"].shape[3]),
        "shape": tensor_shape(new_payload["z_visual"]),
        "counts_by_subtype": dict(sorted(counts.items())),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "source_jsonl": str(input_path),
        "branch_definitions": {
            "z_visual": "image + visual_prompt",
            "z_fact_text": "trusted_prompt_fact, text-only",
            "z_counterfact_text": "trusted_prompt_counterfact, text-only",
        },
    }
    new_payload["schema"] = schema
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_payload, output_path)
    write_jsonl(metadata_path, new_payload["metadata"])
    torch.save(yesno_payload, yesno_path)
    manifest = {
        **schema,
        "metadata_output": str(metadata_path),
        "yesno_output": str(yesno_path),
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    print(f"Wrote subtype activations to {output_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(f"Wrote yes/no direction to {yesno_path}")
    print(f"Counts by subtype: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
