#!/usr/bin/env python3
"""Merge sharded subtype minimal-pair activation caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default="")
    parser.add_argument("--yesno-output", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("merge_subtype_minpair_activations.py requires torch.") from exc


def torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    torch = load_torch()
    output_path = resolve(args.output)
    metadata_path = resolve(args.metadata_output) if str(args.metadata_output).strip() else output_path.with_suffix(".meta.jsonl")
    yesno_path = resolve(args.yesno_output) if str(args.yesno_output).strip() else output_path.with_name(output_path.stem + ".yesno.pt")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite.")
    payloads = []
    for raw_path in args.activation_files:
        path = resolve(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing activation shard: {path}")
        payload = torch_load(torch, path)
        for key in ("z_visual", "z_fact_text", "z_counterfact_text", "metadata"):
            if key not in payload:
                raise ValueError(f"{path} missing key {key}")
        payloads.append((path, payload))
    payloads.sort(key=lambda item: str(item[0]))
    metadata = []
    tensors: dict[str, list[Any]] = {"z_visual": [], "z_fact_text": [], "z_counterfact_text": []}
    seen_ids: set[str] = set()
    for path, payload in payloads:
        shard_meta = list(payload.get("metadata", []))
        for row in shard_meta:
            row_id = str(row.get("id", ""))
            if row_id in seen_ids:
                raise ValueError(f"Duplicate row id while merging: {row_id}")
            seen_ids.add(row_id)
            metadata.append(dict(row))
        for key in tensors:
            tensor = payload[key]
            if int(tensor.shape[0]) != len(shard_meta):
                raise ValueError(f"{path} key {key} N={tensor.shape[0]} but metadata len={len(shard_meta)}")
            tensors[key].append(tensor)
    order = sorted(range(len(metadata)), key=lambda idx: (int(metadata[idx].get("row_index", idx)), str(metadata[idx].get("id", ""))))
    merged = {
        key: torch.cat(value, dim=0)[order]
        for key, value in tensors.items()
    }
    merged_metadata = [metadata[idx] for idx in order]
    schema = dict(payloads[0][1].get("schema", {}))
    schema.update(
        {
            "script": "scripts/merge_subtype_minpair_activations.py",
            "num_shards_merged": len(payloads),
            "shard_files": [str(path) for path, _payload in payloads],
            "shape": [int(dim) for dim in merged["z_visual"].shape],
        }
    )
    merged["metadata"] = merged_metadata
    merged["schema"] = schema
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    write_jsonl(metadata_path, merged_metadata)
    yesno_payloads = []
    for path, _payload in payloads:
        candidate = path.with_name(path.stem + ".yesno.pt")
        if candidate.exists():
            yesno_payloads.append(torch_load(torch, candidate))
    if yesno_payloads:
        directions = [payload["yesno_direction"].float() for payload in yesno_payloads if "yesno_direction" in payload]
        if directions:
            torch.save(
                {
                    "yesno_direction": torch.stack(directions, dim=0).mean(dim=0),
                    "schema": {
                        "mode": "answer_token",
                        "source": "mean of shard yes/no directions",
                        "num_shards": len(directions),
                        "shape": [int(dim) for dim in directions[0].shape],
                    },
                },
                yesno_path,
            )
    write_json(output_path.with_suffix(".manifest.json"), schema | {"metadata_output": str(metadata_path), "yesno_output": str(yesno_path)})
    print(f"Wrote merged subtype activation cache to {output_path}")
    print(f"Wrote merged metadata to {metadata_path}")
    if yesno_path.exists():
        print(f"Wrote merged yes/no direction to {yesno_path}")
    print(f"num_rows={len(merged_metadata)} shape={[int(dim) for dim in merged['z_visual'].shape]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
