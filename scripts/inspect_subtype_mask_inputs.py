#!/usr/bin/env python3
"""Inspect subtype mask steering inputs and write a Markdown report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def load_jsonl(path: str | Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_pt(path: str | Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def summarize_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            out[key] = {"type": "Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, list):
            out[key] = {"type": "list", "len": len(value)}
        elif isinstance(value, dict):
            out[key] = {"type": "dict", "len": len(value), "keys": list(value.keys())[:40]}
        else:
            out[key] = {"type": type(value).__name__}
    return out


def get_vectors(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if "vectors" in payload and isinstance(payload["vectors"], Mapping):
        return payload["vectors"]
    return {k: v for k, v in payload.items() if torch.is_tensor(v)}


def subtype(row: Mapping[str, Any]) -> str:
    if row.get("subtype"):
        return str(row["subtype"])
    meta = row.get("metadata")
    if isinstance(meta, Mapping) and meta.get("subtype"):
        return str(meta["subtype"])
    return ""


def expert_type(row: Mapping[str, Any]) -> str:
    if row.get("expert_type"):
        return str(row["expert_type"])
    meta = row.get("metadata")
    if isinstance(meta, Mapping) and meta.get("expert_type"):
        return str(meta["expert_type"])
    st = subtype(row)
    return st.split("_", 1)[0] if "_" in st else st


def md_table(headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def count_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    counts = Counter(subtype(r) for r in rows)
    yes_no = defaultdict(Counter)
    sources = defaultdict(Counter)
    for row in rows:
        st = subtype(row)
        yes_no[st][str(row.get("gt_answer", "")).lower()] += 1
        sources[st][str(row.get("source", ""))] += 1
    out = []
    for st in sorted(counts):
        out.append(
            {
                "subtype": st,
                "count": counts[st],
                "yes": yes_no[st].get("yes", 0),
                "no": yes_no[st].get("no", 0),
                "sources": dict(sources[st]),
            }
        )
    return out


def image_overlap(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_type: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        img = str(row.get("image_id") or row.get("image_path") or "")
        if img:
            by_type[expert_type(row)].add(img)
    result = []
    keys = sorted(by_type)
    for a in keys:
        for b in keys:
            inter = len(by_type[a] & by_type[b])
            result.append({"type_a": a, "type_b": b, "overlap": inter})
    return result


def main() -> int:
    args = parse_args()
    train_rows = load_jsonl(args.train_jsonl)
    val_rows = load_jsonl(args.val_jsonl)
    act_payload = load_pt(args.activations)
    vec_payload = load_pt(args.vectors)
    vectors = get_vectors(vec_payload)

    vector_rows = []
    for key, tensor in sorted(vectors.items()):
        if torch.is_tensor(tensor):
            vector_rows.append(
                {
                    "key": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "flat_norm": f"{float(tensor.float().norm().item()):.4f}",
                }
            )

    sample_rows = []
    for row in val_rows[:5]:
        sample_rows.append(
            {
                "id": row.get("id", ""),
                "subtype": subtype(row),
                "gt_answer": row.get("gt_answer", ""),
                "question": row.get("question", ""),
                "image_path": row.get("image_path", ""),
            }
        )

    lines: List[str] = []
    lines.append("# Subtype Mask Steering Input Inspection")
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    lines.append(f"- Train JSONL: `{args.train_jsonl}`")
    lines.append(f"- Val JSONL: `{args.val_jsonl}`")
    lines.append(f"- Activations: `{args.activations}`")
    lines.append(f"- Vectors: `{args.vectors}`")
    lines.append("")

    lines.append("## Train Counts")
    lines.append("")
    lines.append(md_table(["subtype", "count", "yes", "no", "sources"], count_rows(train_rows)))
    lines.append("")
    lines.append("## Val Counts")
    lines.append("")
    lines.append(md_table(["subtype", "count", "yes", "no", "sources"], count_rows(val_rows)))
    lines.append("")
    lines.append("## Expert-Type Image Overlap In Train")
    lines.append("")
    lines.append(md_table(["type_a", "type_b", "overlap"], image_overlap(train_rows)))
    lines.append("")

    lines.append("## Activation Payload Schema")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summarize_payload(act_payload), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Vector Payload Schema")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summarize_payload(vec_payload), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Vector Keys")
    lines.append("")
    lines.append(md_table(["key", "shape", "dtype", "flat_norm"], vector_rows))
    lines.append("")

    lines.append("## Val Sample Preview")
    lines.append("")
    lines.append(md_table(["id", "subtype", "gt_answer", "question", "image_path"], sample_rows))
    lines.append("")

    lines.append("## Runner Capability Notes")
    lines.append("")
    lines.append("- `ExpertSteeringController` supports `head_select=expert_map`, so direction vectors and head masks can be separated via an external JSON head map.")
    lines.append("- `scripts/eval_subtype_mask_steering.py` writes a runtime vector file keyed by direction/mask pair and a matching expert head map.")
    lines.append("- This keeps official LLaVA generation and the existing o_proj pre-hook path.")
    lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote inspection report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
