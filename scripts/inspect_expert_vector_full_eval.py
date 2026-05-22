#!/usr/bin/env python3
"""Inspect candidate expert vectors and build a runtime vector-only bundle.

This script is intentionally read-only for existing experiment outputs.  It
chooses one vector each for global/category/attribute/relation steering and
writes a small runtime file with keys:

    global, cat, attr, rel

The runtime file is what the vector-only benchmark evaluator consumes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Candidate:
    role: str
    path: Path
    key: str
    tensor: Any
    layers: list[int]
    source_note: str
    priority: int


ROLE_KEY_PRIORITY = {
    "global": ["g_all_clean", "global_all_clean", "global_all", "g_all", "global"],
    "cat": ["g_cat_clean", "cat_clean", "cat", "coco_cat", "global_plus_cat_res"],
    "attr": ["g_attr_clean", "attr_clean", "attr", "global_plus_attr_res"],
    "rel": ["g_rel_clean", "rel_clean", "rel", "global_plus_rel_res"],
}


PREFERRED_FILES = [
    "data/clean_type_minpair_v2/vectors/condition_vectors.pt",
    "data/subtype_minpair_v1/vectors/subtype_vectors.pt",
    "data/gqa_typeaware_v1/steering/gqa_typeaware_expert_vectors.pt",
    "data/gqa_typeaware_v1/steering/gqa_global_residual_vectors.pt",
]


SCAN_DIRS = [
    "data/clean_type_minpair_v2/vectors",
    "data/subtype_minpair_v1/vectors",
    "data/after_template_disjoint_v2",
    "data/after_template_disjoint_v1",
    "data/after_style_v1",
    "data/pope_cat_expert_eval",
    "data/gqa_typeaware_v1/steering",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/expert_vector_full_eval_v1")
    parser.add_argument("--inspect-output", default="")
    parser.add_argument("--runtime-vector-output", default="")
    parser.add_argument("--resolved-output", default="")
    parser.add_argument("--extra-vector-files", nargs="*", default=[])
    parser.add_argument("--allow-layer-intersection", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def import_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("This script requires torch to inspect .pt vector files.") from exc


def torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def candidate_paths(extra_files: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in [*PREFERRED_FILES, *extra_files]:
        path = resolve(raw)
        if path.exists() and path not in paths:
            paths.append(path)
    for raw_dir in SCAN_DIRS:
        root = resolve(raw_dir)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.pt")):
            if path not in paths:
                paths.append(path)
    return paths


def vector_items(payload: Any) -> dict[str, Any]:
    items: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        vectors = payload.get("vectors")
        if isinstance(vectors, Mapping):
            for key, value in vectors.items():
                if hasattr(value, "ndim") and int(value.ndim) == 3:
                    items[str(key)] = value
        for key, value in payload.items():
            if key == "vectors":
                continue
            if hasattr(value, "ndim") and int(value.ndim) == 3:
                items.setdefault(str(key), value)
    return items


def payload_layers(payload: Any, layer_count: int) -> list[int]:
    if isinstance(payload, Mapping) and isinstance(payload.get("layers"), (list, tuple)):
        layers = [int(x) for x in payload["layers"]]
        if len(layers) == layer_count:
            return layers
    return list(range(layer_count))


def path_priority(path: Path) -> int:
    rel = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
    for index, preferred in enumerate(PREFERRED_FILES):
        if rel.replace("\\", "/") == preferred:
            return index
    return 100


def find_role_candidates(torch: Any, paths: list[Path]) -> tuple[list[Candidate], list[dict[str, Any]], list[str]]:
    candidates: list[Candidate] = []
    inventory: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            payload = torch_load(torch, path)
            items = vector_items(payload)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if not items:
            continue
        layers = None
        for key, tensor in sorted(items.items()):
            shape = [int(x) for x in tensor.shape]
            if layers is None:
                layers = payload_layers(payload, shape[0])
            info = tensor_info(torch, tensor, layers)
            info.update({"path": str(path), "key": key})
            inventory.append(info)
            for role, key_order in ROLE_KEY_PRIORITY.items():
                if key not in key_order:
                    continue
                candidates.append(
                    Candidate(
                        role=role,
                        path=path,
                        key=key,
                        tensor=tensor.detach().cpu().float(),
                        layers=list(layers),
                        source_note=source_note(path, key),
                        priority=(path_priority(path) * 100) + key_order.index(key),
                    )
                )
    return candidates, inventory, errors


def source_note(path: Path, key: str) -> str:
    rel = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
    if "clean_type_minpair_v2" in rel:
        return f"clean_type_minpair_v2:{key}"
    if "subtype_minpair_v1" in rel:
        return f"subtype_minpair_v1:{key}"
    if "gqa_typeaware_v1" in rel:
        return f"gqa_typeaware_v1:{key}"
    if "after_template" in rel:
        return f"after_template:{key}"
    if "pope_cat" in rel:
        return f"pope_cat_experiment:{key}"
    return f"{rel}:{key}"


def tensor_info(torch: Any, tensor: Any, layers: list[int]) -> dict[str, Any]:
    tensor = tensor.detach().cpu().float()
    finite = bool(torch.isfinite(tensor).all().item())
    norm = float(tensor.norm().item())
    head_norms = tensor.norm(dim=-1)
    flat = head_norms.reshape(-1)
    k = min(10, int(flat.numel()))
    values, indices = torch.topk(flat, k=k, largest=True, sorted=True)
    heads = []
    num_heads = int(head_norms.shape[1])
    for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        layer_idx = int(idx // num_heads)
        head = int(idx % num_heads)
        heads.append({"rank": rank, "layer": int(layers[layer_idx]), "head": head, "norm": float(value)})
    return {
        "shape": [int(x) for x in tensor.shape],
        "dtype": str(tensor.dtype),
        "norm": norm,
        "finite": finite,
        "layers": list(layers),
        "covers_32_layers": len(layers) == 32 and list(layers) == list(range(32)),
        "head_norm_top10": heads,
    }


def select_candidates(candidates: list[Candidate]) -> dict[str, Candidate]:
    selected: dict[str, Candidate] = {}
    for role in ["global", "cat", "attr", "rel"]:
        role_candidates = [item for item in candidates if item.role == role]
        if not role_candidates:
            raise FileNotFoundError(f"No candidate vector found for role={role}")
        role_candidates.sort(
            key=lambda item: (
                0 if (len(item.layers) == 32 and item.layers == list(range(32))) else 1,
                item.priority,
                str(item.path),
                item.key,
            )
        )
        selected[role] = role_candidates[0]
    return selected


def align_vectors(torch: Any, selected: Mapping[str, Candidate], allow_intersection: bool) -> tuple[dict[str, Any], list[int]]:
    layer_sets = [set(item.layers) for item in selected.values()]
    common_layers = sorted(set.intersection(*layer_sets))
    if not common_layers:
        raise ValueError("Selected vectors have no common layer ids.")
    if common_layers != list(range(32)) and not allow_intersection:
        detail = {role: item.layers for role, item in selected.items()}
        raise ValueError(
            "Selected vectors do not cover all 32 layers. "
            "Pass --allow-layer-intersection only for diagnostics, not for the main requested full-layer eval. "
            f"Selected layers: {detail}"
        )
    out: dict[str, Any] = {}
    for role, item in selected.items():
        index_by_layer = {int(layer): idx for idx, layer in enumerate(item.layers)}
        indices = torch.tensor([index_by_layer[layer] for layer in common_layers], dtype=torch.long)
        out[role] = item.tensor.index_select(0, indices).float().contiguous()
    return out, common_layers


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6f}"
        return str(value)
    return str(value)


def md_table(headers: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def write_inspect(
    path: Path,
    *,
    inventory: list[dict[str, Any]],
    selected: Mapping[str, Candidate],
    runtime_vector_path: Path,
    runtime_layers: list[int],
    errors: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_rows = []
    for role, item in selected.items():
        stats = tensor_info(import_torch(), item.tensor, item.layers)
        selected_rows.append(
            {
                "role": role,
                "path": item.path,
                "key": item.key,
                "shape": stats["shape"],
                "dtype": stats["dtype"],
                "norm": stats["norm"],
                "covers_32_layers": stats["covers_32_layers"],
                "source": item.source_note,
            }
        )
    inv_rows = [
        {
            "path": row["path"],
            "key": row["key"],
            "shape": row["shape"],
            "dtype": row["dtype"],
            "norm": row["norm"],
            "covers_32_layers": row["covers_32_layers"],
        }
        for row in inventory
    ]
    lines = [
        "# Expert Vector Full Eval Inspection",
        "",
        "## Final Vector Sources",
        "",
        md_table(["role", "path", "key", "shape", "dtype", "norm", "covers_32_layers", "source"], selected_rows),
        "",
        "## Runtime Bundle",
        "",
        f"- Runtime vector file: `{runtime_vector_path}`",
        f"- Runtime keys: `global`, `cat`, `attr`, `rel`",
        f"- Runtime layers: `{runtime_layers}`",
        "- Head selection for evaluation: vector norm top64 over all runtime layers.",
        "",
        "## Candidate Vector Inventory",
        "",
        md_table(["path", "key", "shape", "dtype", "norm", "covers_32_layers"], inv_rows) if inv_rows else "No vector tensors found.",
        "",
        "## Selected Top10 Heads",
        "",
    ]
    torch = import_torch()
    for role, item in selected.items():
        stats = tensor_info(torch, item.tensor, item.layers)
        lines.extend(
            [
                f"### {role}",
                "",
                md_table(["rank", "layer", "head", "norm"], stats["head_norm_top10"]),
                "",
            ]
        )
    if errors:
        lines.extend(["## Load Errors", "", *[f"- `{err}`" for err in errors[:50]], ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    torch = import_torch()
    output_root = resolve(args.output_root)
    inspect_output = resolve(args.inspect_output) if args.inspect_output else output_root / "INSPECT.md"
    runtime_output = resolve(args.runtime_vector_output) if args.runtime_vector_output else output_root / "vectors" / "expert_vectors_runtime.pt"
    resolved_output = resolve(args.resolved_output) if args.resolved_output else output_root / "resolved_vectors.json"
    for path in [inspect_output, runtime_output, resolved_output]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}. Pass --overwrite.")

    paths = candidate_paths(args.extra_vector_files)
    candidates, inventory, errors = find_role_candidates(torch, paths)
    selected = select_candidates(candidates)
    runtime_vectors, runtime_layers = align_vectors(torch, selected, bool(args.allow_layer_intersection))
    sample = next(iter(runtime_vectors.values()))
    payload = {
        "vectors": runtime_vectors,
        "layers": runtime_layers,
        "num_heads": int(sample.shape[1]),
        "head_dim": int(sample.shape[2]),
        "hidden_size": int(sample.shape[1] * sample.shape[2]),
        "metadata": {
            "created_by": "scripts/inspect_expert_vector_full_eval.py",
            "purpose": "vector_only_global_cat_attr_rel_full_eval",
            "selected_sources": {
                role: {"path": str(item.path), "key": item.key, "source": item.source_note}
                for role, item in selected.items()
            },
        },
    }
    runtime_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, runtime_output)
    resolved = {
        "runtime_vector_file": str(runtime_output),
        "layers": runtime_layers,
        "selected": {
            role: {
                "path": str(item.path),
                "key": item.key,
                "source": item.source_note,
                "shape": [int(x) for x in item.tensor.shape],
                "norm": float(item.tensor.norm().item()),
            }
            for role, item in selected.items()
        },
        "inventory_count": len(inventory),
        "load_errors": errors,
    }
    write_json(resolved_output, resolved)
    write_inspect(
        inspect_output,
        inventory=inventory,
        selected=selected,
        runtime_vector_path=runtime_output,
        runtime_layers=runtime_layers,
        errors=errors,
    )
    print(f"Wrote inspection report to {inspect_output}")
    print(f"Wrote runtime vector bundle to {runtime_output}")
    print(f"Wrote resolved vector manifest to {resolved_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
