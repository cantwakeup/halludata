"""Lightweight activation-cache read/write helpers."""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for manifests."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path) -> str:
    """Compute a SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write dictionaries as UTF-8 JSONL and return the number of rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file into dictionaries."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            rows.append(payload)
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write pretty JSON with stable UTF-8 formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def read_json(path: str | Path) -> dict[str, Any]:
    """Read one JSON object from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _import_torch_or_none() -> Any | None:
    """Import torch lazily, returning None when it is not installed or usable."""

    try:
        import torch

        return torch
    except Exception:
        return None


def _save_activations(path: Path, cache_dict: dict[str, Any]) -> None:
    """Save activation tensors with torch when available, otherwise use pickle fallback."""

    torch = _import_torch_or_none()
    if torch is not None:
        torch.save(cache_dict, path)
        return
    with path.open("wb") as handle:
        pickle.dump(cache_dict, handle)


def _load_activations(path: Path) -> dict[str, Any]:
    """Load activation tensors from torch or pickle fallback formats."""

    torch = _import_torch_or_none()
    if torch is not None:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            with path.open("rb") as handle:
                return pickle.load(handle)
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load activation cache {path}. Install torch if this file was produced with torch.save."
            ) from exc


def save_activation_cache(
    out_dir: str | Path,
    cache_dict: dict[str, Any],
    metadata_rows: Iterable[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Path]:
    """Write activations, metadata rows, and a manifest into one cache directory."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_path = output_dir / "activations.pt"
    metadata_path = output_dir / "metadata.jsonl"
    manifest_path = output_dir / "activation_manifest.json"
    _save_activations(activation_path, cache_dict)
    write_jsonl(metadata_path, metadata_rows)
    write_json(manifest_path, manifest)
    return {
        "activations": activation_path,
        "metadata": metadata_path,
        "manifest": manifest_path,
    }


def load_activation_cache(out_dir_or_file: str | Path) -> dict[str, Any]:
    """Load an activation cache directory or a direct `activations.pt` path."""

    path = Path(out_dir_or_file)
    if path.is_dir():
        activation_path = path / "activations.pt"
        metadata_path = path / "metadata.jsonl"
        manifest_path = path / "activation_manifest.json"
    else:
        activation_path = path
        metadata_path = path.parent / "metadata.jsonl"
        manifest_path = path.parent / "activation_manifest.json"
    payload = {
        "activations": _load_activations(activation_path),
        "metadata": read_jsonl(metadata_path) if metadata_path.exists() else [],
        "manifest": read_json(manifest_path) if manifest_path.exists() else {},
    }
    return payload


def tensor_shape(value: Any) -> list[int]:
    """Return the shape of a tensor-like object or nested Python lists."""

    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(item) for item in shape]
    current = value
    result: list[int] = []
    while isinstance(current, list):
        result.append(len(current))
        current = current[0] if current else []
    return result
