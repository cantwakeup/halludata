"""Small file I/O helpers for mock fact-counterfact data pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from expert_data.schemas import FactRecord, PairRecord


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return a dictionary payload."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return payload


def read_json(path: str | Path) -> Any:
    """Read a JSON file and return the decoded payload."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON payload with stable formatting and return its path."""

    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            records.append(payload)
    return records


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path when it does not exist."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def load_fact_records(path: str | Path) -> list[FactRecord]:
    """Load typed fact records from a JSONL fact index."""

    return [FactRecord.from_dict(payload) for payload in read_jsonl(path)]


def _record_to_dict(record: FactRecord | PairRecord | dict[str, Any]) -> dict[str, Any]:
    """Normalize supported output records into plain dictionaries."""

    if isinstance(record, FactRecord):
        return record.to_dict()
    if isinstance(record, PairRecord):
        return record.to_dict()
    if hasattr(record, "to_dict"):
        return dict(record.to_dict())
    return dict(record)


def write_jsonl(
    path: str | Path,
    records: Iterable[FactRecord | PairRecord | dict[str, Any]],
) -> int:
    """Write records to JSONL and return the number of rows emitted."""

    output_path = ensure_parent_dir(path)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_record_to_dict(record), ensure_ascii=False) + "\n")
            count += 1
    return count
