"""Resolve COCO image IDs into filesystem paths for activation extraction."""

from __future__ import annotations

from pathlib import Path

from expert_data.io_utils import read_json


class CocoImageResolver:
    """Resolve image paths from a COCO image root and optional instances JSON."""

    def __init__(self, image_root: str | Path, instances_json: str | Path | None = None) -> None:
        """Build an image resolver using COCO `images[].file_name` when available."""

        self.image_root = Path(image_root)
        self.instances_json = Path(instances_json) if instances_json else None
        self._file_name_by_id: dict[str, str] = {}
        if self.instances_json is not None:
            payload = read_json(self.instances_json)
            images = payload.get("images", []) if isinstance(payload, dict) else []
            for image_info in images:
                if not isinstance(image_info, dict):
                    continue
                if "id" in image_info and image_info.get("file_name"):
                    self._file_name_by_id[str(image_info["id"])] = str(image_info["file_name"])

    def _candidate_paths(self, image_id: str | int) -> list[Path]:
        """Return candidate filesystem paths for an image ID."""

        image_key = str(image_id)
        candidates: list[Path] = []
        if image_key in self._file_name_by_id:
            candidates.append(self.image_root / self._file_name_by_id[image_key])
        try:
            candidates.append(self.image_root / f"{int(image_key):012d}.jpg")
        except ValueError:
            pass
        return candidates

    def resolve(self, image_id: str | int) -> str:
        """Resolve one image ID to an existing file path or raise a clear error."""

        candidates = self._candidate_paths(image_id)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        attempted = ", ".join(str(candidate) for candidate in candidates) or "<no candidate paths>"
        raise FileNotFoundError(f"Could not resolve image_id={image_id}; attempted: {attempted}")
