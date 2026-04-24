"""Test-only temporary workspace helpers.

The local Windows sandbox used for development can create directories while
refusing directory removal. Python's `tempfile.TemporaryDirectory` treats that
as an unusable temp directory, so tests use this small tolerant context manager
instead.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path


class TemporaryWorkspace:
    """Create a unique workspace under the repo and ignore cleanup failures."""

    def __init__(self, prefix: str = "test_") -> None:
        """Prepare a unique path for one test case."""

        self.root = Path(__file__).resolve().parents[1] / ".test_tmp"
        self.path = self.root / f"{prefix}{uuid.uuid4().hex}"

    def __enter__(self) -> str:
        """Create and return the workspace path as a string."""

        self.path.mkdir(parents=True, exist_ok=False)
        return str(self.path)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Best-effort cleanup that never masks test failures."""

        shutil.rmtree(self.path, ignore_errors=True)
