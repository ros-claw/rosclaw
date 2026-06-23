"""JSONL writer for practice events.

Appends ``PracticeEventEnvelope`` records as one JSON object per line.
Supports optional rotation by file size and atomic-ish writes via a temporary
file rename.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.practice.jsonl_writer")


class JsonlWriter:
    """Append-only JSONL writer with optional size-based rotation."""

    def __init__(
        self,
        path: Path | str,
        rotate_mb: float | None = None,
    ):
        self._path = Path(path)
        self._rotate_bytes = int(rotate_mb * 1024 * 1024) if rotate_mb else None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._file = None
        self._open()

    def _open(self) -> None:
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    def write(self, record: dict[str, Any]) -> None:
        """Serialize *record* to one JSON line and append."""
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            logger.error("Failed to serialize JSONL record: %s", e)
            return

        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

            if self._rotate_bytes and self._path.stat().st_size >= self._rotate_bytes:
                self._rotate()

    def _rotate(self) -> None:
        self._file.close()
        suffix = 1
        while True:
            rotated = self._path.with_suffix(f".jsonl.{suffix:03d}")
            if not rotated.exists():
                self._path.rename(rotated)
                break
            suffix += 1
        self._open()

    def flush(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()
                os.fsync(self._file.fileno())

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self.flush()
                self._file.close()
                self._file = None

    def write_atomic(self, record: dict[str, Any]) -> None:
        """Write a single record atomically using a temp file + rename.

        Useful for small metadata files where rotation is not needed.
        """
        line = json.dumps(record, ensure_ascii=False, default=str)
        fd, tmp = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=self._path.name + ".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
