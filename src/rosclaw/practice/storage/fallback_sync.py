"""Sync offline fallback JSON files back to SeekDB.

Scans the fallback directory for previously-failed PraxisEvent JSON files,
POSTs them to SeekDB, and moves successfully-uploaded files to an archive
subdirectory.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("rosclaw.practice.storage.fallback_sync")


class FallbackSync:
    """Re-submit fallback JSON files to SeekDB."""

    def __init__(
        self,
        seekdb_url: str = "http://localhost:2881",
        fallback_dir: Path | str = "/data/rosclaw/practice/fallback",
        table: str = "praxis_events",
        timeout_sec: float = 2.0,
    ):
        self._seekdb_url = seekdb_url.rstrip("/")
        self._endpoint = f"{self._seekdb_url}/api/v1/insert"
        self._fallback_dir = Path(fallback_dir)
        self._table = table
        self._timeout = timeout_sec
        self._archived_dir = self._fallback_dir / "archived"

    def sync(self) -> dict[str, Any]:
        """Return summary with attempted/success/failed counts."""
        self._fallback_dir.mkdir(parents=True, exist_ok=True)
        self._archived_dir.mkdir(parents=True, exist_ok=True)

        summary = {"attempted": 0, "success": 0, "failed": 0, "errors": []}
        files = sorted(self._fallback_dir.glob("*.json"))
        for path in files:
            summary["attempted"] += 1
            try:
                with open(path, encoding="utf-8") as f:
                    event_data = json.load(f)
                response = requests.post(
                    self._endpoint,
                    json={"table": self._table, "data": event_data},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                target = self._archived_dir / path.name
                suffix = 1
                stem = path.stem
                ext = path.suffix
                while target.exists():
                    target = self._archived_dir / f"{stem}.{suffix:03d}{ext}"
                    suffix += 1
                shutil.move(str(path), str(target))
                summary["success"] += 1
            except Exception as e:
                summary["failed"] += 1
                summary["errors"].append(f"{path.name}: {e}")
                logger.error("Failed to sync fallback %s: %s", path, e)
        return summary

    def list_pending(self) -> list[Path]:
        self._fallback_dir.mkdir(parents=True, exist_ok=True)
        return sorted(self._fallback_dir.glob("*.json"))
