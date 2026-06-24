"""Local JSONL event storage helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def append_event(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON object as a line to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def read_events(
    path: Path,
    *,
    limit: int | None = None,
    days: int | None = None,
) -> list[dict[str, Any]]:
    """Read events from a JSONL file, optionally limited or filtered by age."""
    path = Path(path)
    if not path.exists():
        return []

    cutoff = None
    if days is not None:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat().replace("+00:00", "Z")

    results: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if cutoff is not None:
                    ts = record.get("created_at", "")
                    if isinstance(ts, str) and ts < cutoff:
                        continue
                results.append(record)
                if limit is not None and len(results) >= limit:
                    break
    except OSError:
        return []
    return results


def count_events(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    path = Path(path)
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def directory_size_mb(path: Path) -> float:
    """Return total size of all files under `path` in megabytes."""
    path = Path(path)
    if not path.exists():
        return 0.0
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        pass
    return round(total / (1024 * 1024), 2)


def event_file_for_date(
    home: Path,
    kind: str,
    date: datetime | None = None,
) -> Path:
    """Return the JSONL path for telemetry or feedback events on a given date."""
    date = date or datetime.now(UTC)
    return home / kind / "events" / f"{date.date().isoformat()}.jsonl"
