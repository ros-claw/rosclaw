"""Idempotent merge utilities for agent onboarding files."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class MergeConflictError(Exception):
    """Raised when JSON merge encounters a conflict that requires human review."""

    def __init__(self, conflicts: list[str]) -> None:
        self.conflicts = conflicts
        super().__init__(f"Merge conflicts detected: {conflicts}")


def managed_block_merge(
    existing_content: str,
    new_content: str,
    begin_marker: str,
    end_marker: str,
) -> str:
    """Replace the managed block in existing_content with the one from new_content.

    If existing_content does not contain the markers, the new content is returned
    verbatim. Content outside the managed block is preserved.
    """
    if begin_marker not in existing_content or end_marker not in existing_content:
        return new_content

    begin_idx = existing_content.index(begin_marker)
    end_idx = existing_content.index(end_marker) + len(end_marker)

    if begin_marker not in new_content or end_marker not in new_content:
        # New content has no managed block; preserve existing by returning as-is.
        return existing_content

    new_begin_idx = new_content.index(begin_marker)
    new_end_idx = new_content.index(end_marker) + len(end_marker)
    new_block = new_content[new_begin_idx:new_end_idx]

    return existing_content[:begin_idx] + new_block + existing_content[end_idx:]


def _json_merge(
    base: Any,
    override: Any,
    path: str,
    conflicts: list[str],
) -> Any:
    """Recursively merge override into base, collecting conflicts."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged_dict = dict(base)
        for key, value in override.items():
            current_path = f"{path}.{key}" if path else key
            if key not in merged_dict:
                merged_dict[key] = value
            else:
                merged_dict[key] = _json_merge(merged_dict[key], value, current_path, conflicts)
        return merged_dict

    if isinstance(base, list) and isinstance(override, list):
        # For lists we concatenate uniquely while preserving order.
        seen = {json.dumps(item, sort_keys=True): item for item in base}
        merged_list = list(base)
        for item in override:
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                merged_list.append(item)
                seen[key] = item
        return merged_list

    if base == override:
        return override

    conflicts.append(path)
    # Keep the existing value on conflict; caller is informed via conflicts list.
    return base


def json_merge_with_conflict_detection(
    base: dict[str, Any],
    override: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Deep-merge override into base and return the result plus conflict paths."""
    conflicts: list[str] = []
    merged = _json_merge(base, override, "", conflicts)
    if not isinstance(merged, dict):
        raise MergeConflictError(["root type mismatch"])
    return merged, conflicts


def backup_file(path: Path, backups_dir: Path | None = None) -> Path:
    """Create a timestamped backup of path and return the backup path."""
    if not path.exists():
        return path

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    backup_name = f"{path.name}.{timestamp}.bak"
    backup_dir = path.parent / ".backups" if backups_dir is None else backups_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / backup_name
    shutil.copy2(path, backup_path)
    return backup_path


def read_json_if_exists(path: Path) -> dict[str, Any]:
    """Read JSON from path or return an empty dict."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_write_json(path: Path, data: dict[str, Any], indent: int = 2) -> None:
    """Write JSON atomically using a temporary file and rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically using a temporary file and rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(content)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
