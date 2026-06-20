"""Staged installation rollback support.

``RollbackContext`` keeps a snapshot of every file that an installation step
mutates. If anything fails, ``rollback()`` restores the originals and removes
the staging directory. On success, ``commit()`` discards the snapshots.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.mcp.onboarding.errors import RollbackError


@dataclass
class FileSnapshot:
    """A single file backup."""

    original: Path
    backup: Path


class RollbackContext:
    """Track file changes and enable rollback.

    The context is intentionally simple: each ``backup(path)`` call copies the
    current file (if it exists) into the staging directory. ``rollback()``
    restores files in reverse order and deletes newly-created files that were
    not backed up.
    """

    def __init__(self, staging_dir: Path) -> None:
        self.staging_dir = staging_dir
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._backups: list[FileSnapshot] = []
        self._created: list[Path] = []
        self._record: list[dict[str, Any]] = []

    def backup(self, path: Path) -> None:
        """Back up ``path`` if it exists; otherwise remember it as created."""
        if not path.exists():
            self._created.append(path)
            self._record.append({"action": "created", "path": str(path)})
            return
        backup_path = self._unique_backup(path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, backup_path, dirs_exist_ok=True)
        else:
            shutil.copy2(path, backup_path)
        self._backups.append(FileSnapshot(original=path, backup=backup_path))
        self._record.append({"action": "backed_up", "path": str(path), "backup": str(backup_path)})

    def _unique_backup(self, path: Path) -> Path:
        """Generate a non-conflicting backup path inside staging."""
        relative = path.relative_to(Path("/")) if path.is_absolute() else path
        base = self.staging_dir / relative
        candidate = base
        counter = 1
        while candidate.exists():
            candidate = base.with_suffix(f"{base.suffix}.bak{counter}")
            counter += 1
        return candidate

    def rollback(self) -> None:
        """Restore all backups and remove created files."""
        errors: list[str] = []
        # Restore in reverse order so nested overrides are unwound correctly.
        for snapshot in reversed(self._backups):
            try:
                if snapshot.original.exists() and snapshot.original.is_dir():
                    shutil.rmtree(snapshot.original)
                snapshot.backup.parent.mkdir(parents=True, exist_ok=True)
                if snapshot.backup.is_dir():
                    shutil.copytree(snapshot.backup, snapshot.original, dirs_exist_ok=True)
                else:
                    shutil.copy2(snapshot.backup, snapshot.original)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{snapshot.original}: {exc}")
        for created in reversed(self._created):
            try:
                if created.is_dir():
                    shutil.rmtree(created)
                elif created.exists():
                    created.unlink()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{created}: {exc}")
        if errors:
            raise RollbackError("Rollback completed with errors: " + "; ".join(errors))

    def commit(self) -> None:
        """Discard backups on successful installation."""
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)

    def record(self) -> list[dict[str, Any]]:
        """Return the rollback journal for debugging."""
        return list(self._record)
