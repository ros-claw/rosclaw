"""Multi-body registry persistence and management.

A single ROSClaw workspace may host multiple physical or simulated bodies. This
module persists a lightweight registry mapping body IDs to their storage
 directories, and tracks which body is currently "active".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rosclaw.body.schema import BodyRegistry as BodyRegistrySchema
from rosclaw.body.schema import BodyRegistryEntry, _utc_now


class BodyRegistryError(RuntimeError):
    """Raised for registry-level mistakes (duplicate ID, invalid ID, etc.)."""


@dataclass
class BodyRemoval:
    """Result of a body removal operation."""

    body_id: str
    archived: bool
    archive_path: Path | None = None


class BodyRegistryManager:
    """Manage the workspace-level body registry.

    Backward compatibility:
      - If ``workspace/body_registry.yaml`` exists, it is the source of truth.
      - If it does not exist but ``workspace/body/`` does, the workspace is
        treated as a legacy single-body workspace. The legacy body is imported
        as body ID ``default`` the first time a mutating operation runs.
    """

    _ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)
        self.registry_path = self.workspace / "body_registry.yaml"
        self.bodies_root = self.workspace / "bodies"
        self.archive_root = self.bodies_root / "_archive"
        self._legacy_body_dir = self.workspace / "body"
        self._cache: BodyRegistrySchema | None = None
        self._cache_mtime: float | None = None

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------
    def load(self) -> BodyRegistrySchema:
        """Load the registry, auto-migrating legacy workspaces on first read.

        The result is cached in memory based on the registry file's mtime so
        repeated reads in the same manager instance do not re-parse YAML.
        """
        if self.registry_path.exists():
            mtime = self.registry_path.stat().st_mtime
            if self._cache is not None and self._cache_mtime == mtime:
                return self._cache
            with open(self.registry_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._cache = BodyRegistrySchema.from_dict(data)
            self._cache_mtime = mtime
            return self._cache

        registry = BodyRegistrySchema()
        if self._legacy_body_dir.exists():
            entry = self._legacy_entry()
            registry.bodies[entry.body_id] = entry
            registry.current_body_id = entry.body_id
        self._cache = registry
        self._cache_mtime = None
        return registry

    def save(self, registry: BodyRegistrySchema) -> None:
        """Persist the registry to disk and refresh the in-memory cache."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(registry.to_dict(), f, sort_keys=False, allow_unicode=True)
        self._cache = registry
        self._cache_mtime = self.registry_path.stat().st_mtime

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def list_bodies(self) -> list[BodyRegistryEntry]:
        """Return all registered bodies, current body first."""
        registry = self.load()
        current_id = registry.current_body_id
        bodies = list(registry.bodies.values())
        bodies.sort(key=lambda e: (0 if e.body_id == current_id else 1, e.body_id.lower()))
        return bodies

    def get_current_body_id(self) -> str:
        """Return the currently active body ID, migrating legacy if needed."""
        registry = self.load()
        if registry.current_body_id in registry.bodies:
            return registry.current_body_id
        if registry.bodies:
            first_id = next(iter(registry.bodies))
            registry.current_body_id = first_id
            self.save(registry)
            return first_id
        return "default"

    def set_current_body_id(self, body_id: str) -> None:
        """Set the active body ID."""
        body_id = self._normalize_id(body_id)
        registry = self.load()
        if body_id not in registry.bodies:
            raise BodyRegistryError(f"Body not found: {body_id}")
        registry.current_body_id = body_id
        self.save(registry)

    def get_body(self, body_id: str) -> BodyRegistryEntry | None:
        """Look up a single body entry by ID."""
        body_id = self._normalize_id(body_id)
        return self.load().bodies.get(body_id)

    def has_body(self, body_id: str) -> bool:
        """Return True if the given body ID is registered."""
        return self.get_body(body_id) is not None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def create_body(
        self,
        body_id: str,
        profile_id: str,
        nickname: str = "",
        profile_version: str = "",
        tags: list[str] | None = None,
        force: bool = False,
    ) -> BodyRegistryEntry:
        """Register a new body and create its directory scaffold.

        The newly created body becomes the current body so that subsequent
        commands operate on it without an explicit ``switch``.

        If a legacy ``body/`` directory exists but no registry file exists yet,
        the legacy body is imported first so the new registry has a ``default``
        entry before the new body is added.
        """
        body_id = self._normalize_id(body_id)
        self._validate_id(body_id)

        registry = self.load()

        # Migrate legacy body if present.
        if not self.registry_path.exists() and self._legacy_body_dir.exists():
            legacy = self._legacy_entry()
            registry.bodies[legacy.body_id] = legacy
            if registry.current_body_id not in registry.bodies:
                registry.current_body_id = legacy.body_id

        existing = registry.bodies.get(body_id)
        if existing is not None and not force:
            raise BodyRegistryError(f"Body already exists: {body_id}")

        now = _utc_now()
        entry = BodyRegistryEntry(
            body_id=body_id,
            nickname=nickname or body_id,
            profile_id=profile_id,
            profile_version=profile_version or "latest",
            created_at=now,
            updated_at=now,
            path=f"bodies/{body_id}",
            tags=list(tags or []),
        )
        registry.bodies[body_id] = entry
        registry.current_body_id = body_id
        self.save(registry)

        body_dir = self._body_dir(body_id)
        body_dir.mkdir(parents=True, exist_ok=True)
        (body_dir / "refs").mkdir(exist_ok=True)
        (body_dir / "snapshots").mkdir(exist_ok=True)
        (body_dir / "generated").mkdir(exist_ok=True)

        return entry

    def remove_body(self, body_id: str, archive: bool = False) -> BodyRemoval:
        """Remove a body from the registry and optionally archive its data."""
        body_id = self._normalize_id(body_id)
        registry = self.load()
        if body_id not in registry.bodies:
            raise BodyRegistryError(f"Body not found: {body_id}")

        registry.bodies.pop(body_id)

        # If we just removed the active body, point current at another body (if
        # any) before persisting so the registry is always internally consistent.
        if registry.current_body_id == body_id:
            registry.current_body_id = next(iter(registry.bodies), "default")

        self.save(registry)

        body_dir = self._body_dir(body_id)
        archive_path: Path | None = None
        if body_dir.exists():
            if archive:
                self.archive_root.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                archive_path = self.archive_root / f"{body_id}-{ts}"
                # Avoid collisions
                counter = 0
                original = archive_path
                while archive_path.exists():
                    counter += 1
                    archive_path = original.parent / f"{original.name}_{counter}"
                body_dir.rename(archive_path)
            else:
                import shutil

                shutil.rmtree(body_dir)

        return BodyRemoval(body_id=body_id, archived=archive, archive_path=archive_path)

    def migrate_legacy_body(self) -> BodyRegistryEntry | None:
        """Explicitly import a legacy ``workspace/body/`` directory.

        Returns the migrated entry, or None if there is nothing to migrate.
        """
        if not self._legacy_body_dir.exists():
            return None
        if self.registry_path.exists():
            registry = self.load()
            for entry in registry.bodies.values():
                if entry.path in ("body", "") or not entry.path:
                    return entry
            return None

        entry = self._legacy_entry()
        registry = BodyRegistrySchema(
            current_body_id=entry.body_id,
            bodies={entry.body_id: entry},
        )
        self.save(registry)
        return entry

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Return a concise statistics dict for CLI display."""
        registry = self.load()
        by_profile: dict[str, int] = {}
        for entry in registry.bodies.values():
            by_profile[entry.profile_id] = by_profile.get(entry.profile_id, 0) + 1
        archived_count = 0
        if self.archive_root.exists():
            archived_count = sum(1 for p in self.archive_root.iterdir() if p.is_dir())
        return {
            "total": len(registry.bodies),
            "current": registry.current_body_id,
            "by_profile": by_profile,
            "archived": archived_count,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _body_dir(self, body_id: str) -> Path:
        """Return the storage directory for a registered body."""
        entry = self.get_body(body_id)
        if entry and entry.path:
            if entry.path == "body":
                return self.workspace / "body"
            return self.workspace / entry.path
        return self.bodies_root / body_id

    def _legacy_entry(self) -> BodyRegistryEntry:
        """Build a registry entry for a legacy single-body workspace."""
        now = _utc_now()
        profile_id = ""
        # Try to infer profile_id from body.yaml if available.
        body_yaml_path = self._legacy_body_dir / "body.yaml"
        if body_yaml_path.exists():
            with open(body_yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            profile_id = data.get("model_ref", {}).get("profile_id", "")
        return BodyRegistryEntry(
            body_id="default",
            nickname="legacy-default",
            profile_id=profile_id,
            created_at=now,
            updated_at=now,
            path="body",
        )

    def _normalize_id(self, body_id: str) -> str:
        return body_id.strip().lower()

    def _validate_id(self, body_id: str) -> None:
        """Validate a body ID for format and emptiness only.

        Duplicate detection is the responsibility of the caller (e.g.
        ``create_body``) so it can be bypassed via ``force=True``.
        """
        if not body_id:
            raise BodyRegistryError("Body ID cannot be empty")
        if not self._ID_PATTERN.match(body_id):
            raise BodyRegistryError(
                f"Invalid body ID: {body_id}. Use only letters, numbers, hyphens, and underscores."
            )
