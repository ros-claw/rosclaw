"""Multi-body registry persistence and management.

A single ROSClaw workspace may host multiple physical or simulated bodies. This
module persists a lightweight registry mapping body IDs to their storage
 directories, and tracks which body is currently "active".
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rosclaw.body.compiler import compute_checksum
from rosclaw.body.notes import MaintenanceLog
from rosclaw.body.schema import BodyRegistry as BodyRegistrySchema
from rosclaw.body.schema import BodyRegistryEntry, BodyYaml, CalibrationYaml, EurdfProfile, _utc_now
from rosclaw.eurdf.registry import RobotRegistry


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
        source: str = "builtin",
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

        # Resolve the robot profile.
        resolved_profile_id = self._resolve_profile_alias(profile_id)
        version = profile_version or "latest"
        if version == "latest":
            version = "1.0.0"
        profile = RobotRegistry().get(resolved_profile_id)
        if profile is None:
            raise BodyRegistryError(f"e-URDF profile not found: {profile_id}")

        eurdf = EurdfProfile.from_robot_complete_profile(profile)
        now = _utc_now()

        # Remove any existing data when forcing.
        if existing is not None:
            self._remove_body_data(body_id, archive=False)
            registry = self.load()

        body_dir = self._body_dir(body_id)
        refs_dir = body_dir / "refs"
        snapshots_dir = body_dir / "snapshots"
        generated_dir = body_dir / "generated"
        body_dir.mkdir(parents=True, exist_ok=True)
        refs_dir.mkdir(exist_ok=True)
        snapshots_dir.mkdir(exist_ok=True)
        generated_dir.mkdir(exist_ok=True)

        # Write normalized e-URDF profile.
        eurdf_path = refs_dir / "eurdf.profile.yaml"
        with open(eurdf_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(eurdf.to_dict(), f, sort_keys=False, allow_unicode=True)
        checksum = compute_checksum(eurdf_path)

        # Write lock file.
        lock = {
            "schema_version": "rosclaw.eurdf_lock.v1",
            "profile_id": resolved_profile_id,
            "profile_version": version,
            "uri": f"rosclaw://eurdf/{profile_id}@{version}",
            "source": source,
            "checksum": checksum,
            "locked_at": now,
        }
        with open(refs_dir / "eurdf.lock", "w", encoding="utf-8") as f:
            yaml.safe_dump(lock, f, sort_keys=False, allow_unicode=True)

        # Write body.yaml.
        body_yaml = BodyYaml(
            body_instance={
                "id": body_id,
                "nickname": nickname or body_id,
                "robot_model": profile_id,
                "serial_number": "UNKNOWN",
                "owner": "local",
                "deployment_site": "lab",
                "created_at": now,
                "updated_at": now,
            },
            model_ref={
                "eurdf_uri": f"rosclaw://eurdf/{profile_id}@{version}",
                "profile_id": profile_id,
                "profile_version": version,
                "profile_checksum": checksum,
                "lock_file": "refs/eurdf.lock",
            },
            calibration={
                "file": "calibration.yaml",
                "checksum": "sha256:uninitialized",
                "last_calibrated_at": None,
                "status": "factory_default",
            },
            maintenance={
                "log_file": "maintenance.log",
                "last_event_at": None,
                "safety_relevant_open_items": [],
            },
            installed_components=_default_installed_components(eurdf),
            capabilities={"enabled": [], "disabled": [], "degraded": []},
            prohibited_capabilities=[],
            safety_overrides={},
            runtime_state={
                "battery_percent": None,
                "last_seen_at": None,
                "health": "unknown",
                "online": False,
            },
            fingerprint={
                "effective_body_hash": None,
                "last_compiled_at": None,
                "last_skill_check_at": None,
            },
            compatibility_summary={
                "compatible_skills": 0,
                "degraded_skills": 0,
                "blocked_skills": 0,
                "unknown_skills": 0,
            },
        )
        with open(body_dir / "body.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(body_yaml.to_dict(), f, sort_keys=False, allow_unicode=True)

        # Write calibration.yaml.
        calibration = CalibrationYaml(
            body_instance_id=body_id,
            model_ref=f"rosclaw://eurdf/{resolved_profile_id}@{version}",
            validation={
                "status": "factory_default",
                "last_validated_at": None,
                "errors": [],
                "warnings": [],
            },
        )
        with open(body_dir / "calibration.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(calibration.to_dict(), f, sort_keys=False, allow_unicode=True)

        # Write initial maintenance log.
        MaintenanceLog(body_dir / "maintenance.log").write_init_event(
            body_id, f"rosclaw://eurdf/{profile_id}@{version}"
        )

        entry = BodyRegistryEntry(
            body_id=body_id,
            nickname=nickname or body_id,
            profile_id=profile_id,
            profile_version=version,
            created_at=now,
            updated_at=now,
            path=f"bodies/{body_id}",
            source=source,
            tags=list(tags or []),
        )
        registry.bodies[body_id] = entry
        registry.current_body_id = body_id
        self.save(registry)

        return entry

    # Common user-facing profile IDs → zoo directory names.
    _PROFILE_ALIASES: dict[str, str] = {
        "unitree-g1": "g1",
        "unitree-go2": "unitree_go2",
        "franka-panda": "franka_panda",
        "fetch": "fetch_robot",
    }

    def _resolve_profile_alias(self, profile_id: str) -> str:
        """Normalize and resolve a profile ID alias.

        Maps common user-facing names (e.g. ``unitree-g1``) to the e-URDF zoo
        directory name (``g1``). Unrecognized IDs are returned lower-cased and
        stripped so that direct directory names still work.
        """
        normalized = profile_id.strip().lower()
        return self._PROFILE_ALIASES.get(normalized, normalized)

    def remove_body(self, body_id: str, archive: bool = True) -> BodyRemoval:
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

        removal = self._remove_body_data(body_id, archive=archive)
        # Reload cache so subsequent reads see the removal immediately.
        self._cache = None
        self._cache_mtime = None
        return removal

    def _remove_body_data(self, body_id: str, archive: bool = True) -> BodyRemoval:
        """Delete or archive a body's on-disk data without touching the registry."""
        body_id = self._normalize_id(body_id)
        body_dir = self._body_dir(body_id)
        archive_path: Path | None = None
        archived = False
        if body_dir.exists():
            if archive:
                self.archive_root.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                archive_path = self.archive_root / f"{body_id}-{ts}"
                # Avoid collisions when multiple removals share a timestamp.
                counter = 0
                original = archive_path
                while archive_path.exists():
                    counter += 1
                    archive_path = original.parent / f"{original.name}_{counter}"
                body_dir.rename(archive_path)
                archived = True
            else:
                shutil.rmtree(body_dir)
        return BodyRemoval(body_id=body_id, archived=archived, archive_path=archive_path)

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


def _default_installed_components(eurdf: EurdfProfile) -> dict[str, Any]:
    """Build default installed_components from e-URDF sensors/actuators."""
    sensors = {}
    for sensor in eurdf.sensors:
        name = sensor.get("name")
        if name:
            sensors[name] = {
                "installed": True,
                "status": "available",
                "provider_ref": None,
                "notes": [],
            }
    actuators = {}
    for actuator in eurdf.actuators:
        name = actuator.get("name")
        if name:
            actuators[name] = {"installed": True, "status": "available", "notes": []}
    return {"sensors": sensors, "actuators": actuators}
