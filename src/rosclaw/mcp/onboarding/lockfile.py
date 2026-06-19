"""MCP lockfile for reproducible resolutions.

Stores resolved manifest IDs with exact versions in
``~/.rosclaw/mcp/lock.yaml``. The lockfile is updated after every successful
install so that subsequent operations (list, health) do not accidentally
upgrade a server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rosclaw.firstboot.workspace import resolve_home


@dataclass
class LockedPackage:
    """One locked manifest resolution."""

    manifest_id: str
    version: str
    name: str
    server_name: str
    locked_at: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "name": self.name,
            "server_name": self.server_name,
            "locked_at": self.locked_at,
        }
        if self.extra:
            result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LockedPackage":
        extra = {
            k: v
            for k, v in data.items()
            if k not in {"manifest_id", "version", "name", "server_name", "locked_at"}
        }
        return cls(
            manifest_id=data["manifest_id"],
            version=data["version"],
            name=data.get("name", ""),
            server_name=data.get("server_name", ""),
            locked_at=data.get("locked_at", datetime.now(UTC).isoformat().replace("+00:00", "Z")),
            extra=extra,
        )


class Lockfile:
    """Read/write the MCP resolution lockfile."""

    FILENAME = "lock.yaml"

    def __init__(self, home: Path | None = None) -> None:
        self.home = resolve_home(str(home) if home else None)
        self.path = self.home / "mcp" / self.FILENAME

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, LockedPackage]:
        """Load locks keyed by manifest_id."""
        if not self.path.exists():
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(k): LockedPackage.from_dict(v)
            for k, v in data.get("packages", {}).items()
            if isinstance(v, dict)
        }

    def save(self, packages: dict[str, LockedPackage]) -> None:
        """Write the lockfile atomically."""
        self._ensure_dir()
        payload = {
            "schema_version": "1.0",
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "packages": {k: v.to_dict() for k, v in packages.items()},
        }
        tmp = self.path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, manifest_id: str) -> LockedPackage | None:
        """Return the lock entry for a manifest ID."""
        return self.load().get(manifest_id)

    def get_locked_version(self, manifest_id: str) -> str | None:
        """Return the locked version, if any."""
        lock = self.get(manifest_id)
        return lock.version if lock else None

    def set(self, package: LockedPackage) -> None:
        """Add or update a lock entry keyed by manifest_id."""
        packages = self.load()
        packages[package.manifest_id] = package
        self.save(packages)

    def remove(self, manifest_id: str) -> LockedPackage | None:
        """Remove a lock entry and return it if it existed."""
        packages = self.load()
        package = packages.pop(manifest_id, None)
        if package is not None:
            self.save(packages)
        return package
