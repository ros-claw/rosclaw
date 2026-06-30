"""Installed Hardware MCP registry.

Stores locally installed MCP servers in ``~/.rosclaw/mcp/installed.yaml``.
Each record tracks the manifest, version, installation path, body binding key,
and current status so that ``rosclaw mcp list`` and ``health`` can operate
without re-resolving the hub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rosclaw.firstboot.workspace import resolve_home


@dataclass
class InstalledRecord:
    """One installed Hardware MCP server record."""

    server_name: str
    manifest_id: str
    name: str
    version: str
    installed_at: str
    artifact_type: str
    server_dir: str
    runtime_config_path: str | None = None
    body_binding_key: str | None = None
    eurdf_profile: str | None = None
    status: str = "installed"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "server_name": self.server_name,
            "manifest_id": self.manifest_id,
            "name": self.name,
            "version": self.version,
            "installed_at": self.installed_at,
            "artifact_type": self.artifact_type,
            "server_dir": self.server_dir,
        }
        if self.runtime_config_path:
            result["runtime_config_path"] = self.runtime_config_path
        if self.body_binding_key:
            result["body_binding_key"] = self.body_binding_key
        if self.eurdf_profile:
            result["eurdf_profile"] = self.eurdf_profile
        result["status"] = self.status
        if self.extra:
            result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstalledRecord:
        extra = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "server_name",
                "manifest_id",
                "name",
                "version",
                "installed_at",
                "artifact_type",
                "server_dir",
                "runtime_config_path",
                "body_binding_key",
                "eurdf_profile",
                "status",
            }
        }
        return cls(
            server_name=data["server_name"],
            manifest_id=data["manifest_id"],
            name=data["name"],
            version=data["version"],
            installed_at=data.get(
                "installed_at", datetime.now(UTC).isoformat().replace("+00:00", "Z")
            ),
            artifact_type=data.get("artifact_type", "pypi"),
            server_dir=data["server_dir"],
            runtime_config_path=data.get("runtime_config_path"),
            body_binding_key=data.get("body_binding_key"),
            eurdf_profile=data.get("eurdf_profile"),
            status=data.get("status", "installed"),
            extra=extra,
        )


class InstalledRegistry:
    """Read/write the installed MCP registry."""

    FILENAME = "installed.yaml"

    def __init__(self, home: Path | str | None = None) -> None:
        self.home = resolve_home(str(home) if home else None)
        self.path = self.home / "mcp" / self.FILENAME

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, InstalledRecord]:
        """Load the registry as a mapping keyed by server_name."""
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
            str(k): InstalledRecord.from_dict(v)
            for k, v in data.get("servers", {}).items()
            if isinstance(v, dict)
        }

    def save(self, servers: dict[str, InstalledRecord]) -> None:
        """Write the registry atomically."""
        self._ensure_dir()
        payload = {
            "schema_version": "1.0",
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "servers": {k: v.to_dict() for k, v in servers.items()},
        }
        tmp = self.path.with_suffix(".yaml.tmp")
        tmp.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        tmp.replace(self.path)

    def get(self, server_name: str) -> InstalledRecord | None:
        """Return one installed record or None."""
        return self.load().get(server_name)

    def list(self) -> list[InstalledRecord]:
        """Return all installed records."""
        return list(self.load().values())

    def add(self, record: InstalledRecord) -> None:
        """Add or update a record keyed by server_name."""
        servers = self.load()
        servers[record.server_name] = record
        self.save(servers)

    def remove(self, server_name: str) -> InstalledRecord | None:
        """Remove a record and return it if it existed."""
        servers = self.load()
        record = servers.pop(server_name, None)
        if record is not None:
            self.save(servers)
        return record

    def update_status(self, server_name: str, status: str) -> InstalledRecord | None:
        """Update the status field of an existing record."""
        servers = self.load()
        record = servers.get(server_name)
        if record is None:
            return None
        record.status = status
        self.save(servers)
        return record
