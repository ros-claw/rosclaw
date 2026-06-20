"""Hardware MCP permission store.

Persists granted/denied permission IDs per server in
``~/.rosclaw/mcp/permissions.yaml``. The store follows the manifest permission
levels: safe, guarded, sensitive, dangerous, forbidden_by_default.

* ``safe`` permissions are auto-granted.
* ``guarded`` / ``sensitive`` permissions require confirmation.
* ``dangerous`` permissions require explicit ``--allow-dangerous``.
* ``forbidden_by_default`` permissions are never auto-granted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.mcp.onboarding.schema import PermissionDecl, Permissions


AUTO_GRANT_LEVELS = {"safe"}
REQUIRES_CONFIRMATION_LEVELS = {"guarded", "sensitive"}
REQUIRES_EXPLICIT_FLAG_LEVELS = {"dangerous"}
FORBIDDEN_LEVELS = {"forbidden_by_default"}


@dataclass
class PermissionState:
    """Effective permission state for one MCP server."""

    granted: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "granted": list(self.granted),
            "denied": list(self.denied),
            "pending": list(self.pending),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PermissionState":
        data = data or {}
        return cls(
            granted=list(data.get("granted", [])),
            denied=list(data.get("denied", [])),
            pending=list(data.get("pending", [])),
        )


class PermissionStore:
    """Read/write the MCP permission store."""

    FILENAME = "permissions.yaml"

    def __init__(self, home: Path | None = None) -> None:
        self.home = resolve_home(str(home) if home else None)
        self.path = self.home / "mcp" / self.FILENAME

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, PermissionState]:
        """Load permission states keyed by server_name."""
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
            str(k): PermissionState.from_dict(v)
            for k, v in data.get("servers", {}).items()
            if isinstance(v, dict)
        }

    def save(self, servers: dict[str, PermissionState]) -> None:
        """Write the permission store atomically."""
        self._ensure_dir()
        payload = {
            "schema_version": "1.0",
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "servers": {k: v.to_dict() for k, v in servers.items()},
        }
        tmp = self.path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, server_name: str) -> PermissionState:
        """Return the permission state for a server (creating if absent)."""
        return self.load().get(server_name, PermissionState())

    def grant(self, server_name: str, permission_id: str) -> None:
        """Grant a permission to a server."""
        servers = self.load()
        state = servers.setdefault(server_name, PermissionState())
        if permission_id in state.denied:
            state.denied.remove(permission_id)
        if permission_id not in state.granted:
            state.granted.append(permission_id)
        if permission_id in state.pending:
            state.pending.remove(permission_id)
        self.save(servers)

    def deny(self, server_name: str, permission_id: str) -> None:
        """Deny a permission for a server."""
        servers = self.load()
        state = servers.setdefault(server_name, PermissionState())
        if permission_id in state.granted:
            state.granted.remove(permission_id)
        if permission_id in state.pending:
            state.pending.remove(permission_id)
        if permission_id not in state.denied:
            state.denied.append(permission_id)
        self.save(servers)

    def is_granted(self, server_name: str, permission_id: str) -> bool:
        """Check whether a permission is currently granted."""
        return permission_id in self.get(server_name).granted

    def is_denied(self, server_name: str, permission_id: str) -> bool:
        """Check whether a permission is currently denied."""
        return permission_id in self.get(server_name).denied

    def compute_effective(
        self,
        server_name: str,
        permissions: Permissions,
        allow_dangerous: bool = False,
    ) -> PermissionState:
        """Compute the effective permission state from the manifest model.

        Args:
            server_name: Server to compute permissions for.
            permissions: Manifest permission declaration.
            allow_dangerous: Whether dangerous permissions may be auto-granted.

        Returns:
            PermissionState with granted, denied, and pending lists.
        """
        stored = self.get(server_name)
        granted: list[str] = list(stored.granted)
        denied: list[str] = list(stored.denied)
        pending: list[str] = []

        def classify(decl: PermissionDecl) -> None:
            if decl.id in granted or decl.id in denied:
                return
            if decl.level in AUTO_GRANT_LEVELS:
                granted.append(decl.id)
            elif decl.level in FORBIDDEN_LEVELS:
                denied.append(decl.id)
            elif decl.level in REQUIRES_EXPLICIT_FLAG_LEVELS:
                if allow_dangerous:
                    granted.append(decl.id)
                else:
                    pending.append(decl.id)
            else:
                pending.append(decl.id)

        for decl in permissions.required:
            classify(decl)
        for decl in permissions.optional:
            classify(decl)

        return PermissionState(granted=granted, denied=denied, pending=pending)

    def apply_effective(
        self,
        server_name: str,
        permissions: Permissions,
        allow_dangerous: bool = False,
    ) -> PermissionState:
        """Compute and persist the effective permission state."""
        state = self.compute_effective(server_name, permissions, allow_dangerous)
        servers = self.load()
        servers[server_name] = state
        self.save(servers)
        return state

    def list_required_ungranted(
        self,
        server_name: str,
        permissions: Permissions,
    ) -> list[str]:
        """Return required permission IDs that are not effectively granted.

        Safe auto-grants are taken into account, so safe permissions do not
        appear in the result unless they were explicitly denied.
        """
        state = self.compute_effective(server_name, permissions)
        return [
            decl.id
            for decl in permissions.required
            if decl.id not in state.granted and decl.id not in state.denied
        ]
