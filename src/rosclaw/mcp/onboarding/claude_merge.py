"""Merge Hardware MCP server fragments into the project ``.mcp.json``.

Preserves servers that are not managed by ROSClaw. Managed servers carry an
``x-rosclaw.managed`` metadata block so that upgrades and health checks can
identify them.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.mcp.onboarding.errors import ClaudeMergeError

MANAGED_KEY = "x-rosclaw.managed"


@dataclass
class ClaudeMergeResult:
    """Result of merging a managed server into ``.mcp.json``."""

    path: Path
    server_name: str
    action: str  # created, upgraded, renamed, replaced, skipped
    previous_backup: Path | None = None
    unmanaged_conflict_server: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "server_name": self.server_name,
            "action": self.action,
            "previous_backup": str(self.previous_backup) if self.previous_backup else None,
            "unmanaged_conflict_server": self.unmanaged_conflict_server,
            "errors": list(self.errors),
        }


def _find_project_root(start: Path | None = None) -> Path:
    """Locate the project root from the current working directory."""
    start = start or Path.cwd()
    markers = {".git", "pyproject.toml", "setup.py", ".rosclaw", "CLAUDE.md"}
    for path in [start, *start.parents]:
        if any((path / marker).exists() for marker in markers):
            return path
    return start


def _is_managed_server(server_config: dict[str, Any]) -> bool:
    """Return True if the server config is managed by ROSClaw."""
    return isinstance(server_config.get(MANAGED_KEY), dict)


def _backup_existing(path: Path) -> Path:
    """Create a timestamped backup of an existing ``.mcp.json``."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    backup = path.with_suffix(f".mcp.json.bak.{timestamp}")
    shutil.copy2(path, backup)
    return backup


def _write_mcp_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write ``.mcp.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".mcp.json.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


class ClaudeMcpMerge:
    """Merge a Hardware MCP manifest fragment into ``.mcp.json``."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = _find_project_root(project_root)
        self.mcp_json_path = self.project_root / ".mcp.json"

    def load(self) -> dict[str, Any]:
        """Load the existing ``.mcp.json`` or an empty dict."""
        if not self.mcp_json_path.exists():
            return {}
        try:
            return json.loads(self.mcp_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ClaudeMergeError(
                f"Invalid JSON in {self.mcp_json_path}: {exc}"
            ) from exc

    def merge(
        self,
        server_name: str,
        manifest_id: str,
        version: str,
        mcp_json_fragment: dict[str, Any],
        conflict: str = "abort",
        dry_run: bool = False,
    ) -> ClaudeMergeResult:
        """Merge ``mcpJson`` fragment for ``server_name``.

        Args:
            server_name: The MCP server name (key in ``mcpServers``).
            manifest_id: Canonical manifest ID for managed metadata.
            version: Installed manifest version.
            mcp_json_fragment: Dict containing ``mcpServers`` and optional extra keys.
            conflict: How to handle an unmanaged name collision:
                ``abort`` (default), ``rename``, or ``replace``.
            dry_run: If True, do not write files.

        Returns:
            ``ClaudeMergeResult`` describing what happened.
        """
        servers_fragment = mcp_json_fragment.get("mcpServers", {})
        if server_name not in servers_fragment:
            raise ClaudeMergeError(
                f"mcpJson fragment does not contain server '{server_name}'"
            )

        server_config = dict(servers_fragment[server_name])
        server_config[MANAGED_KEY] = {
            "manifest_id": manifest_id,
            "version": version,
            "managed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

        existing = self.load()
        existing_servers: dict[str, Any] = existing.setdefault("mcpServers", {})

        result = ClaudeMergeResult(
            path=self.mcp_json_path,
            server_name=server_name,
            action="created",
        )

        if server_name in existing_servers:
            existing_config = existing_servers[server_name]
            if _is_managed_server(existing_config):
                # Managed upgrade path.
                managed_meta = existing_config.get(MANAGED_KEY, {})
                if managed_meta.get("manifest_id") == manifest_id:
                    result.action = "upgraded"
                else:
                    result.action = "replaced"
            else:
                # Unmanaged collision.
                result.unmanaged_conflict_server = server_name
                if conflict == "abort":
                    raise ClaudeMergeError(
                        f"Unmanaged MCP server '{server_name}' already exists in "
                        f"{self.mcp_json_path}. Use --conflict=rename or --conflict=replace."
                    )
                if conflict == "rename":
                    new_name = f"{server_name}-unmanaged"
                    counter = 1
                    while new_name in existing_servers:
                        new_name = f"{server_name}-unmanaged-{counter}"
                        counter += 1
                    existing_servers[new_name] = existing_config
                    result.unmanaged_conflict_server = new_name
                    result.action = "renamed"
                elif conflict == "replace":
                    result.action = "replaced"
                else:
                    raise ClaudeMergeError(
                        f"Unknown conflict strategy: {conflict}"
                    )

        existing_servers[server_name] = server_config

        # Merge any top-level keys from the fragment other than mcpServers.
        for key, value in mcp_json_fragment.items():
            if key == "mcpServers":
                continue
            if key in existing and isinstance(existing[key], dict) and isinstance(value, dict):
                existing[key].update(value)
            else:
                existing[key] = value

        if dry_run:
            result.action = f"dry-run:{result.action}"
            return result

        backup: Path | None = None
        if self.mcp_json_path.exists():
            backup = _backup_existing(self.mcp_json_path)
            result.previous_backup = backup

        _write_mcp_json(self.mcp_json_path, existing)
        return result

    def remove_managed_server(self, server_name: str, dry_run: bool = False) -> ClaudeMergeResult:
        """Remove a managed server from ``.mcp.json``."""
        existing = self.load()
        servers = existing.get("mcpServers", {})
        if server_name not in servers:
            return ClaudeMergeResult(
                path=self.mcp_json_path,
                server_name=server_name,
                action="not_found",
            )

        config = servers[server_name]
        if not _is_managed_server(config):
            raise ClaudeMergeError(
                f"Server '{server_name}' is not managed by ROSClaw; refusing to remove"
            )

        result = ClaudeMergeResult(
            path=self.mcp_json_path,
            server_name=server_name,
            action="removed",
        )
        if dry_run:
            result.action = "dry-run:removed"
            return result

        backup = _backup_existing(self.mcp_json_path) if self.mcp_json_path.exists() else None
        result.previous_backup = backup
        del servers[server_name]
        if not servers:
            existing.pop("mcpServers", None)
        _write_mcp_json(self.mcp_json_path, existing)
        return result

    def list_managed_servers(self) -> dict[str, dict[str, Any]]:
        """Return all ROSClaw-managed servers from ``.mcp.json``."""
        existing = self.load()
        return {
            name: config
            for name, config in existing.get("mcpServers", {}).items()
            if _is_managed_server(config)
        }
