"""Safe merge of installed MCP servers into Claude Code MCP configuration.

The installer uses :class:`McpMerger` to add (and later remove) ``stdio`` MCP
server entries.  It updates:

* ``.mcp.json`` in the project root (Claude Code's primary config file)
* ``~/.rosclaw/runtime/mcp.d/<server>.json`` fragment for diagnostics and
  manual recovery

Every mutation creates a backup first and writes atomically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rosclaw.hub.cache import HubCache
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef
from rosclaw.hub.schema import AssetManifest


def _sanitize_server_name(ref: AssetRef) -> str:
    """Create a filesystem-safe MCP server name from an asset reference."""
    base = f"{ref.type}-{ref.namespace}-{ref.name}-{ref.version}"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", base).strip("-").lower()


def _default_mcp_config_path(project_root: Path) -> Path:
    """Return the default ``.mcp.json`` path in the project root."""
    return project_root / ".mcp.json"


class McpMerger:
    """Merge and un-merge ROSClaw Hub MCP server entries."""

    def __init__(
        self,
        project_root: str | Path,
        home: str | Path | None = None,
        *,
        cache: HubCache | None = None,
        mcp_config_path: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.cache = cache or HubCache(home)
        home_path = self.cache.home
        self.fragments_dir = home_path / "runtime" / "mcp.d"
        self.fragments_dir.mkdir(parents=True, exist_ok=True)
        if mcp_config_path is not None:
            self.mcp_config_path = Path(mcp_config_path)
        else:
            self.mcp_config_path = _default_mcp_config_path(self.project_root)

    def _load(self) -> dict[str, Any]:
        if not self.mcp_config_path.exists():
            return {"version": "1.0.0", "servers": {}}
        try:
            text = self.mcp_config_path.read_text(encoding="utf-8")
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HubError(
                code=HubErrorCode.MANIFEST_INVALID,
                message=f"Corrupt MCP config: {self.mcp_config_path}",
            ) from exc
        if not isinstance(data, dict):
            data = {"version": "1.0.0", "servers": {}}
        if "servers" not in data or not isinstance(data["servers"], dict):
            data["servers"] = {}
        return data

    def _write(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        if self.mcp_config_path.exists():
            self.cache.backup_file(self.mcp_config_path)
        self.cache._atomic_write(self.mcp_config_path, payload)

    def _entry_from_manifest(
        self,
        manifest: AssetManifest,
        asset_dir: Path,
    ) -> dict[str, Any]:
        """Build a Claude Code ``servers`` entry from a hardware_mcp manifest."""
        entrypoints: dict[str, Any] = manifest.install.get("entrypoints", {})
        mcp_ep = entrypoints.get("mcp", {})
        command = mcp_ep.get("command")
        if not command:
            raise HubError(
                code=HubErrorCode.MANIFEST_INVALID,
                message="MCP entrypoint is missing a command",
            )
        ref = AssetRef(
            type=manifest.asset.type.value,
            namespace=manifest.asset.namespace,
            name=manifest.asset.name,
            version=manifest.asset.version,
        )
        argv = command.split()
        env = mcp_ep.get("env") or {}
        # Mark this server as managed so the uninstaller can find it.
        return {
            "command": argv[0],
            "args": argv[1:],
            "env": {str(k): str(v) for k, v in env.items()},
            "transport": mcp_ep.get("transport", "stdio"),
            "rosclaw": {
                "ref": str(ref),
                "type": manifest.asset.type.value,
                "asset_dir": str(asset_dir),
                "installed_by": "rosclaw hub",
            },
        }

    def add_server(
        self,
        manifest: AssetManifest,
        asset_dir: Path,
    ) -> str:
        """Add or update an MCP server entry for *manifest*.

        Returns:
            The server name used in the ``servers`` map.
        """
        entry = self._entry_from_manifest(manifest, asset_dir)
        ref = entry["rosclaw"]["ref"]
        server_name = _sanitize_server_name(
            AssetRef(
                type=manifest.asset.type.value,
                namespace=manifest.asset.namespace,
                name=manifest.asset.name,
                version=manifest.asset.version,
            )
        )

        data = self._load()
        data["servers"][server_name] = entry
        self._write(data)

        fragment_path = self.fragments_dir / f"{server_name}.json"
        fragment = {
            "server_name": server_name,
            "ref": ref,
            "entry": entry,
        }
        self.cache._atomic_write(
            fragment_path,
            json.dumps(fragment, indent=2, ensure_ascii=False).encode("utf-8"),
        )
        return server_name

    def remove_server(self, ref: AssetRef) -> bool:
        """Remove the MCP server entry matching *ref*.

        Returns:
            True if an entry was removed from ``.mcp.json``.
        """
        data = self._load()
        servers: dict[str, Any] = data.get("servers", {})
        ref_str = str(ref)
        removed = False
        to_remove: list[str] = []
        for name, entry in servers.items():
            rosclaw_meta = entry.get("rosclaw", {})
            if rosclaw_meta.get("ref") == ref_str:
                to_remove.append(name)
        for name in to_remove:
            del servers[name]
            removed = True
            fragment_path = self.fragments_dir / f"{name}.json"
            if fragment_path.exists():
                self.cache.backup_file(fragment_path)
                fragment_path.unlink()
        if removed:
            self._write(data)
        return removed

    def list_servers(self) -> dict[str, dict[str, Any]]:
        """Return all ``rosclaw``-managed MCP server entries."""
        data = self._load()
        servers: dict[str, Any] = data.get("servers", {})
        return {
            name: entry
            for name, entry in servers.items()
            if isinstance(entry, dict) and "rosclaw" in entry
        }

    def is_managed(self, ref: AssetRef) -> bool:
        """Return whether *ref* already has a managed MCP server entry."""
        return str(ref) in {e.get("rosclaw", {}).get("ref") for e in self.list_servers().values()}
