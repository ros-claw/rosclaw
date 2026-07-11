"""Hardware MCP hub client.

Supports three resolution sources, in order of preference when ``offline=True``:

1. Local on-disk cache under ``~/.rosclaw/mcp/cache/``.
2. Built-in default registry (bundled with rosclaw).
3. Remote HTTP hub (``ROSCLAW_MCP_HUB`` or explicit ``hub_url``).

When ``offline=False`` the client also consults the public ROSClaw Hub at
``https://www.rosclaw.io/api/registry``.  Hub packages are returned as
lightweight metadata and normalized into the same ``McpManifest`` model used by
the built-in registry, so ``install --dry-run``, ``list``, and ``health`` all
work uniformly.  If the network is unreachable or the remote endpoint returns an
error, the client falls back to the cache and built-ins.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.mcp.onboarding.errors import ManifestError, ManifestNotFoundError
from rosclaw.mcp.onboarding.schema import McpManifest

# Public ROSClaw Hub endpoints.
DEFAULT_HUB_URL = "https://www.rosclaw.io"
REGISTRY_PATH = "/api/registry"
PACKAGE_LIST_PATH = "/api/mcp-packages"
REMOTE_CANONICAL_PREFIX = "io.rosclaw.hub."


# Minimal built-in registry for offline operation.
_BUILTIN_REGISTRY: dict[str, dict[str, Any]] = {
    "io.rosclaw.hardware.unitree-g1": {
        "$schema": "https://schemas.rosclaw.io/mcp/hardware-manifest.schema.json",
        "schemaVersion": "1.0.0",
        "id": "io.rosclaw.hardware.unitree-g1",
        "name": "unitree-g1",
        "displayName": "Unitree G1 Humanoid",
        "version": "1.0.0",
        "channel": "stable",
        "description": "ROSClaw Hardware MCP for Unitree G1 humanoid robot.",
        "tags": ["humanoid", "unitree", "ros2"],
        "categories": ["hardware"],
        "license": "MIT",
        "publisher": {
            "name": "ROSClaw Project",
            "namespace": "io.rosclaw.hardware",
            "homepage": "https://rosclaw.io",
            "support": "mcp@rosclaw.io",
            "verified": True,
        },
        "artifact": {
            "type": "python",
            "package": "rosclaw-mcp-unitree-g1",
            "version": "1.0.0",
            "entrypoint": "rosclaw_mcp_unitree_g1.server:main",
            "install": "pip install rosclaw-mcp-unitree-g1==1.0.0",
        },
        "mcp": {
            "serverName": "unitree-g1",
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "rosclaw_mcp_unitree_g1.server"],
            },
            "capabilities": {"tools": True, "resources": True},
        },
        "hardware": {
            "type": "robot",
            "vendor": "Unitree",
            "models": ["G1"],
            "connection": {"modes": ["ros2"], "defaultMode": "ros2"},
        },
        "eurdf": {
            "profiles": [
                {"id": "unitree-g1", "version": "1.0.0", "required": True},
            ],
            "defaultProfile": "unitree-g1",
        },
        "bodyBinding": {
            "bodyType": "unitree-g1",
            "bindingKey": "unitree_g1",
            "requiredFields": [
                "robot.model",
                "robot.serial",
                "robot.eurdf_profile",
            ],
            "writePaths": {
                "mcpBinding": "mcp_bindings.unitree_g1",
            },
            "template": {
                "mcp_bindings": {
                    "unitree_g1": {
                        "server": "rosclaw-unitree-g1",
                        "manifest_id": "io.rosclaw.hardware.unitree-g1",
                    }
                }
            },
        },
        "permissions": {
            "required": [
                {"id": "mcp:tools:read", "level": "safe", "description": "List and call MCP tools"},
                {"id": "mcp:resources:read", "level": "safe", "description": "Read resource URIs"},
                {
                    "id": "mcp:prompts:read",
                    "level": "guarded",
                    "description": "Read safety prompts",
                },
            ],
            "optional": [
                {"id": "mcp:logging:write", "level": "safe", "description": "Forward server logs"},
            ],
        },
        "install": {
            "preflight": [
                {"id": "ros_distro", "command": "python3 --version", "required": True},
                {"id": "python_version", "command": "python3 --version", "required": True},
            ],
            "postInstall": [{"id": "relink_body", "action": "rosclaw body relink"}],
        },
        "health": {
            "checks": [
                {"id": "install_integrity", "category": "install", "required": True},
                {"id": "protocol_stdio", "category": "protocol", "required": True},
                {"id": "eurdf_binding", "category": "binding", "required": True},
            ],
        },
        "claude": {
            "scopeDefault": "project",
            "mcpJson": {
                "mcpServers": {
                    "rosclaw-unitree-g1": {
                        "type": "stdio",
                        "command": "${ROSCLAW_HOME:-${HOME}/.rosclaw}/mcp/bin/rosclaw-mcp-run",
                        "args": ["unitree-g1"],
                        "env": {
                            "ROSCLAW_HOME": "${ROSCLAW_HOME:-${HOME}/.rosclaw}",
                            "ROS_DOMAIN_ID": "${ROS_DOMAIN_ID:-42}",
                        },
                    }
                }
            },
            "claudeMdSnippet": "This project has ROSClaw Hardware MCP enabled for Unitree G1. Use MCP tools for robot state, diagnostics, motion validation, and guarded execution. Never call raw execution tools unless explicitly approved.",
        },
        "compatibility": {
            "rosDistros": ["humble", "jazzy"],
            "python": ">=3.10",
            "rosclaw": ">=1.0.0",
        },
        "lifecycle": {"deprecated": False},
    },
    "io.rosclaw.hardware.realsense-d455": {
        "$schema": "https://schemas.rosclaw.io/mcp/hardware-manifest.schema.json",
        "schemaVersion": "1.0.0",
        "id": "io.rosclaw.hardware.realsense-d455",
        "name": "realsense-d455",
        "displayName": "Intel RealSense D455",
        "version": "1.0.0",
        "channel": "stable",
        "description": "ROSClaw Hardware MCP for Intel RealSense D455 depth camera.",
        "tags": ["camera", "realsense", "sensor"],
        "categories": ["hardware", "sensor"],
        "license": "MIT",
        "publisher": {
            "name": "ROSClaw Project",
            "namespace": "io.rosclaw.hardware",
            "homepage": "https://rosclaw.io",
            "support": "mcp@rosclaw.io",
            "verified": True,
        },
        "artifact": {
            "type": "python",
            "package": "rosclaw-mcp-realsense-d455",
            "version": "1.0.0",
            "entrypoint": "rosclaw_mcp_realsense_d455.server:main",
            "install": "pip install rosclaw-mcp-realsense-d455==1.0.0",
        },
        "mcp": {
            "serverName": "realsense-d455",
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "rosclaw_mcp_realsense_d455.server"],
            },
            "capabilities": {"tools": True, "resources": True},
        },
        "hardware": {
            "type": "sensor",
            "vendor": "Intel",
            "models": ["RealSense D455"],
            "connection": {"modes": ["ros2"], "defaultMode": "ros2"},
        },
        "eurdf": {
            "profiles": [
                {"id": "realsense-d455", "version": "1.0.0", "required": False},
            ],
            "defaultProfile": "realsense-d455",
        },
        "bodyBinding": {
            "bodyType": "sensor",
            "bindingKey": "realsense_d455",
            "writePaths": {
                "mcpBinding": "mcp_bindings.realsense_d455",
            },
            "template": {
                "mcp_bindings": {
                    "realsense_d455": {
                        "server": "rosclaw-realsense-d455",
                        "manifest_id": "io.rosclaw.hardware.realsense-d455",
                    }
                }
            },
        },
        "permissions": {
            "required": [
                {"id": "mcp:tools:read", "level": "safe", "description": "List and call MCP tools"},
                {"id": "mcp:resources:read", "level": "safe", "description": "Read resource URIs"},
            ],
            "optional": [
                {"id": "mcp:logging:write", "level": "safe", "description": "Forward server logs"},
            ],
        },
        "install": {
            "preflight": [
                {"id": "ros_distro", "command": "python3 --version", "required": True},
            ],
            "postInstall": [{"id": "relink_body", "action": "rosclaw body relink"}],
        },
        "health": {
            "checks": [
                {"id": "install_integrity", "category": "install", "required": True},
                {"id": "protocol_stdio", "category": "protocol", "required": True},
                {"id": "eurdf_binding", "category": "binding", "required": False},
            ],
        },
        "claude": {
            "scopeDefault": "project",
            "mcpJson": {
                "mcpServers": {
                    "rosclaw-realsense-d455": {
                        "type": "stdio",
                        "command": "${ROSCLAW_HOME:-${HOME}/.rosclaw}/mcp/bin/rosclaw-mcp-run",
                        "args": ["realsense-d455"],
                        "env": {
                            "ROSCLAW_HOME": "${ROSCLAW_HOME:-${HOME}/.rosclaw}",
                        },
                    }
                }
            },
        },
        "compatibility": {
            "rosDistros": ["humble", "jazzy"],
            "python": ">=3.10",
            "rosclaw": ">=1.0.0",
        },
        "lifecycle": {"deprecated": False},
    },
    "io.rosclaw.hardware.librealsense-mcp": {
        "$schema": "https://schemas.rosclaw.io/mcp/hardware-manifest.schema.json",
        "schemaVersion": "1.0.0",
        "id": "io.rosclaw.hardware.librealsense-mcp",
        "name": "librealsense-mcp",
        "displayName": "Intel RealSense (pyrealsense2) MCP",
        "version": "1.0.0",
        "channel": "stable",
        "description": "MCP server for Intel RealSense via pyrealsense2 SDK. Install from source with --from-git.",
        "tags": ["camera", "realsense", "sensor", "pyrealsense2"],
        "categories": ["hardware", "sensor"],
        "license": "MIT",
        "publisher": {
            "name": "ROSClaw Project",
            "namespace": "io.rosclaw.hardware",
            "homepage": "https://rosclaw.io",
            "support": "mcp@rosclaw.io",
            "verified": True,
        },
        "artifact": {
            "type": "git",
            "url": "https://github.com/ros-claw/librealsense-mcp",
            "entrypoint": "mcp_server.py",
        },
        "mcp": {
            "serverName": "librealsense-mcp",
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": ["mcp_server.py"],
            },
            "capabilities": {"tools": True},
        },
        "hardware": {
            "type": "sensor",
            "vendor": "Intel",
            "models": ["RealSense D400 Series"],
            "connection": {"modes": ["usb"], "defaultMode": "usb"},
        },
        "permissions": {
            "required": [
                {"id": "mcp:tools:read", "level": "safe", "description": "List and call MCP tools"},
            ],
        },
        "health": {
            "checks": [
                {"id": "install_integrity", "category": "install", "required": True},
                {"id": "protocol_stdio", "category": "protocol", "required": True},
            ],
        },
        "compatibility": {
            "python": ">=3.10",
            "rosclaw": ">=1.0.0",
        },
        "lifecycle": {"deprecated": False},
    },
    "io.rosclaw.hardware.realsense-ros-mcp": {
        "$schema": "https://schemas.rosclaw.io/mcp/hardware-manifest.schema.json",
        "schemaVersion": "1.0.0",
        "id": "io.rosclaw.hardware.realsense-ros-mcp",
        "name": "realsense-ros-mcp",
        "displayName": "Intel RealSense ROS2 MCP",
        "version": "1.0.0",
        "channel": "stable",
        "description": "MCP server for Intel RealSense via ROS2/realsense2_camera. Install from source with --from-git.",
        "tags": ["camera", "realsense", "sensor", "ros2"],
        "categories": ["hardware", "sensor"],
        "license": "MIT",
        "publisher": {
            "name": "ROSClaw Project",
            "namespace": "io.rosclaw.hardware",
            "homepage": "https://rosclaw.io",
            "support": "mcp@rosclaw.io",
            "verified": True,
        },
        "artifact": {
            "type": "git",
            "url": "https://github.com/ros-claw/realsense-ros-mcp",
            "entrypoint": "mcp_server.py",
        },
        "mcp": {
            "serverName": "realsense-ros-mcp",
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": ["mcp_server.py"],
            },
            "capabilities": {"tools": True},
        },
        "hardware": {
            "type": "sensor",
            "vendor": "Intel",
            "models": ["RealSense D400 Series"],
            "connection": {"modes": ["ros2"], "defaultMode": "ros2"},
        },
        "permissions": {
            "required": [
                {"id": "mcp:tools:read", "level": "safe", "description": "List and call MCP tools"},
            ],
        },
        "health": {
            "checks": [
                {"id": "install_integrity", "category": "install", "required": True},
                {"id": "protocol_stdio", "category": "protocol", "required": True},
            ],
        },
        "compatibility": {
            "python": ">=3.10",
            "rosclaw": ">=1.0.0",
        },
        "lifecycle": {"deprecated": False},
    },
}


def _remote_canonical_id(pkg_name: str) -> str:
    """Return a canonical manifest ID for a Hub package name."""
    return f"{REMOTE_CANONICAL_PREFIX}{pkg_name.replace('/', '.')}"


def _pkg_name_from_remote_canonical(manifest_id: str) -> str | None:
    """Reverse a remote canonical ID back to ``owner/repo``."""
    if not manifest_id.startswith(REMOTE_CANONICAL_PREFIX):
        return None
    suffix = manifest_id[len(REMOTE_CANONICAL_PREFIX) :]
    if "." not in suffix:
        return None
    return suffix.replace(".", "/", 1)


def _metadata_to_manifest(metadata: dict[str, Any], pkg_name: str | None = None) -> dict[str, Any]:
    """Convert a lightweight Hub package record into a full hardware manifest dict.

    The remote registry only carries distribution metadata (git URL, entry point,
    dependencies, etc.).  This helper fills in safe defaults for theHardware MCP
    schema so that planning, listing, and dry-run health checks work out of the
    box.  Real installation of remote/git artifacts still requires a remote
    installer implementation.
    """
    name = metadata.get("name") or pkg_name or "unknown"
    short_name = name.split("/")[-1]
    version = metadata.get("version") or "1.0.0"
    owner = name.split("/")[0] if "/" in name else "unknown"
    author = metadata.get("author") or owner
    git_url = metadata.get("git_url") or ""
    entry_point = metadata.get("entry_point") or "server.py"
    dependencies = list(metadata.get("dependencies") or [])
    description = metadata.get("description") or ""

    install_command: str | None = None
    if dependencies:
        install_command = f"{sys.executable} -m pip install {' '.join(dependencies)}"

    return {
        "$schema": "https://schemas.rosclaw.io/mcp/hardware-manifest.schema.json",
        "schemaVersion": "1.0.0",
        "id": _remote_canonical_id(name),
        "name": short_name,
        "displayName": description or short_name,
        "version": version,
        "channel": "stable",
        "description": description,
        "tags": list(metadata.get("tags") or []),
        "categories": ["hardware"],
        "publisher": {
            "name": author,
            "namespace": owner,
            "verified": False,
        },
        "artifact": {
            "type": "remote",
            "url": git_url,
            "package": name,
            "entrypoint": entry_point,
        },
        "mcp": {
            "serverName": short_name,
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": [entry_point],
            },
            "capabilities": {"tools": True},
        },
        "install": {
            "supportedPlatforms": [],
            "preflight": [],
            "runtimes": (
                [{"type": "python", "installCommand": install_command}] if install_command else []
            ),
        },
        "health": {
            "checks": [{"id": "install_integrity", "category": "install", "required": True}]
        },
        "permissions": {
            "required": [
                {
                    "id": "mcp:tools:read",
                    "level": "safe",
                    "description": "List and call MCP tools",
                }
            ]
        },
        "hardware": {"type": "unknown"},
        "compatibility": {},
        "lifecycle": {"deprecated": False},
    }


class HubClient:
    """Client for discovering and fetching Hardware MCP manifests."""

    DEFAULT_HUB_URL = DEFAULT_HUB_URL

    def __init__(
        self,
        hub_url: str | None = None,
        home: Path | None = None,
        offline: bool = False,
        cache_writes: bool = True,
    ) -> None:
        self.hub_url = (
            hub_url or os.environ.get("ROSCLAW_MCP_HUB") or self.DEFAULT_HUB_URL
        ).rstrip("/")
        self.home: Path = resolve_home(str(home) if home else None)
        self.cache_dir: Path = self.home / "mcp" / "cache"
        self.offline = offline
        self.cache_writes = cache_writes
        self._session: Any | None = None

    def _ensure_cache_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, manifest_id: str, version: str) -> Path:
        safe_id = manifest_id.replace("/", "--")
        return self.cache_dir / safe_id / f"{version}.json"

    def _load_cache(self, manifest_id: str, version: str) -> dict[str, Any] | None:
        path = self._cache_path(manifest_id, version)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return cast(dict[str, Any], json.load(f))
        except (json.JSONDecodeError, OSError):
            return None

    def _save_cache(self, manifest_id: str, version: str, data: dict[str, Any]) -> None:
        if not self.cache_writes:
            return
        self._ensure_cache_dir()
        path = self._cache_path(manifest_id, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _http_get(self, url: str) -> Any:
        """Fetch JSON from a remote hub."""
        try:
            import requests
        except ImportError as exc:
            raise ManifestNotFoundError(f"Remote hub fetch requires 'requests': {exc}") from exc

        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"Accept": "application/json"})
        response = self._session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def _fetch_remote_metadata(self, pkg_name: str) -> dict[str, Any] | None:
        """Query ``/api/registry?pkg=...`` and return metadata on success."""
        if self.offline or not pkg_name:
            return None
        try:
            url = f"{self.hub_url}{REGISTRY_PATH}?pkg={pkg_name}"
            data = self._http_get(url)
        except Exception:
            return None
        if not isinstance(data, dict) or data.get("status") != "success":
            return None
        return data

    def list_manifest_ids(self) -> list[str]:
        """Return all known manifest IDs from hub, cache, and built-ins."""
        ids: set[str] = set(_BUILTIN_REGISTRY.keys())
        if self.cache_dir.exists():
            for d in self.cache_dir.iterdir():
                if d.is_dir() and any(d.glob("*.json")):
                    ids.add(d.name.replace("--", "/"))
        if not self.offline:
            try:
                url = f"{self.hub_url}{PACKAGE_LIST_PATH}"
                data = self._http_get(url)
                for item in data or []:
                    pkg_name = item.get("name")
                    if pkg_name:
                        ids.add(_remote_canonical_id(pkg_name))
            except Exception:
                pass
        return sorted(ids)

    def fetch_manifest(
        self,
        manifest_id: str,
        version: str | None = None,
    ) -> McpManifest:
        """Fetch a manifest by ID and optional version.

        Resolution order:
        1. If version specified: cache, then built-in, then hub.
        2. If version not specified: built-in latest stable, then hub registry, then cache.

        Remote Hub packages are normalized from the lightweight registry
        response into full ``McpManifest`` objects so the rest of the onboarding
        pipeline can treat built-in and Hub manifests identically.
        """
        data: dict[str, Any] | None = None

        if version:
            data = self._load_cache(manifest_id, version)
            if data is None:
                builtin = _BUILTIN_REGISTRY.get(manifest_id)
                if builtin and builtin.get("version") == version:
                    data = builtin
            if data is None and not self.offline:
                pkg_name = _pkg_name_from_remote_canonical(manifest_id)
                if pkg_name:
                    metadata = self._fetch_remote_metadata(pkg_name)
                    if metadata and metadata.get("version") == version:
                        data = _metadata_to_manifest(metadata, pkg_name)
                        self._save_cache(data["id"], version, data)
                if data is None and ("/" in manifest_id or "-" in manifest_id):
                    metadata = self._fetch_remote_metadata(manifest_id)
                    if metadata and metadata.get("version") == version:
                        data = _metadata_to_manifest(metadata, metadata.get("name"))
                        self._save_cache(data["id"], version, data)
        else:
            builtin = _BUILTIN_REGISTRY.get(manifest_id)
            if builtin:
                version = builtin["version"]
                data = builtin
            elif not self.offline:
                pkg_name = _pkg_name_from_remote_canonical(manifest_id) or manifest_id
                metadata = self._fetch_remote_metadata(pkg_name)
                if metadata:
                    data = _metadata_to_manifest(metadata, metadata.get("name"))
                    version = data["version"]
                    self._save_cache(data["id"], version, data)

        if data is None:
            raise ManifestNotFoundError(
                f"Manifest not found: {manifest_id}" + (f"@{version}" if version else "")
            )

        try:
            return McpManifest.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"Invalid manifest {manifest_id}: {exc}") from exc

    def fetch_index(self) -> dict[str, dict[str, Any]]:
        """Return a lightweight index of available versions per manifest ID."""
        index: dict[str, dict[str, Any]] = {}
        for manifest_id, data in _BUILTIN_REGISTRY.items():
            entry = index.setdefault(manifest_id, {"versions": [], "aliases": []})
            entry["versions"].append(
                {
                    "version": data["version"],
                    "channel": data.get("channel", "stable"),
                }
            )
        if not self.offline:
            try:
                url = f"{self.hub_url}{PACKAGE_LIST_PATH}"
                data = self._http_get(url)
                for item in data or []:
                    pkg_name = item.get("name")
                    if not pkg_name:
                        continue
                    manifest_id = _remote_canonical_id(pkg_name)
                    short_name = pkg_name.split("/")[-1]
                    entry = index.setdefault(manifest_id, {"versions": [], "aliases": []})
                    entry["aliases"] = [short_name, pkg_name]
                    entry["pkg_name"] = pkg_name
                    entry["versions"].append(
                        {
                            "version": item.get("version", "1.0.0"),
                            "channel": "stable",
                        }
                    )
            except Exception:
                pass
        return index

    def clear_cache(self) -> int:
        """Remove all cached manifests. Returns number of files deleted."""
        count = 0
        if self.cache_dir.exists():
            for path in self.cache_dir.rglob("*.json"):
                path.unlink()
                count += 1
        return count

    @classmethod
    def get_builtin_manifest_ids(cls) -> list[str]:
        """Return IDs of all built-in manifests."""
        return sorted(_BUILTIN_REGISTRY.keys())
