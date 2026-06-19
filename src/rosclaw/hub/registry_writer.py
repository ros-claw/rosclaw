"""Runtime registry writer for ROSClaw Hub assets.

The installer uses :class:`RegistryWriter` to maintain five JSON registry files
under ``~/.rosclaw/runtime/registries/``:

* ``skills.json``
* ``providers.json``
* ``hardware_mcp.json``
* ``digital_twins.json``
* ``cognitive_wiki.json``

Each file contains a versioned list of active assets.  Writes are atomic and
backed up before modification.
"""

from __future__ import annotations

import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from rosclaw.hub.cache import HubCache
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef, ref_from_dict
from rosclaw.hub.schema import AssetManifest, AssetType

_REGISTRY_FILENAMES: dict[str, str] = {
    AssetType.skill.value: "skills.json",
    AssetType.provider.value: "providers.json",
    AssetType.hardware_mcp.value: "hardware_mcp.json",
    AssetType.digital_twin.value: "digital_twins.json",
    AssetType.cognitive_wiki.value: "cognitive_wiki.json",
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _entrypoint_command(entrypoint: dict[str, Any] | None) -> str | None:
    if not entrypoint:
        return None
    return entrypoint.get("command")


def _entrypoint_env(entrypoint: dict[str, Any] | None) -> dict[str, str]:
    if not entrypoint:
        return {}
    env = entrypoint.get("env") or {}
    return {str(k): str(v) for k, v in env.items()}


def _manifest_ref(manifest: AssetManifest) -> AssetRef:
    """Return a fully-qualified :class:`AssetRef` from a manifest."""
    return ref_from_dict(
        {
            "type": manifest.asset.type.value,
            "namespace": manifest.asset.namespace,
            "name": manifest.asset.name,
            "version": manifest.asset.version,
        }
    )


def _skill_entry(
    manifest: AssetManifest,
    asset_dir: Path,
) -> dict[str, Any]:
    ref = _manifest_ref(manifest)
    entrypoints: dict[str, Any] = manifest.install.get("entrypoints", {})
    skill_ep = entrypoints.get("skill", {})
    special = manifest.special.get("skill", {})
    return {
        "ref": str(ref),
        "type": AssetType.skill.value,
        "namespace": ref.namespace,
        "name": ref.name,
        "version": ref.version,
        "title": manifest.asset.title,
        "summary": manifest.asset.summary,
        "publisher": {
            "id": manifest.publisher.id,
            "display_name": manifest.publisher.display_name,
            "trust_level": manifest.publisher.trust_level.value,
        },
        "asset_dir": str(asset_dir),
        "entrypoint": _entrypoint_command(skill_ep),
        "capabilities": special.get("required_capabilities", []),
        "parameters": special.get("parameters", {}),
        "components": special.get("components", {}),
        "runtime": special.get("runtime", {}),
    }


def _provider_entry(
    manifest: AssetManifest,
    asset_dir: Path,
) -> dict[str, Any]:
    ref = _manifest_ref(manifest)
    entrypoints: dict[str, Any] = manifest.install.get("entrypoints", {})
    provider_ep = entrypoints.get("provider", {})
    special = manifest.special.get("provider", {})
    return {
        "ref": str(ref),
        "type": AssetType.provider.value,
        "namespace": ref.namespace,
        "name": ref.name,
        "version": ref.version,
        "title": manifest.asset.title,
        "summary": manifest.asset.summary,
        "publisher": {
            "id": manifest.publisher.id,
            "display_name": manifest.publisher.display_name,
            "trust_level": manifest.publisher.trust_level.value,
        },
        "asset_dir": str(asset_dir),
        "entrypoint": _entrypoint_command(provider_ep),
        "provider_kind": special.get("provider_kind"),
        "capability_routes": special.get("capability_routes", []),
        "backend": special.get("backend", {}),
        "model": special.get("model", {}),
        "secrets": special.get("secrets", {}),
    }


def _hardware_mcp_entry(
    manifest: AssetManifest,
    asset_dir: Path,
) -> dict[str, Any]:
    ref = _manifest_ref(manifest)
    entrypoints: dict[str, Any] = manifest.install.get("entrypoints", {})
    mcp_ep = entrypoints.get("mcp", {})
    special = manifest.special.get("hardware_mcp", {})
    command = _entrypoint_command(mcp_ep) or ""
    argv = shlex.split(command) if command else []
    transport = mcp_ep.get("transport") or special.get("transport", {}).get("kind", "stdio")
    return {
        "ref": str(ref),
        "type": AssetType.hardware_mcp.value,
        "namespace": ref.namespace,
        "name": ref.name,
        "version": ref.version,
        "title": manifest.asset.title,
        "summary": manifest.asset.summary,
        "publisher": {
            "id": manifest.publisher.id,
            "display_name": manifest.publisher.display_name,
            "trust_level": manifest.publisher.trust_level.value,
        },
        "asset_dir": str(asset_dir),
        "command": command,
        "args": argv[1:] if len(argv) > 1 else [],
        "program": argv[0] if argv else None,
        "env": _entrypoint_env(mcp_ep),
        "transport": transport,
        "vendor": special.get("vendor"),
        "model": special.get("model"),
        "eurdf_binding": special.get("eurdf_binding", {}),
    }


def _digital_twin_entry(
    manifest: AssetManifest,
    asset_dir: Path,
) -> dict[str, Any]:
    ref = _manifest_ref(manifest)
    entrypoints: dict[str, Any] = manifest.install.get("entrypoints", {})
    twin_ep = entrypoints.get("twin", {})
    special = manifest.special.get("digital_twin", {})
    return {
        "ref": str(ref),
        "type": AssetType.digital_twin.value,
        "namespace": ref.namespace,
        "name": ref.name,
        "version": ref.version,
        "title": manifest.asset.title,
        "summary": manifest.asset.summary,
        "publisher": {
            "id": manifest.publisher.id,
            "display_name": manifest.publisher.display_name,
            "trust_level": manifest.publisher.trust_level.value,
        },
        "asset_dir": str(asset_dir),
        "entrypoint": _entrypoint_command(twin_ep),
        "simulator": special.get("simulator", {}),
        "eurdf_binding": special.get("eurdf_binding", {}),
        "assets": special.get("assets", {}),
        "firewall": special.get("firewall", {}),
        "tasks": special.get("tasks", []),
    }


def _cognitive_wiki_entry(
    manifest: AssetManifest,
    asset_dir: Path,
) -> dict[str, Any]:
    ref = _manifest_ref(manifest)
    entrypoints: dict[str, Any] = manifest.install.get("entrypoints", {})
    wiki_ep = entrypoints.get("wiki", {})
    special = manifest.special.get("cognitive_wiki", {})
    return {
        "ref": str(ref),
        "type": AssetType.cognitive_wiki.value,
        "namespace": ref.namespace,
        "name": ref.name,
        "version": ref.version,
        "title": manifest.asset.title,
        "summary": manifest.asset.summary,
        "publisher": {
            "id": manifest.publisher.id,
            "display_name": manifest.publisher.display_name,
            "trust_level": manifest.publisher.trust_level.value,
        },
        "asset_dir": str(asset_dir),
        "entrypoint": _entrypoint_command(wiki_ep),
        "bundle_path": special.get("bundle_path"),
        "graph_schema": special.get("graph_schema"),
        "knowledge_format": special.get("knowledge_format"),
        "ingest": special.get("runtime", {}),
    }


_ENTRY_BUILDERS: dict[str, Any] = {
    AssetType.skill.value: _skill_entry,
    AssetType.provider.value: _provider_entry,
    AssetType.hardware_mcp.value: _hardware_mcp_entry,
    AssetType.digital_twin.value: _digital_twin_entry,
    AssetType.cognitive_wiki.value: _cognitive_wiki_entry,
}


def registry_entry_from_manifest(
    manifest: AssetManifest,
    asset_dir: Path,
) -> dict[str, Any]:
    """Build a runtime registry entry for *manifest*."""
    builder = _ENTRY_BUILDERS.get(manifest.asset.type.value)
    if builder is None:
        raise HubError(
            code=HubErrorCode.INCOMPATIBLE_RUNTIME,
            message=f"No registry builder for asset type {manifest.asset.type.value}",
        )
    return cast(dict[str, Any], builder(manifest, asset_dir))


class RegistryWriter:
    """Atomic writer for runtime registry JSON files."""

    def __init__(
        self,
        home: str | Path | None = None,
        *,
        cache: HubCache | None = None,
    ) -> None:
        if cache is not None:
            self.cache = cache
        else:
            self.cache = HubCache(home)
        home_path = self.cache.home
        self.registries_dir = home_path / "runtime" / "registries"
        self.registries_dir.mkdir(parents=True, exist_ok=True)

    def _registry_path(self, asset_type: str) -> Path:
        filename = _REGISTRY_FILENAMES.get(asset_type)
        if filename is None:
            raise HubError(
                code=HubErrorCode.INCOMPATIBLE_RUNTIME,
                message=f"Unknown asset type for registry: {asset_type!r}",
            )
        return self.registries_dir / filename

    def _load(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"version": "1.0", "updated_at": "", "assets": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HubError(
                code=HubErrorCode.MANIFEST_INVALID,
                message=f"Corrupt registry file: {path}",
            ) from exc
        if "assets" not in data:
            data["assets"] = {}
        if isinstance(data["assets"], list):
            data["assets"] = {a.get("ref", str(a)): a for a in data["assets"]}
        return cast(dict[str, Any], data)

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        data["updated_at"] = _now()
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        if path.exists():
            self.cache.backup_file(path)
        self.cache._atomic_write(path, payload)

    def add_asset(
        self,
        manifest: AssetManifest,
        asset_dir: Path,
    ) -> Path:
        """Add or replace an asset entry in its type-specific registry.

        Returns:
            Path to the written registry file.
        """
        asset_type = manifest.asset.type.value
        entry = registry_entry_from_manifest(manifest, asset_dir)
        ref = entry["ref"]
        path = self._registry_path(asset_type)
        data = self._load(path)
        data["assets"][ref] = entry
        self._write(path, data)
        return path

    def remove_asset(self, ref: AssetRef) -> bool:
        """Remove an asset entry from its type-specific registry.

        Returns:
            True if the entry existed and was removed.
        """
        path = self._registry_path(ref.type)
        if not path.exists():
            return False
        data = self._load(path)
        key = str(ref)
        if key not in data["assets"]:
            return False
        del data["assets"][key]
        self._write(path, data)
        return True

    def list_assets(self, asset_type: str) -> list[dict[str, Any]]:
        """Return all entries for *asset_type*."""
        path = self._registry_path(asset_type)
        data = self._load(path)
        return list(data["assets"].values())

    def registry_path(self, asset_type: str) -> Path:
        """Return the filesystem path for *asset_type* registry."""
        return self._registry_path(asset_type)
