"""Hardware MCP alias resolution and version solving.

* ``AliasResolver`` maps short names like ``unitree-g1`` / ``g1`` to canonical
  manifest IDs such as ``io.rosclaw.hardware.unitree-g1``.
* ``VersionSolver`` picks the exact version to install, honouring explicit
  requests, the lockfile, channel, and compatibility constraints.
"""

from __future__ import annotations

import platform as _platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

import rosclaw
from rosclaw.mcp.onboarding.errors import (
    AliasResolutionError,
    ManifestNotFoundError,
    VersionResolutionError,
)
from rosclaw.mcp.onboarding.hub_client import HubClient
from rosclaw.mcp.onboarding.lockfile import Lockfile
from rosclaw.mcp.onboarding.schema import CompatibilityDecl, McpManifest

# Canonical namespace prefix for ROSClaw hardware manifests.
CANONICAL_PREFIX = "io.rosclaw.hardware."


class AliasResolver:
    """Resolve aliases and short names to canonical manifest IDs."""

    # Hard-coded short aliases for the built-in registry.
    BUILTIN_ALIASES: dict[str, str] = {
        "unitree-g1": "io.rosclaw.hardware.unitree-g1",
        "g1": "io.rosclaw.hardware.unitree-g1",
        "unitreeg1": "io.rosclaw.hardware.unitree-g1",
        "realsense-d455": "io.rosclaw.hardware.realsense-d455",
        "realsense": "io.rosclaw.hardware.realsense-d455",
        "d455": "io.rosclaw.hardware.realsense-d455",
    }

    def __init__(self, hub: HubClient | None = None, home: Path | None = None) -> None:
        self.hub = hub or HubClient(home=home)

    def _is_canonical(self, value: str) -> bool:
        return value.startswith(CANONICAL_PREFIX)

    def _canonicalize(self, value: str) -> str:
        return f"{CANONICAL_PREFIX}{value}"

    def resolve(self, alias: str) -> str:
        """Resolve a short name or alias to a canonical manifest ID.

        Resolution order:
        1. If already canonical, return as-is.
        2. Built-in alias table.
        3. Hub index (name or id match).
        """
        cleaned = alias.strip().lower()
        if self._is_canonical(alias):
            return alias

        if cleaned in self.BUILTIN_ALIASES:
            return self.BUILTIN_ALIASES[cleaned]

        # Try hub index by exact name/id match.
        try:
            index = self.hub.fetch_index()
            for manifest_id, meta in index.items():
                if manifest_id.lower() == cleaned:
                    return manifest_id
                name = manifest_id.replace(CANONICAL_PREFIX, "")
                if name == cleaned:
                    return manifest_id
                for alt in meta.get("aliases", []):
                    if alt.lower() == cleaned:
                        return manifest_id
        except Exception:
            pass

        # Public Hub packages use owner/repository names. A package may be
        # queryable from /api/registry before it appears in the list index.
        if re.fullmatch(r"[a-z0-9_.-]+/[a-z0-9_.-]+", cleaned):
            try:
                return self.hub.fetch_manifest(cleaned).id
            except Exception:
                pass

        # Last resort: assume the alias is the short name under our namespace.
        if re.match(r"^[a-z0-9_-]+$", cleaned):
            return self._canonicalize(cleaned)

        raise AliasResolutionError(f"Cannot resolve alias to manifest ID: {alias}")

    def resolve_or_canonical(self, value: str) -> str:
        """Resolve alias; if already canonical, validate existence if possible."""
        if self._is_canonical(value):
            return value
        return self.resolve(value)


@dataclass
class SolvedVersion:
    """Result of version solving."""

    manifest_id: str
    version: str
    source: str
    manifest: McpManifest | None = None


class VersionSolver:
    """Select an exact manifest version to install.

    Priority:
    1. Explicit version argument.
    2. Lockfile entry for the manifest ID.
    3. Latest stable version compatible with the local environment.
    """

    def __init__(
        self,
        hub: HubClient | None = None,
        lockfile: Lockfile | None = None,
    ) -> None:
        self.hub = hub or HubClient()
        self.lockfile = lockfile or Lockfile()

    def solve(
        self,
        manifest_id: str,
        explicit_version: str | None = None,
        channel: str = "stable",
    ) -> SolvedVersion:
        """Return the version to install for ``manifest_id``."""
        if explicit_version:
            manifest = self._fetch(manifest_id, explicit_version)
            return SolvedVersion(
                manifest_id=manifest_id,
                version=explicit_version,
                source="explicit",
                manifest=manifest,
            )

        locked_version = self.lockfile.get_locked_version(manifest_id)
        if locked_version:
            manifest = self._fetch(manifest_id, locked_version)
            return SolvedVersion(
                manifest_id=manifest_id,
                version=locked_version,
                source="lockfile",
                manifest=manifest,
            )

        manifest = self._latest_compatible(manifest_id, channel=channel)
        return SolvedVersion(
            manifest_id=manifest_id,
            version=manifest.version,
            source="hub",
            manifest=manifest,
        )

    def _fetch(self, manifest_id: str, version: str) -> McpManifest:
        try:
            return self.hub.fetch_manifest(manifest_id, version)
        except ManifestNotFoundError:
            raise VersionResolutionError(
                f"Version {version} not available for {manifest_id}"
            ) from None

    def _latest_compatible(
        self,
        manifest_id: str,
        channel: str = "stable",
    ) -> McpManifest:
        """Find the latest stable-compatible version."""
        index = self.hub.fetch_index()
        entry = index.get(manifest_id)
        if entry is None:
            # Fallback: fetch without version (built-in default).
            try:
                return self.hub.fetch_manifest(manifest_id)
            except ManifestNotFoundError as exc:
                raise VersionResolutionError(str(exc)) from exc

        versions = entry.get("versions", [])
        candidates: list[tuple[Version, dict[str, Any]]] = []
        for v in versions:
            ver_str = v.get("version")
            if not ver_str:
                continue
            if v.get("channel", "stable") != channel:
                continue
            try:
                candidates.append((Version(ver_str), v))
            except InvalidVersion:
                continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates:
            raise VersionResolutionError(f"No {channel} versions found for {manifest_id}")

        for parsed, _v in candidates:
            manifest = self._fetch(manifest_id, str(parsed))
            if self._is_compatible(manifest.compatibility):
                return manifest

        raise VersionResolutionError(f"No compatible {channel} version found for {manifest_id}")

    def _is_compatible(self, compatibility: CompatibilityDecl | None) -> bool:
        """Check whether the local environment satisfies constraints."""
        if compatibility is None:
            return True

        if compatibility.python and not _satisfies_python_constraint(compatibility.python):
            return False

        return not (
            compatibility.rosclaw and not _satisfies_rosclaw_constraint(compatibility.rosclaw)
        )


def _satisfies_python_constraint(spec: str) -> bool:
    """Check current Python version against a packaging specifier."""
    try:
        current = Version(_platform.python_version())
        return current in SpecifierSet(spec)
    except (InvalidVersion, ValueError):
        return False


def _satisfies_rosclaw_constraint(spec: str) -> bool:
    """Check installed rosclaw version against a packaging specifier."""
    try:
        current = Version(rosclaw.__version__)
        return current in SpecifierSet(spec)
    except (InvalidVersion, ValueError):
        return False
