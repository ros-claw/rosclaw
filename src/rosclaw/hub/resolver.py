"""Semver resolver for ROSClaw Hub asset references.

The resolver turns a possibly under-specified ``rosclaw://`` reference into a
concrete versioned asset by scanning local cached manifests.  It supports:

* exact versions (``@1.0.0``)
* latest (no version)
* channels (``@stable``, ``@beta``, ``@experimental``)
* simple semver ranges using ``>``, ``>=``, ``<``, ``<=``, ``==``, ``!=``
  combined with commas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import semver

from rosclaw.hub.cache import HubCache
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef
from rosclaw.hub.schema import AssetManifest, LifecycleStatus, load_manifest

CHANNELS = {status.value for status in LifecycleStatus}


def _parse_version(version: str) -> semver.VersionInfo | None:
    """Parse a semantic version string, returning None on failure."""
    try:
        return semver.VersionInfo.parse(version)
    except (ValueError, TypeError):
        return None


def _is_semver(version: str) -> bool:
    """Return True if *version* parses as a semantic version."""
    return _parse_version(version) is not None


def _version_key(version: str) -> tuple[int, semver.VersionInfo | str]:
    """Return a sortable key for version strings."""
    parsed = _parse_version(version)
    if parsed is not None:
        return (0, parsed)
    return (1, version)


def _compare_versions(a: str, b: str) -> int:
    """Compare two semantic version strings (-1, 0, 1)."""
    va = _parse_version(a)
    vb = _parse_version(b)
    if va is None or vb is None:
        raise ValueError(f"Cannot compare non-semver versions: {a!r}, {b!r}")
    result: int = va.compare(b)
    return result


_OP_FUNCS = {
    "==": lambda c: c == 0,
    "!=": lambda c: c != 0,
    ">": lambda c: c > 0,
    ">=": lambda c: c >= 0,
    "<": lambda c: c < 0,
    "<=": lambda c: c <= 0,
}


def _matches_spec(version: str, spec: str) -> bool:
    """Check whether *version* satisfies *spec*.

    *spec* may be:
    - an exact semver string (``1.0.0``)
    - a single comparison (``>=1.0.0``)
    - a comma-separated list of comparisons (``>=1.0.0,<2.0.0``)
    """
    spec = spec.strip()
    if spec == version:
        return True

    if _is_semver(spec) and _is_semver(version):
        return _compare_versions(version, spec) == 0

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        return False

    for part in parts:
        match = re.match(r"^(?P<op>[<>=!]+)(?P<ver>.+)$", part)
        if not match:
            return False
        op, target = match.group("op"), match.group("ver").strip()
        func = _OP_FUNCS.get(op)
        if func is None or not _is_semver(version) or not _is_semver(target):
            return False
        if not func(_compare_versions(version, target)):
            return False
    return True


def _channel_from_manifest(manifest: AssetManifest) -> str:
    """Return the channel declared by a manifest, falling back to lifecycle status."""
    lifecycle = manifest.lifecycle or {}
    return lifecycle.get("channel") or lifecycle.get("status") or "stable"


@dataclass(frozen=True)
class ResolvedAsset:
    """A concrete asset reference plus its manifest and manifest path."""

    ref: AssetRef
    manifest_path: Path
    manifest: AssetManifest


class Resolver:
    """Resolve asset references against local cached manifests."""

    def __init__(
        self,
        search_paths: list[str | Path] | None = None,
        *,
        cache: HubCache | None = None,
    ) -> None:
        self.search_paths: list[Path] = [Path(p) for p in (search_paths or [])]
        if cache is not None:
            self.search_paths.append(cache.manifests_dir)
        self._candidates: list[tuple[AssetRef, Path, AssetManifest]] | None = None

    def _scan(self) -> list[tuple[AssetRef, Path, AssetManifest]]:
        """Lazy-load all manifests under the search paths."""
        if self._candidates is not None:
            return self._candidates

        candidates: list[tuple[AssetRef, Path, AssetManifest]] = []
        seen: set[Path] = set()
        for base in self.search_paths:
            if not base.exists():
                continue
            for manifest_path in base.rglob("*.yaml"):
                manifest_path = manifest_path.resolve()
                if manifest_path in seen:
                    continue
                seen.add(manifest_path)
                try:
                    manifest = load_manifest(manifest_path)
                except Exception:  # noqa: BLE001 - skip unreadable manifests
                    continue
                ref = ref_from_manifest(manifest)
                candidates.append((ref, manifest_path, manifest))

        self._candidates = candidates
        return candidates

    def _matching_candidates(
        self,
        ref: AssetRef,
    ) -> list[tuple[AssetRef, Path, AssetManifest]]:
        """Return candidates matching type/namespace/name."""
        return [
            (c_ref, path, manifest)
            for c_ref, path, manifest in self._scan()
            if c_ref.type == ref.type
            and c_ref.namespace == ref.namespace
            and c_ref.name == ref.name
        ]

    def resolve(self, ref: str | AssetRef) -> ResolvedAsset:
        """Resolve a reference to a concrete versioned asset.

        Raises:
            HubError: If no matching asset is found or the reference is
                ambiguous in a way that cannot be resolved.
        """
        if isinstance(ref, str):
            from rosclaw.hub.refs import parse_ref

            ref = parse_ref(ref)

        candidates = self._matching_candidates(ref)
        if not candidates:
            raise HubError(
                code=HubErrorCode.ASSET_NOT_FOUND,
                message=f"No local manifest found for {ref.canonical()}",
                suggested_fix="Run `rosclaw hub sync` to refresh the local catalog.",
            )

        version_spec = (ref.version or "").strip()

        if not version_spec:
            # latest semver
            chosen = self._choose_latest(candidates)
            return ResolvedAsset(
                ref=AssetRef(
                    type=ref.type,
                    namespace=ref.namespace,
                    name=ref.name,
                    version=chosen[0].version,
                ),
                manifest_path=chosen[1],
                manifest=chosen[2],
            )

        if version_spec.lower() in CHANNELS:
            channel = version_spec.lower()
            channel_candidates = [
                (c_ref, path, manifest)
                for c_ref, path, manifest in candidates
                if _channel_from_manifest(manifest).lower() == channel
            ]
            if not channel_candidates:
                raise HubError(
                    code=HubErrorCode.ASSET_NOT_FOUND,
                    message=f"No {channel} release found for {ref.canonical()}",
                )
            chosen = self._choose_latest(channel_candidates)
            return ResolvedAsset(
                ref=AssetRef(
                    type=ref.type,
                    namespace=ref.namespace,
                    name=ref.name,
                    version=chosen[0].version,
                ),
                manifest_path=chosen[1],
                manifest=chosen[2],
            )

        # Treat version_spec as an exact version or semver range.
        spec_candidates = [
            (c_ref, path, manifest)
            for c_ref, path, manifest in candidates
            if _matches_spec(c_ref.version or "", version_spec)
        ]
        if not spec_candidates:
            raise HubError(
                code=HubErrorCode.ASSET_NOT_FOUND,
                message=f"No version satisfies '{version_spec}' for {ref.canonical()}",
            )
        chosen = self._choose_latest(spec_candidates)
        return ResolvedAsset(
            ref=AssetRef(
                type=ref.type,
                namespace=ref.namespace,
                name=ref.name,
                version=chosen[0].version,
            ),
            manifest_path=chosen[1],
            manifest=chosen[2],
        )

    def _choose_latest(
        self,
        candidates: list[tuple[AssetRef, Path, AssetManifest]],
    ) -> tuple[AssetRef, Path, AssetManifest]:
        """Return the candidate with the highest semver version."""
        return max(
            candidates,
            key=lambda item: _version_key(item[0].version or ""),
        )


def resolve_dependencies(
    manifest: AssetManifest,
    resolver: Resolver,
) -> list[ResolvedAsset]:
    """Resolve declared asset dependencies.

    Only required dependencies are resolved; optional ones are skipped unless
    already present.
    """
    deps: list[dict[str, Any]] = manifest.dependencies.get("assets", [])
    resolved: list[ResolvedAsset] = []
    for dep in deps:
        ref_str = dep.get("ref") if isinstance(dep, dict) else str(dep)
        if not ref_str:
            continue
        required = dep.get("required", True) if isinstance(dep, dict) else True
        if not required:
            continue
        resolved.append(resolver.resolve(ref_str))
    return resolved


def ref_from_manifest(manifest: AssetManifest) -> AssetRef:
    """Build an :class:`AssetRef` from a loaded manifest."""
    asset = manifest.asset
    return AssetRef(
        type=asset.type.value,
        namespace=asset.namespace,
        name=asset.name,
        version=asset.version,
    )
