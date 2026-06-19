"""Tests for the ROSClaw Hub asset resolver."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rosclaw.hub.errors import HubError
from rosclaw.hub.refs import AssetRef
from rosclaw.hub.resolver import Resolver, resolve_dependencies


def _write_manifest(
    path: Path, version: str, channel: str = "stable", name: str = "pick-place"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "hub.asset.v1",
        "asset": {
            "type": "skill",
            "namespace": "rosclaw",
            "name": name,
            "version": version,
            "title": "Pick and Place",
            "summary": "A test skill",
            "description": "Test",
            "tags": [],
        },
        "publisher": {
            "id": "rosclaw",
            "display_name": "ROSClaw Team",
            "trust_level": "official",
        },
        "visibility": {"scope": "public"},
        "lifecycle": {"status": "stable", "channel": channel, "deprecated": False},
        "compatibility": {"rosclaw": {"min_version": "1.0.0"}},
        "permissions": {"hardware": {"real_robot_execution": False}},
        "license": {
            "spdx": "MIT",
            "commercial_use": True,
            "redistribution": True,
            "attribution_required": True,
            "export_control": "none",
        },
        "data_rights": {
            "contains_training_data": False,
            "contains_robot_logs": False,
            "contains_personal_data": False,
            "allowed_usage": ["research"],
            "restrictions": [],
        },
        "security": {
            "signing": {"required": False},
            "checksums": {"algorithm": "sha256", "file": "checksums.txt"},
        },
        "artifacts": [],
        "install": {"mode": "declarative", "entrypoints": {}},
        "special": {"skill": {"parameters": {}}},
    }
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")


def test_resolve_exact_version(tmp_path: Path) -> None:
    """Resolver selects the exact requested version."""
    _write_manifest(tmp_path / "1.0.0.yaml", "1.0.0")
    _write_manifest(tmp_path / "1.5.0.yaml", "1.5.0")
    resolver = Resolver(search_paths=[tmp_path])
    resolved = resolver.resolve("rosclaw://skill/rosclaw/pick-place@1.0.0")
    assert resolved.ref.version == "1.0.0"


def test_resolve_latest(tmp_path: Path) -> None:
    """Resolver picks the highest semver version when no version is given."""
    _write_manifest(tmp_path / "1.0.0.yaml", "1.0.0")
    _write_manifest(tmp_path / "1.10.0.yaml", "1.10.0")
    _write_manifest(tmp_path / "2.0.0.yaml", "2.0.0")
    resolver = Resolver(search_paths=[tmp_path])
    resolved = resolver.resolve("rosclaw://skill/rosclaw/pick-place")
    assert resolved.ref.version == "2.0.0"


def test_resolve_channel(tmp_path: Path) -> None:
    """Resolver can select by lifecycle channel."""
    _write_manifest(tmp_path / "1.0.0.yaml", "1.0.0", channel="stable")
    _write_manifest(tmp_path / "2.0.0-beta.yaml", "2.0.0-beta", channel="beta")
    resolver = Resolver(search_paths=[tmp_path])
    stable = resolver.resolve("rosclaw://skill/rosclaw/pick-place@stable")
    assert stable.manifest.lifecycle["channel"] == "stable"
    beta = resolver.resolve("rosclaw://skill/rosclaw/pick-place@beta")
    assert beta.manifest.lifecycle["channel"] == "beta"


def test_resolve_semver_range(tmp_path: Path) -> None:
    """Resolver supports comma-separated semver ranges."""
    _write_manifest(tmp_path / "1.0.0.yaml", "1.0.0")
    _write_manifest(tmp_path / "1.5.0.yaml", "1.5.0")
    _write_manifest(tmp_path / "2.0.0.yaml", "2.0.0")
    resolver = Resolver(search_paths=[tmp_path])
    resolved = resolver.resolve("rosclaw://skill/rosclaw/pick-place@>=1.0.0,<2.0.0")
    assert resolved.ref.version == "1.5.0"


def test_resolve_not_found(tmp_path: Path) -> None:
    """Resolving an unknown asset raises ASSET_NOT_FOUND."""
    resolver = Resolver(search_paths=[tmp_path])
    with pytest.raises(HubError) as exc_info:
        resolver.resolve("rosclaw://skill/rosclaw/missing@1.0.0")
    assert exc_info.value.code.value == "HUB_ASSET_NOT_FOUND"


def test_resolve_version_not_satisfied(tmp_path: Path) -> None:
    """Resolving an unavailable version raises ASSET_NOT_FOUND."""
    _write_manifest(tmp_path / "1.0.0.yaml", "1.0.0")
    resolver = Resolver(search_paths=[tmp_path])
    with pytest.raises(HubError) as exc_info:
        resolver.resolve("rosclaw://skill/rosclaw/pick-place@2.0.0")
    assert "No version satisfies" in exc_info.value.message


def test_resolve_dependencies(tmp_path: Path) -> None:
    """resolve_dependencies uses the resolver for declared asset deps."""
    _write_manifest(tmp_path / "driver-1.0.0.yaml", "1.0.0", name="driver")
    dep_manifest = {
        "schema_version": "hub.asset.v1",
        "asset": {
            "type": "skill",
            "namespace": "rosclaw",
            "name": "dependent",
            "version": "1.0.0",
            "title": "Dependent",
            "summary": "Depends on driver",
        },
        "publisher": {"id": "rosclaw", "display_name": "ROSClaw Team"},
        "visibility": {"scope": "public"},
        "lifecycle": {"status": "stable", "channel": "stable"},
        "compatibility": {"rosclaw": {"min_version": "1.0.0"}},
        "dependencies": {
            "assets": [{"ref": "rosclaw://skill/rosclaw/driver@1.0.0", "required": True}]
        },
        "permissions": {"hardware": {"real_robot_execution": False}},
        "license": {
            "spdx": "MIT",
            "commercial_use": True,
            "redistribution": True,
            "attribution_required": True,
            "export_control": "none",
        },
        "data_rights": {
            "contains_training_data": False,
            "contains_robot_logs": False,
            "contains_personal_data": False,
            "allowed_usage": ["research"],
            "restrictions": [],
        },
        "security": {"signing": {"required": False}},
        "artifacts": [],
        "install": {"mode": "declarative"},
        "special": {"skill": {}},
    }
    (tmp_path / "dependent-1.0.0.yaml").write_text(yaml.safe_dump(dep_manifest), encoding="utf-8")
    resolver = Resolver(search_paths=[tmp_path])
    parent = resolver.resolve("rosclaw://skill/rosclaw/dependent@1.0.0")
    deps = resolve_dependencies(parent.manifest, resolver)
    assert len(deps) == 1
    assert deps[0].ref.name == "driver"
    assert deps[0].ref.version == "1.0.0"


def test_resolved_asset_ref_type(tmp_path: Path) -> None:
    """resolve() returns an AssetRef with a concrete version."""
    _write_manifest(tmp_path / "1.0.0.yaml", "1.0.0")
    resolver = Resolver(search_paths=[tmp_path])
    resolved = resolver.resolve("rosclaw://skill/rosclaw/pick-place")
    assert resolved.ref == AssetRef(
        type="skill",
        namespace="rosclaw",
        name="pick-place",
        version="1.0.0",
    )
