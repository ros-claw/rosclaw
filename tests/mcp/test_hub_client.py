"""Tests for the Hardware MCP hub client."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.mcp.onboarding.errors import ManifestNotFoundError
from rosclaw.mcp.onboarding.hub_client import HubClient


def test_fetch_builtin_manifest_offline(fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    manifest = hub.fetch_manifest("io.rosclaw.hardware.unitree-g1")
    assert manifest.id == "io.rosclaw.hardware.unitree-g1"
    assert manifest.version == "1.0.0"


def test_fetch_builtin_by_version(fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    manifest = hub.fetch_manifest("io.rosclaw.hardware.realsense-d455", version="1.0.0")
    assert manifest.name == "realsense-d455"


def test_fetch_missing_manifest_raises(fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    with pytest.raises(ManifestNotFoundError):
        hub.fetch_manifest("io.rosclaw.hardware.missing", version="9.9.9")


def test_fetch_index_contains_builtins(fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    index = hub.fetch_index()
    assert "io.rosclaw.hardware.unitree-g1" in index
    assert any(v["version"] == "1.0.0" for v in index["io.rosclaw.hardware.unitree-g1"]["versions"])


def test_list_manifest_ids(fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    ids = hub.list_manifest_ids()
    assert "io.rosclaw.hardware.unitree-g1" in ids
    assert "io.rosclaw.hardware.realsense-d455" in ids


def test_cache_roundtrip(fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    manifest_id = "io.rosclaw.hardware.unitree-g1"
    manifest = hub.fetch_manifest(manifest_id)
    # Second fetch should hit cache path but also built-in; verify cache exists.
    cache_path = hub._cache_path(manifest_id, manifest.version)
    # Built-in path does not save to cache; explicitly save a copy.
    hub._save_cache(manifest_id, manifest.version, manifest.to_dict())
    assert cache_path.exists()

    cached = hub._load_cache(manifest_id, manifest.version)
    assert cached is not None
    assert cached["id"] == manifest_id


def test_clear_cache(fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    manifest_id = "io.rosclaw.hardware.unitree-g1"
    manifest = hub.fetch_manifest(manifest_id)
    hub._save_cache(manifest_id, manifest.version, manifest.to_dict())
    count = hub.clear_cache()
    assert count == 1


def test_get_builtin_manifest_ids() -> None:
    ids = HubClient.get_builtin_manifest_ids()
    assert "io.rosclaw.hardware.unitree-g1" in ids
    assert "io.rosclaw.hardware.realsense-d455" in ids
