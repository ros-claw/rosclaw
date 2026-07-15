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


# ── Remote Hub fixtures/helpers ────────────────────────────────────────────────


def _g1_remote_metadata() -> dict[str, object]:
    return {
        "status": "success",
        "type": "mcp_server",
        "name": "ros-claw/g1-mcp",
        "git_url": "https://github.com/ros-claw/g1-mcp.git",
        "description": "Unitree G1 humanoid robot MCP server",
        "entry_point": "g1_mcp_server.py",
        "version": "0.1.0",
        "author": "ROSClaw Team",
        "dependencies": ["unitree-sdk"],
    }


def _remote_package_list() -> list[dict[str, object]]:
    return [{"name": "ros-claw/g1-mcp", "version": "0.1.0"}]


# ── Remote Hub tests ───────────────────────────────────────────────────────────


def test_fetch_remote_manifest_by_canonical_id(fake_home: Path, monkeypatch) -> None:
    hub = HubClient(home=fake_home, offline=False)

    def _fake_get(url: str):
        if "registry" in url:
            return _g1_remote_metadata()
        if "mcp-packages" in url:
            return _remote_package_list()
        raise RuntimeError(url)

    monkeypatch.setattr(hub, "_http_get", _fake_get)

    manifest = hub.fetch_manifest("io.rosclaw.hub.ros-claw.g1-mcp")
    assert manifest.id == "io.rosclaw.hub.ros-claw.g1-mcp"
    assert manifest.name == "g1-mcp"
    assert manifest.version == "0.1.0"
    assert manifest.artifact is not None
    assert manifest.artifact.type == "remote"
    assert manifest.artifact.url == "https://github.com/ros-claw/g1-mcp.git"


def test_fetch_remote_manifest_by_pkg_name(fake_home: Path, monkeypatch) -> None:
    hub = HubClient(home=fake_home, offline=False)
    monkeypatch.setattr(
        hub,
        "_http_get",
        lambda url: _g1_remote_metadata() if "registry" in url else _remote_package_list(),
    )

    manifest = hub.fetch_manifest("ros-claw/g1-mcp")
    assert manifest.id == "io.rosclaw.hub.ros-claw.g1-mcp"
    assert manifest.display_name == "Unitree G1 humanoid robot MCP server"


def test_remote_manifest_cache_can_be_disabled(fake_home: Path, monkeypatch) -> None:
    hub = HubClient(home=fake_home, offline=False, cache_writes=False)
    monkeypatch.setattr(hub, "_http_get", lambda _url: _g1_remote_metadata())

    manifest = hub.fetch_manifest("ros-claw/g1-mcp")

    assert manifest.id == "io.rosclaw.hub.ros-claw.g1-mcp"
    assert not hub.cache_dir.exists()


def test_fetch_remote_manifest_version_mismatch(fake_home: Path, monkeypatch) -> None:
    hub = HubClient(home=fake_home, offline=False)
    monkeypatch.setattr(
        hub,
        "_http_get",
        lambda url: _g1_remote_metadata() if "registry" in url else _remote_package_list(),
    )

    with pytest.raises(ManifestNotFoundError):
        hub.fetch_manifest("io.rosclaw.hub.ros-claw.g1-mcp", version="9.9.9")


def test_list_manifest_ids_includes_remote(fake_home: Path, monkeypatch) -> None:
    hub = HubClient(home=fake_home, offline=False)
    monkeypatch.setattr(
        hub,
        "_http_get",
        lambda url: _remote_package_list() if "mcp-packages" in url else [],
    )

    ids = hub.list_manifest_ids()
    assert "io.rosclaw.hub.ros-claw.g1-mcp" in ids
    assert "io.rosclaw.hardware.unitree-g1" in ids


def test_fetch_index_includes_remote(fake_home: Path, monkeypatch) -> None:
    hub = HubClient(home=fake_home, offline=False)
    monkeypatch.setattr(
        hub,
        "_http_get",
        lambda url: _remote_package_list() if "mcp-packages" in url else _g1_remote_metadata(),
    )

    index = hub.fetch_index()
    assert "io.rosclaw.hub.ros-claw.g1-mcp" in index
    entry = index["io.rosclaw.hub.ros-claw.g1-mcp"]
    assert any(v["version"] == "0.1.0" for v in entry["versions"])
    assert "g1-mcp" in entry.get("aliases", [])


def test_offline_mode_skips_remote(fake_home: Path, monkeypatch) -> None:
    hub = HubClient(home=fake_home, offline=True)
    calls: list[str] = []
    monkeypatch.setattr(hub, "_http_get", lambda url: calls.append(url) or [])

    ids = hub.list_manifest_ids()
    assert "io.rosclaw.hub.ros-claw.g1-mcp" not in ids
    assert not calls


def test_remote_hub_failure_falls_back_to_builtin(fake_home: Path, monkeypatch) -> None:
    hub = HubClient(home=fake_home, offline=False)

    def _boom(url: str) -> None:
        raise ConnectionError("hub unavailable")

    monkeypatch.setattr(hub, "_http_get", _boom)

    manifest = hub.fetch_manifest("io.rosclaw.hardware.unitree-g1")
    assert manifest.id == "io.rosclaw.hardware.unitree-g1"
