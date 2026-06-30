"""Tests for RealSense MCP offline registry entries."""
from __future__ import annotations

from rosclaw.mcp.onboarding.hub_client import HubClient
from rosclaw.mcp.onboarding.resolver import AliasResolver


def test_offline_index_includes_realsense_cameras() -> None:
    client = HubClient(offline=True)
    index = client.fetch_index()
    assert "io.rosclaw.hardware.realsense-d405" in index
    assert "io.rosclaw.hardware.realsense-d435i" in index


def test_offline_manifest_d405() -> None:
    client = HubClient(offline=True)
    manifest = client.fetch_manifest("io.rosclaw.hardware.realsense-d405")
    assert manifest.id == "io.rosclaw.hardware.realsense-d405"
    assert manifest.name == "realsense-d405"
    assert manifest.hardware.type == "sensor"
    assert "d405" in manifest.eurdf.default_profile


def test_offline_manifest_d435i() -> None:
    client = HubClient(offline=True)
    manifest = client.fetch_manifest("io.rosclaw.hardware.realsense-d435i")
    assert manifest.id == "io.rosclaw.hardware.realsense-d435i"
    assert manifest.compatibility.ros_distros == ["humble", "jazzy"]


def test_alias_resolver_d405() -> None:
    resolver = AliasResolver()
    assert resolver.resolve("d405") == "io.rosclaw.hardware.realsense-d405"
    assert resolver.resolve("realsense-d405") == "io.rosclaw.hardware.realsense-d405"


def test_alias_resolver_d435i() -> None:
    resolver = AliasResolver()
    assert resolver.resolve("d435i") == "io.rosclaw.hardware.realsense-d435i"
    assert resolver.resolve("realsense-d435i") == "io.rosclaw.hardware.realsense-d435i"
