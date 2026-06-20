"""Tests for the Hardware MCP version solver."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.mcp.onboarding.errors import VersionResolutionError
from rosclaw.mcp.onboarding.hub_client import HubClient
from rosclaw.mcp.onboarding.lockfile import LockedPackage, Lockfile
from rosclaw.mcp.onboarding.resolver import VersionSolver
from rosclaw.mcp.onboarding.schema import McpManifest


def test_explicit_version_wins(unitree_manifest: McpManifest, fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    lockfile = Lockfile(home=fake_home)
    solver = VersionSolver(hub=hub, lockfile=lockfile)

    result = solver.solve(unitree_manifest.id, explicit_version="1.0.0")
    assert result.version == "1.0.0"
    assert result.source == "explicit"
    assert result.manifest is not None
    assert result.manifest.id == unitree_manifest.id


def test_lockfile_version_second_priority(unitree_manifest: McpManifest, fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    lockfile = Lockfile(home=fake_home)
    lockfile.set(
        LockedPackage(
            manifest_id=unitree_manifest.id,
            version="0.9.0",
            name=unitree_manifest.name,
            server_name=unitree_manifest.server_name,
            locked_at="2025-01-01T00:00:00Z",
        )
    )
    solver = VersionSolver(hub=hub, lockfile=lockfile)

    # Even though built-in is 1.0.0, lockfile pins 0.9.0.
    with pytest.raises(VersionResolutionError):
        solver.solve(unitree_manifest.id)


def test_latest_built_in_selected_when_no_lock(unitree_manifest: McpManifest, fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    solver = VersionSolver(hub=hub)
    result = solver.solve(unitree_manifest.id)
    assert result.version == unitree_manifest.version
    assert result.source == "hub"


def test_unknown_manifest_raises(fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    solver = VersionSolver(hub=hub)
    with pytest.raises(VersionResolutionError):
        solver.solve("io.rosclaw.hardware.unknown")


def test_python_constraint_check_passes_for_built_in(unitree_manifest: McpManifest, fake_home: Path) -> None:
    hub = HubClient(home=fake_home, offline=True)
    solver = VersionSolver(hub=hub)
    result = solver.solve(unitree_manifest.id)
    assert result.manifest is not None
    assert result.manifest.compatibility is not None
    assert result.manifest.compatibility.python == ">=3.10"


def test_channel_filter_excludes_non_matching() -> None:
    # No built-ins under beta channel, so solver should fail.
    solver = VersionSolver()
    with pytest.raises(VersionResolutionError):
        solver.solve("io.rosclaw.hardware.unitree-g1", channel="beta")
