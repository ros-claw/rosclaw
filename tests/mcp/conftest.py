"""Shared pytest fixtures for ROSClaw Hardware MCP onboarding tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from rosclaw.mcp.onboarding.schema import McpManifest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def unitree_manifest_dict() -> dict[str, Any]:
    """Golden Unitree G1 manifest as a dict."""
    return json.loads((FIXTURES_DIR / "manifests" / "unitree-g1.json").read_text(encoding="utf-8"))


@pytest.fixture
def realsense_manifest_dict() -> dict[str, Any]:
    """Golden RealSense D455 manifest as a dict."""
    return json.loads((FIXTURES_DIR / "manifests" / "realsense-d455.json").read_text(encoding="utf-8"))


@pytest.fixture
def unitree_manifest(unitree_manifest_dict: dict[str, Any]) -> McpManifest:
    return McpManifest.from_dict(unitree_manifest_dict)


@pytest.fixture
def realsense_manifest(realsense_manifest_dict: dict[str, Any]) -> McpManifest:
    return McpManifest.from_dict(realsense_manifest_dict)


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Temporary ROSCLAW_HOME directory."""
    home = tmp_path / ".rosclaw"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    return home


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Temporary project root containing a marker file."""
    root = tmp_path / "project"
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").mkdir(exist_ok=True)
    return root


@pytest.fixture
def body_yaml_empty(fake_home: Path) -> Path:
    """Copy the empty body.yaml fixture into fake_home."""
    dest = fake_home / "body" / "body.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES_DIR / "body" / "empty.yaml", dest)
    return dest


@pytest.fixture
def body_yaml_unitree(fake_home: Path) -> Path:
    """Copy the Unitree G1 body.yaml fixture into fake_home."""
    dest = fake_home / "body" / "body.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES_DIR / "body" / "unitree-g1.yaml", dest)
    return dest


@pytest.fixture
def claude_mcp_empty(project_root: Path) -> Path:
    """Install an empty .mcp.json in the project root."""
    dest = project_root / ".mcp.json"
    shutil.copy(FIXTURES_DIR / "claude" / "mcp.empty.json", dest)
    return dest


@pytest.fixture
def claude_mcp_with_existing(project_root: Path) -> Path:
    """Install a .mcp.json with an unmanaged existing server."""
    dest = project_root / ".mcp.json"
    shutil.copy(FIXTURES_DIR / "claude" / "mcp.with-existing.json", dest)
    return dest


@pytest.fixture
def claude_mcp_conflict(project_root: Path) -> Path:
    """Install a .mcp.json with a conflicting managed server."""
    dest = project_root / ".mcp.json"
    shutil.copy(FIXTURES_DIR / "claude" / "mcp.conflict.json", dest)
    return dest


class _FakeRobotProfile:
    """Minimal stand-in for RobotCompleteProfile used by hash computation."""

    def __init__(self, robot_id: str):
        self.robot_id = robot_id

    def to_dict(self) -> dict[str, Any]:
        return {"robot_id": self.robot_id}


class _FakeRobotRegistry:
    """In-memory RobotRegistry replacement for binding/health tests."""

    def __init__(self, installed: set[str] | None = None):
        self._installed: set[str] = set(installed or ())

    def get(self, profile_id: str) -> Any | None:
        if profile_id in self._installed:
            return _FakeRobotProfile(profile_id)
        return None

    def install(self, profile_id: str) -> None:
        self._installed.add(profile_id)

    def has(self, profile_id: str) -> bool:
        return profile_id in self._installed


@pytest.fixture
def fake_robot_registry() -> _FakeRobotRegistry:
    """Return a fresh in-memory robot registry."""
    return _FakeRobotRegistry()


@pytest.fixture
def monkeypatch_registry(
    fake_robot_registry: _FakeRobotRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeRobotRegistry:
    """Patch RobotRegistry so BodyBindingManager uses the fake robot registry."""
    import rosclaw.mcp.onboarding.binding as binding_module

    monkeypatch.setattr(binding_module, "RobotRegistry", lambda: fake_robot_registry)
    return fake_robot_registry


@pytest.fixture
def installed_unitree(
    fake_home: Path,
    unitree_manifest: McpManifest,
) -> None:
    """Record a fake installed unitree-g1 server and runtime artifacts."""
    from rosclaw.mcp.onboarding.installed import InstalledRegistry, InstalledRecord
    from rosclaw.mcp.onboarding.runner import ensure_runner

    runtime_path, runner_path = ensure_runner(unitree_manifest, fake_home)
    server_dir = fake_home / "mcp" / "servers" / unitree_manifest.server_name
    server_dir.mkdir(parents=True, exist_ok=True)
    registry = InstalledRegistry(home=fake_home)
    registry.add(
        InstalledRecord(
            server_name=unitree_manifest.server_name,
            manifest_id=unitree_manifest.id,
            name=unitree_manifest.name,
            version=unitree_manifest.version,
            installed_at="2025-01-01T00:00:00Z",
            artifact_type="python",
            server_dir=str(fake_home / "mcp" / "servers" / unitree_manifest.server_name),
            runtime_config_path=str(runtime_path),
            body_binding_key="unitree_g1",
            eurdf_profile="unitree-g1",
        )
    )
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
