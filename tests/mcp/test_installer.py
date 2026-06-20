"""Tests for the Hardware MCP staged installer."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from rosclaw.mcp.onboarding.errors import PermissionDeniedError, PreflightError
from rosclaw.mcp.onboarding.installer import InstallEngine, PythonPackageInstaller
from rosclaw.mcp.onboarding.schema import Artifact, McpManifest, PermissionDecl, Permissions


class _FakeInstaller:
    def install(
        self,
        manifest: McpManifest,
        artifact: Artifact,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return {
            "server_dir": "/fake/server/dir",
            "command": "fake-install",
            "dry_run": dry_run,
        }


@pytest.fixture
def patched_select_installer(monkeypatch: pytest.MonkeyPatch) -> None:
    import rosclaw.mcp.onboarding.installer as installer_module

    monkeypatch.setattr(installer_module, "_select_installer", lambda artifact: _FakeInstaller())


def test_plan_returns_dry_run_info(unitree_manifest: McpManifest, fake_home: Path) -> None:
    engine = InstallEngine(home=fake_home)
    plan = engine.plan("unitree-g1")
    assert plan.manifest.id == "io.rosclaw.hardware.unitree-g1"
    assert plan.solved.version == "1.0.0"
    assert plan.installer_type == "python"
    assert plan.install_command is not None
    assert plan.body_patch.get("mcp_bindings.unitree_g1")
    assert plan.permission_state is not None


def test_install_dry_run_does_not_mutate(
    unitree_manifest: McpManifest,
    fake_home: Path,
    project_root: Path,
    monkeypatch_registry: Any,
    patched_select_installer: None,
) -> None:
    monkeypatch_registry.install("unitree-g1")
    body_path = fake_home / "body" / "body.yaml"
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text("schema_version: rosclaw.body.v1\n", encoding="utf-8")

    engine = InstallEngine(home=fake_home, project_root=project_root)
    result = engine.install("unitree-g1", dry_run=True)

    assert result.success
    assert result.installed_record is not None
    assert result.installed_record.status == "planned"
    assert not (fake_home / "mcp" / "installed.yaml").exists()


def test_install_writes_registry_and_runtime(
    unitree_manifest: McpManifest,
    fake_home: Path,
    project_root: Path,
    monkeypatch_registry: Any,
    patched_select_installer: None,
) -> None:
    monkeypatch_registry.install("unitree-g1")
    body_path = fake_home / "body" / "body.yaml"
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text("schema_version: rosclaw.body.v1\n", encoding="utf-8")

    engine = InstallEngine(home=fake_home, project_root=project_root)
    result = engine.install("unitree-g1", dry_run=False)

    assert result.success, result.errors
    assert result.runtime_config_path is not None
    assert result.runner_script_path is not None
    assert result.installed_record.status == "installed"
    assert result.claude_result is not None


def test_install_preflight_failure_aborts(
    fake_home: Path,
    project_root: Path,
    unitree_manifest: McpManifest,
) -> None:
    from rosclaw.mcp.onboarding.preflight import PreflightRunner

    class _FailingPreflight(PreflightRunner):
        def run(self, manifest: McpManifest, dry_run: bool = False):
            raise PreflightError("simulated preflight failure")

    engine = InstallEngine(
        home=fake_home,
        project_root=project_root,
        preflight_runner=_FailingPreflight(),
    )
    result = engine.install("unitree-g1", dry_run=True)
    assert not result.success
    assert "simulated preflight failure" in result.errors


def test_install_forbidden_permission_raises(
    fake_home: Path,
    project_root: Path,
    monkeypatch_registry: Any,
    patched_select_installer: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rosclaw.mcp.onboarding.hub_client import HubClient

    hub = HubClient(home=fake_home, offline=True)
    manifest = hub.fetch_manifest("io.rosclaw.hardware.unitree-g1")
    manifest.permissions = Permissions(
        required=[PermissionDecl(id="mcp:bad", level="forbidden_by_default")]
    )
    monkeypatch.setattr(hub, "fetch_manifest", lambda manifest_id, version=None: manifest)

    engine = InstallEngine(home=fake_home, project_root=project_root, hub=hub)
    with pytest.raises(PermissionDeniedError):
        engine.install("unitree-g1", dry_run=False)


def test_python_installer_dry_run_returns_command(unitree_manifest: McpManifest) -> None:
    artifact = Artifact(type="python", package="unitree-g1-mcp")
    installer = PythonPackageInstaller()
    info = installer.install(unitree_manifest, artifact, dry_run=True)
    assert info["dry_run"] is True
    assert "unitree-g1-mcp" in info["command"]


def test_select_installer_unsupported_type() -> None:
    from rosclaw.mcp.onboarding.installer import _select_installer
    from rosclaw.mcp.onboarding.errors import InstallationError

    with pytest.raises(InstallationError):
        _select_installer(Artifact(type="unsupported"))
