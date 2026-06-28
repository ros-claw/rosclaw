"""Tests for MCP health binding checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rosclaw.mcp.onboarding.claude_merge import ClaudeMcpMerge
from rosclaw.mcp.onboarding.health import HealthRunner
from rosclaw.mcp.onboarding.schema import McpManifest


@pytest.fixture
def health_runner(
    fake_home: Path,
    project_root: Path,
    monkeypatch_registry: Any,
) -> HealthRunner:
    monkeypatch_registry.install("unitree-g1")
    return HealthRunner(
        home=fake_home,
        claude_merge=ClaudeMcpMerge(project_root=project_root),
    )


def test_binding_check_passes_with_linked_body(
    installed_unitree: None,
    health_runner: HealthRunner,
    unitree_manifest: McpManifest,
    body_yaml_unitree: Path,
) -> None:
    report = health_runner.check(unitree_manifest.server_name)
    binding = next(c for c in report.checks if c.category == "binding")
    assert binding.passed, binding.message


def test_binding_check_fails_when_body_not_linked(
    installed_unitree: None,
    health_runner: HealthRunner,
    unitree_manifest: McpManifest,
    fake_home: Path,
) -> None:
    report = health_runner.check(unitree_manifest.server_name)
    binding = next(c for c in report.checks if c.category == "binding")
    assert not binding.passed
    assert "body not linked" in binding.message.lower()


def test_binding_check_fails_when_binding_key_missing(
    installed_unitree: None,
    health_runner: HealthRunner,
    unitree_manifest: McpManifest,
    fake_home: Path,
) -> None:
    body_path = fake_home / "body" / "body.yaml"
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text("schema_version: rosclaw.body.v1\n", encoding="utf-8")

    report = health_runner.check(unitree_manifest.server_name)
    binding = next(c for c in report.checks if c.category == "binding")
    assert not binding.passed
    assert "binding key" in binding.message.lower()


def test_optional_profile_missing_does_not_fail(
    realsense_manifest: McpManifest,
    fake_home: Path,
    project_root: Path,
    body_yaml_empty: Path,
    monkeypatch_registry: Any,
) -> None:
    # realsense e-URDF profile is optional; binding check should pass without it.
    runner = HealthRunner(
        home=fake_home,
        claude_merge=ClaudeMcpMerge(project_root=project_root),
    )
    report = runner.check(realsense_manifest.server_name, manifest=realsense_manifest)
    binding = next(c for c in report.checks if c.category == "binding")
    assert binding.passed, binding.message


def test_required_profile_missing_fails(
    unitree_manifest_dict: dict[str, Any],
    fake_home: Path,
    project_root: Path,
    body_yaml_empty: Path,
) -> None:
    # Use a manifest that references a profile which definitely does not exist.
    data = dict(unitree_manifest_dict)
    data["eurdf"] = {
        "profiles": [{"id": "missing-robot-xyz", "version": "1.0.0", "required": True}],
        "defaultProfile": "missing-robot-xyz",
    }
    manifest = McpManifest.from_dict(data)
    runner = HealthRunner(
        home=fake_home,
        claude_merge=ClaudeMcpMerge(project_root=project_root),
    )
    report = runner.check(manifest.server_name, manifest=manifest)
    binding = next(c for c in report.checks if c.category == "binding")
    assert not binding.passed
    assert "required e-URDF profile not installed" in binding.message


def test_no_binding_check_when_manifest_lacks_binding(
    unitree_manifest: McpManifest,
    fake_home: Path,
    project_root: Path,
) -> None:
    manifest = unitree_manifest
    manifest.body_binding = None
    runner = HealthRunner(
        home=fake_home,
        claude_merge=ClaudeMcpMerge(project_root=project_root),
    )
    report = runner.check(unitree_manifest.server_name, manifest=manifest)
    binding = next((c for c in report.checks if c.category == "binding"), None)
    if binding is not None:
        assert binding.passed
        assert "no body binding" in binding.message.lower()
