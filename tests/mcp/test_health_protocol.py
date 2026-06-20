"""Tests for MCP health protocol checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rosclaw.mcp.onboarding.claude_merge import ClaudeMcpMerge
from rosclaw.mcp.onboarding.health import HealthRunner
from rosclaw.mcp.onboarding.schema import HealthCheck, McpManifest


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


@pytest.fixture
def managed_mcp_json(
    project_root: Path,
    unitree_manifest: McpManifest,
) -> Path:
    merger = ClaudeMcpMerge(project_root=project_root)
    merger.merge(
        server_name=unitree_manifest.server_name,
        manifest_id=unitree_manifest.id,
        version=unitree_manifest.version,
        mcp_json_fragment=unitree_manifest.claude.mcp_json,
        dry_run=False,
    )
    return merger.mcp_json_path


def test_check_not_installed_server_returns_failed(
    fake_home: Path,
    project_root: Path,
) -> None:
    runner = HealthRunner(home=fake_home, claude_merge=ClaudeMcpMerge(project_root=project_root))
    report = runner.check("unitree-g1")
    assert report.overall == "failed"
    assert any(c.check_id == "installed" and not c.passed for c in report.checks)


def test_check_all_empty_registry_returns_empty(
    fake_home: Path,
    project_root: Path,
) -> None:
    runner = HealthRunner(home=fake_home, claude_merge=ClaudeMcpMerge(project_root=project_root))
    assert runner.check_all() == []


def test_protocol_check_resolvable_without_full(
    installed_unitree: None,
    health_runner: HealthRunner,
    unitree_manifest: McpManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rosclaw.mcp.onboarding.health._command_resolvable",
        lambda cmd, env=None: (True, "/fake/python"),
    )
    report = health_runner.check(unitree_manifest.server_name, full=False)
    proto = next(c for c in report.checks if c.category == "protocol")
    assert proto.passed
    assert "resolvable" in proto.message


def test_protocol_full_handshake_success(
    installed_unitree: None,
    health_runner: HealthRunner,
    unitree_manifest: McpManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rosclaw.mcp.onboarding.health._command_resolvable",
        lambda cmd, env=None: (True, "/fake/python"),
    )
    monkeypatch.setattr(
        "rosclaw.mcp.onboarding.health._handshake_stdio",
        lambda command, args, env, timeout: (True, "initialize OK"),
    )
    report = health_runner.check(unitree_manifest.server_name, full=True)
    proto = next(c for c in report.checks if c.category == "protocol")
    assert proto.passed
    assert "initialize OK" in proto.message


def test_protocol_full_handshake_failure(
    installed_unitree: None,
    health_runner: HealthRunner,
    unitree_manifest: McpManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rosclaw.mcp.onboarding.health._command_resolvable",
        lambda cmd, env=None: (True, "/fake/python"),
    )
    monkeypatch.setattr(
        "rosclaw.mcp.onboarding.health._handshake_stdio",
        lambda command, args, env, timeout: (False, "timeout"),
    )
    report = health_runner.check(unitree_manifest.server_name, full=True)
    proto = next(c for c in report.checks if c.category == "protocol")
    assert not proto.passed


def test_install_integrity_check_passes(
    installed_unitree: None,
    health_runner: HealthRunner,
    unitree_manifest: McpManifest,
) -> None:
    report = health_runner.check(unitree_manifest.server_name)
    integrity = next(c for c in report.checks if c.check_id == "install_integrity")
    assert integrity.passed


def test_agent_check_passes_when_managed(
    installed_unitree: None,
    health_runner: HealthRunner,
    unitree_manifest: McpManifest,
    managed_mcp_json: Path,
) -> None:
    report = health_runner.check(unitree_manifest.server_name)
    agent = next((c for c in report.checks if c.category == "agent"), None)
    if agent is None:
        pytest.skip("manifest has no agent health check")
    assert agent.passed


def test_health_report_to_dict(unitree_manifest: McpManifest) -> None:
    from rosclaw.mcp.onboarding.health import HealthReport, HealthResult

    report = HealthReport(
        server_name=unitree_manifest.server_name,
        checks=[HealthResult(check_id="x", category="install", passed=True, required=True)],
    )
    data = report.to_dict()
    assert data["overall"] == "unknown"
    assert data["checks"][0]["check_id"] == "x"


def test_health_overall_calculation(unitree_manifest: McpManifest) -> None:
    from rosclaw.mcp.onboarding.health import HealthReport, HealthResult

    report = HealthReport(server_name=unitree_manifest.server_name)
    report.checks = [
        HealthResult(check_id="a", category="install", passed=True, required=True),
        HealthResult(check_id="b", category="install", passed=False, required=False),
    ]
    assert report._calculate_overall(report.checks) == "degraded"

    report.checks.append(
        HealthResult(check_id="c", category="install", passed=False, required=True),
    )
    assert report._calculate_overall(report.checks) == "failed"


def test_unknown_check_category_fails(
    unitree_manifest: McpManifest,
    fake_home: Path,
    project_root: Path,
) -> None:
    manifest = unitree_manifest
    manifest.health.checks = [HealthCheck(id="weird", category="unknown_category", required=True)]
    runner = HealthRunner(home=fake_home, claude_merge=ClaudeMcpMerge(project_root=project_root))
    report = runner.check(unitree_manifest.server_name, manifest=manifest)
    check = report.checks[0]
    assert not check.passed
    assert "Unknown check category" in check.message
