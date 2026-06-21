"""Tests for ROSClaw Hub health checks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from rosclaw.hub.health import (
    HealthResult,
    HealthStatus,
    SingleCheckResult,
    aggregate_health_status,
    run_health_checks,
)
from rosclaw.hub.schema import AssetManifest, load_manifest


@pytest.fixture
def valid_manifest() -> AssetManifest:
    """Load a valid manifest fixture as a base for health tests."""
    fixture = (
        Path(__file__).parents[1] / "fixtures" / "hub_assets" / "skill_valid" / "manifest.yaml"
    )
    return load_manifest(fixture)


def _manifest_with_checks(manifest: AssetManifest, checks: list[dict[str, Any]]) -> AssetManifest:
    """Return a copy of *manifest* with the given health checks."""
    manifest.install = {"health_checks": checks}
    return manifest


def test_no_checks_declared(valid_manifest) -> None:
    """An asset with no health checks is considered healthy."""
    valid_manifest.install = {}
    result = run_health_checks(valid_manifest, Path("/nonexistent"))
    assert result.status == HealthStatus.HEALTHY
    assert "No health checks declared" in result.summary


def test_file_exists_pass(valid_manifest, tmp_path) -> None:
    """file_exists passes when the target is present."""
    target = tmp_path / "marker.txt"
    target.write_text("ok", encoding="utf-8")
    manifest = _manifest_with_checks(
        valid_manifest,
        [{"name": "has-marker", "type": "file_exists", "target": "marker.txt"}],
    )
    result = run_health_checks(manifest, tmp_path)
    assert result.status == HealthStatus.HEALTHY
    assert result.checks[0].status == HealthStatus.HEALTHY
    assert "Found" in result.checks[0].message


def test_file_exists_fail(valid_manifest, tmp_path) -> None:
    """file_exists fails when the target is missing."""
    manifest = _manifest_with_checks(
        valid_manifest,
        [{"name": "missing", "type": "file_exists", "target": "not-there.txt"}],
    )
    result = run_health_checks(manifest, tmp_path)
    assert result.status == HealthStatus.UNHEALTHY
    assert result.checks[0].status == HealthStatus.UNHEALTHY
    assert "Missing file" in result.checks[0].message


def test_python_import_pass(valid_manifest, tmp_path) -> None:
    """python_import passes when the module is importable from the asset dir."""
    (tmp_path / "my_health_module.py").write_text("x = 1\n", encoding="utf-8")
    manifest = _manifest_with_checks(
        valid_manifest,
        [{"name": "import-ok", "type": "python_import", "target": "my_health_module"}],
    )
    result = run_health_checks(manifest, tmp_path)
    assert result.status == HealthStatus.HEALTHY
    assert result.checks[0].status == HealthStatus.HEALTHY
    assert "Successfully imported" in result.checks[0].message
    # Ensure sys.path is restored.
    assert str(tmp_path) not in sys.path


def test_python_import_fail(valid_manifest, tmp_path) -> None:
    """python_import fails for a missing module."""
    manifest = _manifest_with_checks(
        valid_manifest,
        [{"name": "import-bad", "type": "python_import", "target": "no_such_module_12345"}],
    )
    result = run_health_checks(manifest, tmp_path)
    assert result.status == HealthStatus.UNHEALTHY
    assert result.checks[0].status == HealthStatus.UNHEALTHY
    assert "Cannot import" in result.checks[0].message


def test_python_import_no_target(valid_manifest, tmp_path) -> None:
    """python_import without a target is unhealthy."""
    manifest = _manifest_with_checks(
        valid_manifest,
        [{"name": "import-empty", "type": "python_import"}],
    )
    result = run_health_checks(manifest, tmp_path)
    assert result.status == HealthStatus.UNHEALTHY
    assert "No target module specified" in result.checks[0].message


def test_unknown_check_type_skipped(valid_manifest, tmp_path) -> None:
    """Unknown check types are skipped but the aggregate remains healthy."""
    manifest = _manifest_with_checks(
        valid_manifest,
        [{"name": "future-check", "type": "future_magic_check"}],
    )
    result = run_health_checks(manifest, tmp_path)
    assert result.status == HealthStatus.HEALTHY
    assert result.checks[0].status == HealthStatus.SKIPPED
    assert "Unknown health check type" in result.checks[0].message


def test_dry_run_skips_all_checks(valid_manifest, tmp_path) -> None:
    """Dry-run mode skips every declared check."""
    manifest = _manifest_with_checks(
        valid_manifest,
        [
            {"name": "a", "type": "file_exists", "target": "x"},
            {"name": "b", "type": "python_import", "target": "y"},
        ],
    )
    result = run_health_checks(manifest, tmp_path, dry_run=True)
    assert result.status == HealthStatus.HEALTHY
    assert all(c.status == HealthStatus.SKIPPED for c in result.checks)
    assert "Skipped in dry-run mode" in result.checks[0].message


def test_mcp_checks_skipped(valid_manifest, tmp_path) -> None:
    """MCP health checks are skipped locally."""
    manifest = _manifest_with_checks(
        valid_manifest,
        [
            {"name": "mcp-list", "type": "mcp_list_tools"},
            {"name": "mcp-call", "type": "mcp_call"},
        ],
    )
    result = run_health_checks(manifest, tmp_path)
    assert result.status == HealthStatus.HEALTHY
    assert all(c.status == HealthStatus.SKIPPED for c in result.checks)
    assert "MCP health checks require a running MCP session" in result.checks[0].message


def test_mujoco_load_skipped_when_mujoco_missing(valid_manifest, tmp_path, monkeypatch) -> None:
    """mujoco_load is skipped when the simulator is not available."""
    monkeypatch.setitem(sys.modules, "mujoco", None)  # type: ignore[arg-type]
    manifest = _manifest_with_checks(
        valid_manifest,
        [{"name": "mujoco", "type": "mujoco_load", "target": "model.xml"}],
    )
    result = run_health_checks(manifest, tmp_path)
    assert result.status == HealthStatus.HEALTHY
    assert result.checks[0].status == HealthStatus.SKIPPED
    assert "mujoco is not installed" in result.checks[0].message


def test_check_exception_becomes_unhealthy(valid_manifest, tmp_path, monkeypatch) -> None:
    """An unexpected exception from a check function is recorded as unhealthy."""

    def _boom(*args, **kwargs) -> SingleCheckResult:  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr("rosclaw.hub.health._CHECK_FUNCS", {"explodes": _boom})
    manifest = _manifest_with_checks(
        valid_manifest,
        [{"name": "kaboom", "type": "explodes"}],
    )
    result = run_health_checks(manifest, tmp_path)
    assert result.status == HealthStatus.UNHEALTHY
    assert "boom" in result.checks[0].message


def test_aggregate_health_status_unhealthy() -> None:
    """Aggregate status is unhealthy when any check is unhealthy."""
    checks = [
        SingleCheckResult("a", "file_exists", HealthStatus.HEALTHY),
        SingleCheckResult("b", "python_import", HealthStatus.UNHEALTHY),
    ]
    assert aggregate_health_status(checks) == HealthStatus.UNHEALTHY


def test_aggregate_health_status_skipped() -> None:
    """Aggregate status is healthy when checks are only healthy or skipped."""
    checks = [
        SingleCheckResult("a", "file_exists", HealthStatus.HEALTHY),
        SingleCheckResult("b", "mcp_list_tools", HealthStatus.SKIPPED),
    ]
    assert aggregate_health_status(checks) == HealthStatus.HEALTHY


def test_aggregate_health_status_empty() -> None:
    """Aggregate status is healthy for an empty list (no failures to report)."""
    assert aggregate_health_status([]) == HealthStatus.HEALTHY


def test_health_result_healthy_property() -> None:
    """HealthResult.healthy reflects the aggregate status."""
    assert HealthResult(status=HealthStatus.HEALTHY).healthy is True
    assert HealthResult(status=HealthStatus.UNHEALTHY).healthy is False
