"""Post-install health checks for ROSClaw Hub assets.

Health checks are declared in ``manifest.install.health_checks``.  The runner
supports lightweight checks that do not require real hardware:

* ``file_exists`` – verify a path relative to the asset directory.
* ``python_import`` – import a Python module (asset dir is added to
  ``sys.path`` temporarily).
* ``mcp_list_tools`` / ``mcp_call`` – reserved for MCP servers; currently
  skipped in local/offline mode to avoid spawning untrusted executables.
* ``mujoco_load`` – reserved for MuJoCo XML loading; skipped unless a
  simulator is available.

The installer uses the aggregate status to decide whether an asset is
``healthy`` or ``unhealthy``.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.hub.schema import AssetManifest


class HealthStatus:
    """String constants for health check results."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    SKIPPED = "skipped"
    PENDING = "pending"


@dataclass
class SingleCheckResult:
    """Result of one named health check."""

    name: str
    check_type: str
    status: str
    message: str = ""
    duration_sec: float = 0.0


@dataclass
class HealthResult:
    """Aggregate health result for an asset."""

    status: str = HealthStatus.PENDING
    checks: list[SingleCheckResult] = field(default_factory=list)
    summary: str = ""

    @property
    def healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


def _check_file_exists(
    asset_dir: Path,
    check: dict[str, Any],
    *,
    timeout: float,
) -> SingleCheckResult:
    """Verify that *target* exists relative to *asset_dir*."""
    target = check.get("target", "")
    path = asset_dir / target if target else asset_dir
    exists = path.exists()
    return SingleCheckResult(
        name=check.get("name", "file_exists"),
        check_type="file_exists",
        status=HealthStatus.HEALTHY if exists else HealthStatus.UNHEALTHY,
        message=(f"Found {path}" if exists else f"Missing file: {path}"),
    )


def _check_python_import(
    asset_dir: Path,
    check: dict[str, Any],
    *,
    timeout: float,
) -> SingleCheckResult:
    """Import a Python module with *asset_dir* temporarily on ``sys.path``."""
    module_name = check.get("target", "")
    if not module_name:
        return SingleCheckResult(
            name=check.get("name", "python_import"),
            check_type="python_import",
            status=HealthStatus.UNHEALTHY,
            message="No target module specified",
        )

    with contextlib.suppress(ValueError):
        sys.path.insert(0, str(asset_dir))
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        return SingleCheckResult(
            name=check.get("name", "python_import"),
            check_type="python_import",
            status=HealthStatus.UNHEALTHY,
            message=f"Cannot import {module_name}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - any import failure is unhealthy
        return SingleCheckResult(
            name=check.get("name", "python_import"),
            check_type="python_import",
            status=HealthStatus.UNHEALTHY,
            message=f"Import of {module_name} failed: {exc}",
        )
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(asset_dir))

    return SingleCheckResult(
        name=check.get("name", "python_import"),
        check_type="python_import",
        status=HealthStatus.HEALTHY,
        message=f"Successfully imported {module_name}",
    )


def _check_mcp(
    asset_dir: Path,
    check: dict[str, Any],
    *,
    timeout: float,
) -> SingleCheckResult:
    """MCP checks are skipped locally to avoid spawning untrusted executables."""
    check_type = check.get("type", "mcp_list_tools")
    return SingleCheckResult(
        name=check.get("name", check_type),
        check_type=check_type,
        status=HealthStatus.SKIPPED,
        message=("MCP health checks require a running MCP session; skipped in offline/local mode"),
    )


def _check_mujoco_load(
    asset_dir: Path,
    check: dict[str, Any],
    *,
    timeout: float,
) -> SingleCheckResult:
    """MuJoCo XML loading is skipped unless the simulator is available."""
    target = check.get("target", "")
    try:
        import mujoco
    except ImportError:
        return SingleCheckResult(
            name=check.get("name", "mujoco_load"),
            check_type="mujoco_load",
            status=HealthStatus.SKIPPED,
            message="mujoco is not installed; skipping XML load check",
        )

    path = asset_dir / target if target else asset_dir
    if not path.exists():
        return SingleCheckResult(
            name=check.get("name", "mujoco_load"),
            check_type="mujoco_load",
            status=HealthStatus.UNHEALTHY,
            message=f"MuJoCo model file not found: {path}",
        )

    try:
        mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # noqa: BLE001
        return SingleCheckResult(
            name=check.get("name", "mujoco_load"),
            check_type="mujoco_load",
            status=HealthStatus.UNHEALTHY,
            message=f"Failed to load MuJoCo model {path}: {exc}",
        )

    return SingleCheckResult(
        name=check.get("name", "mujoco_load"),
        check_type="mujoco_load",
        status=HealthStatus.HEALTHY,
        message=f"Loaded MuJoCo model {path}",
    )


_CHECK_FUNCS: dict[
    str,
    Any,
] = {
    "file_exists": _check_file_exists,
    "python_import": _check_python_import,
    "mcp_list_tools": _check_mcp,
    "mcp_call": _check_mcp,
    "mujoco_load": _check_mujoco_load,
}


def run_health_checks(
    manifest: AssetManifest,
    asset_dir: Path,
    *,
    dry_run: bool = False,
    timeout: float = 30.0,
) -> HealthResult:
    """Run all declared health checks for *manifest*.

    Args:
        manifest: The installed asset manifest.
        asset_dir: Directory where the asset was installed.
        dry_run: If True, mark every check as skipped without executing it.
        timeout: Default per-check timeout in seconds.

    Returns:
        :class:`HealthResult` with the aggregate status.
    """
    install_section: dict[str, Any] = manifest.install or {}
    checks: list[dict[str, Any]] = install_section.get("health_checks", [])
    if not checks:
        return HealthResult(
            status=HealthStatus.HEALTHY,
            summary="No health checks declared",
        )

    results: list[SingleCheckResult] = []
    has_unhealthy = False
    has_skipped = False

    for check in checks:
        if not isinstance(check, dict):
            continue
        check_type = check.get("type", "")
        if dry_run:
            result = SingleCheckResult(
                name=str(check.get("name", check_type)),
                check_type=check_type,
                status=HealthStatus.SKIPPED,
                message="Skipped in dry-run mode",
            )
        else:
            func = _CHECK_FUNCS.get(check_type)
            if func is None:
                result = SingleCheckResult(
                    name=str(check.get("name", check_type)),
                    check_type=check_type,
                    status=HealthStatus.SKIPPED,
                    message=f"Unknown health check type: {check_type}",
                )
            else:
                try:
                    result = func(asset_dir, check, timeout=timeout)
                except Exception as exc:  # noqa: BLE001
                    result = SingleCheckResult(
                        name=str(check.get("name", check_type)),
                        check_type=check_type,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check raised: {exc}",
                    )

        results.append(result)
        if result.status == HealthStatus.UNHEALTHY:
            has_unhealthy = True
        elif result.status == HealthStatus.SKIPPED:
            has_skipped = True

    if has_unhealthy:
        aggregate = HealthStatus.UNHEALTHY
        summary = "One or more health checks failed"
    elif has_skipped:
        aggregate = HealthStatus.HEALTHY
        summary = "Required checks passed; some checks skipped"
    else:
        aggregate = HealthStatus.HEALTHY
        summary = "All health checks passed"

    return HealthResult(status=aggregate, checks=results, summary=summary)


def aggregate_health_status(checks: list[SingleCheckResult]) -> str:
    """Reduce a list of check results to a single status string."""
    if any(c.status == HealthStatus.UNHEALTHY for c in checks):
        return HealthStatus.UNHEALTHY
    if any(c.status == HealthStatus.SKIPPED for c in checks):
        return HealthStatus.HEALTHY
    if all(c.status == HealthStatus.HEALTHY for c in checks):
        return HealthStatus.HEALTHY
    return HealthStatus.PENDING
