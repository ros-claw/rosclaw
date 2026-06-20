"""Hardware MCP health checks.

Runs the checks declared in ``manifest.health.checks`` against the local
installation, runtime configuration, body binding, permissions, and project
``.mcp.json``. Protocol checks can optionally perform a real MCP initialize
handshake when ``--full`` is passed.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.mcp.onboarding.binding import BodyBindingManager
from rosclaw.mcp.onboarding.claude_merge import ClaudeMcpMerge
from rosclaw.mcp.onboarding.errors import EurdfProfileMissingError
from rosclaw.mcp.onboarding.hub_client import HubClient
from rosclaw.mcp.onboarding.installed import InstalledRegistry
from rosclaw.mcp.onboarding.permissions import PermissionStore
from rosclaw.mcp.onboarding.schema import HealthCheck, McpManifest


@dataclass
class HealthResult:
    """Result of a single health check."""

    check_id: str
    category: str
    passed: bool
    required: bool
    message: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "category": self.category,
            "passed": self.passed,
            "required": self.required,
            "message": self.message,
            "duration_ms": self.duration_ms,
        }


@dataclass
class HealthReport:
    """Aggregated health report for one MCP server."""

    server_name: str
    manifest_id: str | None = None
    version: str | None = None
    overall: str = "unknown"  # ok, degraded, failed
    checks: list[HealthResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_name": self.server_name,
            "manifest_id": self.manifest_id,
            "version": self.version,
            "overall": self.overall,
            "checks": [c.to_dict() for c in self.checks],
            "skipped": list(self.skipped),
        }

    def summarize(self) -> dict[str, Any]:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        failed = [c.to_dict() for c in self.checks if not c.passed]
        return {
            "server_name": self.server_name,
            "overall": self.overall,
            "total": total,
            "passed": passed,
            "failed": failed,
        }

    def _calculate_overall(self, checks: list[HealthResult]) -> str:
        if any(not c.passed and c.required for c in checks):
            return "failed"
        if any(not c.passed for c in checks):
            return "degraded"
        return "ok"


def _expand_command(cmd: str, env: dict[str, str] | None = None) -> str:
    """Expand environment variables in a command string."""
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return os.path.expandvars(cmd)


def _command_resolvable(cmd: str, env: dict[str, str] | None = None) -> tuple[bool, str]:
    """Return whether ``cmd`` can be resolved on PATH or is absolute."""
    expanded = _expand_command(cmd, env)
    if not expanded:
        return False, "empty command"
    if os.path.isabs(expanded) and os.path.exists(expanded):
        return True, expanded
    resolved = shutil.which(expanded.split()[0])
    if resolved:
        return True, resolved
    return False, f"command not found: {expanded}"


async def _handshake_stdio(
    command: str,
    args: list[str],
    env: dict[str, str],
    timeout_ms: int,
) -> tuple[bool, str]:
    """Attempt an MCP initialize handshake over stdio."""
    try:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except Exception as exc:  # noqa: BLE001
        return False, f"mcp library not available: {exc}"

    params = StdioServerParameters(command=command, args=args, env=env)
    try:
        async with stdio_client(params) as (read, write), ClientSession(
            read, write
        ) as session:
            await asyncio.wait_for(
                session.initialize(),
                timeout=timeout_ms / 1000.0,
            )
            return True, "initialize OK"
    except TimeoutError:
        return False, f"handshake timed out after {timeout_ms}ms"
    except Exception as exc:  # noqa: BLE001
        return False, f"handshake failed: {exc}"


def _run_handshake_stdio(
    command: str,
    args: list[str],
    env: dict[str, str],
    timeout_ms: int,
) -> tuple[bool, str]:
    """Synchronous wrapper around ``_handshake_stdio`` for testability.

    If ``_handshake_stdio`` has been replaced with a synchronous test double,
    invoke it directly instead of running it through ``asyncio.run``.
    """
    if asyncio.iscoroutinefunction(_handshake_stdio):
        return asyncio.run(_handshake_stdio(command, args, env, timeout_ms))
    return _handshake_stdio(command, args, env, timeout_ms)


class HealthRunner:
    """Run health checks for installed Hardware MCP servers."""

    def __init__(
        self,
        home: Path | str | None = None,
        registry: InstalledRegistry | None = None,
        hub: HubClient | None = None,
        permission_store: PermissionStore | None = None,
        body_binding: BodyBindingManager | None = None,
        claude_merge: ClaudeMcpMerge | None = None,
    ) -> None:
        self.home = resolve_home(str(home) if home else None)
        self.registry = registry or InstalledRegistry(home=self.home)
        self.hub = hub or HubClient(home=self.home)
        self.permission_store = permission_store or PermissionStore(home=self.home)
        self.body_binding = body_binding or BodyBindingManager(workspace=self.home)
        self.claude_merge = claude_merge or ClaudeMcpMerge()

    def check(
        self,
        server_name: str,
        manifest: McpManifest | None = None,
        full: bool = False,
    ) -> HealthReport:
        """Run health checks for ``server_name``.

        If ``manifest`` is not provided, the installed registry and hub/cache
        are used to resolve it.
        """
        record = self.registry.get(server_name)
        if manifest is None:
            if record is None:
                return HealthReport(
                    server_name=server_name,
                    overall="failed",
                    checks=[
                        HealthResult(
                            check_id="installed",
                            category="install",
                            passed=False,
                            required=True,
                            message=f"'{server_name}' is not installed",
                        )
                    ],
                )
            manifest = self.hub.fetch_manifest(record.manifest_id, record.version)

        report = HealthReport(
            server_name=server_name,
            manifest_id=manifest.id,
            version=manifest.version,
        )

        health = manifest.health
        if health is None or not health.checks:
            report.checks.append(
                HealthResult(
                    check_id="manifest_health",
                    category="install",
                    passed=True,
                    required=False,
                    message="No health checks declared",
                )
            )
            report.overall = "ok"
            return report

        for check in health.checks:
            if check.category in {"hardware", "safety"} and not full:
                report.skipped.append(check.id)
                continue
            result = self._run_check(check, manifest, record, full=full)
            report.checks.append(result)

        report.overall = report._calculate_overall(report.checks)
        return report

    def check_all(self, full: bool = False) -> list[HealthReport]:
        """Run health checks for every installed server."""
        reports: list[HealthReport] = []
        for record in self.registry.list():
            reports.append(self.check(record.server_name, full=full))
        return reports

    def _run_check(
        self,
        check: HealthCheck,
        manifest: McpManifest,
        record: Any,
        full: bool,
    ) -> HealthResult:
        start = time.monotonic()
        try:
            if check.category == "install":
                passed, message = self._check_install(check, manifest, record)
            elif check.category == "protocol":
                passed, message = self._check_protocol(check, manifest, record, full=full)
            elif check.category == "binding":
                passed, message = self._check_binding(check, manifest)
            elif check.category == "permissions":
                passed, message = self._check_permissions(check, manifest)
            elif check.category == "agent":
                passed, message = self._check_agent(check, manifest)
            elif check.category in {"hardware", "safety"}:
                passed, message = self._check_hardware(check, manifest, full=full)
            else:
                passed, message = False, f"Unknown check category: {check.category}"
        except Exception as exc:  # noqa: BLE001
            passed, message = False, f"exception: {exc}"
        duration_ms = int((time.monotonic() - start) * 1000)
        return HealthResult(
            check_id=check.id,
            category=check.category,
            passed=passed,
            required=check.required,
            message=message,
            duration_ms=duration_ms,
        )

    def _check_install(
        self,
        check: HealthCheck,
        manifest: McpManifest,
        record: Any,
    ) -> tuple[bool, str]:
        if check.id == "install_integrity":
            if record is None:
                return False, "not installed"
            runtime_path = Path(record.runtime_config_path) if record.runtime_config_path else None
            if not runtime_path or not runtime_path.exists():
                return False, f"runtime config missing: {record.runtime_config_path}"
            runner_path = self.home / "mcp" / "bin" / "rosclaw-mcp-run"
            if not runner_path.exists() or not os.access(runner_path, os.X_OK):
                return False, f"runner script missing or not executable: {runner_path}"
            if record.server_dir and not Path(record.server_dir).exists():
                return False, f"server directory missing: {record.server_dir}"
            return True, "install integrity OK"
        return True, f"check '{check.id}' not specialized"

    def _check_protocol(
        self,
        check: HealthCheck,
        manifest: McpManifest,
        record: Any,
        full: bool,
    ) -> tuple[bool, str]:
        if manifest.mcp is None or manifest.mcp.transport is None:
            return False, "manifest has no transport"
        transport = manifest.mcp.transport
        command = _expand_command(transport.command, transport.env)
        args = [os.path.expandvars(a) for a in transport.args]

        resolvable, resolved = _command_resolvable(command, transport.env)
        if not resolvable:
            return False, resolved

        if not full:
            return True, f"command resolvable: {resolved}"

        # Full protocol handshake.
        env = dict(os.environ)
        env.update({k: os.path.expandvars(v) for k, v in transport.env.items()})
        timeout = (manifest.health.startup_timeout_ms if manifest.health else 5000)
        try:
            passed, message = _run_handshake_stdio(resolved, args, env, timeout)
        except Exception as exc:  # noqa: BLE001
            passed, message = False, f"handshake error: {exc}"
        return passed, message

    def _check_binding(
        self,
        check: HealthCheck,
        manifest: McpManifest,
    ) -> tuple[bool, str]:
        binding = manifest.body_binding
        if binding is None:
            return True, "no body binding declared"

        # e-URDF profile presence. Required missing profiles fail immediately;
        # optional-only missing profiles are allowed to pass without a linked body.
        eurdf = manifest.eurdf
        if eurdf and eurdf.profiles:
            try:
                profile_id, _ = self.body_binding.ensure_eurdf(eurdf, dry_run=True)
            except EurdfProfileMissingError:
                return False, "required e-URDF profile not installed"
            if profile_id is None:
                has_required = any(p.required for p in eurdf.profiles)
                if has_required:
                    return False, "required e-URDF profile not installed"
                return True, "optional e-URDF profile not installed"

        if not self.body_binding.resolver.is_linked():
            return False, f"body not linked at {self.body_binding.resolver.body_yaml_path}"
        body_path = self.body_binding.resolver.body_yaml_path
        try:
            with open(body_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:  # noqa: BLE001
            return False, f"cannot read body.yaml: {exc}"

        # Verify every declared write path is present in body.yaml.
        target_paths = list(binding.write_paths.values()) if binding.write_paths else []
        if not target_paths:
            target_paths = [binding.binding_key]
        for target_path in target_paths:
            current: Any = data
            for part in target_path.split("."):
                if not isinstance(current, dict):
                    return False, f"binding key '{target_path}' not present"
                current = current.get(part)
            if current is None:
                return False, f"binding key '{target_path}' not present"

        return True, "binding OK"

    def _check_permissions(
        self,
        check: HealthCheck,
        manifest: McpManifest,
    ) -> tuple[bool, str]:
        permissions = manifest.permissions
        if permissions is None:
            return True, "no permissions declared"
        state = self.permission_store.get(manifest.server_name)
        missing_required = [
            decl.id
            for decl in permissions.required
            if decl.id not in state.granted and decl.id not in state.denied
        ]
        forbidden_granted = [
            decl.id
            for decl in permissions.required
            if decl.level == "forbidden_by_default" and decl.id in state.granted
        ]
        if forbidden_granted:
            return False, f"forbidden permissions granted: {', '.join(forbidden_granted)}"
        if missing_required:
            return False, f"required permissions pending: {', '.join(missing_required)}"
        return True, "permissions OK"

    def _check_agent(
        self,
        check: HealthCheck,
        manifest: McpManifest,
    ) -> tuple[bool, str]:
        managed = self.claude_merge.list_managed_servers()
        claude_server_name = manifest.claude_server_name
        if claude_server_name in managed:
            return True, "managed server present in .mcp.json"
        return False, "managed server not found in .mcp.json"

    def _check_hardware(
        self,
        check: HealthCheck,
        manifest: McpManifest,
        full: bool,
    ) -> tuple[bool, str]:
        if not full:
            return True, "skipped (use --full to run hardware checks)"
        # P1 placeholder: real hardware checks require device-specific logic.
        return True, "hardware check skipped (P1 placeholder)"
