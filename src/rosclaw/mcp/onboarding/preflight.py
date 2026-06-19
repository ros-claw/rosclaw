"""Hardware MCP preflight checks.

Runs the commands declared in ``manifest.install.preflight`` before any files
are written or packages installed. Required failures abort installation.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from rosclaw.mcp.onboarding.errors import PreflightError
from rosclaw.mcp.onboarding.schema import InstallDecl, McpManifest


@dataclass
class PreflightResult:
    """Result of a single preflight check."""

    id: str
    command: str
    passed: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    required: bool = True
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command,
            "passed": self.passed,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "required": self.required,
            "message": self.message,
        }


class PreflightRunner:
    """Execute preflight checks for a manifest."""

    DEFAULT_TIMEOUT = 60

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    def run(self, manifest: McpManifest, dry_run: bool = False) -> list[PreflightResult]:
        """Run all preflight checks and return results.

        Raises:
            PreflightError: if a required check fails (or would fail in dry-run).
        """
        install = manifest.install or InstallDecl()
        results: list[PreflightResult] = []
        for check in install.preflight:
            if dry_run:
                result = PreflightResult(
                    id=check.id,
                    command=check.command,
                    passed=True,
                    message="dry-run: skipped",
                )
            else:
                result = self._execute(check.id, check.command, check.required)
            results.append(result)
            if check.required and not result.passed:
                raise PreflightError(
                    f"Preflight check '{check.id}' failed: {result.message}"
                )
        return results

    def _execute(self, check_id: str, command: str, required: bool) -> PreflightResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                executable="/bin/bash",
            )
        except subprocess.TimeoutExpired as exc:
            return PreflightResult(
                id=check_id,
                command=command,
                passed=False,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                message=f"timed out after {self.timeout}s",
                required=required,
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                id=check_id,
                command=command,
                passed=False,
                stderr=str(exc),
                message=f"could not execute: {exc}",
                required=required,
            )

        passed = proc.returncode == 0
        return PreflightResult(
            id=check_id,
            command=command,
            passed=passed,
            stdout=proc.stdout.strip(),
            stderr=proc.stderr.strip(),
            returncode=proc.returncode,
            required=required,
            message="" if passed else f"exit code {proc.returncode}",
        )

    def summarize(self, results: list[PreflightResult]) -> dict[str, Any]:
        """Return a concise summary for CLI output."""
        return {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": [r.to_dict() for r in results if not r.passed],
        }
