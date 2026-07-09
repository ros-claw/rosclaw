"""LeRobot integration subprocess helpers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    """Result of a subprocess command."""

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    command: list[str]


def run_command(
    cmd: list[str],
    *,
    timeout: float = 60.0,
    check: bool = False,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> CommandResult:
    """Run a subprocess command and return a structured result.

    This wrapper guarantees no exception escapes: failures are captured in the
    returned ``CommandResult``.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
            cwd=cwd,
        )
        return CommandResult(
            ok=result.returncode == 0,
            returncode=result.returncode,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
            command=cmd,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            ok=False,
            returncode=-1,
            stdout="",
            stderr=f"Command timed out after {timeout}s: {cmd}",
            command=cmd,
        )
    except Exception as exc:  # noqa: BLE001
        return CommandResult(
            ok=False,
            returncode=-1,
            stdout="",
            stderr=f"Failed to run command {cmd}: {exc}",
            command=cmd,
        )


def which(command: str) -> str | None:
    """Return the absolute path to ``command`` if it is on PATH."""
    return shutil.which(command)
