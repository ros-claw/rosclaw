"""Isolated subprocess runner for the integrated auto CLI."""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest


def _subprocess_env() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    python_path = [str(project_root / "src")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


@pytest.fixture
def run_auto_cli(
    tmp_path: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Run an auto CLI module with an isolated local store."""
    env = _subprocess_env()

    def run(
        *args: str,
        module: str = "rosclaw.auto.cli",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", module, *args],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
            check=False,
        )

    return run


@pytest.fixture
def run_isolated_python(
    tmp_path: Path,
) -> Callable[[str], subprocess.CompletedProcess[str]]:
    """Run a Python snippet against this checkout in a fresh interpreter."""
    env = _subprocess_env()

    def run(code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
            check=False,
        )

    return run
