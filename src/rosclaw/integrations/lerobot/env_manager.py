"""LeRobot isolated environment management.

Creates and populates a dedicated Python 3.12+ virtual environment for
LeRobot, keeping it separate from the ROSClaw core runtime.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.runtime import (
    LeRobotRuntime,
    find_python312,
    inspect_lerobot_runtime,
)
from rosclaw.integrations.lerobot.schemas import InstallReport
from rosclaw.integrations.lerobot.subprocess_runner import run_command


class LeRobotEnvManager:
    """Manage isolated LeRobot virtual environments."""

    def create_isolated_env(
        self,
        runtime_path: Path,
        python_executable: Path | str | None = None,
        force: bool = False,
    ) -> LeRobotRuntime:
        """Create (or reuse) an isolated LeRobot runtime.

        Raises a ``RuntimeError`` on fatal failures so the installer can turn
        them into a structured ``InstallReport``.
        """
        runtime_path = Path(runtime_path).expanduser()
        resolved_runtime_path = runtime_path.expanduser().resolve()
        if resolved_runtime_path in {Path("/"), Path.home().resolve()}:
            raise RuntimeError(f"Refusing unsafe LeRobot runtime path: {runtime_path}")
        if runtime_path.is_symlink():
            raise RuntimeError(f"Refusing symlink LeRobot runtime path: {runtime_path}")
        python_path = find_python312(python_executable)
        if python_path is None:
            raise RuntimeError(
                "Cannot create isolated LeRobot runtime because python3.12 was not found.\n\n"
                "Please install Python 3.12 or provide an existing runtime:\n\n"
                "  rosclaw setup lerobot --profile core --mode external --python /path/to/python3.12"
            )

        runtime_python = runtime_path / "bin" / "python"

        if runtime_path.exists():
            is_venv = runtime_python.exists() and (runtime_path / "pyvenv.cfg").exists()
            if is_venv and not force:
                return inspect_lerobot_runtime(
                    runtime_python,
                    mode="isolated",
                    runtime_path=runtime_path,
                )
            if not is_venv:
                raise RuntimeError(
                    f"Runtime path exists but does not look like a venv: {runtime_path}"
                )
            shutil.rmtree(runtime_path)

        # Create venv.
        create_result = run_command(
            [str(python_path), "-m", "venv", str(runtime_path)],
            timeout=300.0,
        )
        if not create_result.ok:
            raise RuntimeError(
                f"Failed to create isolated LeRobot runtime at {runtime_path}: "
                f"{create_result.stderr}"
            )

        # Upgrade pip.
        run_command(
            [str(runtime_python), "-m", "pip", "install", "--upgrade", "pip"],
            timeout=300.0,
        )

        return inspect_lerobot_runtime(
            runtime_python,
            mode="isolated",
            runtime_path=runtime_path,
        )

    def install_lerobot(
        self,
        runtime: LeRobotRuntime,
        profile: str = "core",
        upgrade: bool = False,
        index_url: str | None = None,
        extra_index_url: str | None = None,
        packages: list[str] | None = None,
    ) -> InstallReport:
        """Install LeRobot packages into an existing runtime."""
        pip_cmd = [str(runtime.python_executable), "-m", "pip", "install"]
        if upgrade:
            pip_cmd.append("--upgrade")
        if index_url:
            pip_cmd.extend(["--index-url", index_url])
        if extra_index_url:
            pip_cmd.extend(["--extra-index-url", extra_index_url])
        pip_cmd.extend(packages or ["lerobot"])

        pip_result = run_command(pip_cmd, timeout=600.0)
        details: dict[str, Any] = {
            "command": pip_cmd,
            "ok": pip_result.ok,
            "returncode": pip_result.returncode,
            "stderr": pip_result.stderr,
        }

        if not pip_result.ok:
            return InstallReport(
                ok=False,
                profile=profile,
                dry_run=False,
                message=f"pip install failed: {pip_result.stderr}",
                error_code="pip_install_failed",
                mode="isolated",
                runtime=runtime,
                details=details,
            )

        # Re-inspect runtime after installation.
        runtime = inspect_lerobot_runtime(
            runtime.python_executable,
            mode=runtime.mode,
            runtime_path=runtime.runtime_path,
        )
        if runtime.state == "error":
            return InstallReport(
                ok=False,
                profile=profile,
                dry_run=False,
                message=runtime.error or "Installed LeRobot runtime failed validation",
                error_code="lerobot_runtime_invalid",
                mode="isolated",
                runtime=runtime,
                details=details,
            )

        return InstallReport(
            ok=True,
            profile=profile,
            dry_run=False,
            message="LeRobot installed successfully in isolated runtime.",
            mode="isolated",
            runtime=runtime,
            lerobot_version=runtime.lerobot_version,
            python_executable=str(runtime.python_executable),
            pip_executable=str(runtime.pip_executable) if runtime.pip_executable else None,
            details=details,
        )
