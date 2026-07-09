"""LeRobot integration installer."""

from __future__ import annotations

import os
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import yaml

from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.integrations.lerobot.profiles import load_profile
from rosclaw.integrations.lerobot.schemas import InstallReport
from rosclaw.integrations.lerobot.subprocess_runner import (
    CommandResult,
    run_command,
    which,
)


class LeRobotInstaller:
    """Install or dry-run the LeRobot integration."""

    def __init__(
        self,
        python_executable: str | None = None,
        pip_executable: str | None = None,
        config_dir: Path | None = None,
    ):
        self.python_executable = python_executable or sys.executable
        self.pip_executable = pip_executable or self._find_pip()
        self.config_dir = config_dir or get_rosclaw_home() / "integrations"

    @staticmethod
    def _find_pip() -> str:
        for candidate in ("pip", "pip3"):
            path = which(candidate)
            if path:
                return path
        return "pip"

    def install(
        self,
        profile_name: str,
        *,
        dry_run: bool = False,
        upgrade: bool = False,
        env: dict[str, str] | None = None,
    ) -> InstallReport:
        """Install LeRobot for the requested profile."""
        try:
            profile = load_profile(profile_name)
        except FileNotFoundError as exc:
            return InstallReport(
                ok=False,
                profile=profile_name,
                dry_run=dry_run,
                message=f"Profile '{profile_name}' not found: {exc}",
            )

        env = env or {}
        env = {**os.environ, **env}
        if "HF_ENDPOINT" not in env:
            env["HF_ENDPOINT"] = "https://hf-mirror.com"

        details: dict[str, Any] = {
            "profile": profile_name,
            "python": self.python_executable,
            "pip": self.pip_executable,
            "packages": profile.pip,
            "checks": profile.checks,
        }

        if dry_run:
            details["dry_run"] = True
            return InstallReport(
                ok=True,
                profile=profile_name,
                dry_run=True,
                message=(
                    f"Dry-run: would install profile '{profile_name}' "
                    f"with packages {profile.pip} and run checks {profile.checks}."
                ),
                python_executable=self.python_executable,
                pip_executable=self.pip_executable,
                details=details,
            )

        # Fast path: skip pip install when LeRobot is already importable and the
        # caller is not requesting an upgrade. This avoids blocking on PyPI when
        # a local/venv installation is already present.
        skip_pip = env.get("ROSCLAW_LEROBOT_SKIP_PIP_INSTALL", "").lower() in (
            "1",
            "true",
            "yes",
        )
        already_importable = self._is_lerobot_importable(env)
        if not skip_pip and already_importable and not upgrade:
            skip_pip = True

        if skip_pip:
            reason = (
                "ROSCLAW_LEROBOT_SKIP_PIP_INSTALL is set"
                if env.get("ROSCLAW_LEROBOT_SKIP_PIP_INSTALL", "").lower()
                in ("1", "true", "yes")
                else "LeRobot is already importable and --upgrade was not set"
            )
            details["pip_install"] = {"skipped": True, "reason": reason}
        else:
            # Install packages.
            install_cmd = [self.pip_executable, "install"]
            if upgrade:
                install_cmd.append("--upgrade")
            install_cmd.extend(profile.pip)
            pip_result = run_command(install_cmd, env=env, timeout=600.0)
            details["pip_install"] = {
                "ok": pip_result.ok,
                "returncode": pip_result.returncode,
                "stderr": pip_result.stderr,
            }

            if not pip_result.ok:
                return InstallReport(
                    ok=False,
                    profile=profile_name,
                    dry_run=False,
                    message=f"pip install failed: {pip_result.stderr}",
                    python_executable=self.python_executable,
                    pip_executable=self.pip_executable,
                    details=details,
                )

        # Run post-install checks.
        check_results: list[CommandResult] = []
        for check in profile.checks:
            check_path = which(check)
            if check_path:
                result = run_command([check_path], env=env, timeout=60.0)
            else:
                result = run_command([self.python_executable, "-m", check], env=env, timeout=60.0)
            check_results.append(result)
            if not result.ok:
                return InstallReport(
                    ok=False,
                    profile=profile_name,
                    dry_run=False,
                    message=f"Post-install check '{check}' failed: {result.stderr}",
                    python_executable=self.python_executable,
                    pip_executable=self.pip_executable,
                    details=details,
                )

        details["checks"] = [
            {"command": r.command, "ok": r.ok, "returncode": r.returncode}
            for r in check_results
        ]

        # Write integration config.
        self.config_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.config_dir / "lerobot.yaml"
        config = {
            "enabled": True,
            "profile": profile_name,
            "python": self.python_executable,
            "pip": self.pip_executable,
            "installed_at": None,
            "capabilities": profile.enabled_capabilities,
        }
        try:
            from datetime import datetime

            config["installed_at"] = datetime.now(UTC).isoformat()
        except Exception:
            pass
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        # Probe LeRobot version.
        lerobot_version = self._probe_lerobot_version(env)

        return InstallReport(
            ok=True,
            profile=profile_name,
            dry_run=False,
            message=f"LeRobot profile '{profile_name}' installed successfully.",
            lerobot_version=lerobot_version,
            python_executable=self.python_executable,
            pip_executable=self.pip_executable,
            details=details,
        )

    def _is_lerobot_importable(self, env: dict[str, str]) -> bool:
        """Check whether ``lerobot`` can be imported without importing it directly."""
        result = run_command(
            [
                self.python_executable,
                "-c",
                "import importlib.util; import sys; "
                "spec = importlib.util.find_spec('lerobot'); "
                "sys.exit(0 if spec is not None else 1)",
            ],
            env=env,
            timeout=30.0,
        )
        return result.ok

    def _probe_lerobot_version(self, env: dict[str, str]) -> str | None:
        """Probe the installed LeRobot version without importing it directly."""
        result = run_command(
            [
                self.python_executable,
                "-c",
                "import importlib; m = importlib.import_module('lerobot'); print(getattr(m, '__version__', 'unknown'))",
            ],
            env=env,
            timeout=30.0,
        )
        return result.stdout if result.ok else None


def install_lerobot(
    profile: str = "core",
    *,
    dry_run: bool = False,
    upgrade: bool = False,
    env: dict[str, str] | None = None,
) -> InstallReport:
    """Convenience entry point used by the CLI."""
    installer = LeRobotInstaller()
    return installer.install(profile, dry_run=dry_run, upgrade=upgrade, env=env)
