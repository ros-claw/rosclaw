"""LeRobot integration installer with runtime-aware install modes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.config import (
    build_lerobot_config,
    get_default_runtime_path,
    save_lerobot_config,
)
from rosclaw.integrations.lerobot.env_manager import LeRobotEnvManager
from rosclaw.integrations.lerobot.profiles import load_profile
from rosclaw.integrations.lerobot.runtime import (
    LeRobotRuntime,
    RuntimeMode,
    current_rosclaw_runtime,
    find_python312,
    inspect_lerobot_runtime,
)
from rosclaw.integrations.lerobot.schemas import InstallReport, LeRobotSetupErrorCode
from rosclaw.integrations.lerobot.subprocess_runner import run_command, which


class LeRobotInstaller:
    """Install, register, or dry-run the LeRobot integration."""

    def __init__(
        self,
        config_dir: Path | None = None,
    ):
        self.config_dir = config_dir
        self.env_manager = LeRobotEnvManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def install(
        self,
        profile_name: str = "core",
        *,
        mode: str = "auto",
        python: Path | str | None = None,
        runtime_path: Path | str | None = None,
        upgrade: bool = False,
        force: bool = False,
        dry_run: bool = False,
        index_url: str | None = None,
        extra_index_url: str | None = None,
        env: dict[str, str] | None = None,
    ) -> InstallReport:
        """Install or register LeRobot for the requested profile and mode."""
        try:
            profile = load_profile(profile_name)
        except FileNotFoundError as exc:
            return InstallReport(
                ok=False,
                profile=profile_name,
                dry_run=dry_run,
                message=f"Profile '{profile_name}' not found: {exc}",
                error_code=None,
                mode=mode,
            )

        merged_env = self._prepare_env(env)
        current = current_rosclaw_runtime()
        resolved_mode = self._resolve_mode(mode, current)

        details: dict[str, Any] = {
            "profile": profile_name,
            "mode": resolved_mode,
            "rosclaw_python": str(current.executable),
            "rosclaw_python_version": current.version,
        }

        if dry_run:
            plan = self._build_plan(
                resolved_mode,
                profile,
                current,
                python=python,
                runtime_path=runtime_path,
            )
            details["plan"] = plan
            return InstallReport(
                ok=True,
                profile=profile_name,
                dry_run=True,
                message=f"Dry-run: resolved mode='{resolved_mode}'. Planned steps:\n" + "\n".join(f"  - {step}" for step in plan),
                mode=resolved_mode,
                details=details,
            )

        if resolved_mode == "current-env":
            return self._install_current_env(
                profile_name,
                profile,
                current,
                upgrade=upgrade,
                index_url=index_url,
                extra_index_url=extra_index_url,
                env=merged_env,
            )

        if resolved_mode == "isolated":
            return self._install_isolated(
                profile_name,
                profile,
                current,
                python=python,
                runtime_path=runtime_path,
                upgrade=upgrade,
                force=force,
                index_url=index_url,
                extra_index_url=extra_index_url,
                env=merged_env,
            )

        if resolved_mode == "external":
            return self._register_external(
                profile_name,
                profile,
                current,
                python=python,
                env=merged_env,
            )

        return InstallReport(
            ok=False,
            profile=profile_name,
            dry_run=False,
            message=f"Unknown install mode: {resolved_mode}",
            error_code=None,
            mode=resolved_mode,
            details=details,
        )

    # ------------------------------------------------------------------
    # Mode resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_mode(mode: str, current: Any) -> RuntimeMode:
        if mode != "auto":
            return mode  # type: ignore[return-value]
        if current.major == 3 and current.minor is not None and current.minor >= 12:
            return "current-env"
        return "isolated"

    # ------------------------------------------------------------------
    # current-env
    # ------------------------------------------------------------------
    def _install_current_env(
        self,
        profile_name: str,
        profile: Any,
        current: Any,
        *,
        upgrade: bool,
        index_url: str | None,
        extra_index_url: str | None,
        env: dict[str, str],
    ) -> InstallReport:
        if current.major != 3 or (current.minor is not None and current.minor < 12):
            message = (
                f"ERROR: current-env mode requires Python >= 3.12.\n"
                f"Current Python: {current.version or f'{current.major}.{current.minor}'}\n\n"
                "Use one of:\n"
                "  rosclaw setup lerobot --profile core --mode isolated\n"
                "  rosclaw setup lerobot --profile core --mode external --python /path/to/python3.12"
            )
            return InstallReport(
                ok=False,
                profile=profile_name,
                dry_run=False,
                message=message,
                error_code=LeRobotSetupErrorCode.PYTHON_TOO_OLD,
                mode="current-env",
                python_executable=str(current.executable),
            )

        skip_pip = env.get("ROSCLAW_LEROBOT_SKIP_PIP_INSTALL", "").lower() in (
            "1",
            "true",
            "yes",
        )
        already_importable = self._is_lerobot_importable(current.executable, env)
        if not skip_pip and already_importable and not upgrade:
            skip_pip = True

        details: dict[str, Any] = {
            "profile": profile_name,
            "mode": "current-env",
            "python": str(current.executable),
        }

        if skip_pip:
            reason = (
                "ROSCLAW_LEROBOT_SKIP_PIP_INSTALL is set"
                if env.get("ROSCLAW_LEROBOT_SKIP_PIP_INSTALL", "").lower()
                in ("1", "true", "yes")
                else "LeRobot is already importable and --upgrade was not set"
            )
            details["pip_install"] = {"skipped": True, "reason": reason}
        else:
            pip_cmd = [str(current.executable), "-m", "pip", "install"]
            if upgrade:
                pip_cmd.append("--upgrade")
            if index_url:
                pip_cmd.extend(["--index-url", index_url])
            if extra_index_url:
                pip_cmd.extend(["--extra-index-url", extra_index_url])
            pip_cmd.extend(profile.pip)

            pip_result = run_command(pip_cmd, env=env, timeout=600.0)
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
                    error_code=LeRobotSetupErrorCode.PIP_INSTALL_FAILED,
                    mode="current-env",
                    python_executable=str(current.executable),
                    details=details,
                )

        runtime = inspect_lerobot_runtime(
            current.executable,
            mode="current-env",
        )
        if runtime.state == "error":
            error_code = (
                LeRobotSetupErrorCode.LEROBOT_VERSION_UNSUPPORTED
                if "Unsupported LeRobot version" in (runtime.error or "")
                else LeRobotSetupErrorCode.LEROBOT_IMPORT_FAILED
            )
            return InstallReport(
                ok=False,
                profile=profile_name,
                dry_run=False,
                message=runtime.error or "LeRobot runtime inspection failed",
                error_code=error_code,
                mode="current-env",
                runtime=runtime,
                details=details,
            )

        report = self._run_post_install_checks(profile, runtime, env, details)
        if report is not None:
            return report

        config_error = self._write_config(profile_name, "current-env", runtime, details)
        if config_error:
            return self._config_write_failure(
                profile_name, "current-env", runtime, details, config_error
            )

        return InstallReport(
            ok=True,
            profile=profile_name,
            dry_run=False,
            message=f"LeRobot profile '{profile_name}' installed in current environment.",
            mode="current-env",
            runtime=runtime,
            lerobot_version=runtime.lerobot_version,
            python_executable=str(runtime.python_executable),
            pip_executable=str(runtime.pip_executable) if runtime.pip_executable else None,
            details=details,
        )

    # ------------------------------------------------------------------
    # isolated
    # ------------------------------------------------------------------
    def _install_isolated(
        self,
        profile_name: str,
        profile: Any,
        current: Any,
        *,
        python: Path | str | None,
        runtime_path: Path | str | None,
        upgrade: bool,
        force: bool,
        index_url: str | None,
        extra_index_url: str | None,
        env: dict[str, str],
    ) -> InstallReport:
        target_path = Path(runtime_path) if runtime_path else get_default_runtime_path()
        python312 = find_python312(python)

        if python312 is None:
            message = (
                "ERROR: Cannot create isolated LeRobot runtime because python3.12 was not found.\n\n"
                "Please install Python 3.12 or provide an existing runtime:\n\n"
                "  rosclaw setup lerobot --profile core --mode external --python /path/to/python3.12"
            )
            return InstallReport(
                ok=False,
                profile=profile_name,
                dry_run=False,
                message=message,
                error_code=LeRobotSetupErrorCode.PYTHON312_NOT_FOUND,
                mode="isolated",
                details={"runtime_path": str(target_path)},
            )

        details: dict[str, Any] = {
            "profile": profile_name,
            "mode": "isolated",
            "python312": str(python312),
            "runtime_path": str(target_path),
        }

        try:
            runtime = self.env_manager.create_isolated_env(
                target_path,
                python_executable=python312,
                force=force,
            )
        except RuntimeError as exc:
            return InstallReport(
                ok=False,
                profile=profile_name,
                dry_run=False,
                message=str(exc),
                error_code=LeRobotSetupErrorCode.VENV_CREATE_FAILED,
                mode="isolated",
                details=details,
            )

        if runtime.state in ("ready", "degraded") and runtime.lerobot_version is not None:
            if not upgrade:
                details["pip_install"] = {"skipped": True, "reason": "LeRobot already importable and --upgrade not set"}
        else:
            install_report = self.env_manager.install_lerobot(
                runtime,
                profile=profile_name,
                upgrade=upgrade,
                index_url=index_url,
                extra_index_url=extra_index_url,
                packages=profile.pip,
            )
            details["pip_install"] = install_report.details.get("pip_install", install_report.details)
            if not install_report.ok:
                return InstallReport(
                    ok=False,
                    profile=profile_name,
                    dry_run=False,
                    message=install_report.message,
                    error_code=LeRobotSetupErrorCode.PIP_INSTALL_FAILED,
                    mode="isolated",
                    runtime=runtime,
                    details=details,
                )
            runtime = install_report.runtime or runtime

        report = self._run_post_install_checks(profile, runtime, env, details)
        if report is not None:
            return report

        config_error = self._write_config(profile_name, "isolated", runtime, details)
        if config_error:
            return self._config_write_failure(
                profile_name, "isolated", runtime, details, config_error
            )

        return InstallReport(
            ok=True,
            profile=profile_name,
            dry_run=False,
            message=f"LeRobot isolated runtime created at {target_path}.",
            mode="isolated",
            runtime=runtime,
            lerobot_version=runtime.lerobot_version,
            python_executable=str(runtime.python_executable),
            pip_executable=str(runtime.pip_executable) if runtime.pip_executable else None,
            details=details,
        )

    # ------------------------------------------------------------------
    # external
    # ------------------------------------------------------------------
    def _register_external(
        self,
        profile_name: str,
        profile: Any,
        current: Any,
        *,
        python: Path | str | None,
        env: dict[str, str],
    ) -> InstallReport:
        if python is None:
            return InstallReport(
                ok=False,
                profile=profile_name,
                dry_run=False,
                message="external mode requires --python /path/to/python3.12",
                error_code=LeRobotSetupErrorCode.EXTERNAL_PYTHON_NOT_FOUND,
                mode="external",
            )

        python_path = Path(python)
        if not python_path.exists():
            return InstallReport(
                ok=False,
                profile=profile_name,
                dry_run=False,
                message=f"External Python executable not found: {python_path}",
                error_code=LeRobotSetupErrorCode.EXTERNAL_PYTHON_NOT_FOUND,
                mode="external",
            )

        runtime = inspect_lerobot_runtime(python_path, mode="external")
        if runtime.state == "error":
            error_code = (
                LeRobotSetupErrorCode.EXTERNAL_PYTHON_TOO_OLD
                if "requires Python" in (runtime.error or "")
                else (
                    LeRobotSetupErrorCode.LEROBOT_VERSION_UNSUPPORTED
                    if "Unsupported LeRobot version" in (runtime.error or "")
                    else LeRobotSetupErrorCode.LEROBOT_IMPORT_FAILED
                )
            )
            return InstallReport(
                ok=False,
                profile=profile_name,
                dry_run=False,
                message=runtime.error or "External runtime inspection failed",
                error_code=error_code,
                mode="external",
                runtime=runtime,
            )

        details: dict[str, Any] = {
            "profile": profile_name,
            "mode": "external",
            "python": str(runtime.python_executable),
        }

        report = self._run_post_install_checks(profile, runtime, env, details)
        if report is not None:
            return report

        config_error = self._write_config(profile_name, "external", runtime, details)
        if config_error:
            return self._config_write_failure(
                profile_name, "external", runtime, details, config_error
            )

        return InstallReport(
            ok=True,
            profile=profile_name,
            dry_run=False,
            message=f"LeRobot external runtime registered: {runtime.python_executable}",
            mode="external",
            runtime=runtime,
            lerobot_version=runtime.lerobot_version,
            python_executable=str(runtime.python_executable),
            pip_executable=str(runtime.pip_executable) if runtime.pip_executable else None,
            details=details,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_env(self, env: dict[str, str] | None) -> dict[str, str]:
        return {**os.environ, **(env or {})}

    def _build_plan(
        self,
        mode: RuntimeMode,
        profile: Any,
        current: Any,
        *,
        python: Path | str | None,
        runtime_path: Path | str | None,
    ) -> list[str]:
        plan: list[str] = []
        if mode == "current-env":
            plan.append(f"Verify current Python >= 3.12 ({current.version})")
            plan.append(f"pip install {' '.join(profile.pip)}")
            plan.append("Run post-install checks: " + ", ".join(profile.checks))
            plan.append("Write config with install_mode=current-env")
        elif mode == "isolated":
            target = runtime_path or get_default_runtime_path()
            plan.append(f"Find Python 3.12 executable (preferred: {python or 'auto'})")
            plan.append(f"Create isolated venv at {target}")
            plan.append("Upgrade pip in isolated runtime")
            plan.append(f"pip install {' '.join(profile.pip)} in isolated runtime")
            plan.append("Run lerobot-info smoke test")
            plan.append("Write config with install_mode=isolated")
        elif mode == "external":
            plan.append(f"Inspect external Python: {python or 'MISSING'}")
            plan.append("Verify Python >= 3.12 and import lerobot")
            plan.append("Run lerobot-info smoke test")
            plan.append("Write config with install_mode=external")
        return plan

    def _is_lerobot_importable(self, python_executable: Path | str, env: dict[str, str]) -> bool:
        result = run_command(
            [
                str(python_executable),
                "-c",
                "from importlib.metadata import version; import sys; "
                "parts = tuple(int(p) for p in version('lerobot').split('.')[:2]); "
                "sys.exit(0 if parts == (0, 6) else 1)",
            ],
            env=env,
            timeout=30.0,
        )
        return result.ok

    def _run_post_install_checks(
        self,
        profile: Any,
        runtime: LeRobotRuntime,
        env: dict[str, str],
        details: dict[str, Any],
    ) -> InstallReport | None:
        check_results: list[dict[str, Any]] = []
        for check in profile.checks:
            if check == "lerobot-info":
                info_exe = runtime.lerobot_info_executable
                if info_exe is not None:
                    if info_exe == runtime.python_executable:
                        result = run_command(
                            [
                                str(runtime.python_executable),
                                "-m",
                                "lerobot.scripts.lerobot_info",
                            ],
                            env=env,
                            timeout=60.0,
                        )
                    else:
                        result = run_command([str(info_exe)], env=env, timeout=60.0)
                else:
                    result = run_command(
                        ["lerobot-info"],
                        env=env,
                        timeout=60.0,
                    )
            else:
                check_path = which(check)
                if check_path:
                    result = run_command([check_path], env=env, timeout=60.0)
                else:
                    result = run_command(
                        [str(runtime.python_executable), "-m", check],
                        env=env,
                        timeout=60.0,
                    )

            check_results.append(
                {
                    "command": result.command,
                    "ok": result.ok,
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                }
            )
            if not result.ok:
                details["checks"] = check_results
                return InstallReport(
                    ok=False,
                    profile=profile.name,
                    dry_run=False,
                    message=f"Post-install check '{check}' failed: {result.stderr}",
                    error_code=LeRobotSetupErrorCode.LEROBOT_INFO_FAILED,
                    mode=runtime.mode,
                    runtime=runtime,
                    details=details,
                )

        details["checks"] = check_results
        return None

    def _write_config(
        self,
        profile_name: str,
        mode: RuntimeMode,
        runtime: LeRobotRuntime,
        details: dict[str, Any],
    ) -> str | None:
        current = current_rosclaw_runtime()
        config = build_lerobot_config(
            profile=profile_name,
            mode=mode,
            runtime=runtime,
            rosclaw_python=str(current.executable),
            rosclaw_version=current.version or "",
        )
        try:
            save_lerobot_config(config)
        except Exception as exc:
            details["config_write_error"] = str(exc)
            return str(exc)
        return None

    @staticmethod
    def _config_write_failure(
        profile_name: str,
        mode: RuntimeMode,
        runtime: LeRobotRuntime,
        details: dict[str, Any],
        error: str,
    ) -> InstallReport:
        return InstallReport(
            ok=False,
            profile=profile_name,
            dry_run=False,
            message=f"LeRobot installed but config write failed: {error}",
            error_code=LeRobotSetupErrorCode.CONFIG_WRITE_FAILED,
            mode=mode,
            runtime=runtime,
            details=details,
        )


def install_lerobot(
    profile: str = "core",
    *,
    mode: str = "auto",
    python: Path | str | None = None,
    runtime_path: Path | str | None = None,
    upgrade: bool = False,
    force: bool = False,
    dry_run: bool = False,
    index_url: str | None = None,
    extra_index_url: str | None = None,
    env: dict[str, str] | None = None,
) -> InstallReport:
    """Convenience entry point used by the CLI."""
    installer = LeRobotInstaller()
    return installer.install(
        profile,
        mode=mode,
        python=python,
        runtime_path=runtime_path,
        upgrade=upgrade,
        force=force,
        dry_run=dry_run,
        index_url=index_url,
        extra_index_url=extra_index_url,
        env=env,
    )
