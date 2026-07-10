"""LeRobot integration doctor with dual-runtime awareness."""

from __future__ import annotations

import os
import sys
from typing import Any

from rosclaw.integrations.lerobot.config import (
    get_lerobot_config_path,
    load_lerobot_config,
)
from rosclaw.integrations.lerobot.profiles import list_profile_names
from rosclaw.integrations.lerobot.runtime import (
    LeRobotRuntime,
    current_rosclaw_runtime,
    inspect_lerobot_runtime,
)
from rosclaw.integrations.lerobot.schemas import LeRobotDoctorReport
from rosclaw.integrations.lerobot.smoke_report import (
    get_validation_status,
    read_latest_smoke_report,
)
from rosclaw.integrations.lerobot.subprocess_runner import run_command, which
from rosclaw.integrations.registry import IntegrationCapability


def _lerobot_capabilities(
    registry_check: dict[str, bool],
    subprocess_available: bool,
    in_process_available: bool,
) -> list[IntegrationCapability]:
    """Build the P1 capability list for the doctor report."""
    from rosclaw.integrations.lerobot.capabilities import get_lerobot_capabilities

    caps = get_lerobot_capabilities()
    overrides = {
        "provider_type_lerobot_policy": registry_check.get("provider_type_lerobot_policy", False),
        "dataset_export_lerobot": registry_check.get("dataset_export_lerobot", False),
        "worker_subprocess": subprocess_available,
        "worker_in_process": in_process_available,
    }
    for cap in caps:
        if cap.name in overrides:
            cap.enabled = overrides[cap.name]
    return caps


class LeRobotDoctor:
    """Diagnose the LeRobot integration environment."""

    def run(
        self,
        *,
        registry_check: dict[str, bool] | None = None,
    ) -> LeRobotDoctorReport:
        """Return a diagnostic report distinguishing ROSClaw and LeRobot runtimes."""
        registry_check = registry_check or {}

        rosclaw_runtime = current_rosclaw_runtime()
        rosclaw_python_version = rosclaw_runtime.version
        rosclaw_python_executable = str(rosclaw_runtime.executable)

        # In-process LeRobot availability in the current interpreter.
        in_process_importable = self._is_lerobot_importable_current()

        # Load stored config (if any) and inspect configured LeRobot runtime.
        config = load_lerobot_config()
        config_path = get_lerobot_config_path()
        config_enabled = bool(config.get("enabled"))
        lerobot_runtime: LeRobotRuntime | None = None
        worker_subprocess_available = False
        worker_in_process_available = in_process_importable

        if config_enabled and config.get("lerobot_runtime"):
            runtime_cfg = config["lerobot_runtime"]
            python_exe = runtime_cfg.get("python_executable")
            if python_exe:
                mode = runtime_cfg.get("mode", "external")
                runtime_path = runtime_cfg.get("runtime_path")
                lerobot_runtime = inspect_lerobot_runtime(
                    python_exe,
                    mode=mode,
                    runtime_path=runtime_path,
                )
                worker_subprocess_available = lerobot_runtime.subprocess_available

        # Fallback: if no config but LeRobot is importable in-process, describe it.
        if lerobot_runtime is None and in_process_importable:
            lerobot_runtime = inspect_lerobot_runtime(sys.executable, mode="current-env")
            worker_subprocess_available = lerobot_runtime.subprocess_available

        # Fallback PATH probe for lerobot-info when nothing else is configured.
        info_path_fallback: str | None = None
        info_output_fallback = ""
        info_ok_fallback = False
        if lerobot_runtime is None:
            info_path_fallback = which("lerobot-info")
            if info_path_fallback:
                result = run_command([info_path_fallback], timeout=60.0)
                info_output_fallback = result.stdout
                info_ok_fallback = result.ok

        # Determine status.
        if lerobot_runtime is None:
            status = "not_installed"
        elif lerobot_runtime.state == "ready":
            status = "installed"
        elif lerobot_runtime.state in ("degraded", "error"):
            status = "degraded"
        else:
            status = "not_installed"

        if not config_enabled and lerobot_runtime is not None:
            status = "degraded"

        capabilities = _lerobot_capabilities(
            registry_check,
            worker_subprocess_available,
            worker_in_process_available,
        )

        current_lerobot_version = lerobot_runtime.lerobot_version if lerobot_runtime else None
        current_python_executable = (
            str(lerobot_runtime.python_executable) if lerobot_runtime else None
        )
        latest_report = read_latest_smoke_report()
        validation_status = get_validation_status(
            report=latest_report,
            current_lerobot_version=current_lerobot_version,
            current_python_executable=current_python_executable,
        )

        message = self._build_message(
            status,
            rosclaw_runtime,
            lerobot_runtime,
            config_enabled,
        )

        # Preserve legacy fields for consumers that expect them.
        legacy_python_version = rosclaw_python_version
        legacy_python_executable = rosclaw_python_executable
        legacy_lerobot_importable = in_process_importable
        legacy_lerobot_version = lerobot_runtime.lerobot_version if lerobot_runtime else None
        legacy_info_path = (
            str(lerobot_runtime.lerobot_info_executable)
            if lerobot_runtime and lerobot_runtime.lerobot_info_executable
            else info_path_fallback
        )
        legacy_info_ok = (
            lerobot_runtime.subprocess_available
            if lerobot_runtime
            else info_ok_fallback
        )
        legacy_info_output = (
            lerobot_runtime.lerobot_info_output
            if lerobot_runtime
            else info_output_fallback
        )
        legacy_torch = lerobot_runtime.torch_version if lerobot_runtime else None
        legacy_cuda = lerobot_runtime.cuda_available if lerobot_runtime else None

        return LeRobotDoctorReport(
            name="lerobot",
            status=status,
            version=legacy_lerobot_version,
            capabilities=capabilities,
            message=message,
            python_version=legacy_python_version,
            python_executable=legacy_python_executable,
            lerobot_importable=legacy_lerobot_importable,
            lerobot_version=legacy_lerobot_version,
            lerobot_info_path=legacy_info_path,
            lerobot_info_ok=legacy_info_ok,
            lerobot_info_output=legacy_info_output or "",
            torch_available=bool(legacy_torch),
            torch_version=legacy_torch,
            cuda_available=legacy_cuda,
            hf_endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            hf_cache_dir=os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE"),
            config_path=config_path,
            config_enabled=config_enabled,
            provider_type_registered=registry_check.get("provider_type_lerobot_policy", False),
            exporter_registered=registry_check.get("dataset_export_lerobot", False),
            rosclaw_python_version=rosclaw_python_version,
            rosclaw_python_executable=rosclaw_python_executable,
            lerobot_runtime=lerobot_runtime,
            worker_subprocess_available=worker_subprocess_available,
            worker_in_process_available=worker_in_process_available,
            status_detail=self._status_detail(status, lerobot_runtime),
            validation_status=validation_status,
        )

    @staticmethod
    def _is_lerobot_importable_current() -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("lerobot") is not None
        except Exception:
            return False

    @staticmethod
    def _build_message(
        status: str,
        rosclaw_runtime: Any,
        lerobot_runtime: LeRobotRuntime | None,
        config_enabled: bool,
    ) -> str:
        if status == "installed" and lerobot_runtime is not None:
            return (
                f"LeRobot runtime is ready ({lerobot_runtime.mode}).\n"
                f"ROSClaw Python: {rosclaw_runtime.version}\n"
                f"LeRobot Python: {lerobot_runtime.python_version}"
            )

        if lerobot_runtime is not None and lerobot_runtime.state == "degraded":
            return (
                f"LeRobot runtime is degraded ({lerobot_runtime.mode}).\n"
                f"Reason: {lerobot_runtime.error or 'unknown'}\n"
                "Run `rosclaw setup lerobot --profile core` to repair."
            )

        if config_enabled and lerobot_runtime is None:
            return (
                "LeRobot is enabled in ROSClaw config but no runtime information is available.\n"
                "Re-run `rosclaw setup lerobot --profile core` or check your environment."
            )

        profiles = ", ".join(list_profile_names()) or "core"
        message = (
            f"LeRobot is not configured.\n"
            f"Current ROSClaw Python: {rosclaw_runtime.version}\n"
            f"LeRobot requires Python >= 3.12.\n\n"
            "Run one of:\n"
            f"  rosclaw setup lerobot --profile <{profiles}>\n"
            "  rosclaw setup lerobot --profile core --mode isolated\n"
            "  rosclaw setup lerobot --profile core --mode external --python /path/to/python3.12"
        )
        return message

    @staticmethod
    def _status_detail(status: str, lerobot_runtime: LeRobotRuntime | None) -> str | None:
        if status == "installed":
            return "LeRobot runtime ready"
        if lerobot_runtime is None:
            return "No LeRobot runtime configured"
        return lerobot_runtime.error or "LeRobot runtime not ready"


def run_lerobot_doctor(registry_check: dict[str, bool] | None = None) -> LeRobotDoctorReport:
    """Convenience entry point used by the CLI."""
    return LeRobotDoctor().run(registry_check=registry_check)
