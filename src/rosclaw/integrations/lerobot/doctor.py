"""LeRobot integration doctor."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys

import yaml

from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.integrations.lerobot.profiles import list_profile_names
from rosclaw.integrations.lerobot.schemas import LeRobotDoctorReport
from rosclaw.integrations.lerobot.subprocess_runner import run_command, which
from rosclaw.integrations.registry import IntegrationCapability


class LeRobotDoctor:
    """Diagnose the LeRobot integration environment."""

    def __init__(self, python_executable: str | None = None):
        self.python_executable = python_executable or sys.executable

    def run(
        self,
        *,
        registry_check: dict[str, bool] | None = None,
    ) -> LeRobotDoctorReport:
        """Return a diagnostic report."""
        registry_check = registry_check or {}

        python_version = platform.python_version()
        python_executable = self.python_executable

        # LeRobot import probe.
        lerobot_spec = importlib.util.find_spec("lerobot")
        lerobot_importable = lerobot_spec is not None
        lerobot_version: str | None = None
        if lerobot_importable:
            try:
                import lerobot

                lerobot_version = getattr(lerobot, "__version__", None)
            except Exception:
                lerobot_importable = False

        # torch probe.
        torch_available = importlib.util.find_spec("torch") is not None
        torch_version: str | None = None
        cuda_available: bool | None = None
        if torch_available:
            try:
                import torch

                torch_version = getattr(torch, "__version__", None)
                cuda_available = torch.cuda.is_available()
            except Exception:
                torch_available = False

        # lerobot-info probe.
        lerobot_info_path = which("lerobot-info")
        lerobot_info_output = ""
        lerobot_info_ok = False
        if lerobot_info_path:
            result = run_command([lerobot_info_path], timeout=60.0)
            lerobot_info_output = result.stdout
            lerobot_info_ok = result.ok

        # HuggingFace environment.
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        hf_cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")

        # Integration config.
        config_path = get_rosclaw_home() / "integrations" / "lerobot.yaml"
        config_enabled = False
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                config_enabled = bool(cfg.get("enabled"))
            except Exception:
                pass

        # Determine status.
        if lerobot_importable and config_enabled:
            status = "installed"
        elif lerobot_importable or config_enabled:
            status = "degraded"
        else:
            status = "not_installed"

        capabilities = [
            IntegrationCapability(
                name="provider_type_lerobot_policy",
                kind="provider",
                enabled=registry_check.get("provider_type_lerobot_policy", False),
                experimental=True,
                description="LeRobot policy provider dry-run support",
            ),
            IntegrationCapability(
                name="dataset_export_lerobot",
                kind="exporter",
                enabled=registry_check.get("dataset_export_lerobot", False),
                experimental=True,
                description="LeRobot dataset skeleton export",
            ),
        ]

        message = self._build_message(status, lerobot_importable, config_enabled)

        return LeRobotDoctorReport(
            name="lerobot",
            status=status,
            version=lerobot_version,
            capabilities=capabilities,
            message=message,
            python_version=python_version,
            python_executable=python_executable,
            lerobot_importable=lerobot_importable,
            lerobot_version=lerobot_version,
            lerobot_info_path=lerobot_info_path,
            lerobot_info_ok=lerobot_info_ok,
            lerobot_info_output=lerobot_info_output,
            torch_available=torch_available,
            torch_version=torch_version,
            cuda_available=cuda_available,
            hf_endpoint=hf_endpoint,
            hf_cache_dir=hf_cache_dir,
            config_path=config_path,
            config_enabled=config_enabled,
            provider_type_registered=registry_check.get("provider_type_lerobot_policy", False),
            exporter_registered=registry_check.get("dataset_export_lerobot", False),
        )

    @staticmethod
    def _build_message(
        status: str,
        lerobot_importable: bool,
        config_enabled: bool,
    ) -> str:
        if status == "installed":
            return "LeRobot is installed and enabled."
        if lerobot_importable and not config_enabled:
            return (
                "LeRobot is importable but not enabled in ROSClaw. "
                "Run `rosclaw setup lerobot --profile core` to enable it."
            )
        if config_enabled and not lerobot_importable:
            return (
                "LeRobot is enabled in ROSClaw config but the Python package is not importable. "
                "Re-run `rosclaw setup lerobot --profile core` or check your environment."
            )
        profiles = ", ".join(list_profile_names()) or "core, dataset, train"
        return (
            "LeRobot is not installed. "
            f"Run `rosclaw setup lerobot --profile <{profiles}>` to install it."
        )


def run_lerobot_doctor(registry_check: dict[str, bool] | None = None) -> LeRobotDoctorReport:
    """Convenience entry point used by the CLI."""
    return LeRobotDoctor().run(registry_check=registry_check)
