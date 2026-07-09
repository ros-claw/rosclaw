"""LeRobot integration schemas and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.integrations.registry import IntegrationReport


@dataclass
class ProfileSpec:
    """A LeRobot installation profile."""

    name: str
    pip: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    enabled_capabilities: list[str] = field(default_factory=list)


@dataclass
class InstallReport:
    """Result of a LeRobot installation attempt."""

    ok: bool
    profile: str
    dry_run: bool
    message: str
    lerobot_version: str | None = None
    python_executable: str | None = None
    pip_executable: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeRobotDoctorReport(IntegrationReport):
    """Extended doctor report for the LeRobot integration."""

    python_version: str | None = None
    python_executable: str | None = None
    lerobot_importable: bool = False
    lerobot_version: str | None = None
    lerobot_info_path: str | None = None
    lerobot_info_ok: bool = False
    lerobot_info_output: str = ""
    torch_available: bool = False
    torch_version: str | None = None
    cuda_available: bool | None = None
    hf_endpoint: str | None = None
    hf_cache_dir: str | None = None
    config_path: Path | None = None
    config_enabled: bool = False
    provider_type_registered: bool = False
    exporter_registered: bool = False
