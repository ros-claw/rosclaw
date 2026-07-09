"""LeRobot integration for ROSClaw."""

from __future__ import annotations

from rosclaw.integrations.lerobot.capabilities import (
    LeRobotIntegration,
    register_lerobot_capabilities,
)
from rosclaw.integrations.lerobot.doctor import LeRobotDoctor
from rosclaw.integrations.lerobot.installer import LeRobotInstaller
from rosclaw.integrations.lerobot.schemas import (
    InstallReport,
    LeRobotDoctorReport,
    ProfileSpec,
)

__all__ = [
    "LeRobotIntegration",
    "register_lerobot_capabilities",
    "LeRobotDoctor",
    "LeRobotInstaller",
    "InstallReport",
    "LeRobotDoctorReport",
    "ProfileSpec",
]
