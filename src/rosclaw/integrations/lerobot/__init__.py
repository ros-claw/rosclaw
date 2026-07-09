"""LeRobot integration for ROSClaw."""

from __future__ import annotations

from rosclaw.integrations.lerobot.capabilities import (
    LeRobotIntegration,
    register_lerobot_capabilities,
)
from rosclaw.integrations.lerobot.config import (
    build_lerobot_config,
    get_configured_lerobot_runtime,
    get_default_runtime_path,
    get_lerobot_config_path,
    load_lerobot_config,
    migrate_v0_config_to_v1,
    save_lerobot_config,
)
from rosclaw.integrations.lerobot.doctor import LeRobotDoctor
from rosclaw.integrations.lerobot.env_manager import LeRobotEnvManager
from rosclaw.integrations.lerobot.installer import LeRobotInstaller
from rosclaw.integrations.lerobot.runtime import (
    LeRobotRuntime,
    PythonRuntimeInfo,
    current_rosclaw_runtime,
    find_python312,
    inspect_lerobot_runtime,
    inspect_python,
    resolve_lerobot_info,
)
from rosclaw.integrations.lerobot.schemas import (
    InstallReport,
    LeRobotDoctorReport,
    LeRobotSetupErrorCode,
    ProfileSpec,
)

__all__ = [
    "LeRobotIntegration",
    "register_lerobot_capabilities",
    "LeRobotDoctor",
    "LeRobotInstaller",
    "LeRobotEnvManager",
    "InstallReport",
    "LeRobotDoctorReport",
    "LeRobotSetupErrorCode",
    "ProfileSpec",
    "LeRobotRuntime",
    "PythonRuntimeInfo",
    "current_rosclaw_runtime",
    "find_python312",
    "inspect_lerobot_runtime",
    "inspect_python",
    "resolve_lerobot_info",
    "build_lerobot_config",
    "get_configured_lerobot_runtime",
    "get_default_runtime_path",
    "get_lerobot_config_path",
    "load_lerobot_config",
    "migrate_v0_config_to_v1",
    "save_lerobot_config",
]
