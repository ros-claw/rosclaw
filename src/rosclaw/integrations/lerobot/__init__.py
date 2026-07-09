"""LeRobot integration for ROSClaw."""

from __future__ import annotations

from rosclaw.integrations.lerobot.action_adapter import adapt_action_to_proposal
from rosclaw.integrations.lerobot.capabilities import (
    LeRobotIntegration,
    register_lerobot_capabilities,
)
from rosclaw.integrations.lerobot.compatibility import (
    POLICY_COMPATIBILITY_MATRIX,
    PolicyCompatibility,
    build_compatibility_report,
    classify_compatibility_level,
    get_policy_compatibility,
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
from rosclaw.integrations.lerobot.observation_adapter import adapt_observation_for_worker
from rosclaw.integrations.lerobot.policy_cache import (
    MaterializationResult,
    PolicyMaterializationError,
    get_policy_cache_dir,
    materialize_policy_path,
)
from rosclaw.integrations.lerobot.policy_manifest import load_policy_manifest
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
from rosclaw.integrations.lerobot.smoke_policy import (
    DEFAULT_SMOKE_POLICY,
    SmokePolicyOptions,
    run_smoke_policy,
    run_smoke_policy_sync,
)
from rosclaw.integrations.lerobot.smoke_report import (
    SmokeReport,
    get_smoke_report_dir,
    get_validation_status,
    read_latest_smoke_report,
    write_smoke_report,
)
from rosclaw.integrations.lerobot.worker_runner import LeRobotWorkerRunner, run_worker_op
from rosclaw.integrations.lerobot.worker_schema import (
    WorkerAction,
    WorkerError,
    WorkerObservation,
    WorkerRequest,
    WorkerResponse,
    WorkerTiming,
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
    "LeRobotWorkerRunner",
    "run_worker_op",
    "WorkerRequest",
    "WorkerResponse",
    "WorkerAction",
    "WorkerError",
    "WorkerTiming",
    "WorkerObservation",
    "adapt_observation_for_worker",
    "adapt_action_to_proposal",
    "load_policy_manifest",
    "materialize_policy_path",
    "MaterializationResult",
    "PolicyMaterializationError",
    "get_policy_cache_dir",
    "SmokePolicyOptions",
    "run_smoke_policy",
    "run_smoke_policy_sync",
    "DEFAULT_SMOKE_POLICY",
    "SmokeReport",
    "get_smoke_report_dir",
    "read_latest_smoke_report",
    "write_smoke_report",
    "get_validation_status",
    "POLICY_COMPATIBILITY_MATRIX",
    "PolicyCompatibility",
    "build_compatibility_report",
    "classify_compatibility_level",
    "get_policy_compatibility",
]
