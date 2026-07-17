"""LeRobot integration for ROSClaw.

The integration package is imported by both the ROSClaw control process and
the isolated LeRobot worker.  Keep package import lightweight so the worker
does not inherit unrelated control-plane dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "adapt_action_to_proposal": "action_adapter",
    "LeRobotIntegration": "capabilities",
    "register_lerobot_capabilities": "capabilities",
    "POLICY_COMPATIBILITY_MATRIX": "compatibility",
    "PolicyCompatibility": "compatibility",
    "build_compatibility_report": "compatibility",
    "classify_compatibility_level": "compatibility",
    "get_policy_compatibility": "compatibility",
    "build_lerobot_config": "config",
    "get_configured_lerobot_runtime": "config",
    "get_default_runtime_path": "config",
    "get_lerobot_config_path": "config",
    "load_lerobot_config": "config",
    "migrate_v0_config_to_v1": "config",
    "save_lerobot_config": "config",
    "LeRobotDoctor": "doctor",
    "LeRobotEnvManager": "env_manager",
    "LeRobotInstaller": "installer",
    "adapt_observation_for_worker": "observation_adapter",
    "MaterializationResult": "policy_cache",
    "PolicyMaterializationError": "policy_cache",
    "get_policy_cache_dir": "policy_cache",
    "materialize_policy_path": "policy_cache",
    "load_policy_manifest": "policy_manifest",
    "LeRobotRuntime": "runtime",
    "PythonRuntimeInfo": "runtime",
    "current_rosclaw_runtime": "runtime",
    "find_python312": "runtime",
    "inspect_lerobot_runtime": "runtime",
    "inspect_python": "runtime",
    "resolve_lerobot_info": "runtime",
    "InstallReport": "schemas",
    "LeRobotDoctorReport": "schemas",
    "LeRobotSetupErrorCode": "schemas",
    "ProfileSpec": "schemas",
    "DEFAULT_SMOKE_POLICY": "smoke_policy",
    "SmokePolicyOptions": "smoke_policy",
    "run_smoke_policy": "smoke_policy",
    "run_smoke_policy_sync": "smoke_policy",
    "SmokeReport": "smoke_report",
    "get_smoke_report_dir": "smoke_report",
    "get_validation_status": "smoke_report",
    "read_latest_smoke_report": "smoke_report",
    "write_smoke_report": "smoke_report",
    "LeRobotWorkerRunner": "worker_runner",
    "run_worker_op": "worker_runner",
    "WorkerAction": "worker_schema",
    "WorkerError": "worker_schema",
    "WorkerObservation": "worker_schema",
    "WorkerRequest": "worker_schema",
    "WorkerResponse": "worker_schema",
    "WorkerTiming": "worker_schema",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
