"""LeRobot integration schemas and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rosclaw.integrations.registry import IntegrationReport

if TYPE_CHECKING:
    from rosclaw.integrations.lerobot.runtime import LeRobotRuntime


class LeRobotSetupErrorCode(StrEnum):
    """Structured error codes for LeRobot setup failures."""

    PYTHON_TOO_OLD = "python_too_old"
    PYTHON312_NOT_FOUND = "python312_not_found"
    EXTERNAL_PYTHON_NOT_FOUND = "external_python_not_found"
    EXTERNAL_PYTHON_TOO_OLD = "external_python_too_old"
    VENV_CREATE_FAILED = "venv_create_failed"
    PIP_INSTALL_FAILED = "pip_install_failed"
    LEROBOT_IMPORT_FAILED = "lerobot_import_failed"
    LEROBOT_VERSION_UNSUPPORTED = "lerobot_version_unsupported"
    LEROBOT_INFO_FAILED = "lerobot_info_failed"
    CONFIG_WRITE_FAILED = "config_write_failed"
    PIP_NOT_FOUND = "pip_not_found"


class LeRobotWorkerErrorCode(StrEnum):
    """Structured error codes for LeRobot worker/provider failures."""

    RUNTIME_NOT_CONFIGURED = "runtime_not_configured"
    RUNTIME_PYTHON_MISSING = "runtime_python_missing"
    WORKER_SCRIPT_MISSING = "worker_script_missing"
    WORKER_TIMEOUT = "worker_timeout"
    WORKER_INVALID_JSON = "worker_invalid_json"
    WORKER_PROCESS_FAILED = "worker_process_failed"
    POLICY_CONFIG_NOT_FOUND = "policy_config_not_found"
    POLICY_METADATA_PARSE_FAILED = "policy_metadata_parse_failed"
    POLICY_LOAD_FAILED = "policy_load_failed"
    POLICY_INFER_FAILED = "policy_infer_failed"
    POLICY_API_UNSUPPORTED = "policy_api_unsupported"
    OBSERVATION_SCHEMA_MISMATCH = "observation_schema_mismatch"
    IMAGE_FILE_NOT_FOUND = "image_file_not_found"
    NETWORK_DISABLED = "network_disabled"
    NETWORK_DOWNLOAD_REQUIRED = "network_download_required"


@dataclass
class ProfileSpec:
    """A LeRobot installation profile."""

    name: str
    pip: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    enabled_capabilities: list[str] = field(default_factory=list)
    requires_python: str = ">=3.12"
    capabilities: dict[str, Any] = field(default_factory=dict)


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
    error_code: str | None = None
    mode: str | None = None
    runtime: LeRobotRuntime | None = None
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
    rosclaw_python_version: str | None = None
    rosclaw_python_executable: str | None = None
    lerobot_runtime: LeRobotRuntime | None = None
    worker_subprocess_available: bool = False
    worker_in_process_available: bool = False
    status_detail: str | None = None
    validation_status: dict[str, Any] = field(default_factory=dict)
