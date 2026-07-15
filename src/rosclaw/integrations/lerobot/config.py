"""LeRobot integration config persistence and migration.

Config lives at ``~/.rosclaw/integrations/lerobot.yaml``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.integrations.lerobot.runtime import RuntimeMode


def get_lerobot_config_path() -> Path:
    """Return the path to the LeRobot integration config file."""
    return get_rosclaw_home() / "integrations" / "lerobot.yaml"


def get_default_runtime_path() -> Path:
    """Return the default isolated LeRobot runtime path."""
    return get_rosclaw_home() / "envs" / "lerobot"


def load_lerobot_config() -> dict[str, Any]:
    """Load the LeRobot config, migrating old v0 schemas on the fly."""
    config_path = get_lerobot_config_path()
    if not config_path.exists():
        return {}

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    if "lerobot_runtime" not in config:
        config = migrate_v0_config_to_v1(config)

    return config


def save_lerobot_config(config: dict[str, Any]) -> None:
    """Write the LeRobot config to disk."""
    config_path = get_lerobot_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.setdefault("last_checked_at", datetime.now(UTC).isoformat())
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def migrate_v0_config_to_v1(config: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a Round 1 (v0) config to the P0.1 runtime-aware schema."""
    import sys

    migrated: dict[str, Any] = {
        "enabled": config.get("enabled", True),
        "integration": config.get("integration", "lerobot"),
        "profile": config.get("profile", "core"),
        "install_mode": config.get("install_mode"),
        "created_by": config.get("created_by", "rosclaw"),
        "last_checked_at": config.get("last_checked_at") or datetime.now(UTC).isoformat(),
        "rosclaw_runtime": {
            "python_executable": sys.executable,
            "python_version": "",
        },
        "lerobot_runtime": {},
        "hub": {
            "hf_endpoint": config.get("hf_endpoint", "https://huggingface.co"),
            "hf_home": None,
            "hf_cache": "~/.cache/huggingface",
        },
        "capabilities": config.get("capabilities", {}),
        "status": {
            "state": config.get("status", "installed"),
            "doctor_status": config.get("doctor_status", "ok"),
            "last_error": config.get("last_error"),
        },
    }

    old_python = config.get("python")
    old_pip = config.get("pip")
    old_version = config.get("lerobot_version")

    if old_python:
        migrated["lerobot_runtime"]["python_executable"] = old_python
    if old_pip:
        migrated["lerobot_runtime"]["pip_executable"] = old_pip
    if old_version:
        migrated["lerobot_runtime"]["lerobot_version"] = old_version

    # Infer install_mode if missing.
    install_mode = migrated["install_mode"]
    if install_mode is None:
        if old_python and Path(old_python).resolve() == Path(sys.executable).resolve():
            install_mode = "current-env"
        else:
            install_mode = "external"
        migrated["install_mode"] = install_mode

    migrated["lerobot_runtime"].setdefault("mode", install_mode)
    migrated["lerobot_runtime"].setdefault("runtime_id", "default")

    # Preserve legacy details if present.
    if "installed_at" in config:
        migrated["installed_at"] = config["installed_at"]

    return migrated


def build_lerobot_config(
    *,
    profile: str,
    mode: RuntimeMode,
    runtime: Any,
    rosclaw_python: str,
    rosclaw_version: str,
    capabilities: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Build a fresh v1 config from a discovered runtime."""
    caps = capabilities or {
        "provider_type_lerobot_policy": True,
        "dataset_export_lerobot": True,
        "worker_subprocess": True,
        "worker_in_process": runtime.in_process_available,
        "eval_backend_lerobot": False,
        "rollout_backend_lerobot": False,
        "reward_backend_lerobot": False,
    }

    lerobot_runtime: dict[str, Any] = {
        "runtime_id": "default",
        "mode": mode,
        "runtime_path": str(runtime.runtime_path) if runtime.runtime_path else None,
        "python_executable": str(runtime.python_executable),
        "pip_executable": str(runtime.pip_executable) if runtime.pip_executable else None,
        "lerobot_info_executable": (
            str(runtime.lerobot_info_executable) if runtime.lerobot_info_executable else None
        ),
        "python_version": runtime.python_version,
        "lerobot_version": runtime.lerobot_version,
        "torch_version": runtime.torch_version,
        "cuda_available": runtime.cuda_available,
        "state": runtime.state,
        "in_process_available": runtime.in_process_available,
        "subprocess_available": runtime.subprocess_available,
        "error": runtime.error,
    }

    return {
        "enabled": True,
        "integration": "lerobot",
        "profile": profile,
        "install_mode": mode,
        "created_by": "rosclaw",
        "last_checked_at": datetime.now(UTC).isoformat(),
        "rosclaw_runtime": {
            "python_executable": rosclaw_python,
            "python_version": rosclaw_version,
        },
        "lerobot_runtime": lerobot_runtime,
        "hub": {
            "hf_endpoint": "https://huggingface.co",
            "hf_home": None,
            "hf_cache": "~/.cache/huggingface",
        },
        "capabilities": caps,
        "status": {
            "state": "installed" if runtime.state == "ready" else runtime.state,
            "doctor_status": "ok" if runtime.state == "ready" else "degraded",
            "last_error": runtime.error,
        },
    }


def get_configured_lerobot_runtime() -> dict[str, Any] | None:
    """Return the ``lerobot_runtime`` section of the stored config, if any."""
    config = load_lerobot_config()
    return config.get("lerobot_runtime")
