"""Persistent policy runtime config and daemon bookkeeping.

Config lives at ``~/.rosclaw/integrations/lerobot_policy_runtime.yaml``.
This module is free of torch/lerobot imports.
"""

from __future__ import annotations

import os
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rosclaw.firstboot.workspace import get_rosclaw_home


def get_policy_runtime_config_dir() -> Path:
    """Return the directory for runtime state files."""
    return get_rosclaw_home() / "run"


def get_policy_runtime_config_path() -> Path:
    """Return the path to the persistent policy runtime config file."""
    return get_rosclaw_home() / "integrations" / "lerobot_policy_runtime.yaml"


def get_policy_runtime_socket_path() -> Path:
    """Return the default Unix socket path for the runtime daemon."""
    return get_policy_runtime_config_dir() / "lerobot_policy_runtime.sock"


def get_policy_runtime_pid_path() -> Path:
    """Return the default PID file path for the runtime daemon."""
    return get_policy_runtime_config_dir() / "lerobot_policy_runtime.pid"


def load_policy_runtime_config() -> dict[str, Any]:
    """Load the persistent policy runtime config, returning {} if missing."""
    config_path = get_policy_runtime_config_path()
    if not config_path.exists():
        return {}
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def save_policy_runtime_config(config: dict[str, Any]) -> None:
    """Write the persistent policy runtime config to disk."""
    config_path = get_policy_runtime_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.setdefault("updated_at", datetime.now(UTC).isoformat())
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def clear_policy_runtime_config() -> None:
    """Remove the persistent policy runtime config file if it exists."""
    config_path = get_policy_runtime_config_path()
    if config_path.exists():
        config_path.unlink()


def is_process_alive(pid: int) -> bool:
    """Return True if ``pid`` exists and can be signaled."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, OverflowError):
        return False


def read_pid_file(pid_path: Path | None = None) -> int | None:
    """Read a PID from the runtime PID file, returning None if invalid."""
    pid_path = pid_path or get_policy_runtime_pid_path()
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def write_pid_file(pid: int, pid_path: Path | None = None) -> None:
    """Write ``pid`` to the runtime PID file."""
    pid_path = pid_path or get_policy_runtime_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(pid), encoding="utf-8")


def remove_pid_file(pid_path: Path | None = None) -> None:
    """Remove the runtime PID file if it exists."""
    pid_path = pid_path or get_policy_runtime_pid_path()
    if pid_path.exists():
        pid_path.unlink()


def terminate_process(pid: int, grace_period_sec: float = 5.0) -> bool:
    """Send SIGTERM to ``pid`` and escalate to SIGKILL if it does not exit."""
    if not is_process_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return not is_process_alive(pid)

    import time

    deadline = time.monotonic() + grace_period_sec
    while time.monotonic() < deadline:
        if not is_process_alive(pid):
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    return not is_process_alive(pid)


def get_daemon_status(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a snapshot of the daemon status based on config and PID file."""
    config = config if config is not None else load_policy_runtime_config()
    pid = read_pid_file()
    alive = is_process_alive(pid) if pid is not None else False
    socket_path = config.get("socket_path")
    socket_exists = bool(socket_path and Path(socket_path).exists())
    return {
        "running": alive and socket_exists,
        "pid": pid,
        "alive": alive,
        "socket_path": socket_path,
        "socket_exists": socket_exists,
        "policy_path": config.get("policy_path"),
        "device": config.get("device", "cpu"),
        "dtype": config.get("dtype", "auto"),
        "python_executable": config.get("python_executable"),
        "updated_at": config.get("updated_at"),
        "checked_at": datetime.now(UTC).isoformat(),
    }
