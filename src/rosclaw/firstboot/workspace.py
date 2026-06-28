"""ROSClaw First Boot workspace utilities."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.feedback.directories import ensure_feedback_dirs

DEFAULT_DIRS = [
    "config/profiles",
    "logs",
    "cache/wheels",
    "cache/downloads",
    "cache/doctor",
    "artifacts/episodes",
    "artifacts/replays",
    "artifacts/sandbox",
    "artifacts/reports",
    "data/memory",
    "data/practice",
    "data/seekdb",
    "data/registry",
    "skills/installed",
    "skills/candidates",
    "skills/champions",
    "providers/installed",
    "providers/manifests",
    "robots/installed",
    "state/locks",
    "telemetry/events",
    "telemetry/heartbeat",
    "telemetry/uploads",
    "feedback/events",
    "feedback/crashes",
    "feedback/bundles",
    "feedback/redacted",
    "feedback/media/local_only",
    "feedback/consent",
    "tmp",
]


@dataclass(frozen=True)
class PlatformInfo:
    os: str
    arch: str
    is_wsl: bool
    shell: str


def resolve_home(path: str | None = None) -> Path:
    """Resolve ROSClaw home directory.

    Priority: explicit path > ROSCLAW_HOME env > ~/.rosclaw
    """
    raw = path or os.environ.get("ROSCLAW_HOME") or "~/.rosclaw"
    return Path(raw).expanduser().resolve()


def get_rosclaw_home() -> Path:
    """Return the active ROSClaw workspace root.

    Uses ``ROSCLAW_HOME`` when set, otherwise ``~/.rosclaw``.
    All modules that write persistent ROSClaw state should resolve paths
    relative to this directory instead of hard-coding ``Path.home() / '.rosclaw'``.
    """
    return resolve_home()


def detect_platform() -> PlatformInfo:
    """Detect current platform."""
    import platform as _platform

    os_name = _platform.system().lower()
    arch = _platform.machine().lower()
    is_wsl = False
    try:
        with open("/proc/version", encoding="utf-8") as f:  # noqa: PTH123
            is_wsl = "microsoft" in f.read().lower()
    except OSError:
        pass

    shell = os.environ.get("SHELL", "/bin/sh").split("/")[-1]
    return PlatformInfo(os=os_name, arch=arch, is_wsl=is_wsl, shell=shell)


def init_workspace(home: Path, force: bool = False) -> dict:
    """Create the full ROSClaw workspace tree idempotently.

    Args:
        home: Workspace root path.
        force: If True, recreate even if already initialized.

    Returns:
        Workspace state metadata dict.
    """
    home.mkdir(parents=True, exist_ok=True)

    for rel in DEFAULT_DIRS:
        (home / rel).mkdir(parents=True, exist_ok=True)

    ensure_feedback_dirs(home)

    install_id_path = home / "state" / "install_id"
    if not install_id_path.exists() or force:
        install_id_path.write_text(str(uuid.uuid4()), encoding="utf-8")

    state = {
        "schema_version": "1.0",
        "install_id": install_id_path.read_text(encoding="utf-8").strip(),
        "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }

    workspace_state_path = home / "state" / "workspace.json"
    workspace_state_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return state


def backup_file(path: Path) -> Path | None:
    """Backup an existing file to ~/.rosclaw/backups/ with timestamp.

    Returns:
        Path to backup file, or None if source did not exist.
    """
    if not path.exists():
        return None

    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup_dir = path.parent.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{path.name}.{ts}.bak"
    shutil.copy2(path, target)
    return target


def is_workspace_initialized(home: Path) -> bool:
    """Check whether a workspace has been initialized."""
    return (home / "state" / "workspace.json").exists()


def load_install_state(home: Path) -> dict | None:
    """Load install.json if present."""
    path = home / "state" / "install.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_install_state(home: Path, state: dict) -> Path:
    """Write install.json and return its path."""
    path = home / "state" / "install.json"
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def ensure_minimal_workspace(home: Path, platform: PlatformInfo | None = None) -> dict:
    """Create the minimal workspace skeleton used by the bootstrapper.

    This only creates config/, logs/, cache/, state/ and writes install.json
    without generating the full runtime config. It is used by scripts/get.sh.
    """
    home.mkdir(parents=True, exist_ok=True)
    for rel in ("config", "logs", "cache", "state"):
        (home / rel).mkdir(parents=True, exist_ok=True)

    ensure_feedback_dirs(home)

    install_id_path = home / "state" / "install_id"
    if not install_id_path.exists():
        install_id_path.write_text(str(uuid.uuid4()), encoding="utf-8")

    if platform is None:
        platform = detect_platform()

    import sys

    state = {
        "schema_version": "1.0",
        "install_id": install_id_path.read_text(encoding="utf-8").strip(),
        "installed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "installer_version": "1.0.0",
        "install_backend": "unknown",
        "install_channel": os.environ.get("ROSCLAW_CHANNEL", "stable"),
        "platform": {
            "os": platform.os,
            "arch": platform.arch,
            "is_wsl": platform.is_wsl,
            "shell": platform.shell,
        },
        "python": {
            "path": sys.executable,
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "firstboot_completed": False,
        "last_doctor_status": "pending",
    }
    save_install_state(home, state)

    workspace_state = {
        "schema_version": "1.0",
        "install_id": install_id_path.read_text(encoding="utf-8").strip(),
        "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "initialized": True,
    }
    workspace_state_path = home / "state" / "workspace.json"
    workspace_state_path.write_text(
        json.dumps(workspace_state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return state
