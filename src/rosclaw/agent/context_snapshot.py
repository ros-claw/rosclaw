"""Machine-readable context snapshot builder for agent onboarding."""

from __future__ import annotations

from pathlib import Path

from rosclaw.agent.detectors import ProjectProfile
from rosclaw.agent.merge import atomic_write_json, backup_file
from rosclaw.agent.templates import render_context_snapshot


def write_context_snapshot(
    project_root: Path,
    profile: ProjectProfile,
    *,
    backup: bool = True,
) -> Path:
    """Write the .rosclaw/agent/context.snapshot.json file for the project.

    Returns the path written.
    """
    snapshot_dir = project_root / ".rosclaw" / "agent"
    snapshot_path = snapshot_dir / "context.snapshot.json"
    snapshot = render_context_snapshot(profile)

    if snapshot_path.exists() and backup:
        backup_file(snapshot_path, backups_dir=snapshot_dir / ".backups")

    atomic_write_json(snapshot_path, snapshot, indent=2)
    return snapshot_path


__all__ = ["write_context_snapshot"]
