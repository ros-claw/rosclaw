"""Skill runner that maps skill_id to a Python handler and records results.

This is a lightweight executor used by ``rosclaw run skill``.  It does not
require a full behaviour-tree engine; instead it looks up a handler in a
registry, runs it, and optionally records the result to the active practice
session.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.skill.runtime_handlers import (
    realsense_capture_frame,
    realsense_capture_rgbd,
    realsense_camera_info_check,
    realsense_depth_health_check,
    realsense_imu_check,
    scene_risk_scan,
)


@dataclass
class SkillRunResult:
    skill_id: str
    success: bool
    payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    recorded: bool = False


SkillHandler = Callable[..., dict[str, Any]]


class SkillRunner:
    """Dispatch skill_id to a handler and record results."""

    _handlers: dict[str, SkillHandler] = {
        "realsense_capture_frame": realsense_capture_frame,
        "rosclaw-realsense/realsense_capture_frame": realsense_capture_frame,
        "realsense_capture_rgbd": realsense_capture_rgbd,
        "rosclaw-realsense/realsense_capture_rgbd": realsense_capture_rgbd,
        "realsense_camera_info_check": realsense_camera_info_check,
        "rosclaw-realsense/realsense_camera_info_check": realsense_camera_info_check,
        "realsense_depth_health_check": realsense_depth_health_check,
        "rosclaw-realsense/realsense_depth_health_check": realsense_depth_health_check,
        "realsense_imu_check": realsense_imu_check,
        "rosclaw-realsense/realsense_imu_check": realsense_imu_check,
        "scene_risk_scan": scene_risk_scan,
        "rosclaw-realsense/scene_risk_scan": scene_risk_scan,
    }

    def __init__(self, home: Path | None = None) -> None:
        self.home = home or Path(resolve_home())

    def list_skills(self) -> list[str]:
        return list(self._handlers.keys())

    def run(self, skill_id: str, **kwargs: Any) -> SkillRunResult:
        handler = self._handlers.get(skill_id)
        if handler is None:
            return SkillRunResult(
                skill_id=skill_id,
                success=False,
                payload={"error": f"No handler registered for skill '{skill_id}'"},
            )

        start = time.time()
        try:
            payload = handler(**kwargs)
            success = "error" not in payload
        except Exception as exc:
            payload = {"error": str(exc)}
            success = False
        duration_ms = (time.time() - start) * 1000.0

        result = SkillRunResult(
            skill_id=skill_id,
            success=success,
            payload=payload,
            duration_ms=duration_ms,
        )

        # Record to active practice session if one exists
        self._record(result)
        return result

    def _record(self, result: SkillRunResult) -> None:
        """Append the result to the active practice session artifact if present."""
        pid_file = self.home / "practice" / "coordinator.pid"
        if not pid_file.exists():
            return
        try:
            lines = pid_file.read_text(encoding="utf-8").strip().splitlines()
            practice_id = lines[1] if len(lines) > 1 else None
        except Exception:
            return
        if not practice_id:
            return
        artifact_dir = self.home / "practice" / "artifacts" / practice_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        record_path = artifact_dir / "skill_runs.jsonl"
        record = {
            "skill_id": result.skill_id,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "payload": result.payload,
            "timestamp": time.time(),
        }
        with open(record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        result.recorded = True

    @classmethod
    def register(cls, skill_id: str, handler: SkillHandler) -> None:
        cls._handlers[skill_id] = handler
