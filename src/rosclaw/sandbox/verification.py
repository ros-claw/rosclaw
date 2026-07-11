"""Deterministic MuJoCo verification cases for ROSClaw sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rosclaw.sandbox.sandbox_api import Sandbox


@dataclass
class SandboxVerificationResult:
    """Structured result for a sandbox verification case."""

    case: str
    robot_id: str
    world_id: str
    task: str
    passed: bool
    has_physics: bool
    steps: int
    qpos_size: int = 0
    qvel_size: int = 0
    final_time: float = 0.0
    contacts_count: int = 0
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "robot_id": self.robot_id,
            "world_id": self.world_id,
            "task": self.task,
            "passed": self.passed,
            "has_physics": self.has_physics,
            "steps": self.steps,
            "qpos_size": self.qpos_size,
            "qvel_size": self.qvel_size,
            "final_time": self.final_time,
            "contacts_count": self.contacts_count,
            "reason": self.reason,
            "details": self.details,
        }


def run_ur5e_joint_preview(
    *,
    robot_id: str = "universal_robots_ur5e",
    world_id: str = "empty",
    steps: int = 5,
    joint_positions: list[float] | None = None,
) -> SandboxVerificationResult:
    """Load the UR5e MuJoCo model and run a small joint-position preview."""
    case = "ur5e-joint-preview"
    task = "preview a conservative six-axis joint-position command in MuJoCo"
    positions = joint_positions or [0.05, -0.08, 0.06, 0.0, 0.04, 0.0]
    sandbox = Sandbox.create(robot_id=robot_id, world_id=world_id, engine="mujoco")
    try:
        if not sandbox.has_physics:
            return SandboxVerificationResult(
                case=case,
                robot_id=robot_id,
                world_id=world_id,
                task=task,
                passed=False,
                has_physics=False,
                steps=0,
                reason="MuJoCo model did not load; sandbox is running without physics.",
            )

        initial_state = sandbox.get_state() or {}
        initial_time = float(initial_state.get("time", 0.0))
        state: dict[str, Any] | None = None
        for _ in range(max(1, steps)):
            state = sandbox.step(positions)
        observation = sandbox.get_observation(normalize=True) or {}
        state = state or {}
        qpos = state.get("qpos") if isinstance(state.get("qpos"), list) else []
        qvel = state.get("qvel") if isinstance(state.get("qvel"), list) else []
        final_time = float(state.get("time", 0.0))
        contacts = observation.get("contacts", [])
        contacts_count = len(contacts) if isinstance(contacts, list) else 0
        passed = bool(qpos and qvel and final_time > initial_time)
        reason = (
            "MuJoCo model stepped with non-empty qpos/qvel."
            if passed
            else ("MuJoCo step did not produce a valid advancing physics state.")
        )
        return SandboxVerificationResult(
            case=case,
            robot_id=robot_id,
            world_id=world_id,
            task=task,
            passed=passed,
            has_physics=True,
            steps=max(1, steps),
            qpos_size=len(qpos),
            qvel_size=len(qvel),
            final_time=final_time,
            contacts_count=contacts_count,
            reason=reason,
            details={
                "initial_time": initial_time,
                "joint_positions": positions,
                "body_count": len(observation.get("body_positions", {}) or {}),
            },
        )
    finally:
        sandbox.close()


def run_verification_case(
    case: str,
    *,
    robot_id: str | None = None,
    world_id: str = "empty",
    steps: int = 5,
) -> SandboxVerificationResult:
    """Dispatch a named sandbox verification case."""
    if case == "ur5e-joint-preview":
        return run_ur5e_joint_preview(
            robot_id=robot_id or "universal_robots_ur5e",
            world_id=world_id,
            steps=steps,
        )
    raise ValueError(f"Unknown sandbox verification case: {case}")


__all__ = [
    "SandboxVerificationResult",
    "run_ur5e_joint_preview",
    "run_verification_case",
]
