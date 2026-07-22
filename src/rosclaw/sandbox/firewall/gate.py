"""Fast static action checks.

This module deliberately does not claim to execute physics.  Full trajectory
validation lives in :mod:`rosclaw.sandbox.backends.mujoco_cpu`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Decision:
    """Result of a firewall safety check."""

    action: str = "ALLOW"
    is_allowed: bool = True
    risk_score: float = 0.0
    predicted_collision: bool = False
    reason: str = ""
    violated_constraints: list[str] = field(default_factory=list)
    replay_id: str | None = None
    modified_action: dict[str, Any] | None = None
    validation_type: str = "STATIC_POLICY"
    physics_executed: bool = False


class StaticActionGate:
    """Low-latency policy checks using model-derived constraints only."""

    WORKSPACE_RADIUS = 1.5
    MAX_JOINT_VELOCITY = 3.15
    MAX_TCP_FORCE = 150.0
    MAX_TCP_TORQUE = 8.0

    def __init__(
        self,
        robot_id: str,
        world_id: str,
        engine: str = "mujoco",
        *,
        joint_limits: list[tuple[float, float]] | None = None,
    ):
        self.robot_id = robot_id
        self.world_id = world_id
        self.engine = engine
        self.joint_limits = joint_limits or self._load_joint_limits()

    def _load_joint_limits(self) -> list[tuple[float, float]]:
        """Resolve limits from the effective simulation model, never constants."""
        if self.engine.lower() != "mujoco":
            return []
        from rosclaw.sandbox.sandbox_api import Sandbox

        sandbox = Sandbox.create(self.robot_id, self.world_id, self.engine)
        try:
            model = sandbox.physics_model
            if model is None or int(model.nu) <= 0:
                return []
            return [
                (
                    float(model.actuator_ctrlrange[index][0]),
                    float(model.actuator_ctrlrange[index][1]),
                )
                for index in range(int(model.nu))
            ]
        finally:
            sandbox.close()

    def _generate_replay_id(self) -> str:
        return f"sandbox://replay/{uuid.uuid4().hex[:12]}"

    def check(self, action: dict[str, Any]) -> Decision:
        values = action.get("values", [])
        if not values:
            return Decision(action="ALLOW", is_allowed=True, risk_score=0.0)

        if not self.joint_limits:
            return Decision(
                action="BLOCK",
                is_allowed=False,
                risk_score=1.0,
                reason="Static gate blocked: joint limits unavailable",
                violated_constraints=["joint_limits_unavailable"],
                replay_id=self._generate_replay_id(),
            )
        if len(values) != len(self.joint_limits):
            return Decision(
                action="BLOCK",
                is_allowed=False,
                risk_score=1.0,
                reason="Static gate blocked: action dimension mismatch",
                violated_constraints=["action_dimension_mismatch"],
                replay_id=self._generate_replay_id(),
            )

        max_violation = 0.0
        violations = []
        replay_id = self._generate_replay_id()

        # Joint limits
        for i, v in enumerate(values):
            if i < len(self.joint_limits):
                lo, hi = self.joint_limits[i]
                if v < lo or v > hi:
                    max_violation = max(max_violation, abs(v) - max(abs(lo), abs(hi)))
                    violations.append(f"joint_{i}_limit")

        # Velocity limits
        current = action.get("current")
        if current is not None:
            for i, (target, curr) in enumerate(zip(values, current, strict=False)):
                if abs(target - curr) > self.MAX_JOINT_VELOCITY * 0.002:
                    violations.append(f"joint_{i}_velocity")
                    max_violation = max(max_violation, abs(target - curr))

        # Workspace boundary
        tcp_position = self._forward_kinematics(values)
        dist_from_base = sum(x**2 for x in tcp_position) ** 0.5
        if dist_from_base > self.WORKSPACE_RADIUS:
            violations.append("workspace_boundary")
            max_violation = max(max_violation, dist_from_base - self.WORKSPACE_RADIUS)

        # PFL
        planned_force = action.get("force", 0.0)
        planned_torque = action.get("torque", 0.0)
        if planned_force > self.MAX_TCP_FORCE:
            violations.append("pfl_force")
            max_violation = max(max_violation, planned_force - self.MAX_TCP_FORCE)
        if planned_torque > self.MAX_TCP_TORQUE:
            violations.append("pfl_torque")
            max_violation = max(max_violation, planned_torque - self.MAX_TCP_TORQUE)

        # Self-collision
        if self._predict_self_collision(values):
            violations.append("self_collision")
            max_violation = max(max_violation, 0.5)

        risk_score = min(1.0, 0.5 + max_violation * 0.1) if violations else 0.0

        if violations:
            if risk_score >= 0.8 or "self_collision" in violations:
                return Decision(
                    action="BLOCK",
                    is_allowed=False,
                    risk_score=risk_score,
                    predicted_collision=True,
                    reason=f"Firewall blocked: {', '.join(violations)}",
                    violated_constraints=violations,
                    replay_id=replay_id,
                )
            elif risk_score >= 0.5:
                modified = self._suggest_modification(action, violations)
                return Decision(
                    action="MODIFY",
                    is_allowed=False,
                    risk_score=risk_score,
                    predicted_collision=True,
                    reason=f"Firewall modified: {', '.join(violations)}",
                    violated_constraints=violations,
                    replay_id=replay_id,
                    modified_action=modified,
                )
            else:
                return Decision(
                    action="REQUIRE_CONFIRMATION",
                    is_allowed=False,
                    risk_score=risk_score,
                    reason=f"Firewall requires confirmation: {', '.join(violations)}",
                    violated_constraints=violations,
                    replay_id=replay_id,
                )

        return Decision(
            action="ALLOW",
            is_allowed=True,
            risk_score=0.0,
            reason="Within limits",
            replay_id=replay_id,
        )

    def _forward_kinematics(self, joint_positions):
        link_lengths = [0.089, 0.425, 0.392, 0.109, 0.093, 0.082]
        x, y, z = 0.0, 0.0, 0.0
        for i, (q, length) in enumerate(zip(joint_positions[:6], link_lengths, strict=False)):
            if i % 2 == 0:
                x += length * (1.0 if q >= 0 else -1.0)
            else:
                z += length * abs(q) / 3.14
        return (x, y, z)

    def _predict_self_collision(self, joint_positions):
        if len(joint_positions) >= 3:
            elbow = joint_positions[2]
            if abs(elbow) > 3.0:
                return True
        return False

    def _suggest_modification(self, action, violations):
        modified = dict(action)
        values = list(modified.get("values", []))
        for v in violations:
            if "joint_" in v and "_limit" in v:
                idx = int(v.split("_")[1])
                if idx < len(values):
                    lo, hi = self.joint_limits[idx]
                    values[idx] = max(lo, min(hi, values[idx]))
            elif "pfl_force" in v:
                modified["force"] = self.MAX_TCP_FORCE * 0.8
            elif "pfl_torque" in v:
                modified["torque"] = self.MAX_TCP_TORQUE * 0.8
            elif "workspace_boundary" in v:
                values = [v * 0.7 for v in values]
        modified["values"] = values
        modified["_firewall_modified"] = True
        modified["_original_violations"] = violations
        return modified

    def close(self):
        pass


class FirewallGate(StaticActionGate):
    """Compatibility alias for :class:`StaticActionGate`.

    The class name is retained for integrations, while every decision states
    that it is a static policy result and that no physics was executed.
    """
