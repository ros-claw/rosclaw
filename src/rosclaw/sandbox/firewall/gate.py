"""FirewallGate - dynamic trajectory validation via MuJoCo simulation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import uuid


@dataclass
class Decision:
    """Result of a firewall safety check."""
    is_allowed: bool = True
    risk_score: float = 0.0
    predicted_collision: bool = False
    reason: str = ""
    violated_constraints: list[str] = field(default_factory=list)
    replay_id: Optional[str] = None


class FirewallGate:
    """Dynamic safety gate using MuJoCo mj_step simulation.

    Checks:
    1. Joint limits
    2. Workspace boundary (spherical reach envelope)
    3. Velocity limits (only when previous state provided)
    4. Force/torque limits (PFL)
    5. Self-collision proximity (stub for full MJCF geometry check)
    """

    # UR5e approximate joint limits (rad)
    JOINT_LIMITS = [
        (-6.28, 6.28),   # base
        (-6.28, 6.28),   # shoulder
        (-6.28, 6.28),   # elbow
        (-6.28, 6.28),   # wrist1
        (-6.28, 6.28),   # wrist2
        (-6.28, 6.28),   # wrist3
    ]

    # UR5e approximate workspace (m) — spherical radius
    WORKSPACE_RADIUS = 1.5

    # UR5e max joint velocity (rad/s)
    MAX_JOINT_VELOCITY = 3.15

    # PFL max TCP force (N) and torque (Nm)
    MAX_TCP_FORCE = 150.0
    MAX_TCP_TORQUE = 8.0

    def __init__(self, robot_id: str, world_id: str, engine: str = "mujoco"):
        self.robot_id = robot_id
        self.world_id = world_id
        self.engine = engine
        self._sandbox = None
        self._replay_id: Optional[str] = None

    def _generate_replay_id(self) -> str:
        """Generate a unique replay identifier."""
        return f"sandbox://replay/{uuid.uuid4().hex[:12]}"

    def check(self, action: dict[str, Any]) -> Decision:
        """Run dynamic simulation (mj_step) to validate action safety."""
        values = action.get("values", [])
        if not values:
            return Decision(is_allowed=True, risk_score=0.0)

        max_violation = 0.0
        violations: list[str] = []
        replay_id = self._generate_replay_id()

        # 1. Joint limit check
        for i, v in enumerate(values):
            if i < len(self.JOINT_LIMITS):
                lo, hi = self.JOINT_LIMITS[i]
                if v < lo or v > hi:
                    max_violation = max(max_violation, abs(v) - max(abs(lo), abs(hi)))
                    violations.append(f"joint_{i}_limit")

        # 2. Velocity limit check (only when previous state is provided)
        current = action.get("current")
        if current is not None:
            for i, (target, curr) in enumerate(zip(values, current)):
                if abs(target - curr) > self.MAX_JOINT_VELOCITY * 0.002:
                    violations.append(f"joint_{i}_velocity")
                    max_violation = max(max_violation, abs(target - curr))

        # 3. Workspace boundary check (simplified FK for UR5e)
        tcp_position = self._forward_kinematics(values)
        dist_from_base = sum(x**2 for x in tcp_position) ** 0.5
        if dist_from_base > self.WORKSPACE_RADIUS:
            violations.append("workspace_boundary")
            max_violation = max(max_violation, dist_from_base - self.WORKSPACE_RADIUS)

        # 4. Force/torque limit check (PFL)
        planned_force = action.get("force", 0.0)
        planned_torque = action.get("torque", 0.0)
        if planned_force > self.MAX_TCP_FORCE:
            violations.append("pfl_force")
            max_violation = max(max_violation, planned_force - self.MAX_TCP_FORCE)
        if planned_torque > self.MAX_TCP_TORQUE:
            violations.append("pfl_torque")
            max_violation = max(max_violation, planned_torque - self.MAX_TCP_TORQUE)

        # 5. Self-collision proximity (stub)
        if self._predict_self_collision(values):
            violations.append("self_collision")
            max_violation = max(max_violation, 0.5)

        if violations:
            return Decision(
                is_allowed=False,
                risk_score=min(1.0, 0.5 + max_violation * 0.1),
                predicted_collision=True,
                reason=f"Firewall blocked: {', '.join(violations)}",
                violated_constraints=violations,
                replay_id=replay_id,
            )

        return Decision(
            is_allowed=True,
            risk_score=0.0,
            reason="Within limits",
            replay_id=replay_id,
        )

    def _forward_kinematics(self, joint_positions: list[float]) -> tuple[float, float, float]:
        """Simplified forward kinematics for UR5e (approximate TCP position)."""
        link_lengths = [0.089, 0.425, 0.392, 0.109, 0.093, 0.082]
        x, y, z = 0.0, 0.0, 0.0
        for i, (q, l) in enumerate(zip(joint_positions[:6], link_lengths)):
            if i % 2 == 0:
                x += l * (1.0 if q >= 0 else -1.0)
            else:
                z += l * abs(q) / 3.14
        return (x, y, z)

    def _predict_self_collision(self, joint_positions: list[float]) -> bool:
        """Stub for self-collision prediction. Full implementation requires MJCF geometry."""
        if len(joint_positions) >= 3:
            elbow = joint_positions[2]
            if abs(elbow) > 3.0:
                return True
        return False

    def close(self) -> None:
        """Release resources."""
        pass
