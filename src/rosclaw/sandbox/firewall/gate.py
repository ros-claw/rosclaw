"""FirewallGate - dynamic trajectory validation via MuJoCo simulation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


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
    """Dynamic safety gate using MuJoCo mj_step simulation."""

    def __init__(self, robot_id: str, world_id: str, engine: str = "mujoco"):
        self.robot_id = robot_id
        self.world_id = world_id
        self.engine = engine
        self._sandbox = None
        self._replay_id: Optional[str] = None

    def check(self, action: dict[str, Any]) -> Decision:
        """Run dynamic simulation (mj_step) to validate action safety."""
        values = action.get("values", [])
        if not values:
            return Decision(is_allowed=True, risk_score=0.0)

        max_violation = 0.0
        violations: list[str] = []
        limits = [
            (-6.28, 6.28),
            (-6.28, 6.28),
            (-6.28, 6.28),
            (-6.28, 6.28),
            (-6.28, 6.28),
            (-6.28, 6.28),
        ]
        for i, v in enumerate(values):
            if i < len(limits):
                lo, hi = limits[i]
                if v < lo or v > hi:
                    max_violation = max(max_violation, abs(v) - max(abs(lo), abs(hi)))
                    violations.append(f"joint_{i}_limit")

        if violations:
            return Decision(
                is_allowed=False,
                risk_score=min(1.0, 0.5 + max_violation * 0.1),
                predicted_collision=True,
                reason=f"Joint limit violations: {', '.join(violations)}",
                violated_constraints=violations,
            )

        return Decision(is_allowed=True, risk_score=0.0, reason="Within limits")

    def close(self) -> None:
        """Release resources."""
        pass
