"""Thin adapter around the ROSClaw sandbox for MCP tools."""

from __future__ import annotations

from typing import Any


class SandboxClient:
    """Simulation-only client that delegates trajectory validation and stepping."""

    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    def validate_trajectory(
        self,
        trajectory: list[list[float]],
        *,
        safety_level: str = "MODERATE",
    ) -> dict[str, Any]:
        """Validate a trajectory through the sandbox firewall gate.

        Returns a normalized dict even when the underlying adapter returns
        unexpected types.
        """
        result = self._sandbox.validate_trajectory(
            trajectory=trajectory,
            safety_level=safety_level,
        )
        if isinstance(result, dict):
            return result
        return {
            "is_safe": False,
            "risk_score": 1.0,
            "reason": "Sandbox returned an unexpected validation result.",
            "violations": ["invalid_validation_result"],
            "replay_id": None,
        }

    def simulate_step(self, joint_positions: list[float]) -> dict[str, Any]:
        """Run one MuJoCo simulation step and return the physics state."""
        state = self._sandbox.simulate_step(joint_positions)
        has_physics = bool(getattr(self._sandbox, "has_physics", False))
        if isinstance(state, dict) and state:
            return {
                "physics_state": state,
                "mode": "live",
                "has_physics": has_physics,
            }
        return {
            "physics_state": {},
            "mode": "degraded",
            "has_physics": False,
            "note": "Sandbox initialized but no physics state was returned; MuJoCo model may be unavailable.",
        }
