"""Operational envelope estimator.

Derives dynamic operational constraints from ``BodySense``.  The envelope
considers the overall readiness status, thermal limits, and an externally
supplied fatigue risk level.
"""

from __future__ import annotations

from typing import Any

from rosclaw.sense.schemas import BodySense


class OperationalEnvelopeEstimator:
    """Derive dynamic operational envelope from BodySense."""

    def estimate(
        self,
        body_sense: BodySense,
        fatigue_risk: str = "unknown",
    ) -> dict[str, Any]:
        """Return an operational envelope for the given body sense."""
        thermal_limited = self._is_thermal_limited(body_sense)

        status = body_sense.overall_status
        if status == "ready":
            scale = 1.0
        elif status == "caution":
            scale = 0.5
        elif status in ("not_ready", "emergency"):
            scale = 0.0
        else:
            scale = 1.0

        if thermal_limited:
            scale *= 0.5

        if fatigue_risk == "high":
            scale *= 0.5
        elif fatigue_risk == "critical":
            scale *= 0.0

        return {
            "sandbox_only": status in ("not_ready", "emergency"),
            "cooldown_required": thermal_limited
            or "joint_hot" in body_sense.main_reasons
            or "joint_overheat" in body_sense.main_reasons,
            "thermal_limited": thermal_limited,
            "max_velocity_scale": scale,
        }

    def _is_thermal_limited(self, body_sense: BodySense) -> bool:
        risk = body_sense.risk_summary
        return (
            risk.thermal_risk in ("high", "critical")
            or any(
                reason.startswith(("joint_hot", "joint_overheat"))
                or "temperature" in reason
                for reason in body_sense.main_reasons
            )
        )
