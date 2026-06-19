"""Operational envelope estimator stub.

The real implementation will derive dynamic operational constraints from
BodySense.  For Phase 1 it returns a pass-through envelope.
"""

from __future__ import annotations

from typing import Any

from rosclaw.sense.schemas import BodySense


class OperationalEnvelopeEstimator:
    """Derive dynamic operational envelope from BodySense."""

    def estimate(self, body_sense: BodySense) -> dict[str, Any]:
        """Return placeholder operational envelope."""
        envelope: dict[str, Any] = {
            "sandbox_only": body_sense.overall_status in ("not_ready", "emergency"),
            "cooldown_required": "joint_hot" in body_sense.main_reasons
            or "joint_overheat" in body_sense.main_reasons,
            "max_velocity_scale": 1.0,
        }
        if body_sense.overall_status == "caution":
            envelope["max_velocity_scale"] = 0.5
        elif body_sense.overall_status in ("not_ready", "emergency"):
            envelope["max_velocity_scale"] = 0.0
        return envelope
