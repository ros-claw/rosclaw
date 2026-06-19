"""Health estimator for raw BodyState snapshots."""

from __future__ import annotations

from typing import Any

from rosclaw.sense.schemas import BodyState


class HealthEstimator:
    """Evaluate subsystem health from a BodyState snapshot.

    Returns string health levels: unknown, ok, degraded, bad.
    """

    def estimate(self, state: BodyState) -> dict[str, str]:
        """Return a mapping from subsystem name to health level."""
        return {
            "energy": self._energy_health(state),
            "thermal": self._thermal_health(state),
            "joints": self._joint_health(state),
            "balance": self._balance_health(state),
            "contact": self._contact_health(state),
            "perception": self._perception_health(state),
            "communication": self._communication_health(state),
            "compute": self._compute_health(state),
        }

    def _energy_health(self, state: BodyState) -> str:
        if state.energy.battery_percent is None:
            return "unknown"
        if state.energy.battery_percent < 10.0:
            return "bad"
        if state.energy.battery_percent < 25.0:
            return "degraded"
        return "ok"

    def _thermal_health(self, state: BodyState) -> str:
        temps = [
            j.temperature_c for j in state.joints.values()
            if j.temperature_c is not None
        ]
        if not temps:
            return "unknown"
        max_temp = max(temps)
        if max_temp >= 85.0:
            return "bad"
        if max_temp >= 75.0:
            return "degraded"
        return "ok"

    def _joint_health(self, state: BodyState) -> str:
        if not state.joints:
            return "unknown"
        bad = 0
        degraded = 0
        for joint in state.joints.values():
            if joint.tracking_error is not None and joint.tracking_error >= 0.25:
                bad += 1
            elif joint.temperature_c is not None and joint.temperature_c >= 75.0:
                degraded += 1
        if bad > 0:
            return "bad"
        if degraded > 0:
            return "degraded"
        return "ok"

    def _balance_health(self, state: BodyState) -> str:
        if state.balance.support_margin is None:
            return "unknown"
        if state.balance.support_margin < 0.12:
            return "bad"
        if state.balance.support_margin < 0.18:
            return "degraded"
        return "ok"

    def _contact_health(self, state: BodyState) -> str:
        if not state.contact:
            return "unknown"
        for contact in state.contact.values():
            if contact.contact is None:
                return "unknown"
        return "ok"

    def _perception_health(self, state: BodyState) -> str:
        if state.perception.status != "unknown":
            return state.perception.status
        if (
            state.perception.front_camera_fps is None
            and state.perception.target_detector_confidence is None
        ):
            return "unknown"
        return "ok"

    def _communication_health(self, state: BodyState) -> str:
        if state.communication.heartbeat_ok is False:
            return "bad"
        if state.communication.dds_latency_ms is None:
            return "unknown"
        if state.communication.dds_latency_ms >= 100.0:
            return "bad"
        if state.communication.dds_latency_ms >= 50.0:
            return "degraded"
        return "ok"

    def _compute_health(self, state: BodyState) -> str:
        if state.compute.cpu_usage_percent is None:
            return "unknown"
        if state.compute.cpu_usage_percent >= 95.0:
            return "bad"
        if state.compute.cpu_usage_percent >= 85.0:
            return "degraded"
        return "ok"

    def to_dict(self, state: BodyState) -> dict[str, Any]:
        """Return health estimate as a dictionary."""
        return self.estimate(state)
