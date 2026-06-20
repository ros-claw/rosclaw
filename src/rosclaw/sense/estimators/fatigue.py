"""Fatigue estimator.

The estimator combines instantaneous thermal and compute load signals into a
single fatigue score, then smooths it with an exponential moving average so
that transient spikes do not immediately dominate the estimate.
"""

from __future__ import annotations

from typing import Any

from rosclaw.sense.schemas import BodyState

DEFAULT_THERMAL_THRESHOLDS = {
    "warm": 50.0,
    "hot": 65.0,
    "overheat": 80.0,
}

DEFAULT_COMPUTE_THRESHOLDS = {
    "high": 85.0,
    "critical": 95.0,
}


DEFAULT_THERMAL_THRESHOLDS = {
    "warm": 50.0,
    "hot": 65.0,
    "overheat": 80.0,
}

DEFAULT_COMPUTE_THRESHOLDS = {
    "high": 85.0,
    "critical": 95.0,
}


class FatigueEstimator:
    """Estimate robot fatigue level from recent history and current state."""

    def __init__(
        self,
        thresholds: dict[str, Any] | None = None,
        tau: float = 2.0,
    ):
        self.thresholds = thresholds or {}
        self.tau = tau
        self._ema_score = 0.0

    def estimate(
        self,
        state: BodyState,
        prev_state: BodyState | None = None,
        dt: float = 1.0,
    ) -> dict[str, Any]:
        """Return a fatigue estimate for ``state``.

        ``prev_state`` is accepted for API compatibility but the internal
        exponential moving average is what provides temporal smoothing.
        """
        instant = self._instant_score(state)
        # On the very first sample (no prior EMA history) trust the raw
        # instantaneous score so over-stress conditions are visible immediately.
        if self._ema_score == 0.0 and prev_state is None:
            self._ema_score = instant
        else:
            alpha = min(1.0, dt / (dt + self.tau))
            self._ema_score = alpha * instant + (1.0 - alpha) * self._ema_score

        score = float(self._ema_score)
        return {
            "fatigue_score": score,
            "fatigue_risk": self._score_to_risk(score),
            "note": "thermal+compute load EMA",
        }

    def reset(self) -> None:
        """Reset the internal moving average."""
        self._ema_score = 0.0

    def _instant_score(self, state: BodyState) -> float:
        return max(
            self._thermal_score(state),
            self._compute_score(state),
        )

    def _thermal_score(self, state: BodyState) -> float:
        temps = [
            joint.temperature_c
            for joint in state.joints.values()
            if joint.temperature_c is not None
        ]
        if not temps:
            return 0.0
        max_temp = max(temps)
        return self._thermal_score_for_temp(max_temp)

    def _compute_score(self, state: BodyState) -> float:
        score = 0.0
        cpu = state.compute.cpu_usage_percent
        if cpu is not None:
            score = max(score, self._cpu_score(cpu))
        mem = state.compute.memory_usage_percent
        if mem is not None:
            score = max(score, self._memory_score(mem))
        return score

    def _thermal_score_for_temp(self, temp: float) -> float:
        t = self.thresholds.get("joint_temperature_c", {})
        warm = t.get("warm", DEFAULT_THERMAL_THRESHOLDS["warm"])
        hot = t.get("hot", DEFAULT_THERMAL_THRESHOLDS["hot"])
        overheat = t.get("overheat", DEFAULT_THERMAL_THRESHOLDS["overheat"])

        if temp <= warm:
            return 0.0
        if temp <= hot:
            return (temp - warm) / (hot - warm) * 0.3
        if temp <= overheat:
            return 0.3 + (temp - hot) / (overheat - hot) * 0.4
        return 1.0

    def _cpu_score(self, cpu: float) -> float:
        t = self.thresholds.get("compute", {})
        high = t.get("high", DEFAULT_COMPUTE_THRESHOLDS["high"])
        critical = t.get("critical", DEFAULT_COMPUTE_THRESHOLDS["critical"])
        if cpu <= high:
            return 0.0
        if cpu <= critical:
            return (cpu - high) / (critical - high) * 0.5
        return 0.8

    def _memory_score(self, mem: float) -> float:
        t = self.thresholds.get("compute", {})
        critical = t.get("memory_critical", DEFAULT_COMPUTE_THRESHOLDS["critical"])
        if mem >= critical:
            return 0.2
        return 0.0

    def _score_to_risk(self, score: float) -> str:
        if score > 1.0:
            return "critical"
        if score >= 0.5:
            return "high"
        if score >= 0.2:
            return "medium"
        return "low"
