"""Prediction residual encoding (v3 §8.5, PR-PE-4).

After every action, compare predicted vs measured per channel.  Named
residual channels:

    joint_state / visual_pose / contact_outcome / force_peak /
    control_latency / thermal / perception / task_performance

A single residual NEVER triggers learning (v3 §8.5); escalation follows
the existing PredictionMonitor's persistence thresholds
(NORMAL → SUSPECTED_SHIFT → CONFIRMED_SHIFT → …), with RH56-recalibrated
thresholds — never G1 parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .protocols import ResidualEncoder, SelfObservation, SelfPrediction

RESIDUALS_VERSION = "rosclaw.prediction_residuals.v1"


@dataclass(frozen=True)
class PredictionResiduals:
    channels: dict[str, float]
    per_channel: dict[str, dict[str, float]]
    max_abs: float
    norm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "residuals_version": RESIDUALS_VERSION,
            "channels": self.channels,
            "per_channel": self.per_channel,
            "max_abs": round(self.max_abs, 4),
            "norm": round(self.norm, 4),
        }


# RH56-recalibrated residual thresholds (raw units / physical units),
# initial values from the evo-rps campaign telemetry — recalibrated by
# residual evidence, never inherited from G1.
RH56_SHIFT_THRESHOLDS: dict[str, float] = {
    "joint_state": 60.0,  # raw units, ~4x typical first-order miss
    "thermal": 2.0,  # °C prediction miss
    "control_latency": 100.0,  # ms
    "visual_pose": 0.05,  # m
    "force_peak": 80.0,  # raw force units
    "task_performance": 0.15,  # invalid-rate delta
}


class ChannelResidualEncoder(ResidualEncoder):
    """Map prediction/observation channel names onto the §8.5 residual
    families and compute per-family magnitudes."""

    _FAMILY_PREFIX = {
        "joint_state": ("next_pos_", "tracking_error_"),
        "thermal": ("temp_",),
        "control_latency": ("latency_", "time_to_reach_"),
        "visual_pose": ("visual_",),
        "force_peak": ("force_",),
        "task_performance": ("invalid_rate", "task_"),
        "contact_outcome": ("contact_",),
        "perception": ("reliability", "confidence"),
    }

    def encode(self, prediction: SelfPrediction, observation: SelfObservation) -> dict[str, float]:
        residuals: dict[str, float] = {}
        for name, measured in observation.channels.items():
            predicted = prediction.channels.get(name)
            if predicted is None:
                continue  # unpredicted channels are disclosed, not zero
            residuals[name] = measured - predicted
        return residuals

    def encode_families(
        self, prediction: SelfPrediction, observation: SelfObservation
    ) -> PredictionResiduals:
        flat = self.encode(prediction, observation)
        families: dict[str, dict[str, float]] = {}
        for name, value in flat.items():
            family = "other"
            for label, prefixes in self._FAMILY_PREFIX.items():
                if any(name.startswith(prefix) for prefix in prefixes):
                    family = label
                    break
            families.setdefault(family, {})[name] = value
        channels = {
            family: max((abs(v) for v in values.values()), default=0.0)
            for family, values in families.items()
        }
        norm = math.sqrt(sum(v * v for v in flat.values())) if flat else 0.0
        return PredictionResiduals(
            channels=channels,
            per_channel=families,
            max_abs=max((abs(v) for v in flat.values()), default=0.0),
            norm=norm,
        )


def shift_flags(
    residuals: PredictionResiduals, thresholds: dict[str, float] | None = None
) -> dict[str, bool]:
    """Per-family shift flag (persistence/escalation belongs to the
    PredictionMonitor — this is only the per-sample signal)."""
    limits = thresholds or RH56_SHIFT_THRESHOLDS
    return {
        family: value > limits.get(family, float("inf"))
        for family, value in residuals.channels.items()
    }
