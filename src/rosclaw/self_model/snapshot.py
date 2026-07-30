"""SelfSnapshot: immutable pre-action self-state binding (v3 §8.7, PR-PE-4).

Every real action submission binds to a SelfSnapshot: who the body is,
how healthy it is, what it currently perceives, which regime it is in,
what it believes it can do, which model predicted for it, and how
uncertain that prediction was.  ``ActionGateway``'s body_snapshot_hash
upgrades to this hash — a mutated body/perception/model produces a
different hash, and stale snapshots stop matching reality (v3 §8.7).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rosclaw.practice.physical_observation import canonical_hash

SNAPSHOT_VERSION = "rosclaw.self_snapshot.v1"


@dataclass(frozen=True)
class SelfSnapshot:
    body_id: str
    body_hash: str
    health: dict[str, Any]
    perception_confidence: float | None
    regime_label: str | None
    capabilities: dict[str, str]
    forward_model_hash: str | None
    prediction_uncertainty: dict[str, float] | None
    active_policy_hash: str | None
    agency_summary: dict[str, int]
    sequence: int
    timestamp_ns: int = field(default_factory=time.time_ns)

    @property
    def snapshot_hash(self) -> str:
        return canonical_hash(self.to_dict(), prefix="selfsnap")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_VERSION,
            "body_id": self.body_id,
            "body_hash": self.body_hash,
            "health": self.health,
            "perception_confidence": self.perception_confidence,
            "regime_label": self.regime_label,
            "capabilities": self.capabilities,
            "forward_model_hash": self.forward_model_hash,
            "prediction_uncertainty": self.prediction_uncertainty,
            "active_policy_hash": self.active_policy_hash,
            "agency_summary": self.agency_summary,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
        }

    def mutation_report(self, other: SelfSnapshot) -> list[str]:
        """Which binding components changed (v3 §15 self-loop acceptance:
        body mutation invalidates the old snapshot)."""
        changed: list[str] = []
        for name in (
            "body_hash",
            "regime_label",
            "forward_model_hash",
            "active_policy_hash",
        ):
            if getattr(self, name) != getattr(other, name):
                changed.append(name)
        if self.snapshot_hash == other.snapshot_hash:
            return []
        if not changed and (
            self.health != other.health or self.perception_confidence != other.perception_confidence
        ):
            changed.append("health_or_perception")
        return changed


@dataclass(frozen=True)
class CalibratedUncertainty:
    """Prediction uncertainty with calibration bookkeeping (v3 §15:
    prediction uncertainty 有校准).  calibrated=False is the honest
    default until enough predicted-vs-measured pairs exist."""

    per_channel_std: dict[str, float]
    samples: int
    calibrated: bool
    coverage: float | None  # fraction of measured residuals inside ±2σ

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_channel_std": self.per_channel_std,
            "samples": self.samples,
            "calibrated": self.calibrated,
            "coverage": self.coverage,
        }


def calibrate_uncertainty(
    residuals: list[dict[str, float]],
    predicted_std: dict[str, float],
    *,
    min_samples: int = 30,
) -> CalibratedUncertainty:
    """Coverage calibration: with < min_samples the uncertainty is a
    declared PRIOR, never a claimed calibration."""
    coverage: float | None = None
    if residuals:
        inside = 0
        total = 0
        for residual in residuals:
            for channel, value in residual.items():
                std = predicted_std.get(channel)
                if std is None or std <= 0:
                    continue
                total += 1
                inside += int(abs(value) <= 2 * std)
        coverage = inside / total if total else None
    return CalibratedUncertainty(
        per_channel_std=dict(predicted_std),
        samples=len(residuals),
        calibrated=len(residuals) >= min_samples,
        coverage=coverage,
    )
