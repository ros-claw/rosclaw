"""Synchronization configuration for Gate B.1 source-stream resampling.

This module lives in the ROSClaw core Python and must not import torch or
lerobot.  It maps each feature to a resampling method and its constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SyncMethod = Literal[
    "linear",
    "previous",
    "nearest",
    "interval_mean",
    "interval_any",
]


@dataclass
class SyncPolicy:
    """Per-feature synchronization policy."""

    method: SyncMethod = "previous"
    max_gap_ms: float | None = None
    max_skew_ms: float | None = None
    max_age_ms: float | None = None
    emit_peak: bool = False
    emit_peak_abs: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": self.method}
        if self.max_gap_ms is not None:
            out["max_gap_ms"] = self.max_gap_ms
        if self.max_skew_ms is not None:
            out["max_skew_ms"] = self.max_skew_ms
        if self.max_age_ms is not None:
            out["max_age_ms"] = self.max_age_ms
        if self.emit_peak:
            out["emit_peak"] = True
        if self.emit_peak_abs:
            out["emit_peak_abs"] = True
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncPolicy:
        return cls(
            method=data.get("method", "previous"),  # type: ignore[arg-type]
            max_gap_ms=data.get("max_gap_ms"),
            max_skew_ms=data.get("max_skew_ms"),
            max_age_ms=data.get("max_age_ms"),
            emit_peak=bool(data.get("emit_peak", False)),
            emit_peak_abs=bool(data.get("emit_peak_abs", False)),
        )


@dataclass
class SyncConfig:
    """Complete synchronization configuration for one export."""

    target_fps: float = 10.0
    policies: dict[str, SyncPolicy] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_fps": self.target_fps,
            "policies": {k: v.to_dict() for k, v in self.policies.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncConfig:
        raw_policies = data.get("policies", {})
        return cls(
            target_fps=float(data.get("target_fps", 10.0)),
            policies={k: SyncPolicy.from_dict(v) for k, v in raw_policies.items()},
        )

    def policy_for(self, feature_key: str) -> SyncPolicy:
        return self.policies.get(feature_key, SyncPolicy(method="previous"))


def default_sync_config(fps: float = 10.0) -> SyncConfig:
    """Return the built-in default synchronization policies."""
    return SyncConfig(
        target_fps=fps,
        policies={
            "observation.state": SyncPolicy(method="linear", max_gap_ms=100.0),
            "observation.joint_velocity": SyncPolicy(method="linear", max_gap_ms=100.0),
            "observation.joint_effort": SyncPolicy(method="nearest", max_skew_ms=50.0),
            "action": SyncPolicy(method="previous", max_age_ms=200.0),
            "observation.motor_current": SyncPolicy(
                method="interval_mean", emit_peak_abs=True
            ),
            "observation.joint_temperature": SyncPolicy(
                method="previous", max_age_ms=1000.0
            ),
            "observation.force_torque": SyncPolicy(
                method="interval_mean", emit_peak_abs=True
            ),
            "observation.contact": SyncPolicy(method="interval_any"),
            "observation.images.front": SyncPolicy(method="nearest", max_skew_ms=50.0),
            "rosclaw.sandbox.decision": SyncPolicy(method="previous"),
            "rosclaw.failure.active": SyncPolicy(method="interval_any"),
            "rosclaw.intervention.active": SyncPolicy(method="interval_any"),
        },
    )


__all__ = [
    "SyncConfig",
    "SyncMethod",
    "SyncPolicy",
    "default_sync_config",
]
