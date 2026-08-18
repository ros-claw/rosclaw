"""Operating regime data model (数据库优化v4 §4.2/§4.3).

An :class:`OperatingRegime` is a point-in-time description of the physical
working condition a robot is in — thermals, tracking quality, recent
failure rates, control timing, communication health.  Intervention memories
are only meaningful relative to the regime they were validated in; a memory
from a 56–58 °C two-hour thermal-degradation session must not fire in a
48–50 °C short healthy session.

Deterministic labels only (v4 §4.3) — no trained classifier.  Thresholds
live in ``configs/regimes/*.yaml``, never hardcoded here.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RegimeLabel(StrEnum):
    COLD_HEALTHY = "COLD_HEALTHY"
    WARM_STABLE = "WARM_STABLE"
    THERMAL_DRIFT = "THERMAL_DRIFT"
    TRACKING_DEGRADATION = "TRACKING_DEGRADATION"
    THERMAL_TRACKING_DEGRADATION = "THERMAL_TRACKING_DEGRADATION"
    COMMUNICATION_DEGRADED = "COMMUNICATION_DEGRADED"
    SENSOR_UNRELIABLE = "SENSOR_UNRELIABLE"
    CONTROL_TIMING_UNSTABLE = "CONTROL_TIMING_UNSTABLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class OperatingRegime:
    """The current operating regime of one body on one robot (v4 §4.2)."""

    regime_id: str
    timestamp: float

    tenant_id: str | None
    project_id: str | None
    site_id: str | None

    robot_id: str
    body_id: str
    hardware_revision: str | None
    firmware_version: str | None
    calibration_hash: str | None
    control_profile_hash: str | None

    task_id: str | None
    skill_id: str | None
    gesture_name: str | None
    joint_name: str | None

    session_elapsed_sec: float
    rounds_completed: int
    cumulative_action_count: int

    temperature_c: float | None
    temperature_slope_c_per_min: float | None
    ambient_temperature_c: float | None

    position_error_mean: float | None
    position_error_p95: float | None
    time_to_reach_mean_ms: float | None
    time_to_reach_p95_ms: float | None

    recent_invalid_rate: float | None
    recent_failure_rate: float | None
    consecutive_failures: int

    action_frequency_hz: float | None
    gesture_interval_sec: float | None
    control_phase: str | None

    communication_error_rate: float | None
    sensor_health: str | None

    regime_label: str
    confidence: float
    missing_features: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def feature_value(self, name: str) -> float | None:
        value = getattr(self, name, None)
        return float(value) if isinstance(value, (int, float)) else None


@dataclass(frozen=True)
class RegimeThresholds:
    """Deterministic labeling thresholds (loaded from configs/regimes/*.yaml).

    Every value is explicit configuration — the builder never invents one.
    """

    # Temperature bands (°C).
    temperature_warm_c: float = 52.0
    temperature_hot_c: float = 56.0
    # Thermal drift: sustained slope over a minimum window.
    temperature_slope_drift_c_per_min: float = 0.15
    temperature_slope_min_window_sec: float = 300.0
    # Tracking quality.
    position_error_p95_warn: float = 15.0
    time_to_reach_p95_warn_ms: float = 900.0
    # Recent outcome rates (rolling window).
    invalid_rate_warn: float = 0.06
    invalid_rate_high: float = 0.15
    failure_rate_warn: float = 0.10
    consecutive_failures_warn: int = 3
    # Control timing stability (coefficient of variation of gesture interval).
    gesture_interval_cv_warn: float = 0.25
    # Communication health.
    communication_error_rate_warn: float = 0.02
    # Windows (rounds).
    short_window_rounds: int = 10
    medium_window_rounds: int = 30
    # Confidence.
    min_feature_coverage: float = 0.5

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RegimeThresholds:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown regime threshold keys: {unknown}")
        return cls(**{key: raw[key] for key in raw if key in known})


def load_thresholds(path: str) -> RegimeThresholds:
    """Load thresholds from a YAML config (never from code defaults alone)."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("regime_thresholds", raw)
    return RegimeThresholds.from_dict(section)


def new_regime_id() -> str:
    import uuid

    return f"reg_{uuid.uuid4().hex[:16]}"


def empty_regime(
    *,
    robot_id: str,
    body_id: str,
    task_id: str | None = None,
    now: float | None = None,
) -> OperatingRegime:
    """An UNKNOWN regime with every feature missing — the honest baseline
    when no telemetry exists (unknown is not wildcard, v4 §4.4)."""
    return OperatingRegime(
        regime_id=new_regime_id(),
        timestamp=now if now is not None else time.time(),
        tenant_id=None,
        project_id=None,
        site_id=None,
        robot_id=robot_id,
        body_id=body_id,
        hardware_revision=None,
        firmware_version=None,
        calibration_hash=None,
        control_profile_hash=None,
        task_id=task_id,
        skill_id=None,
        gesture_name=None,
        joint_name=None,
        session_elapsed_sec=0.0,
        rounds_completed=0,
        cumulative_action_count=0,
        temperature_c=None,
        temperature_slope_c_per_min=None,
        ambient_temperature_c=None,
        position_error_mean=None,
        position_error_p95=None,
        time_to_reach_mean_ms=None,
        time_to_reach_p95_ms=None,
        recent_invalid_rate=None,
        recent_failure_rate=None,
        consecutive_failures=0,
        action_frequency_hz=None,
        gesture_interval_sec=None,
        control_phase=None,
        communication_error_rate=None,
        sensor_health=None,
        regime_label=RegimeLabel.UNKNOWN.value,
        confidence=0.0,
        missing_features=["all"],
        evidence_refs=[],
    )
