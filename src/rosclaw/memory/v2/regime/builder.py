"""CurrentRegimeBuilder — deterministic regime labeling (数据库优化v4 §4.4).

Inputs: rolling telemetry samples + identity context (robot/body/task/
calibration/control-profile hashes).  Output: one :class:`OperatingRegime`.

Labels (v4 §4.3), evaluated in strict priority order:

1. UNKNOWN — not enough data to say anything honest.
2. SENSOR_UNRELIABLE — sensor_health explicitly not ok.
3. COMMUNICATION_DEGRADED — comm error rate over threshold.
4. THERMAL_TRACKING_DEGRADATION — thermal abnormal AND tracking elevated.
5. THERMAL_DRIFT — sustained positive temperature slope, tracking still normal.
6. TRACKING_DEGRADATION — tracking elevated, thermal normal.
7. CONTROL_TIMING_UNSTABLE — gesture interval CV over threshold.
8. WARM_STABLE — warm but stable.
9. COLD_HEALTHY — the healthy default when all checks pass.

Missing features never count as "matching": a regime with no temperature
data is labeled on the remaining evidence with reduced confidence, and the
missing features are listed explicitly.
"""

from __future__ import annotations

import time
from typing import Any

from .features import TelemetrySample, compute_windows
from .models import (
    OperatingRegime,
    RegimeLabel,
    RegimeThresholds,
    new_regime_id,
)


class CurrentRegimeBuilder:
    """Builds the latest :class:`OperatingRegime` from rolling telemetry."""

    def __init__(self, thresholds: RegimeThresholds | None = None) -> None:
        self._t = thresholds or RegimeThresholds()

    def build(
        self,
        samples: list[TelemetrySample],
        *,
        robot_id: str,
        body_id: str,
        task_id: str | None = None,
        skill_id: str | None = None,
        gesture_name: str | None = None,
        joint_name: str | None = None,
        hardware_revision: str | None = None,
        firmware_version: str | None = None,
        calibration_hash: str | None = None,
        control_profile_hash: str | None = None,
        sensor_health: str | None = None,
        session_started_at: float | None = None,
        rounds_completed: int | None = None,
        now: float | None = None,
    ) -> OperatingRegime:
        now = now if now is not None else time.time()
        windows = compute_windows(
            samples,
            short=self._t.short_window_rounds,
            medium=self._t.medium_window_rounds,
        )
        short = windows["short"]
        long = windows["long"]

        missing = sorted(set(short.missing))
        label, fired = self._label(short, sensor_health, missing)

        coverage = self._coverage(missing)
        confidence = round(self._confidence(label, coverage, fired), 4)

        evidence_refs = [s.evidence_ref for s in samples[-short.sample_count :] if s.evidence_ref]
        return OperatingRegime(
            regime_id=new_regime_id(),
            timestamp=now,
            tenant_id=None,
            project_id=None,
            site_id=None,
            robot_id=robot_id,
            body_id=body_id,
            hardware_revision=hardware_revision,
            firmware_version=firmware_version,
            calibration_hash=calibration_hash,
            control_profile_hash=control_profile_hash,
            task_id=task_id,
            skill_id=skill_id,
            gesture_name=gesture_name,
            joint_name=joint_name,
            session_elapsed_sec=(max(now - session_started_at, 0.0) if session_started_at else 0.0),
            rounds_completed=(rounds_completed if rounds_completed is not None else len(samples)),
            cumulative_action_count=sum(s.action_count for s in samples),
            temperature_c=short.temperature_last,
            temperature_slope_c_per_min=short.temperature_slope_c_per_min,
            ambient_temperature_c=None,
            position_error_mean=short.position_error_mean,
            position_error_p95=short.position_error_p95,
            time_to_reach_mean_ms=short.time_to_reach_mean_ms,
            time_to_reach_p95_ms=short.time_to_reach_p95_ms,
            recent_invalid_rate=short.invalid_rate,
            recent_failure_rate=short.failure_rate,
            consecutive_failures=short.consecutive_failures,
            action_frequency_hz=long.action_frequency_hz,
            gesture_interval_sec=short.gesture_interval_mean_sec,
            control_phase=None,
            communication_error_rate=short.comm_error_rate,
            sensor_health=sensor_health,
            regime_label=label.value,
            confidence=confidence,
            missing_features=missing,
            evidence_refs=evidence_refs[-20:],
        )

    # ------------------------------------------------------------------

    def _label(
        self,
        short: Any,
        sensor_health: str | None,
        missing: list[str],
    ) -> tuple[RegimeLabel, list[str]]:
        t = self._t
        fired: list[str] = []

        if short.sample_count < 2:
            return RegimeLabel.UNKNOWN, ["insufficient_samples"]

        if sensor_health is not None and sensor_health not in ("ok", "healthy"):
            return RegimeLabel.SENSOR_UNRELIABLE, [f"sensor_health={sensor_health}"]

        if (
            short.comm_error_rate is not None
            and short.comm_error_rate >= t.communication_error_rate_warn
        ):
            return RegimeLabel.COMMUNICATION_DEGRADED, [
                f"comm_error_rate={short.comm_error_rate:.3f}>={t.communication_error_rate_warn}"
            ]

        thermal_abnormal = self._thermal_abnormal(short, missing, fired)
        tracking_elevated = self._tracking_elevated(short, fired)

        if thermal_abnormal and tracking_elevated:
            return RegimeLabel.THERMAL_TRACKING_DEGRADATION, fired
        if thermal_abnormal:
            return RegimeLabel.THERMAL_DRIFT, fired
        if tracking_elevated:
            return RegimeLabel.TRACKING_DEGRADATION, fired

        if (
            short.gesture_interval_cv is not None
            and short.gesture_interval_cv >= t.gesture_interval_cv_warn
        ):
            fired.append(
                f"gesture_interval_cv={short.gesture_interval_cv:.3f}>={t.gesture_interval_cv_warn}"
            )
            return RegimeLabel.CONTROL_TIMING_UNSTABLE, fired

        temp = short.temperature_last
        if temp is not None and temp >= t.temperature_warm_c:
            return RegimeLabel.WARM_STABLE, fired
        return RegimeLabel.COLD_HEALTHY, fired

    def _thermal_abnormal(
        self,
        short: Any,
        missing: list[str],
        fired: list[str],
    ) -> bool:
        t = self._t
        temp = short.temperature_last
        slope = short.temperature_slope_c_per_min
        abnormal = False
        if temp is not None and temp >= t.temperature_hot_c:
            fired.append(f"temperature={temp:.1f}>={t.temperature_hot_c}")
            abnormal = True
        # A slope verdict needs enough time behind it to be meaningful:
        # either the full configured window, or a whole short window of
        # samples when the session itself is younger than the window.
        slope_window_ok = (
            short.window_elapsed_sec >= t.temperature_slope_min_window_sec
            or short.sample_count >= t.short_window_rounds
        )
        if slope is not None and slope >= t.temperature_slope_drift_c_per_min and slope_window_ok:
            fired.append(f"temperature_slope={slope:.3f}>={t.temperature_slope_drift_c_per_min}")
            abnormal = True
        return abnormal

    def _tracking_elevated(self, short: Any, fired: list[str]) -> bool:
        t = self._t
        elevated = False
        if (
            short.position_error_p95 is not None
            and short.position_error_p95 >= t.position_error_p95_warn
        ):
            fired.append(
                f"position_error_p95={short.position_error_p95:.1f}>={t.position_error_p95_warn}"
            )
            elevated = True
        if (
            short.time_to_reach_p95_ms is not None
            and short.time_to_reach_p95_ms >= t.time_to_reach_p95_warn_ms
        ):
            fired.append(
                f"time_to_reach_p95={short.time_to_reach_p95_ms:.0f}>={t.time_to_reach_p95_warn_ms}"
            )
            elevated = True
        if short.invalid_rate is not None and short.invalid_rate >= t.invalid_rate_warn:
            fired.append(f"invalid_rate={short.invalid_rate:.3f}>={t.invalid_rate_warn}")
            elevated = True
        if short.failure_rate is not None and short.failure_rate >= t.failure_rate_warn:
            fired.append(f"failure_rate={short.failure_rate:.3f}>={t.failure_rate_warn}")
            elevated = True
        if short.consecutive_failures >= t.consecutive_failures_warn:
            fired.append(f"consecutive_failures={short.consecutive_failures}")
            elevated = True
        return elevated

    def _coverage(self, missing: list[str]) -> float:
        """Fraction of label-relevant features actually present."""
        if "all" in missing:
            return 0.0
        absent = set(missing)
        # WindowStats reports missing sample-level names; map them to the
        # regime features they feed.  recent_invalid_rate is always
        # computable when samples exist.
        name_map = {
            "temperature_c": "temperature_c",
            "temperature_slope_c_per_min": "temperature_slope_c_per_min",
            "position_error_p95": "position_error",
            "time_to_reach_p95_ms": "time_to_reach_ms",
            "recent_invalid_rate": None,
        }
        present = sum(
            1
            for sample_name in name_map.values()
            if sample_name is None or sample_name not in absent
        )
        return present / len(name_map)

    def _confidence(self, label: RegimeLabel, coverage: float, fired: list[str]) -> float:
        if label is RegimeLabel.UNKNOWN:
            return 0.0
        base = 0.4 + 0.6 * coverage
        # A label asserted on explicit rule evidence is more confident than
        # the healthy default (which is an absence-of-evidence judgment).
        if label in (RegimeLabel.COLD_HEALTHY, RegimeLabel.WARM_STABLE):
            base *= 0.9
        elif fired:
            base = min(1.0, base + 0.1)
        if coverage < self._t.min_feature_coverage:
            base *= coverage / max(self._t.min_feature_coverage, 1e-9)
        return max(0.0, min(1.0, base))
