"""Regime change detector tests (PR-MEM-6, v4 §13)."""

from __future__ import annotations

from rosclaw.memory.v2.regime import (
    CurrentRegimeBuilder,
    RegimeChangeDetector,
    RegimeThresholds,
    TelemetrySample,
)
from tests.memory.v2.regime.test_builder import T0


def _regime_at(label: str, temp: float, ts: float):
    return CurrentRegimeBuilder(
        RegimeThresholds(short_window_rounds=5, medium_window_rounds=10)
    ).build(
        [
            TelemetrySample(
                timestamp=ts - i * 8.0,
                temperature_c=temp,
                position_error=5.0,
                time_to_reach_ms=400.0,
                action_count=2,
                gesture_interval_sec=2.5,
                evidence_ref=f"evt_{i}",
            )
            for i in range(5)
        ],
        robot_id="r1",
        body_id="rh56_right_01",
        now=ts,
    )


def test_detector_confirms_transition_after_persistence() -> None:
    detector = RegimeChangeDetector(persistence=3)
    assert detector.observe(_regime_at("COLD_HEALTHY", 48.0, T0)) is None
    # Single flip — not yet confirmed.
    assert detector.observe(_regime_at("WARM_STABLE", 53.0, T0 + 60)) is None
    assert detector.observe(_regime_at("WARM_STABLE", 53.2, T0 + 120)) is None
    transition = detector.observe(_regime_at("WARM_STABLE", 53.4, T0 + 180))
    assert transition is not None
    assert transition.from_label == "COLD_HEALTHY"
    assert transition.to_label == "WARM_STABLE"
    assert transition.changed_features  # temperature moved
    old, new = transition.changed_features["temperature_c"]
    assert new > old


def test_detector_ignores_single_round_flip() -> None:
    detector = RegimeChangeDetector(persistence=3)
    detector.observe(_regime_at("COLD_HEALTHY", 48.0, T0))
    detector.observe(_regime_at("WARM_STABLE", 53.0, T0 + 60))  # 1
    detector.observe(_regime_at("COLD_HEALTHY", 48.0, T0 + 120))  # back
    detector.observe(_regime_at("COLD_HEALTHY", 48.0, T0 + 180))
    detector.observe(_regime_at("COLD_HEALTHY", 48.0, T0 + 240))
    # No confirmed transition — the flip never persisted.
    assert detector._pending_label is None


def test_bocd_shadow_annotates_but_never_gates() -> None:
    detector = RegimeChangeDetector(persistence=2, enable_bocd_shadow=True)
    detector.observe(_regime_at("COLD_HEALTHY", 48.0, T0))
    detector.observe(_regime_at("WARM_STABLE", 53.0, T0 + 60))
    transition = detector.observe(_regime_at("WARM_STABLE", 53.2, T0 + 120))
    assert transition is not None
    assert transition.bocd_shadow is not None
    assert "change_probability" in transition.bocd_shadow
    assert transition.bocd_shadow.get("note") == "shadow_only_not_bayesian"
