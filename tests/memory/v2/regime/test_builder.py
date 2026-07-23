"""Regime builder tests (PR-MEM-6, v4 §13)."""

from __future__ import annotations

from rosclaw.memory.v2.regime import (
    CurrentRegimeBuilder,
    RegimeLabel,
    TelemetrySample,
)

T0 = 1_700_000_000.0


def _samples(
    n: int,
    *,
    temp: float = 48.0,
    temp_step: float = 0.0,
    pos_err: float = 5.0,
    ttr_ms: float = 400.0,
    invalid_every: int = 0,
    interval: float = 2.5,
    spacing: float = 8.0,
) -> list[TelemetrySample]:
    samples = []
    for i in range(n):
        samples.append(
            TelemetrySample(
                timestamp=T0 + i * spacing,
                temperature_c=temp + temp_step * i,
                position_error=pos_err,
                time_to_reach_ms=ttr_ms,
                invalid=bool(invalid_every) and i % invalid_every == invalid_every - 1,
                action_count=2,
                gesture_interval_sec=interval,
                evidence_ref=f"evt_{i}",
            )
        )
    return samples


def test_regime_builder_cold_healthy() -> None:
    builder = CurrentRegimeBuilder()
    regime = builder.build(
        _samples(12, temp=48.0, temp_step=0.01),
        robot_id="r1",
        body_id="rh56_right_01",
        task_id="rh56_rps",
        session_started_at=T0,
        now=T0 + 200,
    )
    assert regime.regime_label == RegimeLabel.COLD_HEALTHY.value
    assert regime.temperature_c == 48.11
    assert regime.recent_invalid_rate == 0.0
    assert regime.missing_features == []
    assert 0.5 < regime.confidence <= 1.0
    assert regime.evidence_refs


def test_regime_builder_thermal_drift() -> None:
    builder = CurrentRegimeBuilder()
    # 0.3 °C/min sustained over the short window, tracking still fine.
    regime = builder.build(
        _samples(10, temp=50.0, temp_step=0.6, spacing=120.0),
        robot_id="r1",
        body_id="rh56_right_01",
        session_started_at=T0,
        now=T0 + 1200,
    )
    assert regime.regime_label == RegimeLabel.THERMAL_DRIFT.value
    assert regime.temperature_slope_c_per_min is not None
    assert regime.temperature_slope_c_per_min >= 0.15


def test_regime_builder_thermal_tracking_degradation() -> None:
    builder = CurrentRegimeBuilder()
    regime = builder.build(
        _samples(10, temp=57.0, pos_err=22.0, invalid_every=3),
        robot_id="r1",
        body_id="rh56_right_01",
        now=T0 + 200,
    )
    assert regime.regime_label == RegimeLabel.THERMAL_TRACKING_DEGRADATION.value


def test_regime_missing_temperature_not_a_match() -> None:
    """v4 §4.4/§13: missing temperature is unknown, never 'temperature ok'."""
    samples = [
        TelemetrySample(
            timestamp=T0 + i * 8.0,
            temperature_c=None,
            position_error=5.0,
            time_to_reach_ms=400.0,
            action_count=2,
            gesture_interval_sec=2.5,
        )
        for i in range(10)
    ]
    regime = CurrentRegimeBuilder().build(
        samples, robot_id="r1", body_id="rh56_right_01", now=T0 + 200
    )
    assert "temperature_c" in regime.missing_features
    assert regime.temperature_c is None
    # The label is computed from remaining evidence (COLD_HEALTHY here) but
    # with reduced confidence — and the missing feature is explicit.
    assert regime.regime_label == RegimeLabel.COLD_HEALTHY.value
    full = CurrentRegimeBuilder().build(
        _samples(10), robot_id="r1", body_id="rh56_right_01", now=T0 + 200
    )
    assert regime.confidence < full.confidence


def test_regime_unknown_on_insufficient_samples() -> None:
    regime = CurrentRegimeBuilder().build(
        _samples(1), robot_id="r1", body_id="rh56_right_01", now=T0 + 10
    )
    assert regime.regime_label == RegimeLabel.UNKNOWN.value
    assert regime.confidence == 0.0


def test_regime_communication_degraded() -> None:
    samples = _samples(10)
    for sample in samples[:3]:
        object.__setattr__(sample, "comm_error", True)
    regime = CurrentRegimeBuilder().build(
        samples, robot_id="r1", body_id="rh56_right_01", now=T0 + 200
    )
    # 3/10 comm errors ≥ 0.02 threshold
    assert regime.regime_label == RegimeLabel.COMMUNICATION_DEGRADED.value


def test_thresholds_from_yaml() -> None:
    from rosclaw.memory.v2.regime import load_thresholds

    thresholds = load_thresholds("configs/regimes/rh56_rps_v1.yaml")
    assert thresholds.temperature_warm_c == 52.0
    assert thresholds.temperature_hot_c == 56.0
    assert thresholds.short_window_rounds == 10
    regime = CurrentRegimeBuilder(thresholds).build(
        _samples(10, temp=53.0),
        robot_id="r1",
        body_id="rh56_right_01",
        now=T0 + 200,
    )
    assert regime.regime_label == RegimeLabel.WARM_STABLE.value


def test_unknown_threshold_keys_rejected(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("regime_thresholds:\n  not_a_threshold: 1.0\n")
    from rosclaw.memory.v2.regime import load_thresholds

    try:
        load_thresholds(str(bad))
    except ValueError as exc:
        assert "not_a_threshold" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown keys must be rejected")
