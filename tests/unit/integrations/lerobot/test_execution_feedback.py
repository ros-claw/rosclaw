"""P5 §13.4 feedback verifier tests."""

from __future__ import annotations

from pathlib import Path

from rosclaw.body.rh56.calibration import load_rh56_calibration
from rosclaw.body.rh56.transport import RH56Feedback
from rosclaw.body.rh56.transport_profile import load_transport_profile
from rosclaw.integrations.lerobot.execution import FeedbackVerifier

CONFIGS = Path(__file__).resolve().parents[4] / "configs"


def _verifier() -> FeedbackVerifier:
    return FeedbackVerifier(
        load_transport_profile(CONFIGS / "rh56_right_rs485_v1.yaml"),
        load_rh56_calibration(CONFIGS / "rh56_right_01_calibration.yaml"),
    )


def _feedback(**overrides) -> RH56Feedback:
    base = RH56Feedback.zero(6)
    base.position = [1000] * 6
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_position_reached() -> None:
    v = _verifier()
    feedback = _feedback(position=[1000, 1000, 1000, 1000, 1000, 1000])
    result = v.verify(target=[1000.0] * 6, feedback=feedback, force_limit_g=100.0)
    assert result.position_reached
    assert v.is_step_ok(result)


def test_position_timeout() -> None:
    v = _verifier()
    # index stuck 200 raw away from target (tolerance 25).
    feedback = _feedback(position=[1000, 1000, 1000, 800, 1000, 1000])
    result = v.verify(target=[1000.0] * 6, feedback=feedback, force_limit_g=100.0)
    assert not result.position_reached
    assert any("position_error:index" in d for d in result.details)
    assert not v.is_step_ok(result)


def test_force_soft_limit() -> None:
    v = _verifier()
    feedback = _feedback(force_g=[0, 0, 0, 150.0, 0, 0])
    result = v.verify(target=[1000.0] * 6, feedback=feedback, force_limit_g=100.0)
    assert result.force_safe  # soft limit is advisory, not a hard failure
    assert any("force_soft_limit:index" in d for d in result.details)


def test_force_hard_limit() -> None:
    v = _verifier()
    feedback = _feedback(force_g=[0, 0, 0, 350.0, 0, 0])
    result = v.verify(target=[1000.0] * 6, feedback=feedback, force_limit_g=100.0)
    assert not result.force_safe
    assert any("force_hard_limit:index" in d for d in result.details)
    assert not v.is_step_ok(result)


def test_status_protection_estop() -> None:
    v = _verifier()
    feedback = _feedback(status_bits=[0, 0, 0x01, 0, 0, 0])
    result = v.verify(target=[1000.0] * 6, feedback=feedback, force_limit_g=100.0)
    assert not result.fault_free
    assert any("status_protection:middle" in d for d in result.details)
    assert not v.is_step_ok(result)


def test_temperature_warning() -> None:
    v = _verifier()
    feedback = _feedback(temperature_c=[32.0, 32.0, 32.0, 56.0, 32.0, 32.0])
    result = v.verify(target=[1000.0] * 6, feedback=feedback, force_limit_g=100.0)
    assert result.temperature_safe  # warning is advisory
    assert any("temperature_warning:index" in d for d in result.details)


def test_temperature_stop() -> None:
    v = _verifier()
    feedback = _feedback(temperature_c=[32.0, 32.0, 32.0, 61.0, 32.0, 32.0])
    result = v.verify(target=[1000.0] * 6, feedback=feedback, force_limit_g=100.0)
    assert not result.temperature_safe
    assert any("temperature_stop:index" in d for d in result.details)
    assert not v.is_step_ok(result)


def test_current_spike() -> None:
    # Current is an auxiliary signal: a spike during motion must not fail the
    # step on its own, but it is recorded for the report.
    v = _verifier()
    feedback = _feedback(current_ma=[0, 0, 0, 900.0, 0, 0])
    result = v.verify(target=[1000.0] * 6, feedback=feedback, force_limit_g=100.0)
    assert v.is_step_ok(result)
