from __future__ import annotations

from dataclasses import fields

from rosclaw.simforge.g1_recovery_quality import (
    G1AbsoluteRecoveryThresholds,
    G1RecoveryQuality,
    evaluate_g1_absolute_recovery_gate,
)


def _quality(**changes: object) -> G1RecoveryQuality:
    values: dict[str, object] = {
        field.name: 0.0
        for field in fields(G1RecoveryQuality)
        if field.init and field.name != "schema_version"
    }
    values.update(
        {
            "post_contact_backward_reversal_m": 0.20,
            "post_contact_pelvis_path_length_m": 0.60,
            "post_contact_support_transition_count": 1,
            "terminal_stable_duration_sec": 0.70,
            "terminal_bilateral_support": True,
            "settling_time_sec": 1.20,
        }
    )
    values.update(changes)
    return G1RecoveryQuality(**values)  # type: ignore[arg-type]


def _result(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "success": True,
        "goal_crossed": True,
        "target_zone_hit": True,
        "ball_speed_mps": 10.1,
        "target_error_m": 0.20,
        "post_kick_fall": False,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "support_foot_slip_m": 0.02,
    }
    value.update(changes)
    return value


def test_absolute_gate_accepts_only_strong_safe_and_quick_recovery() -> None:
    gate = evaluate_g1_absolute_recovery_gate(
        quality=_quality(),
        result=_result(),
        strict_replay=True,
    )

    assert gate.passed is True
    assert gate.reasons == ()
    assert gate.to_dict()["evidence_domain"] == "SIM_ONLY"
    assert gate.to_dict()["hardware_authorized"] is False


def test_absolute_gate_tolerates_only_roundoff_at_frozen_boundary() -> None:
    rounded = evaluate_g1_absolute_recovery_gate(
        quality=_quality(settling_time_sec=1.5 + 2.5e-13),
        result=_result(),
        strict_replay=True,
    )
    real_regression = evaluate_g1_absolute_recovery_gate(
        quality=_quality(settling_time_sec=1.5 + 1e-6),
        result=_result(),
        strict_replay=True,
    )

    assert rounded.passed
    assert not real_regression.passed


def test_absolute_gate_rejects_wandering_even_when_goal_is_good() -> None:
    gate = evaluate_g1_absolute_recovery_gate(
        quality=_quality(
            post_contact_backward_reversal_m=1.194,
            post_contact_pelvis_path_length_m=1.999,
            settling_time_sec=5.518,
        ),
        result=_result(),
        strict_replay=True,
    )

    assert gate.goal_quality_passed is True
    assert gate.physical_recovery_passed is False
    assert gate.passed is False
    assert gate.reasons == (
        "backward_reversal_above_absolute_gate",
        "pelvis_path_above_absolute_gate",
        "settling_time_above_absolute_gate",
    )


def test_absolute_gate_fails_closed_on_missing_or_nonfinite_metrics() -> None:
    result = _result()
    result.pop("support_foot_slip_m")
    gate = evaluate_g1_absolute_recovery_gate(
        quality=_quality(settling_time_sec=None),
        result=result,
        strict_replay=False,
    )

    assert gate.passed is False
    assert "settling_time_missing" in gate.reasons
    assert "safety_below_absolute_gate" in gate.reasons
    assert "strict_replay_missing" in gate.reasons


def test_absolute_thresholds_reject_invalid_configuration() -> None:
    try:
        G1AbsoluteRecoveryThresholds(maximum_pelvis_path_m=float("nan"))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite thresholds must fail closed")
