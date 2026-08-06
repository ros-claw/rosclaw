from __future__ import annotations

from rosclaw.simforge.g1_recovery_awr_validation import (
    _expanded_recovery_scenarios,
    _fresh_recovery_validation_scenarios,
    _recovery_action_indices,
    _task_preserved_dict,
    _value_convergence,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES


def _result(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "success": True,
        "kick_foot_contacted": True,
        "target_error_m": 0.20,
        "ball_speed_mps": 8.0,
        "support_foot_slip_m": 0.01,
        "post_kick_fall": False,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "finite_state": True,
    }
    value.update(changes)
    return value


def test_matched_task_gate_preserves_shot_and_rejects_hidden_regression() -> None:
    passed, reasons = _task_preserved_dict(
        _result(),
        _result(target_error_m=0.24, ball_speed_mps=7.3),
    )
    failed, failure_reasons = _task_preserved_dict(
        _result(),
        _result(success=False, target_error_m=0.30, post_kick_fall=True),
    )

    assert passed
    assert reasons == ()
    assert not failed
    assert "new_post_kick_fall" in failure_reasons
    assert "parent_success_lost" in failure_reasons
    assert "target_error_regressed" in failure_reasons

    inherited, inherited_reasons = _task_preserved_dict(
        _result(success=False, joint_limit_violation=True),
        _result(success=False, joint_limit_violation=True, target_error_m=0.19),
    )
    assert inherited
    assert inherited_reasons == ()


def test_recovery_readout_excludes_kick_yaw_and_distal_wrists() -> None:
    names = {G1_DDS_JOINT_NAMES[index] for index in _recovery_action_indices()}

    assert "left_hip_roll_joint" in names
    assert "right_knee_joint" in names
    assert "waist_roll_joint" in names
    assert "left_shoulder_pitch_joint" in names
    assert "right_hip_yaw_joint" not in names
    assert "right_wrist_yaw_joint" not in names


def test_value_convergence_and_fresh_validation_contracts() -> None:
    updates = tuple({"value_loss": 1.0 - 0.08 * index} for index in range(10))
    scenarios = _fresh_recovery_validation_scenarios(3)

    assert _value_convergence(updates)["converged"]
    assert len(scenarios) == 4
    assert len({item.scenario_commitment for item in scenarios}) == 4
    assert {item.partition.value for item in scenarios} == {"validation", "holdout"}
    training = _expanded_recovery_scenarios("training")
    development = _expanded_recovery_scenarios("development")
    assert len(training) == len(development) == 4
    assert not {item.scenario_commitment for item in training}.intersection(
        item.scenario_commitment for item in development
    )
