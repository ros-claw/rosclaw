from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from rosclaw.simforge.g1_moving_ball import MovingBallInterceptAdapter
from rosclaw.simforge.g1_moving_ball_balance import (
    G1MovingBallBalanceArtifact,
    load_g1_moving_ball_balance_artifact,
    serialize_g1_moving_ball_balance_artifact,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.scenario import generate_goalforge_scenarios


def _scenario():
    base = generate_goalforge_scenarios(
        ledger=SeedLedger(
            task_id="g1_penalty_kick",
            secret=b"moving-ball-adapter-unit-test",
        ),
        partition=Partition.VALIDATION,
        count=1,
        generation=0,
    )[0]
    return replace(
        base,
        ball_x_m=1.12,
        ball_velocity_x_mps=-0.08,
        ball_launch_delay_sec=4.0,
    )


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _balance_artifact() -> G1MovingBallBalanceArtifact:
    return G1MovingBallBalanceArtifact(
        body_hash=_digest("body"),
        motion_hash=_digest("motion"),
        recovery_config_hash=_digest("recovery"),
        training_dataset_hash=_digest("dataset"),
        com_shift_y_m=0.05,
        lateral_com_shift_y_m=0.04,
        minimum_lateral_offset_m=0.002,
        development_case_count=48,
        training_seed=20260802,
    )


def _velocity_balance_artifact() -> G1MovingBallBalanceArtifact:
    return replace(
        _balance_artifact(),
        com_shift_y_m=0.06,
        velocity_com_shift_y_m=0.05,
        maximum_velocity_context_mps=0.075,
        schema_version="rosclaw.g1_goalforge.moving_ball_balance_artifact.v3",
    )


def _large_lateral_balance_artifact() -> G1MovingBallBalanceArtifact:
    return replace(
        _velocity_balance_artifact(),
        large_lateral_com_shift_y_m=0.045,
        minimum_large_lateral_offset_m=0.006,
        schema_version="rosclaw.g1_goalforge.moving_ball_balance_artifact.v4",
    )


def test_moving_ball_adapter_plans_inside_bounded_intercept_envelope() -> None:
    plan = MovingBallInterceptAdapter().plan(_scenario())

    assert plan.eligible
    assert plan.predicted_ball_x_m == pytest.approx(1.0176)
    assert plan.nominal_contact_error_m < 0.02
    assert plan.parameters.policy_type == "parameter"


def test_moving_ball_adapter_rejects_unvalidated_fast_pass() -> None:
    scenario = replace(_scenario(), ball_velocity_x_mps=-0.45)
    plan = MovingBallInterceptAdapter().plan(scenario)

    assert not plan.eligible
    assert "ball_speed_outside_validated_envelope" in plan.reasons


def test_moving_ball_adapter_applies_only_compatible_balance_memory(tmp_path) -> None:
    artifact = _balance_artifact()
    adapter = MovingBallInterceptAdapter(
        artifact,
        expected_body_hash=_digest("body"),
        expected_motion_hash=_digest("motion"),
        expected_recovery_config_hash=_digest("recovery"),
    )

    plan = adapter.plan(_scenario())

    assert plan.parameters.com_shift_y == pytest.approx(0.05)
    assert plan.parameters.dataset_snapshot_hash == artifact.artifact_hash
    with pytest.raises(ValueError, match="Body hash"):
        MovingBallInterceptAdapter(
            artifact,
            expected_body_hash=_digest("other"),
            expected_motion_hash=_digest("motion"),
            expected_recovery_config_hash=_digest("recovery"),
        )
    path = tmp_path / "balance.json"
    path.write_bytes(serialize_g1_moving_ball_balance_artifact(artifact))
    assert (
        load_g1_moving_ball_balance_artifact(
            path,
            expected_body_hash=_digest("body"),
            expected_motion_hash=_digest("motion"),
            expected_recovery_config_hash=_digest("recovery"),
        ).artifact_hash
        == artifact.artifact_hash
    )
    lateral = adapter.plan(replace(_scenario(), ball_y_m=0.003))
    assert lateral.parameters.com_shift_y == pytest.approx(0.04)


def test_moving_ball_v3_uses_observable_velocity_context_before_nominal_shift() -> None:
    artifact = _velocity_balance_artifact()
    adapter = MovingBallInterceptAdapter(
        artifact,
        expected_body_hash=_digest("body"),
        expected_motion_hash=_digest("motion"),
        expected_recovery_config_hash=_digest("recovery"),
    )

    nominal = adapter.plan(_scenario())
    slow = adapter.plan(replace(_scenario(), ball_velocity_x_mps=-0.074))
    lateral = adapter.plan(replace(_scenario(), ball_velocity_x_mps=-0.074, ball_y_m=0.003))

    assert nominal.parameters.com_shift_y == pytest.approx(0.06)
    assert slow.parameters.com_shift_y == pytest.approx(0.05)
    assert lateral.parameters.com_shift_y == pytest.approx(0.04)
    assert artifact.com_shift_for(
        predicted_ball_y_m=0.0,
        predicted_ball_speed_mps=0.075,
    ) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="ball speed"):
        artifact.com_shift_for(predicted_ball_y_m=0.0)


def test_moving_ball_v3_balance_artifact_roundtrips(tmp_path) -> None:
    artifact = _velocity_balance_artifact()
    path = tmp_path / "balance-v3.json"
    path.write_bytes(serialize_g1_moving_ball_balance_artifact(artifact))

    loaded = load_g1_moving_ball_balance_artifact(
        path,
        expected_body_hash=_digest("body"),
        expected_motion_hash=_digest("motion"),
        expected_recovery_config_hash=_digest("recovery"),
    )

    assert loaded == artifact
    assert loaded.artifact_hash == artifact.artifact_hash
    assert "velocity_com_shift_y_m" not in _balance_artifact().to_dict()


def test_moving_ball_v4_uses_large_lateral_context_first() -> None:
    artifact = _large_lateral_balance_artifact()
    adapter = MovingBallInterceptAdapter(
        artifact,
        expected_body_hash=_digest("body"),
        expected_motion_hash=_digest("motion"),
        expected_recovery_config_hash=_digest("recovery"),
    )

    moderate = adapter.plan(replace(_scenario(), ball_y_m=0.004))
    large = adapter.plan(replace(_scenario(), ball_y_m=0.0065))

    assert moderate.parameters.com_shift_y == pytest.approx(0.04)
    assert large.parameters.com_shift_y == pytest.approx(0.045)
    with pytest.raises(ValueError, match="large-lateral threshold"):
        replace(artifact, minimum_large_lateral_offset_m=0.001)


def test_moving_ball_balance_artifact_rejects_unknown_or_unsafe_payload(tmp_path) -> None:
    artifact = _balance_artifact()
    value = artifact.to_dict()
    value["unknown"] = True
    path = tmp_path / "balance.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_g1_moving_ball_balance_artifact(
            path,
            expected_body_hash=_digest("body"),
            expected_motion_hash=_digest("motion"),
            expected_recovery_config_hash=_digest("recovery"),
        )
    with pytest.raises(ValueError, match="COM shift"):
        replace(artifact, com_shift_y_m=float("nan"))
