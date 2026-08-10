from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw.growth.ballistic_contact_impulse_actor import (
    G1BallisticContactImpulseActor,
    derive_g1_ballistic_contact_impulse_actor,
    g1_ballistic_contact_impulse_context_hash,
    g1_ballistic_contact_impulse_effect,
    load_g1_ballistic_contact_impulse_actor,
)


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _actor() -> G1BallisticContactImpulseActor:
    hashes = tuple(_sha(f"probe-{index}") for index in range(8))
    return G1BallisticContactImpulseActor(
        body_hash=_sha("body"),
        implementation_hash=_sha("implementation"),
        experiment_context_hash=_sha("context"),
        source_evidence_hashes=hashes,
        selected_evidence_hash=hashes[0],
        selected_goal_plane_target_error_m=0.12,
        precision_success_count=2,
        rejected_probe_count=6,
        task_space_actor_weight_matrix=((400.0, -40.0, 0.0), (350.0, 0.0, -50.0)),
        maximum_lateral_force_n=250.0,
        maximum_vertical_force_n=250.0,
        maximum_foot_ball_distance_m=0.18,
        start_policy_frame=230,
        end_policy_frame=335,
        foot_strike_point_offset_m=(0.13, 0.0, -0.025),
        qualified_error_max_m=0.16,
    )


def test_impulse_actor_outputs_bounded_direct_joint_torque(monkeypatch) -> None:
    actor = _actor()

    def fake_jac(_model, _data, jacobian, _rotation, _point, _body_id) -> None:
        jacobian[1, 6] = 0.5
        jacobian[1, 7] = -0.25
        jacobian[2, 8] = 0.4

    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mj_jac=fake_jac))
    model = SimpleNamespace(nv=35)
    data = SimpleNamespace(
        xmat=np.asarray([np.eye(3)]),
        xpos=np.asarray([[0.0, 0.0, 0.0]]),
        qvel=np.zeros(35, dtype=np.float64),
    )

    effect = g1_ballistic_contact_impulse_effect(
        model=model,
        data=data,
        right_ankle_body_id=0,
        actor=actor,
        policy_frame=255,
        contact_observed=False,
        ball_position=np.asarray((0.13, 0.0, -0.025)),
    )

    assert effect.active
    assert effect.lateral_force_n == 250.0
    assert effect.vertical_force_n == 250.0
    np.testing.assert_allclose(effect.torque[:3], (125.0, -62.5, 100.0))
    assert np.count_nonzero(effect.torque) == 3


def test_impulse_actor_cannot_authorize_promotion_or_duplicate_support() -> None:
    actor = _actor()

    with pytest.raises(ValueError, match="must remain SIM_ONLY"):
        replace(actor, promotion_authorized=True)
    with pytest.raises(ValueError, match="must be unique"):
        replace(
            actor,
            source_evidence_hashes=(actor.source_evidence_hashes[0],) * 8,
        )


def test_impulse_actor_context_ignores_teacher_but_binds_stable_prior() -> None:
    context = {
        "flow_config": {
            "schema_version": "flow.v1",
            "ballistic_contact_impulse_actor_hash": None,
            "shot_loft_teacher_target_vz_mps": 7.0,
            "football_motion_prior_hash": _sha("stable-prior"),
        },
        "goal_spec": {"target_y_m": 1.0, "target_z_m": 1.35},
        "runup_config": {"start_x_m": -3.4},
        "sonic_runup_config": {"planner_seed": 0},
        "approach_strike_candidate_hash": _sha("candidate"),
    }
    baseline = g1_ballistic_contact_impulse_context_hash(**context)
    teacher_and_runtime = {
        **context,
        "flow_config": {
            **context["flow_config"],
            "schema_version": "flow.v2",
            "ballistic_contact_impulse_actor_hash": _sha("actor"),
            "shot_loft_teacher_target_vz_mps": 0.0,
        },
    }
    changed_prior = {
        **teacher_and_runtime,
        "flow_config": {
            **teacher_and_runtime["flow_config"],
            "football_motion_prior_hash": _sha("plastic-prior"),
        },
    }
    changed_post_contact_recovery = {
        **teacher_and_runtime,
        "flow_config": {
            **teacher_and_runtime["flow_config"],
            "shared_cerebellar_recovery_enabled": True,
            "shot_recovery_step_length_m": 0.0,
            "shot_recovery_step_yaw_rad": 0.0,
            "post_contact_damping_delay_sec": 0.05,
            "post_contact_damping_ramp_sec": 0.2,
        },
    }

    assert g1_ballistic_contact_impulse_context_hash(**teacher_and_runtime) == baseline
    assert g1_ballistic_contact_impulse_context_hash(**changed_post_contact_recovery) == baseline
    assert g1_ballistic_contact_impulse_context_hash(**changed_prior) != baseline


def test_impulse_actor_derivation_binds_success_and_failure_evidence(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    evidence_paths: list[Path] = []
    for index in range(8):
        trajectory = tmp_path / f"trajectory-{index}.npz"
        np.savez_compressed(trajectory, state=np.asarray((index,), dtype=np.float64))
        evidence = {
            "strict_replay": True,
            "claims": {"sim_only_operational_space_loft_teacher": True},
            "trajectory_path": str(trajectory),
            "trajectory_hash": _file_hash(trajectory),
            "body_hash": _sha("body"),
            "implementation_hash": _sha("implementation"),
            "flow_config": {
                "schema_version": "flow.v1",
                "approach_provider": "sonic_fullbody",
                "shot_loft_teacher_target_vy_mps": 10.0,
                "shot_loft_teacher_lateral_gain_n_per_mps": 40.0,
                "shot_loft_teacher_max_lateral_force_n": 250.0,
                "shot_loft_teacher_target_vz_mps": 7.0,
                "shot_loft_teacher_gain_n_per_mps": 50.0,
                "shot_loft_teacher_max_force_n": 250.0,
                "shot_loft_teacher_max_foot_ball_distance_m": 0.18,
                "shot_loft_teacher_start_policy_frame": 230,
                "shot_loft_teacher_end_policy_frame": 335,
            },
            "goal_spec": {"target_y_m": 1.0, "target_z_m": 1.35},
            "runup_config": {"start_x_m": -3.4},
            "sonic_runup_config": {"planner_seed": 0},
            "approach_strike_candidate_hash": _sha("candidate"),
            "result": {
                "goal_plane_target_error_m": 0.10 + 0.02 * index,
                "precision_radius_m": 0.16,
                "kick_contact_observed": True,
                "goal_mouth_hit": True,
                "perceptual_continuity_passed": True,
                "post_kick_fall": False,
                "joint_limit_violation": False,
                "torque_limit_violation": False,
            },
        }
        path = tmp_path / f"evidence-{index}.json"
        path.write_text(json.dumps(evidence), encoding="utf-8")
        evidence_paths.append(path)

    output = tmp_path / "actor.json"
    actor = derive_g1_ballistic_contact_impulse_actor(
        evidence_paths=tuple(evidence_paths),
        output_path=output,
        source_checkout=checkout,
    )
    loaded = load_g1_ballistic_contact_impulse_actor(output)

    assert actor.actor_hash == loaded.actor_hash
    assert actor.precision_success_count == 4
    assert actor.rejected_probe_count == 4
    assert actor.task_space_actor_weight_matrix == (
        (400.0, -40.0, 0.0),
        (350.0, 0.0, -50.0),
    )
