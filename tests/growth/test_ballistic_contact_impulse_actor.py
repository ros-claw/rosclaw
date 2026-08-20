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


def _target_conditioned_actor() -> G1BallisticContactImpulseActor:
    hashes = tuple(_sha(f"target-probe-{index}") for index in range(8))
    return G1BallisticContactImpulseActor(
        body_hash=_sha("body"),
        implementation_hash=_sha("implementation"),
        experiment_context_hash=_sha("context"),
        source_evidence_hashes=hashes,
        selected_evidence_hash=hashes[0],
        selected_goal_plane_target_error_m=0.12,
        precision_success_count=2,
        rejected_probe_count=6,
        task_space_actor_weight_matrix=(
            (0.0, 10.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 20.0, 0.0, 0.0),
        ),
        maximum_lateral_force_n=250.0,
        maximum_vertical_force_n=250.0,
        maximum_foot_ball_distance_m=0.18,
        start_policy_frame=230,
        end_policy_frame=335,
        foot_strike_point_offset_m=(0.13, 0.0, -0.025),
        qualified_error_max_m=0.16,
        reference_forward_ball_speed_mps=10.0,
        minimum_supported_lateral_launch_speed_mps=1.0,
        maximum_supported_lateral_launch_speed_mps=2.0,
        minimum_supported_vertical_launch_speed_mps=4.0,
        maximum_supported_vertical_launch_speed_mps=6.0,
        forward_dynamics_fit_rmse_mps=0.05,
        ridge_regularization=0.05,
        safe_probe_count=6,
        training_target_count=2,
        schema_version="rosclaw.growth.g1_ballistic_contact_impulse_actor.v2",
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


def test_target_conditioned_actor_uses_ball_state_and_fails_outside_envelope(
    monkeypatch,
) -> None:
    actor = _target_conditioned_actor()

    def fake_jac(_model, _data, jacobian, _rotation, _point, _body_id) -> None:
        jacobian[1, 6] = 0.5
        jacobian[2, 7] = 0.25

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
        ball_velocity=np.zeros(3, dtype=np.float64),
        goal_plane_x_m=1.13,
        target_y_m=0.15,
        target_z_m=0.426,
    )

    assert effect.active
    assert effect.target_conditioned
    assert effect.desired_lateral_launch_speed_mps == pytest.approx(1.5)
    assert effect.desired_vertical_launch_speed_mps == pytest.approx(5.0005)
    assert effect.lateral_force_n == pytest.approx(15.0)
    assert effect.vertical_force_n == pytest.approx(100.01)
    np.testing.assert_allclose(effect.torque[:2], (7.5, 25.0025))

    unsupported = g1_ballistic_contact_impulse_effect(
        model=model,
        data=data,
        right_ankle_body_id=0,
        actor=actor,
        policy_frame=255,
        contact_observed=False,
        ball_position=np.asarray((0.13, 0.0, -0.025)),
        ball_velocity=np.zeros(3, dtype=np.float64),
        goal_plane_x_m=1.13,
        target_y_m=0.40,
        target_z_m=0.426,
    )
    assert not unsupported.active
    assert unsupported.target_conditioned
    assert unsupported.desired_lateral_launch_speed_mps == pytest.approx(4.0)
    assert not np.any(unsupported.torque)

    negative_unseen_action = replace(
        actor,
        task_space_actor_weight_matrix=(
            (-100.0, 0.0, 0.0, 0.0, 0.0),
            (-100.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )
    clipped = g1_ballistic_contact_impulse_effect(
        model=model,
        data=data,
        right_ankle_body_id=0,
        actor=negative_unseen_action,
        policy_frame=255,
        contact_observed=False,
        ball_position=np.asarray((0.13, 0.0, -0.025)),
        ball_velocity=np.zeros(3, dtype=np.float64),
        goal_plane_x_m=1.13,
        target_y_m=0.15,
        target_z_m=0.426,
    )
    assert clipped.launch_envelope_supported
    assert clipped.lateral_force_n == 0.0
    assert clipped.vertical_force_n == 0.0
    assert not clipped.active


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


def test_target_conditioned_context_binds_goal_geometry_but_not_goal_point() -> None:
    context = {
        "flow_config": {"approach_provider": "sonic_fullbody"},
        "goal_spec": {
            "plane_x_m": 8.5,
            "width_m": 7.32,
            "height_m": 2.44,
            "target_y_m": 1.32,
            "target_z_m": 1.04,
        },
        "runup_config": {"start_x_m": -3.4},
        "sonic_runup_config": {"planner_seed": 0},
        "approach_strike_candidate_hash": None,
        "target_conditioned": True,
    }
    baseline = g1_ballistic_contact_impulse_context_hash(**context)
    moved_target = {
        **context,
        "goal_spec": {**context["goal_spec"], "target_y_m": 1.6, "target_z_m": 1.2},
    }
    moved_plane = {
        **context,
        "goal_spec": {**context["goal_spec"], "plane_x_m": 9.0},
    }

    assert g1_ballistic_contact_impulse_context_hash(**moved_target) == baseline
    assert g1_ballistic_contact_impulse_context_hash(**moved_plane) != baseline


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
                "contact_task_authority_scale_min": 0.5 if index == 1 else 1.0,
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
    assert actor.precision_success_count == 3
    assert actor.rejected_probe_count == 5
    assert actor.task_space_actor_weight_matrix == (
        (400.0, -40.0, 0.0),
        (350.0, 0.0, -50.0),
    )


def test_target_conditioned_derivation_fits_bound_inverse_dynamics(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    evidence_paths: list[Path] = []
    for index in range(8):
        trajectory = tmp_path / f"target-trajectory-{index}.npz"
        teacher_executed = index != 0
        active = np.asarray((False, teacher_executed), dtype=np.bool_)
        lateral_force = 0.0 if not teacher_executed else 20.0 + 15.0 * index
        vertical_force = 0.0 if not teacher_executed else 30.0 + 12.0 * (index % 4)
        np.savez_compressed(
            trajectory,
            loft_teacher_active=active,
            loft_teacher_lateral_force_n=np.asarray((0.0, lateral_force)),
            loft_teacher_force_n=np.asarray((0.0, vertical_force)),
        )
        launch_y = 1.20 + 0.07 * index
        launch_z = 4.60 + 0.04 * index + 0.08 * (index % 3)
        evidence = {
            "strict_replay": True,
            "claims": {"sim_only_operational_space_loft_teacher": teacher_executed},
            "trajectory_path": str(trajectory),
            "trajectory_hash": _file_hash(trajectory),
            "body_hash": _sha("body"),
            "implementation_hash": _sha("implementation"),
            "flow_config": {
                "schema_version": "flow.v1",
                "approach_provider": "sonic_fullbody",
                "shot_loft_teacher_target_vy_mps": 2.0 + index,
                "shot_loft_teacher_lateral_gain_n_per_mps": 35.0,
                "shot_loft_teacher_max_lateral_force_n": 250.0,
                "shot_loft_teacher_target_vz_mps": 4.0 + 0.2 * index,
                "shot_loft_teacher_gain_n_per_mps": 50.0,
                "shot_loft_teacher_max_force_n": 250.0,
                "shot_loft_teacher_max_foot_ball_distance_m": 0.18,
                "shot_loft_teacher_start_policy_frame": 230,
                "shot_loft_teacher_end_policy_frame": 335,
            },
            "goal_spec": {
                "plane_x_m": 8.5,
                "width_m": 7.32,
                "height_m": 2.44,
                "target_y_m": 1.32 if index < 4 else 1.45,
                "target_z_m": 1.04 if index < 4 else 1.20,
            },
            "runup_config": {"start_x_m": -3.4},
            "sonic_runup_config": {"planner_seed": 0},
            "approach_strike_candidate_hash": _sha("candidate"),
            "result": {
                "goal_plane_target_error_m": 0.10 + 0.03 * index,
                "precision_radius_m": 0.16,
                "ball_launch_velocity_xyz_mps": [9.2 + 0.02 * index, launch_y, launch_z],
                "loft_teacher_executed": teacher_executed,
                "ballistic_contact_impulse_actor_executed": False,
                "kick_contact_observed": True,
                "goal_mouth_hit": True,
                "perceptual_continuity_passed": True,
                "post_kick_fall": False,
                "joint_limit_violation": False,
                "torque_limit_violation": False,
                "contact_task_authority_scale_min": 1.0,
            },
        }
        path = tmp_path / f"target-evidence-{index}.json"
        path.write_text(json.dumps(evidence), encoding="utf-8")
        evidence_paths.append(path)

    output = tmp_path / "target-actor.json"
    actor = derive_g1_ballistic_contact_impulse_actor(
        evidence_paths=tuple(evidence_paths),
        output_path=output,
        source_checkout=checkout,
        target_conditioned=True,
        ridge_regularization=0.05,
    )
    loaded = load_g1_ballistic_contact_impulse_actor(output)

    assert actor.actor_hash == loaded.actor_hash
    assert actor.schema_version.endswith(".v2")
    assert actor.safe_probe_count == 8
    assert actor.training_target_count == 2
    assert actor.precision_success_count == 3
    assert actor.rejected_probe_count == 5
    assert np.asarray(actor.task_space_actor_weight_matrix).shape == (2, 5)
    assert actor.minimum_supported_lateral_launch_speed_mps < 1.20
    assert actor.maximum_supported_vertical_launch_speed_mps > max(
        4.60 + 0.04 * index + 0.08 * (index % 3) for index in range(8)
    )
