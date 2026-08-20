from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw.growth.episodic_contact_evaluation import (
    G1EpisodicContactCaseEvaluation,
    G1EpisodicContactEvaluation,
)
from rosclaw.growth.episodic_contact_memory import (
    G1EpisodicContactMemory,
    G1EpisodicContactPrototype,
    derive_g1_episodic_contact_memory,
    g1_episodic_contact_effect,
    load_g1_episodic_contact_memory,
)


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _memory() -> G1EpisodicContactMemory:
    hashes = tuple(_sha(f"probe-{index}") for index in range(24))
    base_observation = (-0.13, 0.0, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return G1EpisodicContactMemory(
        body_hash=_sha("body"),
        implementation_hash=_sha("implementation"),
        experiment_context_hash=_sha("context"),
        source_evidence_hashes=hashes,
        prototypes=(
            G1EpisodicContactPrototype(
                planner_seed_audit_label=0,
                observation=base_observation,
                goal_plane_dynamics_weight_matrix=(
                    (0.0, 0.2, 1.0),
                    (0.01, 0.0, 0.0),
                    (0.0, 0.01, 0.0),
                ),
                supported_goal_plane_polygon_yz_m=(
                    (0.0, 0.2),
                    (1.0, 0.2),
                    (1.0, 1.2),
                    (0.0, 1.2),
                ),
                source_evidence_hashes=hashes[:8],
                contact_regime="AIRBORNE",
                minimum_lateral_force_n=0.0,
                maximum_lateral_force_n=100.0,
                minimum_vertical_force_n=0.0,
                maximum_vertical_force_n=100.0,
                goal_plane_fit_rmse_m=0.01,
                arrival_time_fit_rmse_sec=0.01,
                maximum_target_prediction_error_m=0.05,
            ),
            G1EpisodicContactPrototype(
                planner_seed_audit_label=7,
                observation=(
                    -0.03,
                    *base_observation[1:],
                ),
                goal_plane_dynamics_weight_matrix=(
                    (0.0, 0.2, 1.5),
                    (0.02, 0.0, 0.0),
                    (0.0, 0.02, 0.0),
                ),
                supported_goal_plane_polygon_yz_m=(
                    (0.0, 0.2),
                    (2.0, 0.2),
                    (2.0, 2.2),
                    (0.0, 2.2),
                ),
                source_evidence_hashes=hashes[8:16],
                contact_regime="BOUNCE",
                minimum_lateral_force_n=0.0,
                maximum_lateral_force_n=100.0,
                minimum_vertical_force_n=0.0,
                maximum_vertical_force_n=100.0,
                goal_plane_fit_rmse_m=0.01,
                arrival_time_fit_rmse_sec=0.01,
                maximum_target_prediction_error_m=0.05,
            ),
        ),
        rejected_context_seed_labels=(21,),
        observation_feature_scales=(
            0.05,
            0.05,
            0.05,
            0.5,
            0.5,
            0.5,
            0.3,
            0.3,
            0.3,
            0.15,
            0.15,
        ),
        maximum_context_distance=0.50,
        minimum_prototype_distance=0.60,
        maximum_foot_ball_distance_m=0.18,
        start_policy_frame=230,
        end_policy_frame=335,
        foot_strike_point_offset_m=(0.13, 0.0, -0.025),
        ridge_regularization=0.05,
        safe_probe_count=16,
        rejected_probe_count=8,
        training_target_count=1,
    )


def test_episodic_memory_routes_by_state_and_fails_closed(monkeypatch) -> None:
    memory = _memory()

    def fake_jac(_model, _data, jacobian, _rotation, _point, _body_id) -> None:
        jacobian[1, 6] = 0.5
        jacobian[2, 7] = 0.25

    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mj_jac=fake_jac))
    model = SimpleNamespace(nv=35)
    data = SimpleNamespace(
        xmat=np.asarray((np.eye(3), np.eye(3))),
        xpos=np.zeros((2, 3), dtype=np.float64),
        xquat=np.asarray(((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))),
        cvel=np.zeros((2, 6), dtype=np.float64),
        qvel=np.zeros(35, dtype=np.float64),
    )
    ball = np.asarray((0.13, 0.0, -0.025))
    active = g1_episodic_contact_effect(
        model=model,
        data=data,
        right_ankle_body_id=0,
        torso_body_id=1,
        memory=memory,
        policy_frame=255,
        contact_observed=False,
        ball_position=ball,
        ball_velocity=np.zeros(3, dtype=np.float64),
        goal_plane_x_m=1.13,
        target_y_m=0.15,
        target_z_m=1.2,
    )

    assert active.active
    assert active.selected_context_seed_label == 0
    assert active.context_distance == pytest.approx(0.0)
    assert active.lateral_force_n == pytest.approx(15.0)
    assert active.vertical_force_n == pytest.approx(100.0)
    assert active.predicted_goal_y_m == pytest.approx(0.15)
    assert active.predicted_goal_z_m == pytest.approx(1.2)
    assert active.predicted_arrival_time_sec == pytest.approx(1.0)
    np.testing.assert_allclose(active.torque[:2], (7.5, 25.0))

    unsupported_target = g1_episodic_contact_effect(
        model=model,
        data=data,
        right_ankle_body_id=0,
        torso_body_id=1,
        memory=memory,
        policy_frame=255,
        contact_observed=False,
        ball_position=ball,
        ball_velocity=np.zeros(3, dtype=np.float64),
        goal_plane_x_m=1.13,
        target_y_m=1.5,
        target_z_m=1.5,
    )
    assert not unsupported_target.active
    assert not unsupported_target.target_envelope_supported
    assert not np.any(unsupported_target.torque)

    data.cvel[0, 3] = 2.0
    rejected = g1_episodic_contact_effect(
        model=model,
        data=data,
        right_ankle_body_id=0,
        torso_body_id=1,
        memory=memory,
        policy_frame=255,
        contact_observed=False,
        ball_position=ball,
        ball_velocity=np.zeros(3, dtype=np.float64),
        goal_plane_x_m=1.13,
        target_y_m=0.15,
        target_z_m=1.2,
    )
    assert not rejected.active
    assert not rejected.context_supported
    assert not np.any(rejected.torque)


def test_episodic_memory_cannot_promote_or_overlap() -> None:
    memory = _memory()
    with pytest.raises(ValueError, match="must remain unpromoted SIM_ONLY"):
        replace(memory, promotion_authorized=True)
    with pytest.raises(ValueError, match="overlap"):
        replace(memory, minimum_prototype_distance=0.49)


def test_episodic_evaluation_cannot_claim_generalization_or_promotion() -> None:
    case = G1EpisodicContactCaseEvaluation(
        planner_seed_audit_label=0,
        baseline_evidence_hash=_sha("baseline-0"),
        candidate_evidence_hash=_sha("candidate-0"),
        baseline_goal_plane_target_error_m=0.4,
        candidate_goal_plane_target_error_m=0.08,
        absolute_error_improvement_m=0.32,
        candidate_active_frames=2,
        selected_context_seed_label=0,
        peak_context_distance=0.1,
        candidate_physically_safe=True,
        precision_improved=True,
    )
    evaluation = G1EpisodicContactEvaluation(
        memory_hash=_sha("memory"),
        memory_artifact_hash=_sha("memory-artifact"),
        evaluation_implementation_hash=_sha("evaluation"),
        body_hash=_sha("body"),
        implementation_hash=_sha("implementation"),
        cases=(case, replace(case, planner_seed_audit_label=7, selected_context_seed_label=7)),
        supported_prototype_count=2,
        active_context_coverage_complete=True,
        stability_anchor_evidence_hash=_sha("anchor"),
        stability_anchor_seed_audit_label=21,
        stability_anchor_out_of_support_frames=2,
        stability_anchor_preserved=True,
        mean_baseline_error_m=0.4,
        mean_candidate_error_m=0.08,
        mean_absolute_error_improvement_m=0.32,
        improved_context_count=2,
        physically_safe_context_count=2,
        development_breakthrough=True,
        verdict="DEVELOPMENT",
    )

    with pytest.raises(ValueError, match="must remain unpromoted SIM_ONLY"):
        replace(evaluation, promotion_authorized=True)
    with pytest.raises(ValueError, match="verdict disagrees"):
        replace(evaluation, verdict="REJECTED")


def test_derive_episodic_memory_binds_supported_and_rejected_contexts(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    evidence_paths: list[Path] = []
    force_pairs = (
        (0.0, 0.0),
        (-30.0, 0.0),
        (30.0, 0.0),
        (0.0, -20.0),
        (0.0, 20.0),
        (-20.0, 15.0),
        (20.0, -15.0),
        (10.0, 10.0),
    )
    for seed_index, seed in enumerate((0, 7, 21)):
        for probe_index, (lateral, vertical) in enumerate(force_pairs):
            teacher = probe_index != 0
            active = np.asarray((False, teacher, False), dtype=np.bool_)
            trajectory = tmp_path / f"trajectory-{seed}-{probe_index}.npz"
            right_foot = np.tile(np.asarray((0.0 + 0.12 * seed_index, 0.0, 0.0)), (3, 1))
            ball_pose = np.zeros((3, 7), dtype=np.float64)
            ball_pose[:, :3] = (0.13, 0.0, 0.11)
            pre_action = np.zeros((3, 11), dtype=np.float64)
            pre_action[1, :3] = right_foot[1] - ball_pose[1, :3]
            np.savez_compressed(
                trajectory,
                loft_teacher_active=active,
                loft_teacher_pre_action_observation=pre_action,
                loft_teacher_pre_action_observation_valid=active,
                loft_teacher_lateral_force_n=np.asarray((0.0, lateral, 0.0)),
                loft_teacher_force_n=np.asarray((0.0, vertical, 0.0)),
                right_foot_position=right_foot,
                right_foot_linear_velocity=np.zeros((3, 3), dtype=np.float64),
                ball_pose=ball_pose,
                pelvis_velocity=np.zeros((3, 6), dtype=np.float64),
                torso_quaternion=np.tile(np.asarray((1.0, 0.0, 0.0, 0.0)), (3, 1)),
                time=np.asarray((4.0, 4.5, 5.0)),
                goal_crossing=np.asarray((False, False, True)),
            )
            safe = seed != 21
            evidence = {
                "strict_replay": True,
                "claims": {"sim_only_operational_space_loft_teacher": teacher},
                "trajectory_path": str(trajectory),
                "trajectory_hash": _file_hash(trajectory),
                "body_hash": _sha("body"),
                "implementation_hash": _sha("implementation"),
                "flow_config": {
                    "schema_version": "flow.v1",
                    "approach_provider": "sonic_fullbody",
                    "episodic_contact_memory_hash": None,
                    "shot_loft_teacher_target_vy_mps": 0.0 if not teacher else 10.0,
                    "shot_loft_teacher_target_vz_mps": 0.0 if not teacher else 7.0,
                    "shot_loft_teacher_max_foot_ball_distance_m": (0.0 if not teacher else 0.18),
                    "shot_loft_teacher_start_policy_frame": 230,
                    "shot_loft_teacher_end_policy_frame": 335,
                },
                "goal_spec": {
                    "plane_x_m": 8.5,
                    "target_y_m": 1.0,
                    "target_z_m": 1.35,
                    "ball_radius_m": 0.11,
                },
                "runup_config": {"start_x_m": -3.4},
                "sonic_runup_config": {"planner_seed": seed},
                "approach_strike_candidate_hash": None,
                "result": {
                    "ball_launch_velocity_xyz_mps": [
                        9.0,
                        1.2 + 0.02 * lateral + 0.001 * vertical,
                        4.0 + 0.001 * lateral + 0.03 * vertical,
                    ],
                    "goal_crossing_xyz_m": [
                        8.5,
                        1.2 + 0.02 * lateral + 0.001 * vertical,
                        0.5 + 0.001 * lateral + 0.03 * vertical,
                    ],
                    "contact_time_sec": 4.5,
                    "loft_teacher_executed": teacher,
                    "ballistic_contact_impulse_actor_executed": False,
                    "kick_contact_observed": True,
                    "perceptual_continuity_passed": True,
                    "post_kick_fall": False,
                    "joint_limit_violation": False,
                    "torque_limit_violation": False,
                    "actuator_saturation": False,
                    "torque_authority_projection_qualified": True,
                    "contact_task_authority_scale_min": 1.0 if safe else 0.5,
                },
            }
            path = tmp_path / f"evidence-{seed}-{probe_index}.json"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            evidence_paths.append(path)

    output = tmp_path / "memory.json"
    memory = derive_g1_episodic_contact_memory(
        evidence_paths=tuple(evidence_paths),
        output_path=output,
        source_checkout=checkout,
    )
    loaded = load_g1_episodic_contact_memory(output)

    assert memory.memory_hash == loaded.memory_hash
    assert tuple(item.planner_seed_audit_label for item in memory.prototypes) == (0, 7)
    assert memory.rejected_context_seed_labels == (21,)
    assert memory.safe_probe_count == 16
    assert memory.rejected_probe_count == 8
    assert all(item.contact_regime == "ROLLING" for item in memory.prototypes)
    assert all(len(item.supported_goal_plane_polygon_yz_m) >= 3 for item in memory.prototypes)
