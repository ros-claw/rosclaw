from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.growth.ballistic_contact_impulse_actor import (
    G1BallisticContactImpulseActor,
    g1_ballistic_contact_impulse_context_hash,
)
from rosclaw.growth.target_conditioned_contact_evaluation import (
    evaluate_g1_target_conditioned_contact_actor,
)


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_conditioned_evaluation_requires_improvement_and_preserved_anchor(
    tmp_path: Path,
) -> None:
    flow = {"approach_provider": "sonic_fullbody"}
    goal = {
        "plane_x_m": 8.5,
        "width_m": 7.32,
        "height_m": 2.44,
        "target_y_m": 2.1,
        "target_z_m": 0.8,
    }
    runup = {"start_x_m": -3.4}
    sonic = {"planner_seed": 0}
    context_hash = g1_ballistic_contact_impulse_context_hash(
        flow_config=flow,
        goal_spec=goal,
        runup_config=runup,
        sonic_runup_config=sonic,
        approach_strike_candidate_hash=None,
        target_conditioned=True,
    )
    hashes = tuple(_sha(f"probe-{index}") for index in range(8))
    actor = G1BallisticContactImpulseActor(
        body_hash=_sha("body"),
        implementation_hash=_sha("implementation"),
        experiment_context_hash=context_hash,
        source_evidence_hashes=hashes,
        selected_evidence_hash=hashes[0],
        selected_goal_plane_target_error_m=0.4,
        precision_success_count=0,
        rejected_probe_count=8,
        task_space_actor_weight_matrix=((0.0,) * 5, (0.0,) * 5),
        maximum_lateral_force_n=30.0,
        maximum_vertical_force_n=30.0,
        minimum_lateral_force_n=0.0,
        minimum_vertical_force_n=0.0,
        maximum_foot_ball_distance_m=0.18,
        start_policy_frame=230,
        end_policy_frame=335,
        foot_strike_point_offset_m=(0.13, 0.0, -0.025),
        qualified_error_max_m=0.1,
        reference_forward_ball_speed_mps=9.0,
        minimum_supported_lateral_launch_speed_mps=2.0,
        maximum_supported_lateral_launch_speed_mps=3.0,
        minimum_supported_vertical_launch_speed_mps=4.0,
        maximum_supported_vertical_launch_speed_mps=5.0,
        forward_dynamics_fit_rmse_mps=0.01,
        ridge_regularization=0.05,
        safe_probe_count=5,
        training_target_count=1,
        schema_version="rosclaw.growth.g1_ballistic_contact_impulse_actor.v2",
    )
    actor_path = tmp_path / "actor.json"
    actor_path.write_text(json.dumps(actor.to_dict()), encoding="utf-8")

    safe = {
        "kick_contact_observed": True,
        "goal_mouth_hit": True,
        "perceptual_continuity_passed": True,
        "post_kick_fall": False,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "torque_authority_projection_qualified": True,
        "contact_task_authority_scale_min": 1.0,
        "post_contact_backward_displacement_m": 0.0,
        "precision_radius_m": 0.1,
    }

    def evidence(name: str, result: dict[str, object], *, anchor: bool = False) -> Path:
        trajectory = tmp_path / f"{name}.npz"
        np.savez_compressed(trajectory, state=np.asarray((1.0,)))
        path = tmp_path / f"{name}.json"
        evidence_goal = {**goal}
        if anchor:
            evidence_goal.update(target_y_m=1.3, target_z_m=1.0)
        evidence_flow = {
            **flow,
            "ballistic_contact_impulse_actor_hash": (
                None
                if result["ballistic_contact_impulse_actor_executed"] is False
                else actor.actor_hash
            ),
        }
        path.write_text(
            json.dumps(
                {
                    "strict_replay": True,
                    "trajectory_path": str(trajectory),
                    "trajectory_hash": _file_hash(trajectory),
                    "body_hash": actor.body_hash,
                    "implementation_hash": actor.implementation_hash,
                    "flow_config": evidence_flow,
                    "goal_spec": evidence_goal,
                    "runup_config": runup,
                    "sonic_runup_config": sonic,
                    "approach_strike_candidate_hash": None,
                    "result": {**safe, **result},
                }
            ),
            encoding="utf-8",
        )
        return path

    baseline = evidence(
        "baseline",
        {
            "goal_plane_target_error_m": 0.05,
            "ballistic_contact_impulse_actor_executed": False,
        },
    )
    candidate = evidence(
        "candidate",
        {
            "goal_plane_target_error_m": 0.02,
            "ballistic_contact_impulse_actor_executed": True,
            "ballistic_contact_impulse_actor_target_conditioned": True,
            "ballistic_contact_impulse_actor_active_frames": 1,
        },
    )
    anchor = evidence(
        "anchor",
        {
            "goal_plane_target_error_m": 0.8,
            "ballistic_contact_impulse_actor_executed": True,
            "ballistic_contact_impulse_actor_target_conditioned": True,
            "ballistic_contact_impulse_actor_active_frames": 0,
            "ballistic_contact_impulse_actor_out_of_envelope_frames": 2,
        },
        anchor=True,
    )
    output = tmp_path / "evaluation.json"
    evaluation = evaluate_g1_target_conditioned_contact_actor(
        actor_path=actor_path,
        baseline_evidence_path=baseline,
        candidate_evidence_path=candidate,
        stability_anchor_evidence_path=anchor,
        output_path=output,
        source_checkout=tmp_path / "checkout",
    )

    assert evaluation.verdict == "DEVELOPMENT"
    assert evaluation.development_breakthrough
    assert evaluation.relative_error_improvement == pytest.approx(0.6)
    assert not evaluation.promotion_authorized
    assert json.loads(output.read_text())["evaluation_hash"] == evaluation.evaluation_hash
