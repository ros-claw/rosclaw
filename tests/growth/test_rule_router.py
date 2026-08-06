from __future__ import annotations

from rosclaw.growth import (
    DataProfile,
    GrowthProblemSignals,
    RouteDisposition,
    route_learners,
)


def _transition_data(**changes: bool) -> DataProfile:
    values = {
        "has_state": True,
        "has_executed_action": True,
        "has_next_state": True,
        "has_reward_vector": True,
        "has_cost_vector": True,
        "fixed_dataset": True,
        "online_rollout_allowed": True,
    }
    values.update(changes)
    return DataProfile(**values)


def test_router_blocks_learning_when_safety_model_is_incomplete() -> None:
    route = route_learners(
        GrowthProblemSignals(local_physics_residual=1.0, safety_model_complete=False),
        _transition_data(),
    )

    assert route.disposition is RouteDisposition.BLOCKED_SAFETY_MODEL
    assert route.learner_ids == ()
    assert route.to_dict()["hardware_authorized"] is False


def test_router_uses_sysid_before_policy_for_regime_shift() -> None:
    route = route_learners(
        GrowthProblemSignals(regime_shift=0.9, repeated_error=0.9),
        DataProfile(),
    )

    assert route.disposition is RouteDisposition.SELECTED
    assert route.learner_ids == ("system_identification",)


def test_router_keeps_raw_kinematics_out_of_offline_rl() -> None:
    route = route_learners(
        GrowthProblemSignals(),
        DataProfile(has_kinematic_reference=True, fixed_dataset=True),
    )

    assert route.learner_ids == ("motion_tracking",)
    assert "iql" not in route.learner_ids


def test_router_can_stage_offline_then_online_residual_learning() -> None:
    route = route_learners(
        GrowthProblemSignals(local_physics_residual=0.8),
        _transition_data(),
    )

    assert route.learner_ids == ("iql", "residual_sac")
    assert route.route_hash.startswith("sha256:")


def test_incomplete_transition_data_reports_exact_missing_requirements() -> None:
    route = route_learners(
        GrowthProblemSignals(local_physics_residual=0.8),
        _transition_data(has_executed_action=False, has_cost_vector=False),
    )

    assert route.disposition is RouteDisposition.NEED_MORE_EVIDENCE
    assert route.learner_ids == ()
    assert route.missing_requirements == (
        "transition.executed_action",
        "transition.cost_vector",
    )


def test_router_combines_chunk_feedback_ood_and_expert_separation() -> None:
    route = route_learners(
        GrowthProblemSignals(out_of_distribution=0.8, gradient_conflict=0.9),
        DataProfile(has_chunk_feedback=True),
    )

    assert route.learner_ids == (
        "advantage_sft",
        "world_model_observer",
        "new_expert_adapter",
    )
