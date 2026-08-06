from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from rosclaw.growth import (
    ActionTraceCommitment,
    CandidateManifest,
    EvidenceLevel,
    ExperienceSegment,
    FailureSignature,
    LearningJob,
    LearningPlan,
    PhysicalAdvantageLabel,
)


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _action(*, projection_applied: bool = False) -> ActionTraceCommitment:
    executed = _hash("executed")
    return ActionTraceCommitment(
        commanded_action_hash=_hash("commanded"),
        executed_action_hash=executed,
        safety_projected_action_hash=_hash("projected") if projection_applied else executed,
        policy_version="champion.g8",
        controller_hash=_hash("controller"),
        projection_applied=projection_applied,
    )


def _failure() -> FailureSignature:
    return FailureSignature(
        primary_type="recovery.oscillation",
        contributors=("angular_momentum.residual",),
        confidence=0.91,
        affected_capability_ids=("g1.recover",),
        reusable_evidence_ids=("approach.good",),
        recommended_learner_ids=("residual.sac",),
    )


def _segment(
    *,
    label: PhysicalAdvantageLabel = PhysicalAdvantageLabel.ADVANTAGE_NEGATIVE,
    costs: dict[str, float] | None = None,
    failure: FailureSignature | None = None,
) -> ExperienceSegment:
    return ExperienceSegment(
        segment_id="segment.recovery.001",
        episode_id="episode.001",
        skill_id="g1.kick",
        phase="recovery",
        start_time_sec=2.0,
        end_time_sec=3.5,
        body_hash=_hash("body"),
        regime_hash=_hash("regime"),
        source_artifact_hash=_hash("trajectory"),
        source_evidence_level=EvidenceLevel.PHYSICS_REPLAY,
        base_policy_version="champion.g8",
        residual_policy_version="candidate.g9",
        state_start_hash=_hash("state-start"),
        observation_sequence_hash=_hash("observations"),
        self_state_hash=_hash("self"),
        world_state_hash=_hash("world"),
        action=_action(),
        reward_vector={"body.recovery": 0.4, "task.goal": 1.0},
        cost_vector=costs or {"safety.fall": 0.0, "safety.torque": 0.0},
        terminal_state_hash=_hash("terminal"),
        advantage_label=label,
        label_confidence=0.83,
        failure_signature=failure if failure is not None else _failure(),
    )


def test_experience_segment_commits_commanded_executed_and_projected_actions() -> None:
    segment = _segment()
    value = segment.to_dict()

    assert value["action"]["commanded_action_hash"] != value["action"]["executed_action_hash"]
    assert (
        value["action"]["executed_action_hash"] == value["action"]["safety_projected_action_hash"]
    )
    assert value["promotion_truth_allowed"] is True
    assert value["hardware_authorized"] is False
    assert segment.segment_hash == _segment().segment_hash


def test_advantage_labels_are_lexicographically_safe() -> None:
    with pytest.raises(ValueError, match="positive advantage"):
        _segment(
            label=PhysicalAdvantageLabel.ADVANTAGE_POSITIVE,
            costs={"safety.fall": 1.0},
            failure=_failure(),
        )
    with pytest.raises(ValueError, match="non-zero safety cost"):
        _segment(
            label=PhysicalAdvantageLabel.UNSAFE_NEGATIVE,
            costs={"safety.fall": 0.0},
        )
    with pytest.raises(ValueError, match="failure signature"):
        replace(_segment(), failure_signature=None)


def _job() -> LearningJob:
    return LearningJob(
        job_id="job.recovery.residual",
        learner_id="residual.sac",
        data_query_hash=_hash("query"),
        target_artifact_kind="residual.policy",
        required_field_ids=("state.proprioception", "action.executed", "outcome.reward"),
        trainable_component_ids=("recovery.adapter",),
    )


def _plan(**changes: object) -> LearningPlan:
    values: dict[str, object] = {
        "campaign_id": "campaign.g1.recovery.001",
        "skill_id": "g1.kick",
        "failure_cluster_hashes": (_hash("failure-cluster"),),
        "jobs": (_job(),),
        "frozen_component_ids": ("body.core", "safety.projector"),
        "trainable_component_ids": ("recovery.adapter",),
        "historical_anchor_bank_hash": _hash("anchors"),
        "boundary_suite_hash": _hash("boundaries"),
        "budget": {"gpu.hours": 8.0, "rollout.steps": 100_000.0},
        "promotion_profile_hash": _hash("promotion"),
    }
    values.update(changes)
    return LearningPlan(**values)  # type: ignore[arg-type]


def test_learning_plan_freezes_body_and_routes_only_declared_components() -> None:
    plan = _plan()

    assert plan.to_dict()["evidence_domain"] == "SIM_ONLY"
    assert plan.plan_hash == _plan().plan_hash
    with pytest.raises(ValueError, match="both frozen and trainable"):
        _plan(trainable_component_ids=("body.core", "recovery.adapter"))


def test_candidate_is_lineaged_but_never_self_authorizes_promotion() -> None:
    manifest = CandidateManifest(
        candidate_id="candidate.g1.recovery.001",
        skill_id="g1.kick",
        parent_artifact_hash=_hash("parent"),
        learning_plan_hash=_plan().plan_hash,
        code_hash=_hash("code"),
        environment_hash=_hash("environment"),
        body_hash=_hash("body"),
        training_data_hashes=(_hash("data"),),
        learned_artifacts={"recovery.adapter": _hash("weights")},
        learned_output_fraction=0.4,
    )

    assert manifest.to_dict()["promotion_truth_allowed"] is False
    assert manifest.to_dict()["hardware_authorized"] is False
    assert manifest.manifest_hash.startswith("sha256:")
