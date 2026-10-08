from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rosclaw.growth import (
    ActionTraceCommitment,
    CandidateManifest,
    CoreGeneralizationGate,
    DerivedExperienceLineage,
    DomainValidationProof,
    EvidenceLevel,
    ExperienceSegment,
    FailureSignature,
    GeneralizationEvidence,
    GeneralizationStatus,
    LearningJob,
    LearningPlan,
    PhysicalAdvantageLabel,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _hash(value: str | bytes) -> str:
    payload = value.encode() if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    return _hash(path.read_bytes())


def _lineage(source_path: Path, event_values: list[dict[str, Any]]) -> DerivedExperienceLineage:
    event_hashes = tuple(
        _hash(json.dumps(event, sort_keys=True, separators=(",", ":"))) for event in event_values
    )
    return DerivedExperienceLineage(
        source_artifact_hash=_file_hash(source_path),
        source_event_hashes=event_hashes,
        transform_hash=_hash("task-neutral-segmenter.v1"),
        clock_id="monotonic.source",
        maximum_skew_sec=0.02,
        observed_skew_sec=0.01 if len(event_hashes) > 1 else 0.0,
        synchronization_receipt_hash=(
            _hash("alignment-receipt") if len(event_hashes) > 1 else None
        ),
    )


def _action(label: str) -> ActionTraceCommitment:
    executed = _hash(f"{label}.executed")
    return ActionTraceCommitment(
        commanded_action_hash=_hash(f"{label}.commanded"),
        executed_action_hash=executed,
        safety_projected_action_hash=executed,
        policy_version="baseline.v1",
        controller_hash=_hash(f"{label}.controller"),
        projection_applied=False,
    )


def _segment(
    *,
    segment_id: str,
    episode_id: str,
    skill_id: str,
    phase: str,
    lineage: DerivedExperienceLineage,
    failure_type: str,
) -> ExperienceSegment:
    failure = FailureSignature(
        primary_type=failure_type,
        contributors=("regime.shift",),
        confidence=0.91,
        affected_capability_ids=(skill_id,),
        reusable_evidence_ids=(),
        recommended_learner_ids=("system_identification",),
    )
    return ExperienceSegment(
        segment_id=segment_id,
        episode_id=episode_id,
        skill_id=skill_id,
        phase=phase,
        start_time_sec=0.1,
        end_time_sec=0.4,
        body_hash=_hash(f"{skill_id}.body"),
        regime_hash=_hash(f"{skill_id}.regime"),
        source_evidence_level=EvidenceLevel.PHYSICS_REPLAY,
        lineage=lineage,
        base_policy_version="baseline.v1",
        residual_policy_version=None,
        state_start_hash=_hash(f"{skill_id}.state"),
        observation_sequence_hash=_hash(f"{skill_id}.observations"),
        self_state_hash=_hash(f"{skill_id}.self"),
        world_state_hash=_hash(f"{skill_id}.world"),
        action=_action(skill_id),
        reward_vector={"task.progress": 0.4},
        cost_vector={"safety.violation": 0.0},
        terminal_state_hash=_hash(f"{skill_id}.terminal"),
        advantage_label=PhysicalAdvantageLabel.ADVANTAGE_NEGATIVE,
        label_confidence=0.82,
        failure_signature=failure,
    )


def test_same_experience_contract_accepts_rh56_and_arm_reach_evidence() -> None:
    fixture_path = REPOSITORY_ROOT / "tests/fixtures/practice/rh56_minimal_loop.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    hand_segment = _segment(
        segment_id="segment.hand.001",
        episode_id="episode.hand.001",
        skill_id="rh56.single_step",
        phase="contact.recovery",
        lineage=_lineage(fixture_path, fixture["events"][1:3]),
        failure_type="contact.force_overshoot",
    )

    reach_path = REPOSITORY_ROOT / "src/rosclaw/simforge/tasks/shield_reach.py"
    reach_segment = _segment(
        segment_id="segment.reach.001",
        episode_id="episode.reach.001",
        skill_id="arm.shield_reach",
        phase="boundary.approach",
        lineage=_lineage(reach_path, [{"case_id": "boundary.001", "risk": 0.82}]),
        failure_type="shield.boundary_error",
    )

    assert hand_segment.to_dict()["source"]["lineage"]["synchronization_receipt_hash"]
    assert hand_segment.segment_hash != reach_segment.segment_hash
    assert hand_segment.to_dict()["hardware_authorized"] is False
    assert reach_segment.to_dict()["promotion_truth_allowed"] is True


def test_multi_event_lineage_fails_closed_without_alignment_receipt() -> None:
    with pytest.raises(ValueError, match="synchronization receipt"):
        DerivedExperienceLineage(
            source_artifact_hash=_hash("source"),
            source_event_hashes=(_hash("event-a"), _hash("event-b")),
            transform_hash=_hash("transform"),
            clock_id="monotonic.source",
            maximum_skew_sec=0.01,
            observed_skew_sec=0.005,
        )


def test_physical_advantage_labels_preserve_safety_cost_semantics() -> None:
    fixture_path = REPOSITORY_ROOT / "tests/fixtures/practice/rh56_minimal_loop.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    segment = _segment(
        segment_id="segment.hand.002",
        episode_id="episode.hand.002",
        skill_id="rh56.single_step",
        phase="contact.recovery",
        lineage=_lineage(fixture_path, fixture["events"][1:2]),
        failure_type="contact.force_overshoot",
    )

    with pytest.raises(ValueError, match="positive advantage"):
        replace(
            segment,
            advantage_label=PhysicalAdvantageLabel.ADVANTAGE_POSITIVE,
            cost_vector={"safety.violation": 1.0},
        )


def _learning_plan() -> LearningPlan:
    job = LearningJob(
        job_id="job.regime.identification",
        learner_id="system_identification",
        data_query_hash=_hash("query"),
        target_artifact_kind="regime.model",
        required_field_ids=("state.self", "action.executed", "outcome.cost"),
        trainable_component_ids=("regime.adapter",),
    )
    return LearningPlan(
        campaign_id="campaign.cross_domain.001",
        skill_id="adaptive.control",
        failure_cluster_hashes=(_hash("failures"),),
        jobs=(job,),
        frozen_component_ids=("safety.projector",),
        trainable_component_ids=("regime.adapter",),
        historical_anchor_bank_hash=_hash("anchors"),
        boundary_suite_hash=_hash("boundaries"),
        budget={"rollout.steps": 1000.0},
        promotion_profile_hash=_hash("promotion"),
    )


def test_learning_plan_and_candidate_never_self_authorize_activation() -> None:
    plan = _learning_plan()
    candidate = CandidateManifest(
        candidate_id="candidate.cross_domain.001",
        skill_id=plan.skill_id,
        parent_artifact_hash=_hash("parent"),
        learning_plan_hash=plan.plan_hash,
        code_hash=_hash("code"),
        environment_hash=_hash("environment"),
        body_hash=_hash("body"),
        training_data_hashes=(_hash("data"),),
        learned_artifacts={"regime.adapter": _hash("weights")},
        learned_output_fraction=0.2,
    )

    assert plan.to_dict()["hardware_authorized"] is False
    assert candidate.to_dict()["promotion_truth_allowed"] is False
    assert candidate.to_dict()["hardware_authorized"] is False


def test_core_generalization_gate_requires_two_real_repository_domains() -> None:
    hand_fixture = REPOSITORY_ROOT / "tests/fixtures/practice/rh56_minimal_loop.json"
    reach_task = REPOSITORY_ROOT / "src/rosclaw/simforge/tasks/shield_reach.py"
    evidence = GeneralizationEvidence(
        component_id="growth.experience",
        source_paths=("src/rosclaw/growth/experience.py",),
        imported_modules=("rosclaw.feedback.contracts", "rosclaw.growth.contracts"),
        adapter_entry_point_group="rosclaw.growth.adapters",
        synthetic_test_report_hash=_hash("synthetic-tests"),
        plugin_removal_test_report_hash=_hash("core-without-plugins"),
        domain_proofs=(
            DomainValidationProof(
                domain_id="dexterous_manipulation",
                task_id="rh56.single_step",
                adapter_id="rh56.practice",
                evidence_hash=_file_hash(hand_fixture),
            ),
            DomainValidationProof(
                domain_id="arm_reach",
                task_id="arm.shield_reach",
                adapter_id="simforge.shield_reach",
                evidence_hash=_file_hash(reach_task),
            ),
        ),
    )
    gate = CoreGeneralizationGate(
        downstream_import_prefixes=("rosclaw_soccer",),
        domain_tokens=("football", "free_kick", "ballistic_contact", "g1_"),
    )

    decision = gate.evaluate(evidence)

    assert decision.status is GeneralizationStatus.PASSED
    assert decision.reasons == ()
    assert decision.hardware_authorized is False


def test_core_generalization_gate_rejects_one_domain_and_reverse_dependency() -> None:
    proof = DomainValidationProof(
        domain_id="dexterous_manipulation",
        task_id="rh56.single_step",
        adapter_id="rh56.practice",
        evidence_hash=_hash("evidence"),
    )
    evidence = GeneralizationEvidence(
        component_id="growth.experience",
        source_paths=("src/rosclaw/growth/experience.py",),
        imported_modules=("rosclaw_soccer.skills",),
        adapter_entry_point_group="rosclaw.growth.adapters",
        synthetic_test_report_hash=_hash("synthetic-tests"),
        plugin_removal_test_report_hash=_hash("core-without-plugins"),
        domain_proofs=(proof,),
    )
    gate = CoreGeneralizationGate(
        downstream_import_prefixes=("rosclaw_soccer",),
        domain_tokens=("football",),
    )

    decision = gate.evaluate(evidence)

    assert decision.status is GeneralizationStatus.REJECTED
    assert decision.reasons == (
        "downstream_import",
        "insufficient_domains",
        "insufficient_tasks",
        "insufficient_adapters",
    )
