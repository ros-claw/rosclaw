from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw.auto.g1_kick.continual_runner import (
    run_goalforge_continual_screening,
)
from rosclaw.auto.g1_kick.parameter_search import build_parameter_candidates
from rosclaw.auto.g1_kick.shot_adapter_train import (
    G1ShotAdapter,
    ShotAdapterRegistry,
    ShotAdapterTeacherSample,
    build_shot_adapter_context,
)
from rosclaw.how.g1_goalforge import GoalForgeHow
from rosclaw.know.g1_goalforge import GoalForgeKnowledge
from rosclaw.memory.g1_goalforge import (
    GoalForgeMemory,
    GoalForgeMemoryEntry,
)
from rosclaw.robot_pack.g1.chaos import run_g1_executor_chaos
from rosclaw.simforge.g1_memory_ablation import run_memory_ablation
from rosclaw.simforge.g1_proof_replay import replay_goalforge_proof_bundle
from rosclaw.simforge.g1_video import build_playback_timeline
from rosclaw.simforge.models import Partition
from rosclaw.simforge.promotion_v4 import (
    GateV4Decision,
    GoalForgeMetrics,
    PromotionEvidenceV4,
    PromotionGateV4,
)
from rosclaw.simforge.proof import (
    CounterfactualMetric,
    CounterfactualRun,
    FaultInjectionResult,
    ModuleProof,
    ProofBundle,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    GoalForgeResult,
    GoalForgeStatus,
    ShotParameters,
    hash_json,
)
from rosclaw.simforge.tasks.g1_goalforge.failure_signature import (
    GoalForgeFailureRouter,
    RecoverabilityV3,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario
from rosclaw.simforge.tasks.g1_goalforge.skill_graph import G1PenaltyKickSkillGraph
from rosclaw.twin.g1_kick import (
    KickPredictionError,
    KickTwinBelief,
    KickTwinEstimator,
)

BODY_HASH = hash_json({"body": "g1-29dof"})
PRIOR_HASH = hash_json({"policy": "fixed-g1-prior"})
REQUIRED_E5_MODULES = (
    "body",
    "provider",
    "failure_router",
    "sandbox",
    "practice",
    "memory",
    "know",
    "how",
    "auto",
    "darwin",
    "registry",
    "rosclawd",
)


def test_shot_parameters_are_bounded_and_learned_requires_dataset() -> None:
    with pytest.raises(ValueError, match="stance_offset_x"):
        ShotParameters(stance_offset_x=0.13)
    with pytest.raises(ValueError, match="dataset snapshot"):
        ShotParameters(policy_type="learned_adapter")
    learned = ShotParameters(
        policy_type="learned_adapter",
        dataset_snapshot_hash=hash_json({"dataset": 1}),
    )
    assert learned.policy_hash.startswith("sha256:")


def test_goalforge_video_timeline_slows_contact_window_deterministically() -> None:
    timeline = build_playback_timeline(13.0, fps=30)
    source_times = [sample.simulation_time_sec for sample in timeline]
    assert source_times == sorted(source_times)
    assert source_times[0] == 0.0
    assert source_times[-1] <= 13.0
    assert len(timeline) > 11 * 30
    slow_motion_frames = sum(4.3 <= value <= 6.4 for value in source_times)
    assert slow_motion_frames >= 150


def test_goalforge_search_covers_coupled_edge_stance_without_expanding_budget() -> None:
    candidates = build_parameter_candidates(
        base=ShotParameters(),
        public_context={
            "ball_x": 1.0,
            "ball_y": 0.10,
            "ball_vx": 0.0,
            "ball_vy": 0.0,
            "target_y": -0.75,
            "target_z": 0.20,
            "support_friction_belief": 0.85,
            "ball_mass_belief": 0.42,
            "control_latency_belief_ms": 20.0,
            "body_calibration_state": 0.0,
        },
        twin=KickTwinBelief.initial(),
        memory=None,
        intervention=None,
    )
    assert len(candidates) <= 32
    sources = {source for _, source in candidates}
    assert {"coupled_lateral_sweep", "stance_lateral_sweep"} <= sources
    edge = next(
        parameters for parameters, source in candidates if source == "edge_stance_lateral_sweep"
    )
    assert edge.stance_offset_y == 0.12
    assert edge.pelvis_yaw_offset == -0.20
    assert edge.foot_yaw_offset == pytest.approx(-0.0595)


def test_holdout_scenario_hides_physical_truth() -> None:
    scenario = _scenario(partition=Partition.HOLDOUT)
    public = scenario.to_dict()
    assert "ball_mass_kg" not in public
    assert "support_ground_friction" not in public
    assert public["observed_context"]["support_friction_belief"] == 0.85
    assert "ball_mass_kg" in scenario.to_private_dict()


def test_twin_updates_from_residual_without_scenario_truth() -> None:
    belief = KickTwinBelief.initial()
    error = KickPredictionError(
        predicted_ball_speed_mps=7.0,
        observed_ball_speed_mps=5.0,
        predicted_target_error_m=0.2,
        observed_target_error_m=0.7,
        predicted_contact_time_sec=5.0,
        observed_contact_time_sec=5.2,
        support_foot_slip_m=0.06,
        torso_response_rad=0.2,
        joint_tracking_rmse_rad=0.04,
        source_episode_hash=hash_json({"episode": 1}),
    )
    updated, record = KickTwinEstimator().update(belief, error)
    assert updated.support_ground_friction.mean < belief.support_ground_friction.mean
    assert updated.control_latency_ms.mean > belief.control_latency_ms.mean
    assert record.hidden_truth_accessed is False
    assert record.parent_belief_hash == belief.belief_hash


def test_failure_router_stops_unreachable_and_bounds_retry() -> None:
    router = GoalForgeFailureRouter()
    unreachable = router.route(
        result=_result(GoalForgeStatus.BALL_OUT_OF_REACH),
        body_hash=BODY_HASH,
        scene_hash=hash_json({"scene": 1}),
        action_id="action-1",
        policy_hash=PRIOR_HASH,
    )
    assert unreachable.recoverability is RecoverabilityV3.UNRECOVERABLE
    assert unreachable.retry_budget == 0
    assert unreachable.recommended_route == ("STOP", "HUMAN")

    target = router.route(
        result=_result(GoalForgeStatus.TARGET_MISS_RIGHT),
        body_hash=BODY_HASH,
        scene_hash=hash_json({"scene": 2}),
        action_id="action-2",
        policy_hash=PRIOR_HASH,
    )
    assert target.recoverability is RecoverabilityV3.RECOVERABLE
    assert target.retry_budget == 2


def test_how_intervention_cannot_expand_bounds() -> None:
    result = _result(GoalForgeStatus.TARGET_MISS_RIGHT)
    signature = GoalForgeFailureRouter().route(
        result=result,
        body_hash=BODY_HASH,
        scene_hash=hash_json({"scene": 3}),
        action_id="action-3",
        policy_hash=ShotParameters().policy_hash,
    )
    intervention = GoalForgeHow().advise(
        signature=signature,
        current=ShotParameters(pelvis_yaw_offset=0.18),
        twin=KickTwinBelief.initial(),
        retry_index=1,
    )
    assert intervention.patch.pelvis_yaw_offset == 0.20
    assert intervention.bounded_retry_index == 1


def test_knowledge_rejects_immutable_safety_mutation() -> None:
    knowledge = GoalForgeKnowledge(body_hash=BODY_HASH, kick_prior_hash=PRIOR_HASH)
    valid, errors = knowledge.validate_candidate(
        candidate=ShotParameters(policy_type="parameter"),
        attempted_mutations={"torque_hard_limit"},
    )
    assert valid is False
    assert errors == ("immutable_mutation=torque_hard_limit",)


def test_memory_is_hard_scoped_to_body_hash() -> None:
    memory = GoalForgeMemory()
    context = {"ball_x": 1.0, "ball_y": 0.0, "target_y": 0.5, "target_z": 0.55}
    for index, body_hash in enumerate((BODY_HASH, hash_json({"other": "body"}))):
        memory.remember(
            GoalForgeMemoryEntry(
                memory_id=f"memory-{index}",
                body_hash=body_hash,
                scenario_commitment=hash_json({"scenario": index}),
                context=tuple(sorted(context.items())),
                safe_patch=ShotParameters(policy_type="parameter"),
                score=1.0,
                successful=True,
                strict_replay=True,
                evidence_hash=hash_json({"evidence": index}),
            )
        )
    recall = memory.recall(body_hash=BODY_HASH, context=context)
    assert len(recall.entries) == 1
    assert recall.entries[0].body_hash == BODY_HASH
    assert recall.rejected_wrong_body == 1


def test_memory_ablation_has_100_pairs_and_reduces_search(tmp_path: Path) -> None:
    result = run_memory_ablation(
        output_path=tmp_path / "memory.json",
        body_hash=BODY_HASH,
    )
    assert result.passed
    assert len(result.pairs) == 100
    assert result.search_reduction >= 0.25
    assert result.wrong_memory_hurt_rate < 0.01


def test_shot_adapter_trains_projects_and_activates() -> None:
    twin = KickTwinBelief.initial()
    samples = []
    dataset_hash = hash_json({"dataset": "training"})
    for index in range(8):
        ball_y = -0.14 + index * 0.04
        target_y = -0.45 + index * 0.13
        observed = {
            "ball_x": 1.0,
            "ball_y": ball_y,
            "ball_vx": 0.0,
            "ball_vy": 0.0,
            "target_y": target_y,
            "target_z": 0.55,
            "support_friction_belief": 0.85,
            "ball_mass_belief": 0.42,
            "control_latency_belief_ms": 20.0,
            "body_calibration_state": 0.0,
        }
        context = build_shot_adapter_context(
            observed_context=observed,
            twin_context=twin.public_context(),
            memory_summary=(0.0,) * 9,
        )
        delta = target_y - ball_y
        samples.append(
            ShotAdapterTeacherSample.from_values(
                context=context,
                best_safe_patch=ShotParameters(
                    pelvis_yaw_offset=max(-0.2, min(0.2, delta * 0.35)),
                    foot_yaw_offset=max(-0.12, min(0.12, delta * 0.055)),
                    policy_type="parameter",
                ),
                teacher_evaluation_hash=hash_json({"teacher": index}),
            )
        )
    adapter = G1ShotAdapter.train(
        samples=tuple(samples),
        dataset_snapshot_hash=dataset_hash,
        epochs=200,
    )
    inference = adapter.infer(dict(samples[3].context))
    assert inference.parameters.policy_type == "learned_adapter"
    assert inference.parameters.dataset_snapshot_hash == dataset_hash
    champion = ShotAdapterRegistry().activate(
        model=adapter,
        body_hash=BODY_HASH,
        kick_prior_hash=PRIOR_HASH,
        validation_evidence_hash=hash_json({"validation": "safe"}),
        fall_rate=0.0,
        torque_violation_rate=0.0,
    )
    assert champion.model_hash == adapter.model_hash


def test_skill_graph_rejects_skips_and_bounds_failures() -> None:
    graph = G1PenaltyKickSkillGraph()
    with pytest.raises(ValueError, match="invalid GoalForge transition"):
        graph.transition("SWING_LEG", verified=set())
    graph.current_state = "SWING_LEG"
    assert graph.record_failure("BALL_NOT_CONTACTED") == "STANDING_READY"
    assert graph.record_failure("BALL_NOT_CONTACTED") == "STANDING_READY"
    assert graph.record_failure("BALL_NOT_CONTACTED") == "STOPPED"


def test_promotion_v4_passes_all_gates_and_rejects_fall() -> None:
    baseline = _metrics(
        goal_success_rate=0.4,
        target_zone_success_rate=0.35,
        mean_target_error_m=0.6,
        first_attempt_success_rate=0.35,
        mean_retries=1.4,
        mean_support_slip_m=0.04,
        mean_post_kick_stability_sec=2.0,
    )
    candidate = _metrics(
        goal_success_rate=0.8,
        target_zone_success_rate=0.75,
        mean_target_error_m=0.3,
        first_attempt_success_rate=0.7,
        retry_success_rate=0.9,
        mean_retries=0.7,
        mean_support_slip_m=0.03,
        mean_post_kick_stability_sec=2.5,
    )
    evidence = PromotionEvidenceV4(
        baseline=baseline,
        candidate_validation=candidate,
        hidden_holdout=candidate,
        candidate_hash=hash_json({"candidate": 1}),
        body_hash=BODY_HASH,
        expected_body_hash=BODY_HASH,
        counterexample_regression_passed=True,
        dds_sim_to_sim_passed=True,
        strict_replay_rate=1.0,
        module_levels=tuple((module, "E5") for module in REQUIRED_E5_MODULES),
        canary_passed=True,
        shards_complete=True,
        holdout_signature_verified=True,
        evidence_commitment_verified=True,
        critical_safety_forgetting=0,
        historical_success_delta=0.0,
    )
    assert PromotionGateV4().evaluate(evidence).decision is GateV4Decision.SIM_CHAMPION
    unsafe = replace(
        evidence,
        candidate_validation=replace(candidate, fall_rate=0.01),
    )
    rejected = PromotionGateV4().evaluate(unsafe)
    assert rejected.decision is GateV4Decision.REJECTED
    assert not next(check for check in rejected.checks if check.gate == "G6").passed
    missing_module = replace(
        evidence,
        module_levels=tuple(item for item in evidence.module_levels if item[0] != "rosclawd"),
    )
    incomplete = PromotionGateV4().evaluate(missing_module)
    assert incomplete.decision is GateV4Decision.NEED_MORE_EVIDENCE
    assert next(check for check in incomplete.checks if check.gate == "G15").missing
    wrong_body = replace(
        evidence,
        expected_body_hash=hash_json({"body": "unqualified"}),
    )
    body_rejected = PromotionGateV4().evaluate(wrong_body)
    assert body_rejected.decision is GateV4Decision.REJECTED
    assert not next(check for check in body_rejected.checks if check.gate == "G13").passed


def test_executor_chaos_is_fail_closed(tmp_path: Path) -> None:
    result = run_g1_executor_chaos(
        output_dir=tmp_path / "chaos",
        source_checkout=Path(__file__).resolve().parents[2],
        body_hash=BODY_HASH,
    )
    assert result.passed
    assert result.old_trigger_replay_count == 0
    assert result.stale_task_verified_count == 0


def test_goalforge_proof_replay_rederives_e5_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    proof = ModuleProof(
        module="memory",
        invoked=True,
        input_refs=("failure://one",),
        output_refs=("memory://one",),
        output_valid=True,
        decision_impacts=("reduced_attempts",),
        counterfactual=CounterfactualRun(
            control_run_id="memory-off",
            treatment_run_id="memory-on",
            same_seed=True,
            same_scenario=True,
            same_body_hash=True,
            decision_changed=True,
            outcome_changed=True,
            metrics=(
                CounterfactualMetric(
                    name="attempts",
                    control=5.0,
                    treatment=1.0,
                    lower_is_better=True,
                ),
            ),
            control_ref="run://memory_off",
            treatment_ref="run://memory_on",
        ),
        fault_injections=(
            FaultInjectionResult(
                name="wrong_body_rejected",
                passed=True,
                evidence_ref="memory://wrong_body",
            ),
        ),
        replay_verified=True,
        replay_ref="memory://replay",
    )
    bundle = ProofBundle(
        run_id="goalforge-proof-test",
        task_id="g1_penalty_kick",
        body_snapshot_hash=BODY_HASH,
        proofs=(proof,),
        evidence_root_ref="goalforge://test",
    )
    path = tmp_path / "proof-bundle.json"
    path.write_text(
        json.dumps(bundle.to_dict()),
        encoding="utf-8",
    )
    replay = replay_goalforge_proof_bundle(
        path,
        requested_modules=("memory",),
    )
    assert replay.passed

    value = json.loads(path.read_text(encoding="utf-8"))
    value["proofs"][0]["fault_injection"][0]["passed"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    rejected = replay_goalforge_proof_bundle(
        path,
        requested_modules=("memory",),
    )
    assert not rejected.passed
    assert "bundle_hash_mismatch" in rejected.errors
    assert "memory:fault_injection_not_passed" in rejected.errors


def test_continual_screening_records_g0_through_g10_without_case_disclosure(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir()
    roles = ("practice", "candidate_search", "falsification", "private_holdout")
    for role in roles:
        rows = [
            {
                "scenario_commitment": hash_json({"role": role, "index": index}),
                "fixed_error_proxy": 0.80 - index * 0.02,
                "candidate_error_proxy": 0.32 - index * 0.02,
                "safe_proxy": True,
            }
            for index in range(3)
        ]
        (private / f"gpu-0-{role}-rows.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    result = run_goalforge_continual_screening(
        four_gpu_root=tmp_path,
        output_path=tmp_path / "continual.json",
    )
    assert result.passed
    assert [record.generation for record in result.records] == list(range(11))
    assert (
        result.records[-1].first_attempt_success_rate > result.records[0].first_attempt_success_rate
    )
    public = json.loads((tmp_path / "continual.json").read_text(encoding="utf-8"))
    assert public["evidence_domain"] == "CUDA_SCREENING"
    assert "scenario_commitment" not in json.dumps(public)


def _scenario(*, partition: Partition) -> GoalForgeScenario:
    return GoalForgeScenario(
        scenario_id="scenario-1",
        partition=partition,
        seed=1,
        seed_commitment=hash_json({"seed": 1}),
        generation=10,
        ball_x_m=1.0,
        ball_y_m=0.0,
        ball_velocity_x_mps=0.0,
        ball_velocity_y_mps=0.0,
        target_y_m=0.5,
        target_z_m=0.55,
        ball_mass_kg=0.42,
        ball_ground_friction=0.05,
        restitution=0.55,
        support_ground_friction=0.7,
        control_latency_ms=20.0,
        observation_noise_m=0.02,
        joint_zero_bias_rad=0.0,
        disturbance_n=0.0,
    )


def _result(status: GoalForgeStatus) -> GoalForgeResult:
    success = status is GoalForgeStatus.SUCCESS
    return GoalForgeResult(
        status=status,
        success=success,
        physics_executed=status not in {GoalForgeStatus.BALL_OUT_OF_REACH},
        contact_observed=True,
        kick_foot_contacted=True,
        goal_crossed=True,
        target_zone_hit=success,
        target_error_m=0.6,
        ball_speed_mps=7.0,
        ball_contact_time_sec=5.2,
        contact_impulse_ns=3.0,
        support_foot_slip_m=0.02,
        com_margin_min_m=0.04,
        torso_roll_peak_rad=0.2,
        torso_pitch_peak_rad=0.2,
        peak_torque_scale=0.8,
        joint_limit_violation=False,
        torque_limit_violation=False,
        actuator_saturation=False,
        post_kick_fall=False,
        post_kick_stability_time_sec=2.0,
        final_pelvis_height_m=0.78,
        physics_steps=100,
        finite_state=True,
        robustness=0.03,
    )


def _metrics(**changes: float) -> GoalForgeMetrics:
    values: dict[str, float | int] = {
        "episodes": 200,
        "goal_success_rate": 0.5,
        "target_zone_success_rate": 0.5,
        "mean_target_error_m": 0.5,
        "mean_ball_speed_mps": 7.0,
        "mean_time_to_kick_sec": 5.2,
        "first_attempt_success_rate": 0.5,
        "retry_success_rate": 0.7,
        "mean_retries": 1.0,
        "fall_rate": 0.0,
        "torque_violation_rate": 0.0,
        "unsafe_allow_rate": 0.0,
        "mean_support_slip_m": 0.03,
        "mean_com_margin_m": 0.04,
        "torso_roll_p95_rad": 0.25,
        "torso_pitch_p95_rad": 0.25,
        "joint_limit_violation_rate": 0.0,
        "mean_post_kick_stability_sec": 2.0,
    }
    values.update(changes)
    return GoalForgeMetrics(**values)
