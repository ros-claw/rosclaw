"""End-to-end physical evidence loops for G1 GoalForge Phase 4."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rosclaw.auto.g1_kick.parameter_search import (
    GoalForgeParameterSearch,
    ParameterSearchOutcome,
)
from rosclaw.auto.g1_kick.shot_adapter_train import (
    G1ShotAdapter,
    ShotAdapterChampion,
    ShotAdapterRegistry,
    ShotAdapterTeacherSample,
    build_shot_adapter_context,
)
from rosclaw.how.g1_goalforge import GoalForgeHow, GoalForgeHowIntervention
from rosclaw.know.g1_goalforge import GoalForgeKnowledge
from rosclaw.memory.g1_goalforge import (
    GoalForgeMemory,
    GoalForgeMemoryEntry,
    memory_context,
)
from rosclaw.practice.g1_goalforge import (
    GoalForgePracticeRecord,
    GoalForgeSemanticEvent,
    KickPracticeDatasetSnapshot,
    build_kick_dataset,
)
from rosclaw.simforge.attestation import (
    create_simforge_signing_key_pair,
    sign_scale_curve,
    verify_scale_curve_signature,
)
from rosclaw.simforge.backends.g1_visual_backend import trajectory_overlay
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
)
from rosclaw.simforge.dataset_snapshot import (
    load_private_holdout,
    load_public_partition,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    ShotParameters,
    hash_json,
)
from rosclaw.simforge.tasks.g1_goalforge.failure_signature import (
    FailureSignatureV3,
    GoalForgeFailureRouter,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario
from rosclaw.simforge.tasks.g1_goalforge.showcase import build_showcase_manifest
from rosclaw.simforge.tasks.g1_goalforge.skill_graph import G1PenaltyKickSkillGraph
from rosclaw.simforge.tasks.g1_goalforge.task import SameSeedKickPair
from rosclaw.simforge.tasks.g1_goalforge.verifier import GoalForgeVerifier
from rosclaw.twin.g1_kick import (
    KickPredictionError,
    KickTwinBelief,
    KickTwinEstimator,
    KickTwinUpdate,
)


@dataclass(frozen=True)
class GoalForgeDemoResult:
    baseline: GoalForgeEpisode
    failure: FailureSignatureV3
    twin_before_hash: str
    twin_after_hash: str
    twin_update: KickTwinUpdate
    intervention: GoalForgeHowIntervention
    search: ParameterSearchOutcome
    retry: GoalForgeEpisode
    new_location: GoalForgeEpisode
    edge_angle_search: ParameterSearchOutcome
    edge_angle: GoalForgeEpisode
    same_seed_pair: SameSeedKickPair
    skill_graph_hash: str
    wrong_candidate_rejected: bool
    report_path: Path

    @property
    def passed(self) -> bool:
        return bool(
            not self.baseline.result.success
            and self.failure.retry_budget > 0
            and self.search.winner is not None
            and self.retry.result.success
            and self.same_seed_pair.causal_passed
            and self.new_location.result.success
            and self.edge_angle_search.winner is not None
            and self.edge_angle.result.success
            and self.wrong_candidate_rejected
            and self.baseline.receipt is not None
            and self.baseline.receipt.strict_replay
            and self.retry.receipt is not None
            and self.retry.receipt.strict_replay
            and self.new_location.receipt is not None
            and self.new_location.receipt.strict_replay
            and self.edge_angle.receipt is not None
            and self.edge_angle.receipt.independently_verified
            and self.edge_angle.receipt.strict_replay
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.g1_goalforge.demo_result.v2",
            "passed": self.passed,
            "baseline": _episode_summary(self.baseline),
            "failure": self.failure.to_dict(),
            "twin": {
                "before_hash": self.twin_before_hash,
                "after_hash": self.twin_after_hash,
                "update": self.twin_update.to_dict(),
            },
            "intervention": {
                "hash": self.intervention.intervention_hash,
                "patch": self.intervention.patch.to_dict(),
                "rationale": list(self.intervention.rationale),
            },
            "search": {
                "search_hash": self.search.search_hash,
                "attempts": self.search.attempts,
                "winner_hash": (self.search.winner.policy_hash if self.search.winner else None),
                "hidden_truth_accessed_by_generator": (
                    self.search.hidden_truth_accessed_by_generator
                ),
            },
            "same_seed_retry": _episode_summary(self.retry),
            "new_location_first_shot": _episode_summary(self.new_location),
            "edge_angle_search": {
                "search_hash": self.edge_angle_search.search_hash,
                "attempts": self.edge_angle_search.attempts,
                "winner_hash": (
                    self.edge_angle_search.winner.policy_hash
                    if self.edge_angle_search.winner
                    else None
                ),
                "hidden_truth_accessed_by_generator": (
                    self.edge_angle_search.hidden_truth_accessed_by_generator
                ),
            },
            "optimized_edge_angle": _episode_summary(self.edge_angle),
            "causal_pair": {
                "same_seed": self.same_seed_pair.same_seed,
                "same_scenario": self.same_seed_pair.same_scenario,
                "only_candidate_changed": self.same_seed_pair.only_candidate_changed,
                "outcome_improved": self.same_seed_pair.outcome_improved,
                "passed": self.same_seed_pair.causal_passed,
            },
            "skill_graph_hash": self.skill_graph_hash,
            "wrong_candidate_rejected": self.wrong_candidate_rejected,
        }


@dataclass(frozen=True)
class GoalForgeFlywheelResult:
    dataset: KickPracticeDatasetSnapshot
    adapter: G1ShotAdapter
    champion: ShotAdapterChampion
    practice_episode_count: int
    development_teacher_count: int
    validation_episode_count: int
    private_holdout_episode_count: int
    fixed_success_rate: float
    online_teacher_success_rate: float
    learned_success_rate: float
    learned_fall_rate: float
    learned_torque_violation_rate: float
    mean_online_search_ms: float
    mean_learned_inference_ms: float
    split_leakage: bool
    ordinary_task_champion_loaded: bool
    ordinary_task_result_hash: str
    private_holdout_attestation: dict[str, Any]
    report_path: Path

    @property
    def passed(self) -> bool:
        return bool(
            self.practice_episode_count >= 24
            and self.development_teacher_count >= 8
            and self.validation_episode_count >= 4
            and self.private_holdout_episode_count >= 4
            and not self.split_leakage
            and self.dataset.base.quality.passes
            and self.adapter.dataset_snapshot_hash == self.dataset.snapshot_hash
            and self.champion.active
            and self.learned_success_rate > self.fixed_success_rate
            and self.learned_success_rate >= self.online_teacher_success_rate - 0.15
            and self.learned_fall_rate == 0.0
            and self.learned_torque_violation_rate == 0.0
            and self.mean_learned_inference_ms < self.mean_online_search_ms
            and self.ordinary_task_champion_loaded
            and self.ordinary_task_result_hash.startswith("sha256:")
            and self.private_holdout_attestation.get("signature_verified") is True
            and self.private_holdout_attestation.get("private_case_results_disclosed") is False
            and int(self.private_holdout_attestation.get("episode_count", 0)) >= 2
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.g1_goalforge.flywheel_result.v1",
            "passed": self.passed,
            "dataset": self.dataset.to_dict(),
            "adapter": {
                "model_hash": self.adapter.model_hash,
                "dataset_snapshot_hash": self.adapter.dataset_snapshot_hash,
                "training_metrics": asdict(self.adapter.metrics),
            },
            "champion": {
                **asdict(self.champion),
                "registry_hash": self.champion.registry_hash,
            },
            "practice_episode_count": self.practice_episode_count,
            "development_teacher_count": self.development_teacher_count,
            "validation_episode_count": self.validation_episode_count,
            "private_holdout_episode_count": self.private_holdout_episode_count,
            "comparison": {
                "fixed_prior_success_rate": self.fixed_success_rate,
                "online_teacher_success_rate": self.online_teacher_success_rate,
                "learned_adapter_success_rate": self.learned_success_rate,
                "learned_fall_rate": self.learned_fall_rate,
                "learned_torque_violation_rate": self.learned_torque_violation_rate,
                "mean_online_search_ms": self.mean_online_search_ms,
                "mean_learned_inference_ms": self.mean_learned_inference_ms,
            },
            "split_leakage": self.split_leakage,
            "ordinary_task": {
                "champion_loaded": self.ordinary_task_champion_loaded,
                "result_hash": self.ordinary_task_result_hash,
            },
            "private_holdout": self.private_holdout_attestation,
        }


def run_goalforge_demo(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
) -> GoalForgeDemoResult:
    root = _new_external_root(output_dir, source_checkout)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    baseline_parameters = ShotParameters()
    scenario = _scenario(
        scenario_id="goalforge-demo-same-seed",
        seed=2026072301,
        partition=Partition.DEVELOPMENT,
        generation=2,
        ball_y=0.0,
        target_y=0.55,
        target_z=0.55,
    )
    baseline = backend.run_and_record(
        scenario=scenario,
        parameters=baseline_parameters,
        output_root=root / "episodes",
        source_checkout=source_checkout,
        practice_id="practice-goalforge-demo",
    )
    verification = GoalForgeVerifier().verify(baseline)
    if not verification.valid or baseline.result.success:
        raise RuntimeError("GoalForge demo baseline did not produce the expected verified miss")
    router = GoalForgeFailureRouter()
    failure = router.route(
        result=baseline.result,
        body_hash=backend.qualification.body_hash,
        scene_hash=scenario.scenario_commitment,
        action_id="demo-action-baseline",
        policy_hash=baseline_parameters.policy_hash,
    )
    twin_before = KickTwinBelief.initial()
    prediction_error = KickPredictionError(
        predicted_ball_speed_mps=7.4,
        observed_ball_speed_mps=baseline.result.ball_speed_mps,
        predicted_target_error_m=0.30,
        observed_target_error_m=baseline.result.target_error_m,
        predicted_contact_time_sec=5.27,
        observed_contact_time_sec=baseline.result.ball_contact_time_sec or 9.0,
        support_foot_slip_m=baseline.result.support_foot_slip_m,
        torso_response_rad=max(
            baseline.result.torso_roll_peak_rad,
            baseline.result.torso_pitch_peak_rad,
        ),
        joint_tracking_rmse_rad=_joint_tracking_rmse(baseline),
        source_episode_hash=baseline.result_hash,
    )
    twin_after, twin_update = KickTwinEstimator().update(
        twin_before,
        prediction_error,
    )
    intervention = GoalForgeHow().advise(
        signature=failure,
        current=baseline_parameters,
        twin=twin_after,
        retry_index=1,
    )
    knowledge = GoalForgeKnowledge(
        body_hash=backend.qualification.body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
    )
    search = GoalForgeParameterSearch(max_candidates=4).run(
        runner=backend,
        scenario=scenario,
        base=baseline_parameters,
        twin=twin_after,
        knowledge=knowledge,
        intervention=intervention,
    )
    if search.winner is None:
        raise RuntimeError("GoalForge bounded search produced no safe candidate")
    retry = backend.run_and_record(
        scenario=scenario,
        parameters=search.winner,
        output_root=root / "episodes",
        source_checkout=source_checkout,
        practice_id="practice-goalforge-demo",
    )
    pair = SameSeedKickPair(
        baseline=baseline,
        retry=retry,
        same_seed=baseline.scenario.seed == retry.scenario.seed,
        same_scenario=(baseline.scenario.scenario_commitment == retry.scenario.scenario_commitment),
        only_candidate_changed=(baseline.parameters.policy_hash != retry.parameters.policy_hash),
        outcome_improved=(
            retry.result.success and retry.result.target_error_m < baseline.result.target_error_m
        ),
    )
    new_scenario = _scenario(
        scenario_id="goalforge-demo-new-location",
        seed=2026072302,
        partition=Partition.VALIDATION,
        generation=2,
        ball_y=0.06,
        target_y=0.35,
        target_z=0.55,
    )
    new_parameters = _context_patch(
        baseline_parameters,
        new_scenario.observed_context(),
        twin_after,
    )
    new_location = backend.run_and_record(
        scenario=new_scenario,
        parameters=new_parameters,
        output_root=root / "episodes",
        source_checkout=source_checkout,
        practice_id="practice-goalforge-demo",
    )
    edge_scenario = _scenario(
        scenario_id="goalforge-demo-optimized-edge-angle",
        seed=2026072403,
        partition=Partition.VALIDATION,
        generation=4,
        ball_y=0.10,
        target_y=-0.75,
        target_z=0.20,
    )
    edge_search = GoalForgeParameterSearch(max_candidates=32).run(
        runner=backend,
        scenario=edge_scenario,
        base=baseline_parameters,
        twin=twin_after,
        knowledge=knowledge,
    )
    if edge_search.winner is None:
        raise RuntimeError("GoalForge edge-angle search produced no safe candidate")
    edge_angle = backend.run_and_record(
        scenario=edge_scenario,
        parameters=edge_search.winner,
        output_root=root / "episodes",
        source_checkout=source_checkout,
        practice_id="practice-goalforge-demo",
    )
    edge_verification = GoalForgeVerifier().verify(edge_angle)
    if not edge_verification.valid or not edge_angle.result.success:
        raise RuntimeError("GoalForge optimized edge-angle shot did not verify as success")
    wrong_candidate_rejected = not knowledge.validate_candidate(
        candidate=replace(
            baseline_parameters,
            swing_amplitude=1.15,
            policy_type="parameter",
        ),
        attempted_mutations={"torque_hard_limit"},
    )[0]
    graph = _execute_skill_graph()
    report_path = root / "goalforge-demo.json"
    result = GoalForgeDemoResult(
        baseline=baseline,
        failure=failure,
        twin_before_hash=twin_before.belief_hash,
        twin_after_hash=twin_after.belief_hash,
        twin_update=twin_update,
        intervention=intervention,
        search=search,
        retry=retry,
        new_location=new_location,
        edge_angle_search=edge_search,
        edge_angle=edge_angle,
        same_seed_pair=pair,
        skill_graph_hash=graph.graph_hash,
        wrong_candidate_rejected=wrong_candidate_rejected,
        report_path=report_path,
    )
    _write_json(report_path, result.to_dict())
    showcase = build_showcase_manifest(pair)
    showcase["shots"].append(
        {
            "label": "new-location-first-shot",
            "color": "blue",
            "result": new_location.result.summary_dict(),
            "trajectory": trajectory_overlay(new_location.trajectory),
        }
    )
    showcase["shots"].append(
        {
            "label": "optimized-edge-angle",
            "color": "gold",
            "result": edge_angle.result.summary_dict(),
            "trajectory": trajectory_overlay(edge_angle.trajectory),
        }
    )
    showcase["module_chain"] = [
        "Body",
        "Provider",
        "Failure Router V3",
        "Practice",
        "Memory",
        "Know",
        "How",
        "Auto",
        "Twin",
        "Sandbox",
        "Verifier",
        "Receipt",
    ]
    _write_json(root / "showcase.json", showcase)
    return result


def run_goalforge_practice_flywheel(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    generation: int = 3,
) -> GoalForgeFlywheelResult:
    root = _new_external_root(output_dir, source_checkout)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    body_hash = backend.qualification.body_hash
    twin = KickTwinBelief.initial()
    memory = GoalForgeMemory()
    scenarios = _practice_scenarios(generation)
    records: list[GoalForgePracticeRecord] = []
    scenario_map = {scenario.scenario_id: scenario for scenario in scenarios}
    teacher_patches: dict[str, ShotParameters] = {}
    online_elapsed: list[float] = []
    fixed_results: dict[str, GoalForgeEpisode] = {}
    teacher_results: dict[str, GoalForgeEpisode] = {}
    router = GoalForgeFailureRouter()
    for index, scenario in enumerate(scenarios):
        baseline_parameters = ShotParameters()
        contextual = _context_patch(
            baseline_parameters,
            scenario.observed_context(),
            twin,
        )
        teacher_patches[scenario.scenario_id] = contextual
        started = time.perf_counter()
        baseline = backend.run_and_record(
            scenario=scenario,
            parameters=baseline_parameters,
            output_root=root / "episodes",
            source_checkout=source_checkout,
            practice_id=f"practice-goalforge-g{generation}",
        )
        candidate = backend.run_and_record(
            scenario=scenario,
            parameters=contextual,
            output_root=root / "episodes",
            source_checkout=source_checkout,
            practice_id=f"practice-goalforge-g{generation}",
        )
        online_elapsed.append((time.perf_counter() - started) * 1000.0)
        fixed_results[scenario.scenario_id] = baseline
        teacher_results[scenario.scenario_id] = candidate
        best = (
            contextual if _safe_score(candidate) >= _safe_score(baseline) else baseline_parameters
        )
        for run_index, episode in enumerate((baseline, candidate)):
            failure = (
                None
                if episode.result.success
                else router.route(
                    result=episode.result,
                    body_hash=body_hash,
                    scene_hash=scenario.scenario_commitment,
                    action_id=f"practice-{index}-{run_index}",
                    policy_hash=episode.parameters.policy_hash,
                )
            )
            events = (
                GoalForgeSemanticEvent(
                    event_type="simulation_receipt",
                    episode_id=episode.receipt.episode_id if episode.receipt else "missing",
                    payload_hash=(
                        episode.receipt.receipt_hash
                        if episode.receipt is not None
                        else hash_json({"missing": True})
                    ),
                ),
            )
            records.append(
                GoalForgePracticeRecord(
                    episode=episode,
                    practice_id=f"practice-goalforge-g{generation}",
                    failure_signature=failure,
                    best_patch=best,
                    semantic_events=events,
                )
            )
        winner_episode = candidate if best == contextual else baseline
        if (
            winner_episode.result.success
            and winner_episode.receipt is not None
            and winner_episode.receipt.strict_replay
        ):
            memory.remember(
                GoalForgeMemoryEntry(
                    memory_id=f"g1-memory-{index}",
                    body_hash=body_hash,
                    scenario_commitment=scenario.scenario_commitment,
                    context=tuple(sorted(memory_context(scenario.observed_context()).items())),
                    safe_patch=best,
                    score=_safe_score(winner_episode),
                    successful=True,
                    strict_replay=True,
                    evidence_hash=winner_episode.receipt.receipt_hash,
                )
            )
    dataset, files = build_kick_dataset(
        records=tuple(records),
        output_dir=root / "dataset",
        source_checkout=source_checkout,
        split_secret=b"goalforge-phase4-split-secret-v1",
        dataset_id=f"dataset_goalforge_g{generation}_v1",
        generation=generation,
        body_hash=body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
    )
    development = load_public_partition(files.development)
    validation = load_public_partition(files.validation)
    private_holdout = load_private_holdout(files.private_holdout)
    development_ids = {record.scenario_id for record in development}
    validation_ids = {record.scenario_id for record in validation}
    holdout_ids = {record.scenario_id for record in private_holdout}
    split_leakage = bool(
        development_ids & validation_ids
        or development_ids & holdout_ids
        or validation_ids & holdout_ids
    )
    for scenario_id in validation_ids:
        scenario_map[scenario_id] = replace(
            scenario_map[scenario_id],
            partition=Partition.VALIDATION,
        )
    for scenario_id in holdout_ids:
        scenario_map[scenario_id] = replace(
            scenario_map[scenario_id],
            partition=Partition.HOLDOUT,
        )
    samples = []
    for scenario_id in sorted(development_ids):
        scenario = scenario_map[scenario_id]
        context = build_shot_adapter_context(
            observed_context=scenario.observed_context(),
            twin_context=twin.public_context(),
            memory_summary=(0.0,) * 9,
        )
        teacher = teacher_patches[scenario_id]
        evidence = teacher_results[scenario_id]
        samples.append(
            ShotAdapterTeacherSample.from_values(
                context=context,
                best_safe_patch=teacher,
                teacher_evaluation_hash=evidence.result_hash,
            )
        )
    adapter = G1ShotAdapter.train(
        samples=tuple(samples),
        dataset_snapshot_hash=dataset.snapshot_hash,
    )
    adapter.export(root / "model")
    learned_results: list[GoalForgeEpisode] = []
    inference_times: list[float] = []
    evaluation_ids = sorted(validation_ids | holdout_ids)
    for scenario_id in evaluation_ids:
        scenario = scenario_map[scenario_id]
        inference = adapter.infer(
            build_shot_adapter_context(
                observed_context=scenario.observed_context(),
                twin_context=twin.public_context(),
                memory_summary=(0.0,) * 9,
            )
        )
        inference_times.append(inference.inference_ms)
        learned_results.append(backend.run(scenario, inference.parameters))
    fall_rate = _rate(episode.result.post_kick_fall for episode in learned_results)
    torque_rate = _rate(episode.result.torque_limit_violation for episode in learned_results)
    registry = ShotAdapterRegistry()
    champion = registry.activate(
        model=adapter,
        body_hash=body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
        validation_evidence_hash=hash_json(
            {
                "validation_scenarios": sorted(validation_ids),
                "learned_result_hashes": [
                    episode.result_hash
                    for episode in learned_results
                    if episode.scenario.scenario_id in validation_ids
                ],
            }
        ),
        fall_rate=fall_rate,
        torque_violation_rate=torque_rate,
    )
    resolved = registry.resolve(
        body_hash=body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
    )
    ordinary_scenario = scenario_map[sorted(validation_ids)[0]]
    ordinary_inference = adapter.infer(
        build_shot_adapter_context(
            observed_context=ordinary_scenario.observed_context(),
            twin_context=twin.public_context(),
            memory_summary=(0.0,) * 9,
        )
    )
    ordinary_result = backend.run(
        ordinary_scenario,
        ordinary_inference.parameters,
    )
    ordinary_loaded = bool(
        resolved.active
        and resolved.model_hash == ordinary_inference.model_hash
        and resolved.dataset_snapshot_hash == ordinary_inference.parameters.dataset_snapshot_hash
        and ordinary_result.result.success
    )
    holdout_results = [
        episode for episode in learned_results if episode.scenario.scenario_id in holdout_ids
    ]
    key_root = root / "holdout-keys"
    key_root.mkdir(mode=0o700)
    private_key = key_root / "private.pem"
    public_key = key_root / "public.pem"
    create_simforge_signing_key_pair(
        private_key_path=private_key,
        public_key_path=public_key,
        source_checkout=source_checkout,
    )
    holdout_value: dict[str, Any] = {
        "schema_version": "rosclaw.g1_goalforge.holdout_aggregate.v1",
        "scenario_set_commitment": dataset.to_dict()["partitions"]["holdout"][
            "scenario_set_commitment"
        ],
        "evidence_commitment": hash_json(
            sorted(episode.result_hash for episode in holdout_results)
        ),
        "episode_count": len(holdout_results),
        "goal_success_rate": _rate(episode.result.success for episode in holdout_results),
        "target_zone_success_rate": _rate(
            episode.result.target_zone_hit for episode in holdout_results
        ),
        "mean_target_error_m": _mean_result(holdout_results, "target_error_m"),
        "mean_ball_speed_mps": _mean_result(holdout_results, "ball_speed_mps"),
        "mean_time_to_kick_sec": _mean_result(
            holdout_results,
            "ball_contact_time_sec",
        ),
        "mean_support_slip_m": _mean_result(
            holdout_results,
            "support_foot_slip_m",
        ),
        "mean_com_margin_m": _mean_result(
            holdout_results,
            "com_margin_min_m",
        ),
        "torso_roll_p95_rad": _p95_result(
            holdout_results,
            "torso_roll_peak_rad",
        ),
        "torso_pitch_p95_rad": _p95_result(
            holdout_results,
            "torso_pitch_peak_rad",
        ),
        "joint_limit_violation_rate": _rate(
            episode.result.joint_limit_violation for episode in holdout_results
        ),
        "mean_post_kick_stability_sec": _mean_result(
            holdout_results,
            "post_kick_stability_time_sec",
        ),
        "fall_rate": _rate(episode.result.post_kick_fall for episode in holdout_results),
        "torque_violation_rate": _rate(
            episode.result.torque_limit_violation for episode in holdout_results
        ),
        "private_case_results_disclosed": False,
        "public_key_path": str(public_key),
        "signature_verified": True,
    }
    holdout_value["attestation"] = sign_scale_curve(
        holdout_value,
        private_key_path=private_key,
    )
    verify_scale_curve_signature(
        holdout_value,
        expected_public_key_path=public_key,
    )
    _write_json(root / "private-holdout-attestation.json", holdout_value)
    fixed_eval = [fixed_results[scenario_id] for scenario_id in evaluation_ids]
    teacher_eval = [teacher_results[scenario_id] for scenario_id in evaluation_ids]
    report_path = root / "goalforge-flywheel.json"
    result = GoalForgeFlywheelResult(
        dataset=dataset,
        adapter=adapter,
        champion=champion,
        practice_episode_count=len(records),
        development_teacher_count=len(samples),
        validation_episode_count=len(validation),
        private_holdout_episode_count=len(private_holdout),
        fixed_success_rate=_rate(episode.result.success for episode in fixed_eval),
        online_teacher_success_rate=_rate(episode.result.success for episode in teacher_eval),
        learned_success_rate=_rate(episode.result.success for episode in learned_results),
        learned_fall_rate=fall_rate,
        learned_torque_violation_rate=torque_rate,
        mean_online_search_ms=sum(online_elapsed) / len(online_elapsed),
        mean_learned_inference_ms=sum(inference_times) / len(inference_times),
        split_leakage=split_leakage,
        ordinary_task_champion_loaded=ordinary_loaded,
        ordinary_task_result_hash=ordinary_result.result_hash,
        private_holdout_attestation=holdout_value,
        report_path=report_path,
    )
    _write_json(report_path, result.to_dict())
    return result


def _scenario(
    *,
    scenario_id: str,
    seed: int,
    partition: Partition,
    generation: int,
    ball_y: float,
    target_y: float,
    target_z: float,
    support_friction: float = 1.0,
    latency_ms: float = 0.0,
) -> GoalForgeScenario:
    return GoalForgeScenario(
        scenario_id=scenario_id,
        partition=partition,
        seed=seed,
        seed_commitment=hash_json({"goalforge_seed": seed}),
        generation=generation,
        ball_x_m=1.0,
        ball_y_m=ball_y,
        ball_velocity_x_mps=0.0,
        ball_velocity_y_mps=0.0,
        target_y_m=target_y,
        target_z_m=target_z,
        ball_mass_kg=0.42,
        ball_ground_friction=0.05,
        restitution=0.55,
        support_ground_friction=support_friction,
        control_latency_ms=latency_ms,
        observation_noise_m=0.0,
        joint_zero_bias_rad=0.0,
        disturbance_n=0.0,
    )


def _practice_scenarios(generation: int) -> tuple[GoalForgeScenario, ...]:
    settings = (
        (0.00, 0.55, 0.55),
        (0.00, -0.35, 0.55),
        (0.06, 0.00, 0.20),
        (-0.05, 0.00, 0.55),
        (0.00, 0.35, 0.55),
        (0.08, 0.45, 0.55),
        (-0.08, -0.45, 0.55),
        (0.04, 0.25, 0.35),
        (-0.03, -0.20, 0.35),
        (0.07, 0.50, 0.45),
        (-0.06, 0.15, 0.45),
        (0.02, -0.50, 0.45),
    )
    return tuple(
        _scenario(
            scenario_id=f"goalforge-practice-{index}",
            seed=2026072400 + index,
            partition=Partition.DEVELOPMENT,
            generation=generation,
            ball_y=ball_y,
            target_y=target_y,
            target_z=target_z,
            support_friction=(0.82 if index == 4 else 1.0),
            latency_ms=0.0,
        )
        for index, (ball_y, target_y, target_z) in enumerate(settings)
    )


def _context_patch(
    base: ShotParameters,
    context: dict[str, float],
    twin: KickTwinBelief,
) -> ShotParameters:
    delta = context["target_y"] - context["ball_y"]
    low_friction = twin.support_ground_friction.mean < 0.70
    return replace(
        base,
        stance_offset_y=max(-0.12, min(0.12, context["ball_y"] * 0.45)),
        pelvis_yaw_offset=max(-0.20, min(0.20, delta * 0.35)),
        com_shift_y=0.03 if low_friction else 0.015,
        swing_speed_scale=0.88 if low_friction else 1.0,
        foot_yaw_offset=max(-0.12, min(0.12, delta * 0.055)),
        recovery_step_length=0.09 if low_friction else 0.055,
        policy_type="parameter",
    )


def _execute_skill_graph() -> G1PenaltyKickSkillGraph:
    graph = G1PenaltyKickSkillGraph()
    verified = {
        "session_live",
        "permit_valid",
        "observation_fresh",
        "support_foot_contact_verified",
        "com_inside_support_margin",
        "ball_observation_fresh",
        "kick_trigger_authorized",
        *(f"{state.lower()}_verified" for state in graph.STATES),
    }
    for state in graph.STATES[1:]:
        graph.transition(state, verified=verified)
    return graph


def _joint_tracking_rmse(episode: GoalForgeEpisode) -> float:
    position = episode.trajectory["joint_position"]
    action = episode.trajectory["policy_action"]
    return float(math.sqrt(float(((position - action) ** 2).mean())))


def _safe_score(episode: GoalForgeEpisode) -> float:
    result = episode.result
    if (
        result.post_kick_fall
        or result.torque_limit_violation
        or result.joint_limit_violation
        or result.actuator_saturation
    ):
        return -1_000_000.0
    error = result.target_error_m if math.isfinite(result.target_error_m) else 10.0
    return 100.0 * float(result.success) - error + result.robustness


def _episode_summary(episode: GoalForgeEpisode) -> dict[str, Any]:
    return {
        "scenario_id": episode.scenario.scenario_id,
        "seed_commitment": episode.scenario.seed_commitment,
        "scenario_commitment": episode.scenario.scenario_commitment,
        "policy_hash": episode.parameters.policy_hash,
        "parameters": episode.parameters.to_dict(),
        "result": episode.result.summary_dict(),
        "receipt": episode.receipt.to_dict() if episode.receipt else None,
        "artifact_root": str(episode.artifact_root) if episode.artifact_root else None,
    }


def _new_external_root(output_dir: Path, source_checkout: Path) -> Path:
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("GoalForge evidence output must stay outside source checkout")
    root.mkdir(parents=True, exist_ok=False)
    return root


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _rate(values: Any) -> float:
    normalized = list(values)
    return sum(map(bool, normalized)) / len(normalized) if normalized else 0.0


def _mean_result(episodes: list[GoalForgeEpisode], field: str) -> float:
    values = [
        float(value)
        for episode in episodes
        if (value := getattr(episode.result, field)) is not None and math.isfinite(float(value))
    ]
    return sum(values) / len(values) if values else 0.0


def _p95_result(episodes: list[GoalForgeEpisode], field: str) -> float:
    values = sorted(float(getattr(episode.result, field)) for episode in episodes)
    if not values:
        return 0.0
    index = max(0, math.ceil(0.95 * len(values)) - 1)
    return values[index]


__all__ = [
    "GoalForgeDemoResult",
    "GoalForgeFlywheelResult",
    "run_goalforge_demo",
    "run_goalforge_practice_flywheel",
]
