"""Matched-elite AWR validation for contact-causal G1 recovery learning."""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.collective.sources.motiondecode.motion_prior import (
    load_g1_motion_prior_artifact,
)
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
    trajectory_digest,
)
from rosclaw.simforge.g1_contact_recovery_torque_policy import (
    G1ContactRecoveryGateConfig,
    G1ContactRecoveryTorquePolicy,
)
from rosclaw.simforge.g1_neural_torque import (
    G1NeuralTorqueArtifact,
    G1NeuralTorquePolicy,
    G1TorqueExplorationConfig,
    load_g1_neural_torque_artifact,
)
from rosclaw.simforge.g1_neural_torque_learning import (
    G1ContinualTorqueActorCritic,
    G1NeuralTorqueLearnerConfig,
    G1NeuralTorqueReplay,
    neural_torque_replay_hash,
    recovery_online_replay,
    teacher_dataset_hash,
    teacher_replay,
)
from rosclaw.simforge.g1_neural_torque_validation import (
    _pilot_scenarios,
    _write_candidate,
)
from rosclaw.simforge.g1_recovery_online_rl_validation import _actor_effect
from rosclaw.simforge.g1_recovery_quality import (
    G1RecoveryQuality,
    measure_g1_recovery_quality,
)
from rosclaw.simforge.g1_stability_plasticity_policy import (
    G1StabilityPlasticityGateConfig,
    G1StabilityPlasticityTorquePolicy,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    GoalForgeResult,
    ShotParameters,
    hash_bytes,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import (
    GoalForgeScenario,
    generate_goalforge_scenarios,
)


def run_g1_recovery_awr_validation(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    motiondecode_pilot_report_path: Path,
    stable_artifact_path: Path,
    recovery_artifact_path: Path,
    output_dir: Path,
    source_checkout: Path,
    device: str = "cuda:1",
    seed: int = 8920,
    exploration_replicates: int = 6,
    value_updates: int = 64,
    actor_updates: int = 4,
    validation_round: int = 1,
) -> dict[str, Any]:
    """Train one recovery generation and promote only independent A/B gains."""

    if not 2 <= exploration_replicates <= 16:
        raise ValueError("recovery AWR exploration replicates must be in [2, 16]")
    if not 8 <= value_updates <= 256:
        raise ValueError("recovery AWR value updates must be in [8, 256]")
    if not 1 <= actor_updates <= 32:
        raise ValueError("recovery AWR actor updates must be in [1, 32]")
    if not 1 <= validation_round <= 99:
        raise ValueError("recovery AWR validation round must be in [1, 99]")
    root = _external_root(output_dir, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    qualification = backend.qualification
    stable = load_g1_neural_torque_artifact(
        stable_artifact_path,
        expected_body_hash=qualification.body_hash,
    )
    retained = load_g1_neural_torque_artifact(
        recovery_artifact_path,
        expected_body_hash=qualification.body_hash,
    )
    prior = load_g1_motion_prior_artifact(motion_prior_path)
    if prior.body_hash != qualification.body_hash:
        raise ValueError("recovery AWR motion prior body mismatch")
    for label, artifact in (("stable", stable), ("recovery", retained)):
        if artifact.parent_policy_hash != qualification.kick_prior_hash:
            raise ValueError(f"{label} artifact parent-policy mismatch")
    motiondecode_audit = _motiondecode_audit(
        motiondecode_pilot_report_path,
        expected_hash=prior.pilot_report_hash,
    )

    baseline_gate = G1StabilityPlasticityGateConfig(
        minimum_recovery_phase=0.20,
        minimum_pelvis_height_m=0.70,
        maximum_projected_gravity_z=-0.88,
        eligibility_warmup_steps=5,
    )
    contact_gate = G1ContactRecoveryGateConfig(
        minimum_policy_phase=0.40,
        minimum_ball_speed_mps=0.50,
        minimum_pelvis_height_m=0.70,
        maximum_projected_gravity_z=-0.88,
        eligibility_warmup_steps=5,
    )
    parameters = ShotParameters()
    training_scenarios, development_scenarios, _ = _pilot_scenarios()
    training_scenarios = (
        *training_scenarios,
        *_expanded_recovery_scenarios("training"),
    )
    development_scenarios = (
        *development_scenarios,
        *_expanded_recovery_scenarios("development"),
    )
    validation_scenarios = _fresh_recovery_validation_scenarios(validation_round)

    parent_training = tuple(
        _run_contact_policy(
            backend,
            scenario,
            parameters,
            stable=stable,
            retained=retained,
            candidate=retained,
            baseline_gate=baseline_gate,
            contact_gate=contact_gate,
        )
        for scenario in training_scenarios
    )
    anchor_episodes = tuple(policy.candidate.episode() for _, policy in parent_training)
    config = G1NeuralTorqueLearnerConfig(
        hidden_dim=retained.hidden_dim,
        sequence_length=32,
        batch_size=64,
        actor_lr=2e-5,
        critic_lr=3e-4,
        behavior_cloning_weight=12.0,
        online_behavior_weight=5.0,
        parent_churn_weight=6.0,
        ewc_weight=60.0,
        initial_alpha=0.002,
        awr_temperature=0.35,
        awr_max_weight=20.0,
        awr_fall_penalty=5.0,
        awr_constraint_penalty=2.0,
        device=device,
        seed=seed,
    )
    learner = G1ContinualTorqueActorCritic(config, safety=retained.safety)
    learner.update_index = retained.update_index
    learner.install_actor_artifact(
        retained,
        expected_body_hash=qualification.body_hash,
        expected_parent_policy_hash=qualification.kick_prior_hash,
    )
    installed_parent_metrics = learner.consolidate_installed_actor(
        anchor_episodes,
        stride=4,
        minimum_end_fraction=contact_gate.minimum_policy_phase,
    )
    anchor = teacher_replay(
        anchor_episodes,
        sequence_length=config.sequence_length,
        stride=10,
        minimum_end_fraction=contact_gate.minimum_policy_phase,
    )

    collection, online_replays = _collect_matched_exploration(
        backend=backend,
        scenarios=training_scenarios,
        parameters=parameters,
        stable=stable,
        retained=retained,
        current=retained,
        baseline_gate=baseline_gate,
        contact_gate=contact_gate,
        config=config,
        seed=seed,
        replicates=exploration_replicates,
    )
    replay = G1NeuralTorqueReplay.combine(anchor, *online_replays)
    replay_path = root / "g1-recovery-awr-replay.npz"
    _write_replay(replay_path, replay)
    value_metrics = tuple(
        learner.update_advantage_weighted(replay, update_actor=False).to_dict()
        for _ in range(value_updates)
    )
    value_checkpoint_path = root / "g1-recovery-awr-value-checkpoint.pt"
    value_checkpoint_path.write_bytes(learner.checkpoint_bytes())
    convergence = _value_convergence(value_metrics)
    fresh_count = int(np.count_nonzero((replay.partitions == 0) & (replay.policy_lags <= 1)))
    elite_scenario_count = len({item["scenario_id"] for item in collection if bool(item["elite"])})
    actor_metrics: tuple[dict[str, Any], ...] = ()
    trust_runs: tuple[dict[str, Any], ...] = ()
    selected: dict[str, Any] | None = None
    parent_development = _evaluate_set(
        backend,
        development_scenarios,
        parameters,
        stable=stable,
        retained=retained,
        candidate=retained,
        baseline_gate=baseline_gate,
        contact_gate=contact_gate,
        stage="recovery_awr_parent_development",
        strict=False,
    )
    if convergence["converged"] and fresh_count >= config.batch_size and elite_scenario_count >= 2:
        parent_snapshot = learner.actor_snapshot()
        actor_metrics = tuple(
            learner.update_advantage_weighted(replay, update_actor=True).to_dict()
            for _ in range(actor_updates)
        )
        proposal_snapshot = learner.actor_snapshot()
        actor_effect = _actor_effect(
            learner,
            parent_snapshot,
            proposal_snapshot,
            replay,
        )
        trust_runs, selected = _trust_search(
            root=root,
            backend=backend,
            scenarios=development_scenarios,
            parameters=parameters,
            learner=learner,
            parent_snapshot=parent_snapshot,
            proposal_snapshot=proposal_snapshot,
            stable=stable,
            retained=retained,
            parent_development=parent_development,
            baseline_gate=baseline_gate,
            contact_gate=contact_gate,
            body_hash=qualification.body_hash,
            parent_policy_hash=qualification.kick_prior_hash,
            replay_hash=neural_torque_replay_hash(replay),
        )
    else:
        actor_effect = None

    candidate = (
        load_g1_neural_torque_artifact(
            Path(str(selected["artifact_path"])),
            expected_body_hash=qualification.body_hash,
        )
        if selected is not None
        else retained
    )
    parent_validation = _evaluate_set(
        backend,
        validation_scenarios,
        parameters,
        stable=stable,
        retained=retained,
        candidate=retained,
        baseline_gate=baseline_gate,
        contact_gate=contact_gate,
        stage="recovery_awr_parent_validation",
        strict=True,
    )
    candidate_validation = _evaluate_set(
        backend,
        validation_scenarios,
        parameters,
        stable=stable,
        retained=retained,
        candidate=candidate,
        baseline_gate=baseline_gate,
        contact_gate=contact_gate,
        stage="recovery_awr_candidate_validation",
        strict=True,
    )
    validation_gate = _matched_gate(
        parent_validation,
        candidate_validation,
        minimum_naturalness_reduction=0.02,
        require_strict=True,
    )
    checks = {
        "motiondecode_snapshot_applicability_known": bool(
            motiondecode_audit["local_snapshot_complete"] is False
            and motiondecode_audit["selected_football_count"] == 0
        ),
        "retained_parent_exact_before_contact": all(
            bool(item["parent_exact_noop"]) for item in collection
        ),
        "matched_exploration_exercised": any(
            int(item["exploration_applied_count"]) > 0 for item in collection
        ),
        "elite_actor_data_available": fresh_count >= config.batch_size,
        "elite_scenario_coverage": elite_scenario_count >= 2,
        "in_sample_value_converged": bool(convergence["converged"]),
        "actor_updated_only_from_elites": bool(actor_metrics)
        and all(item["learning_mode"] == "AWR_IN_SAMPLE" for item in actor_metrics),
        "development_candidate_found": selected is not None,
        "independent_strict_validation_passed": bool(validation_gate["passed"]),
        "sim_only_boundary_preserved": all(
            item["receipt"]["activation_ceiling"] == "SIM_ONLY"
            and not item["receipt"]["hardware_authorized"]
            for item in candidate_validation
        ),
    }
    decision = "CANDIDATE" if all(checks.values()) else "REJECTED"
    blockers = [name for name, passed in checks.items() if not passed]
    blockers.extend(str(reason) for reason in validation_gate["reasons"])
    report = {
        "schema_version": "rosclaw.simforge.g1_recovery_awr_validation.v1",
        "decision": decision,
        "blockers": blockers,
        "body_hash": qualification.body_hash,
        "parent_policy_hash": qualification.kick_prior_hash,
        "stable_artifact_hash": stable.artifact_hash,
        "retained_recovery_artifact_hash": retained.artifact_hash,
        "candidate_artifact_hash": candidate.artifact_hash,
        "motion_prior_artifact_hash": prior.artifact_hash,
        "motiondecode": motiondecode_audit,
        "motiondecode_role": "REFERENCE_ONLY_NOT_TORQUE_ACTIONS",
        "baseline_gate": asdict(baseline_gate),
        "contact_gate": asdict(contact_gate),
        "recovery_action_subspace": [
            G1_DDS_JOINT_NAMES[index] for index in _recovery_action_indices()
        ],
        "learner_config": asdict(config),
        "installed_parent_behavior_loss": list(installed_parent_metrics),
        "anchor_dataset_hash": teacher_dataset_hash(anchor_episodes),
        "online_replay_hash": neural_torque_replay_hash(replay),
        "online_replay_path": str(replay_path),
        "value_checkpoint_path": str(value_checkpoint_path),
        "fresh_elite_transition_count": fresh_count,
        "elite_scenario_count": elite_scenario_count,
        "collection": list(collection),
        "value_updates": list(value_metrics),
        "value_convergence": convergence,
        "actor_updates": list(actor_metrics),
        "actor_effect": actor_effect,
        "parent_development": list(parent_development),
        "trust_runs": list(trust_runs),
        "selected": selected,
        "parent_validation": list(parent_validation),
        "candidate_validation": list(candidate_validation),
        "validation_gate": validation_gate,
        "checks": checks,
        "claims": {
            "contact_causal_recovery": True,
            "matched_elite_advantage_weighted_regression": True,
            "critic_never_scores_unseen_torque": True,
            "direct_joint_torque_actor": True,
            "motiondecode_direct_torque_transfer": False,
            "candidate_promoted": decision == "CANDIDATE",
            "real_robot_evidence": False,
        },
        "evidence_domain": "SIM_ONLY",
        "hardware_authorized": False,
    }
    _atomic_json(root / "g1-recovery-awr-report.json", report)
    return report


def recovery_naturalness_cost(quality: G1RecoveryQuality) -> float:
    """Dimensionless lower-is-better score over the visible failure modes."""

    value = (
        0.35 * quality.tail_wobble_index / 0.25
        + 0.20 * quality.post_contact_backward_reversal_m / 0.25
        + 0.10 * quality.post_contact_lateral_peak_return_m / 0.50
        + 0.10 * min(quality.tail_joint_jerk_rms_rad_s3, 20.0) / 5.0
        + 0.15 * quality.post_contact_pelvis_path_length_m / 1.50
        + 0.10 * min(quality.post_contact_joint_jerk_rms_rad_s3, 2000.0) / 500.0
    )
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("recovery naturalness cost must be finite and non-negative")
    return float(value)


def _run_contact_policy(
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
    *,
    stable: G1NeuralTorqueArtifact,
    retained: G1NeuralTorqueArtifact,
    candidate: G1NeuralTorqueArtifact,
    baseline_gate: G1StabilityPlasticityGateConfig,
    contact_gate: G1ContactRecoveryGateConfig,
    exploration: G1TorqueExplorationConfig | None = None,
) -> tuple[GoalForgeEpisode, G1ContactRecoveryTorquePolicy]:
    def policy(artifact: G1NeuralTorqueArtifact, *, explore: bool = False):
        return G1NeuralTorquePolicy(
            artifact,
            expected_body_hash=backend.qualification.body_hash,
            expected_parent_policy_hash=backend.qualification.kick_prior_hash,
            exploration=exploration if explore else None,
        )

    baseline = G1StabilityPlasticityTorquePolicy(
        policy(stable),
        policy(retained),
        config=baseline_gate,
    )
    overlay = G1ContactRecoveryTorquePolicy(
        baseline,
        policy(candidate, explore=True),
        config=contact_gate,
    )
    return backend.run(scenario, parameters, torque_policy=overlay), overlay


def _collect_matched_exploration(
    *,
    backend: G1MuJoCoBackend,
    scenarios: tuple[GoalForgeScenario, ...],
    parameters: ShotParameters,
    stable: G1NeuralTorqueArtifact,
    retained: G1NeuralTorqueArtifact,
    current: G1NeuralTorqueArtifact,
    baseline_gate: G1StabilityPlasticityGateConfig,
    contact_gate: G1ContactRecoveryGateConfig,
    config: G1NeuralTorqueLearnerConfig,
    seed: int,
    replicates: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[G1NeuralTorqueReplay, ...]]:
    collection: list[dict[str, Any]] = []
    replays: list[G1NeuralTorqueReplay] = []
    for scenario_index, scenario in enumerate(scenarios):
        parent_episode, parent_policy = _run_contact_policy(
            backend,
            scenario,
            parameters,
            stable=stable,
            retained=retained,
            candidate=current,
            baseline_gate=baseline_gate,
            contact_gate=contact_gate,
        )
        parent_quality = _measure_quality(parent_episode)
        parent_cost = (
            recovery_naturalness_cost(parent_quality) if parent_quality is not None else 1_000_000.0
        )
        # Identical parent/candidate tensors must be a trajectory-level no-op.
        exact_episode, _ = _run_contact_policy(
            backend,
            scenario,
            parameters,
            stable=stable,
            retained=retained,
            candidate=current,
            baseline_gate=baseline_gate,
            contact_gate=contact_gate,
        )
        parent_exact = bool(
            parent_episode.result.summary_dict() == exact_episode.result.summary_dict()
            and trajectory_digest(parent_episode.trajectory)
            == trajectory_digest(exact_episode.trajectory)
        )
        for replicate in range(replicates):
            exploration = G1TorqueExplorationConfig(
                noise_std_ratio=0.0015,
                noise_clip_ratio=0.004,
                temporal_correlation=0.995,
                minimum_recovery_phase=contact_gate.minimum_policy_phase,
                minimum_pelvis_height_m=contact_gate.minimum_pelvis_height_m,
                maximum_projected_gravity_z=contact_gate.maximum_projected_gravity_z,
                seed=seed + 1000 * scenario_index + replicate,
            )
            episode, policy = _run_contact_policy(
                backend,
                scenario,
                parameters,
                stable=stable,
                retained=retained,
                candidate=current,
                baseline_gate=baseline_gate,
                contact_gate=contact_gate,
                exploration=exploration,
            )
            quality = _measure_quality(episode)
            cost = recovery_naturalness_cost(quality) if quality is not None else 1_000_000.0
            task_preserved, task_reasons = _task_preserved(
                parent_episode.result,
                episode.result,
            )
            reduction = (parent_cost - cost) / max(parent_cost, 1e-9)
            elite = bool(task_preserved and reduction >= 0.01)
            relative_score = float(
                np.clip(
                    4.0 * reduction + float(task_preserved) - float(not task_preserved), -20, 20
                )
            )
            receipt = policy.build_receipt()
            replay = recovery_online_replay(
                policy.candidate.episode(),
                trajectory=episode.trajectory,
                sequence_length=config.sequence_length,
                recovery_score=relative_score,
                fell=episode.result.post_kick_fall,
                critical_failure=_critical(episode.result),
                projection_fallback_rate=(
                    receipt.candidate_projection_rejection_count / max(1, receipt.inference_count)
                ),
                recovery_start_phase=contact_gate.minimum_policy_phase,
                actor_eligible_mask=policy.candidate_activation_mask(),
                fall_quarantine_sec=0.10,
                stride=10,
            )
            if not elite:
                replay = replace(
                    replay,
                    partitions=np.full(replay.count, 2, dtype=np.int8),
                )
            replays.append(replay)
            collection.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "replicate": replicate,
                    "parent_exact_noop": parent_exact,
                    "parent_naturalness_cost": parent_cost,
                    "candidate_naturalness_cost": cost,
                    "naturalness_reduction": reduction,
                    "task_preserved": task_preserved,
                    "task_reasons": list(task_reasons),
                    "inherited_critical_failure": bool(
                        _critical(parent_episode.result) and _critical(episode.result)
                    ),
                    "elite": elite,
                    "transition_count": replay.count,
                    "fresh_actor_transition_count": int(np.count_nonzero(replay.partitions == 0)),
                    "exploration_applied_count": receipt.exploration_applied_count,
                    "exploration_noise_peak_ratio": receipt.exploration_noise_peak_ratio,
                    "candidate_activation_fraction": receipt.candidate_activation_fraction,
                    "contact_context": (
                        asdict(receipt.contact_context)
                        if receipt.contact_context is not None
                        else None
                    ),
                    "contact_context_hash": receipt.contact_context_hash,
                    "parent_result": parent_episode.result.summary_dict(),
                    "candidate_result": episode.result.summary_dict(),
                    "parent_quality": (
                        parent_quality.to_dict() if parent_quality is not None else None
                    ),
                    "candidate_quality": quality.to_dict() if quality is not None else None,
                }
            )
    return tuple(collection), tuple(replays)


def _trust_search(
    *,
    root: Path,
    backend: G1MuJoCoBackend,
    scenarios: tuple[GoalForgeScenario, ...],
    parameters: ShotParameters,
    learner: G1ContinualTorqueActorCritic,
    parent_snapshot: dict[str, np.ndarray],
    proposal_snapshot: dict[str, np.ndarray],
    stable: G1NeuralTorqueArtifact,
    retained: G1NeuralTorqueArtifact,
    parent_development: tuple[dict[str, Any], ...],
    baseline_gate: G1StabilityPlasticityGateConfig,
    contact_gate: G1ContactRecoveryGateConfig,
    body_hash: str,
    parent_policy_hash: str,
    replay_hash: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any] | None]:
    values: list[dict[str, Any]] = []
    for fraction in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        learner.install_interpolated_actor(
            parent_snapshot,
            proposal_snapshot,
            fraction=fraction,
            action_indices=_recovery_action_indices(),
        )
        path = root / f"recovery-awr-trust-{str(fraction).replace('.', 'p')}.bin"
        _write_candidate(
            path,
            learner,
            body_hash=body_hash,
            parent_policy_hash=parent_policy_hash,
            dataset_hash=canonical_hash(
                {
                    "replay_hash": replay_hash,
                    "trust_fraction": fraction,
                    "action_indices": list(_recovery_action_indices()),
                }
            ),
        )
        artifact = load_g1_neural_torque_artifact(path, expected_body_hash=body_hash)
        candidate = _evaluate_set(
            backend,
            scenarios,
            parameters,
            stable=stable,
            retained=retained,
            candidate=artifact,
            baseline_gate=baseline_gate,
            contact_gate=contact_gate,
            stage=f"recovery_awr_trust_{fraction}_development",
            strict=False,
        )
        gate = _matched_gate(
            parent_development,
            candidate,
            minimum_naturalness_reduction=0.005,
            require_strict=False,
        )
        values.append(
            {
                "fraction": fraction,
                "artifact_path": str(path),
                "artifact_hash": artifact.artifact_hash,
                "accepted": gate["passed"],
                "gate": gate,
                "rollouts": list(candidate),
            }
        )
    eligible = [item for item in values if item["accepted"]]
    selected = min(eligible, key=lambda item: float(item["fraction"])) if eligible else None
    return tuple(values), selected


def _evaluate_set(
    backend: G1MuJoCoBackend,
    scenarios: tuple[GoalForgeScenario, ...],
    parameters: ShotParameters,
    *,
    stable: G1NeuralTorqueArtifact,
    retained: G1NeuralTorqueArtifact,
    candidate: G1NeuralTorqueArtifact,
    baseline_gate: G1StabilityPlasticityGateConfig,
    contact_gate: G1ContactRecoveryGateConfig,
    stage: str,
    strict: bool,
) -> tuple[dict[str, Any], ...]:
    values = []
    for scenario in scenarios:
        episode, policy = _run_contact_policy(
            backend,
            scenario,
            parameters,
            stable=stable,
            retained=retained,
            candidate=candidate,
            baseline_gate=baseline_gate,
            contact_gate=contact_gate,
        )
        quality = _measure_quality(episode)
        naturalness_cost = (
            recovery_naturalness_cost(quality) if quality is not None else 1_000_000.0
        )
        replay_ok = False
        if strict:
            replay, replay_policy = _run_contact_policy(
                backend,
                scenario,
                parameters,
                stable=stable,
                retained=retained,
                candidate=candidate,
                baseline_gate=baseline_gate,
                contact_gate=contact_gate,
            )
            replay_ok = bool(
                episode.result.summary_dict() == replay.result.summary_dict()
                and trajectory_digest(episode.trajectory) == trajectory_digest(replay.trajectory)
                and policy.build_receipt().to_dict() == replay_policy.build_receipt().to_dict()
            )
        values.append(
            {
                "stage": stage,
                "scenario_id": scenario.scenario_id,
                "scenario_commitment": scenario.scenario_commitment,
                "partition": scenario.partition.value,
                "result": episode.result.summary_dict(),
                "quality": quality.to_dict() if quality is not None else None,
                "naturalness_cost": naturalness_cost,
                "trace_hash": trajectory_digest(episode.trajectory),
                "strict_replay": replay_ok,
                "receipt": policy.build_receipt().to_dict(),
            }
        )
    return tuple(values)


def _measure_quality(episode: GoalForgeEpisode) -> G1RecoveryQuality | None:
    try:
        return measure_g1_recovery_quality(episode.trajectory)
    except ValueError:
        return None


def _matched_gate(
    parent: tuple[dict[str, Any], ...],
    candidate: tuple[dict[str, Any], ...],
    *,
    minimum_naturalness_reduction: float,
    require_strict: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if len(parent) != len(candidate) or not parent:
        return {
            "passed": False,
            "reasons": ["matched_rollout_count_mismatch"],
            "mean_naturalness_reduction": -math.inf,
        }
    reductions = []
    for before, after in zip(parent, candidate, strict=True):
        if before["scenario_commitment"] != after["scenario_commitment"]:
            reasons.append("scenario_commitment_mismatch")
            continue
        preserved, item_reasons = _task_preserved_dict(
            before["result"],
            after["result"],
        )
        if not preserved:
            reasons.extend(f"{after['scenario_id']}:{reason}" for reason in item_reasons)
        if require_strict and not (before["strict_replay"] and after["strict_replay"]):
            reasons.append(f"{after['scenario_id']}:strict_replay_missing")
        reductions.append(
            (float(before["naturalness_cost"]) - float(after["naturalness_cost"]))
            / max(float(before["naturalness_cost"]), 1e-9)
        )
    mean_reduction = float(np.mean(reductions)) if reductions else -math.inf
    if mean_reduction < minimum_naturalness_reduction:
        reasons.append("mean_naturalness_reduction_below_gate")
    if any(value < -0.02 for value in reductions):
        reasons.append("per_scenario_naturalness_regression")
    return {
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "mean_naturalness_reduction": mean_reduction,
        "per_scenario_naturalness_reduction": reductions,
        "minimum_naturalness_reduction": minimum_naturalness_reduction,
    }


def _task_preserved(
    parent: GoalForgeResult,
    candidate: GoalForgeResult,
) -> tuple[bool, tuple[str, ...]]:
    return _task_preserved_dict(parent.summary_dict(), candidate.summary_dict())


def _task_preserved_dict(
    parent: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    reasons = []
    for label in (
        "post_kick_fall",
        "joint_limit_violation",
        "torque_limit_violation",
        "actuator_saturation",
    ):
        if candidate.get(label) and not parent.get(label):
            reasons.append(f"new_{label}")
    if parent.get("finite_state", False) and not candidate.get("finite_state", False):
        reasons.append("new_nonfinite_state")
    if parent.get("kick_foot_contacted") and not candidate.get("kick_foot_contacted"):
        reasons.append("ball_contact_lost")
    if parent.get("success") and not candidate.get("success"):
        reasons.append("parent_success_lost")
    before_error = _finite_or_inf(parent.get("target_error_m"))
    after_error = _finite_or_inf(candidate.get("target_error_m"))
    if math.isfinite(before_error) and after_error > before_error + 0.05:
        reasons.append("target_error_regressed")
    if float(candidate.get("ball_speed_mps", 0.0)) < 0.90 * float(
        parent.get("ball_speed_mps", 0.0)
    ):
        reasons.append("ball_speed_regressed")
    maximum_slip = max(0.04, float(parent.get("support_foot_slip_m", 0.0)) + 0.005)
    if float(candidate.get("support_foot_slip_m", math.inf)) > maximum_slip:
        reasons.append("support_slip_regressed")
    return not reasons, tuple(reasons)


def _finite_or_inf(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.inf
    return result if math.isfinite(result) else math.inf


def _critical(result: GoalForgeResult) -> bool:
    return bool(
        result.post_kick_fall
        or result.joint_limit_violation
        or result.torque_limit_violation
        or result.actuator_saturation
        or not result.finite_state
    )


def _recovery_action_indices() -> tuple[int, ...]:
    """Whole-body unloading rows, excluding kick-leg yaw and distal wrists."""

    names = {
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    }
    return tuple(index for index, name in enumerate(G1_DDS_JOINT_NAMES) if name in names)


def _value_convergence(updates: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    if len(updates) < 8:
        return {"converged": False, "reason": "fewer_than_8_value_updates"}
    window = max(4, min(12, len(updates) // 4))
    first = float(np.mean([float(item["value_loss"]) for item in updates[:window]]))
    last = float(np.mean([float(item["value_loss"]) for item in updates[-window:]]))
    ratio = last / max(first, 1e-9)
    return {
        "converged": bool(math.isfinite(ratio) and ratio <= 0.70),
        "window_size": window,
        "first_window_loss": first,
        "last_window_loss": last,
        "last_to_first_ratio": ratio,
        "maximum_ratio": 0.70,
    }


def _motiondecode_audit(path: Path, *, expected_hash: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if hash_bytes(payload) != expected_hash:
        raise ValueError("MotionDecode pilot report hash does not match the motion prior")
    report = json.loads(payload)
    manifest = report.get("source_manifest", {})
    category_counts = report.get("selection_counts", {})
    football_count = int(category_counts.get("football", manifest.get("football_files", 0)))
    return {
        "pilot_report_hash": expected_hash,
        "local_snapshot_complete": bool(manifest.get("local_snapshot_complete", False)),
        "selected_football_count": football_count,
        "selected_category_counts": category_counts,
        "applicability": (
            "BLOCKED_NO_LOCAL_FOOTBALL_CLIPS" if football_count == 0 else "REFERENCE_MOTION_ONLY"
        ),
    }


def _fresh_recovery_validation_scenarios(
    validation_round: int,
) -> tuple[GoalForgeScenario, ...]:
    ledger = SeedLedger(
        task_id=f"g1_recovery_awr_validation_round_{validation_round}",
        secret=(f"rosclaw-phase8-recovery-awr-v1-{validation_round}".encode() * 3)[:64],
    )
    values = []
    for partition, generations in (
        (Partition.VALIDATION, (1, 5)),
        (Partition.HOLDOUT, (3, 8)),
    ):
        for index, generation in enumerate(generations):
            generated = generate_goalforge_scenarios(
                ledger=ledger,
                partition=partition,
                count=index + 1,
                generation=generation,
            )
            values.append(
                replace(
                    generated[index],
                    scenario_id=(
                        f"recovery-awr-{partition.value}-{index:02d}-"
                        f"r{validation_round}-g{generation}"
                    ),
                )
            )
    ledger.assert_disjoint()
    return tuple(values)


def _expanded_recovery_scenarios(split: str) -> tuple[GoalForgeScenario, ...]:
    if split not in {"training", "development"}:
        raise ValueError("expanded recovery split must be training or development")
    ledger = SeedLedger(
        task_id=f"g1_recovery_awr_expanded_{split}",
        secret=(f"rosclaw-phase8-recovery-awr-expanded-{split}".encode() * 3)[:64],
    )
    values = []
    for index, generation in enumerate((1, 3, 5, 8)):
        generated = generate_goalforge_scenarios(
            ledger=ledger,
            partition=Partition.DEVELOPMENT,
            count=index + 1,
            generation=generation,
        )
        values.append(
            replace(
                generated[index],
                scenario_id=f"recovery-awr-{split}-{index:02d}-g{generation}",
            )
        )
    ledger.assert_disjoint()
    return tuple(values)


def _external_root(path: Path, source_checkout: Path) -> Path:
    root = path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("recovery AWR evidence must be outside the source checkout")
    return root


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _write_replay(path: Path, replay: G1NeuralTorqueReplay) -> None:
    np.savez_compressed(
        path,
        observations=replay.observations,
        actions=replay.actions,
        next_observations=replay.next_observations,
        rewards=replay.rewards,
        fall_costs=replay.fall_costs,
        constraint_costs=replay.constraint_costs,
        terminals=replay.terminals,
        parent_actions=replay.parent_actions,
        partitions=replay.partitions,
        policy_lags=replay.policy_lags,
    )


__all__ = ["recovery_naturalness_cost", "run_g1_recovery_awr_validation"]
