"""One fail-closed online actor-critic generation for post-kick recovery."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
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
from rosclaw.simforge.g1_neural_torque import (
    G1NeuralTorqueArtifact,
    G1NeuralTorquePolicy,
    G1TorqueExplorationConfig,
    G1TorqueSafetyConfig,
    load_g1_neural_torque_artifact,
)
from rosclaw.simforge.g1_neural_torque_learning import (
    G1ContinualTorqueActorCritic,
    G1NeuralTorqueLearnerConfig,
    G1NeuralTorqueReplay,
    G1NeuralTorqueUpdate,
    neural_torque_replay_hash,
    recovery_online_replay,
    stale_neural_torque_replay,
    teacher_dataset_hash,
    teacher_replay,
)
from rosclaw.simforge.g1_neural_torque_validation import (
    G1NeuralTorqueRolloutEvidence,
    _aggregate,
    _collect_teacher,
    _critical,
    _online_update_gate,
    _pilot_scenarios,
    _rollout_evidence,
    _write_candidate,
)
from rosclaw.simforge.g1_stability_plasticity_policy import (
    G1StabilityPlasticityGateConfig,
    G1StabilityPlasticityTorquePolicy,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_HARD_TORQUE_LIMITS,
    GoalForgeResult,
    ShotParameters,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import (
    GoalForgeScenario,
    generate_goalforge_scenarios,
)


@dataclass(frozen=True)
class _OnlineGenerationResult:
    selected_artifact: G1NeuralTorqueArtifact
    final_replay: G1NeuralTorqueReplay
    recovery_replays: tuple[G1NeuralTorqueReplay, ...]
    collection: tuple[dict[str, Any], ...]
    critic_updates: tuple[G1NeuralTorqueUpdate, ...]
    actor_updates: tuple[G1NeuralTorqueUpdate, ...]
    parent_development: tuple[G1NeuralTorqueRolloutEvidence, ...]
    trust_runs: tuple[dict[str, Any], ...]
    generations: tuple[dict[str, Any], ...]
    selected: dict[str, Any] | None
    accepted_generation_count: int


def run_g1_recovery_online_rl_validation(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    output_dir: Path,
    source_checkout: Path,
    device: str = "cuda:2",
    seed: int = 8700,
    generations: int = 3,
    critic_updates_per_generation: int = 32,
) -> dict[str, Any]:
    if not 1 <= generations <= 6:
        raise ValueError("recovery online-RL generations must be in [1, 6]")
    if not 8 <= critic_updates_per_generation <= 128:
        raise ValueError("recovery critic updates per generation must be in [8, 128]")
    root = _external_root(output_dir, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    qualification = backend.qualification
    prior = load_g1_motion_prior_artifact(motion_prior_path)
    if prior.body_hash != qualification.body_hash:
        raise ValueError("recovery prior does not match the qualified G1 body")
    training_scenarios, development_scenarios, validation_scenarios = _pilot_scenarios()
    development_scenarios = (
        *development_scenarios,
        *_anticipatory_development_scenarios(),
    )
    parameters = ShotParameters()
    training = tuple(
        _collect_teacher(backend, scenario, parameters)[1] for scenario in training_scenarios
    )
    teacher_validation = tuple(
        _collect_teacher(backend, scenario, parameters)[1] for scenario in development_scenarios
    )
    config = G1NeuralTorqueLearnerConfig(
        hidden_dim=96,
        sequence_length=32,
        batch_size=256,
        actor_lr=2e-4,
        critic_lr=2e-4,
        behavior_cloning_weight=5.0,
        online_behavior_weight=1.0,
        parent_churn_weight=0.5,
        ewc_weight=20.0,
        device=device,
        seed=seed,
    )
    safety = G1TorqueSafetyConfig(
        torque_guard_scale=0.80,
        maximum_mechanical_power_w=4000.0,
        maximum_parent_deviation_ratio=0.05,
        maximum_projection_ratio=0.35,
        maximum_observation_z=5.0,
        minimum_upright_gravity_z=-0.88,
        minimum_pelvis_height_m=0.70,
        recovery_cooldown_steps=100,
        warmup_steps=100,
    )
    stable = G1ContinualTorqueActorCritic(config, safety=safety)
    stable_metrics = stable.pretrain_behavior(
        training, validation=teacher_validation, epochs=6, stride=4
    )
    teacher_hash = teacher_dataset_hash(training)
    stable_path = root / "stable-torque-bc.bin"
    _write_candidate(
        stable_path,
        stable,
        body_hash=qualification.body_hash,
        parent_policy_hash=qualification.kick_prior_hash,
        dataset_hash=teacher_hash,
    )
    plastic = G1ContinualTorqueActorCritic(config, safety=safety)
    plastic.install_motion_prior(
        prior,
        expected_body_hash=qualification.body_hash,
        fraction=1.0,
    )
    plastic_warmup = plastic.pretrain_behavior(
        training, validation=teacher_validation, epochs=2, stride=4
    )
    plastic_metrics = plastic.pretrain_behavior(
        training,
        validation=teacher_validation,
        epochs=8,
        stride=4,
        minimum_end_fraction=0.15,
    )
    plastic_parent_path = root / "recovery-plastic-parent.bin"
    plastic_dataset_hash = canonical_hash(
        {
            "teacher_dataset_hash": teacher_hash,
            "motion_prior_artifact_hash": prior.artifact_hash,
            "anticipatory_kick_recovery_start_fraction": 0.15,
        }
    )
    _write_candidate(
        plastic_parent_path,
        plastic,
        body_hash=qualification.body_hash,
        parent_policy_hash=qualification.kick_prior_hash,
        dataset_hash=plastic_dataset_hash,
    )
    stable_artifact = load_g1_neural_torque_artifact(
        stable_path, expected_body_hash=qualification.body_hash
    )
    plastic_parent_artifact = load_g1_neural_torque_artifact(
        plastic_parent_path, expected_body_hash=qualification.body_hash
    )
    gate = G1StabilityPlasticityGateConfig(
        minimum_recovery_phase=0.20,
        minimum_pelvis_height_m=0.70,
        maximum_projected_gravity_z=-0.88,
        eligibility_warmup_steps=5,
    )
    online = _run_online_generations(
        root=root,
        backend=backend,
        parameters=parameters,
        learner=plastic,
        config=config,
        stable_artifact=stable_artifact,
        plastic_parent_artifact=plastic_parent_artifact,
        plastic_dataset_hash=plastic_dataset_hash,
        gate=gate,
        training_scenarios=training_scenarios,
        development_scenarios=development_scenarios,
        training=training,
        body_hash=qualification.body_hash,
        parent_policy_hash=qualification.kick_prior_hash,
        seed=seed,
        generations=generations,
        critic_updates_per_generation=critic_updates_per_generation,
    )
    selected_artifact = online.selected_artifact
    validation = validation_scenarios[:4]
    parent_validation = tuple(
        _context_evidence(
            backend,
            scenario,
            parameters,
            stable=stable_artifact,
            plastic=plastic_parent_artifact,
            gate=gate,
            stage="recovery_online_parent_validation",
            strict=True,
        )
        for scenario in validation
    )
    candidate_validation = tuple(
        _context_evidence(
            backend,
            scenario,
            parameters,
            stable=stable_artifact,
            plastic=selected_artifact,
            gate=gate,
            stage="recovery_online_candidate_validation",
            strict=True,
        )
        for scenario in validation
    )
    validation_accepted, validation_reasons = _recovery_update_gate(
        parent=parent_validation,
        candidate=candidate_validation,
    )
    validation_meaningful = _meaningful_recovery_improvement(
        _aggregate(parent_validation), _aggregate(candidate_validation)
    )
    if validation_accepted and not validation_meaningful:
        validation_reasons = (*validation_reasons, "no_measurable_recovery_improvement")
        validation_accepted = False
    checks = {
        "dense_replay_aligned": all(item.count > 0 for item in online.recovery_replays),
        "critic_and_actor_updated": bool(online.critic_updates)
        and bool(online.actor_updates)
        and all(item.finite and item.actor_updated for item in online.actor_updates),
        "safe_exploration_exercised": any(
            int(item["exploration_applied_count"]) > 0 for item in online.collection
        )
        and all(
            float(item["exploration_noise_peak_ratio"]) <= 0.012 + 1e-12
            for item in online.collection
        ),
        "locally_safe_hard_state_learning": any(
            int(item["fresh_actor_transition_count"]) > 0
            and bool(
                item["result"]["post_kick_fall"]
                or item["result"]["joint_limit_violation"]
                or item["result"]["torque_limit_violation"]
            )
            for item in online.collection
        ),
        "fresh_resampling_per_actor_update": len(online.actor_updates)
        == len(online.generations),
        "development_trust_region_found": online.accepted_generation_count > 0,
        "strict_validation_replay": all(
            item.strict_replay for item in (*parent_validation, *candidate_validation)
        ),
        "validation_gate_passed": validation_accepted,
        "validation_plasticity_measurable": validation_meaningful,
        "sim_only_boundary_preserved": all(
            item.activation_ceiling == "SIM_ONLY" and not item.hardware_authorized
            for item in candidate_validation
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    blockers.extend(validation_reasons)
    report = {
        "schema_version": "rosclaw.simforge.g1_recovery_online_rl.v3",
        "body_hash": qualification.body_hash,
        "parent_policy_hash": qualification.kick_prior_hash,
        "motion_prior_artifact_hash": prior.artifact_hash,
        "teacher_dataset_hash": teacher_hash,
        "online_replay_hash": neural_torque_replay_hash(online.final_replay),
        "gate": asdict(gate),
        "bc_metrics": {
            "stable_final_validation_loss": stable_metrics[-1].validation_loss,
            "plastic_warmup_validation_loss": plastic_warmup[-1].validation_loss,
            "plastic_recovery_validation_loss": plastic_metrics[-1].validation_loss,
        },
        "online_generation_count_requested": generations,
        "critic_updates_per_generation": critic_updates_per_generation,
        "collection": list(online.collection),
        "critic_updates": [item.to_dict() for item in online.critic_updates],
        "actor_updates": [item.to_dict() for item in online.actor_updates],
        "actor_update": (
            online.actor_updates[-1].to_dict() if online.actor_updates else None
        ),
        "parent_development": [item.to_dict() for item in online.parent_development],
        "trust_runs": list(online.trust_runs),
        "generations": list(online.generations),
        "selected": online.selected,
        "parent_validation": [item.to_dict() for item in parent_validation],
        "candidate_validation": [item.to_dict() for item in candidate_validation],
        "aggregates": {
            "parent_validation": _aggregate(parent_validation),
            "candidate_validation": _aggregate(candidate_validation),
        },
        "checks": checks,
        "decision": "ONLINE_RECOVERY_CANDIDATE" if not blockers else "REJECTED",
        "blockers": blockers,
        "evidence_domain": "SIM_ONLY",
        "hardware_authorized": False,
        "promotion_evidence_eligible": False,
        "claims": {
            "online_actor_critic": True,
            "dense_recovery_reward": True,
            "anticipatory_kick_recovery_control": True,
            "direct_joint_torque_actor": True,
            "candidate_promoted": False,
            "real_robot_evidence": False,
        },
    }
    _atomic_json(root / "g1-recovery-online-rl-report.json", report)
    return report


def _run_online_generations(
    *,
    root: Path,
    backend: G1MuJoCoBackend,
    parameters: ShotParameters,
    learner: G1ContinualTorqueActorCritic,
    config: G1NeuralTorqueLearnerConfig,
    stable_artifact: G1NeuralTorqueArtifact,
    plastic_parent_artifact: G1NeuralTorqueArtifact,
    plastic_dataset_hash: str,
    gate: G1StabilityPlasticityGateConfig,
    training_scenarios: tuple[Any, ...],
    development_scenarios: tuple[Any, ...],
    training: tuple[Any, ...],
    body_hash: str,
    parent_policy_hash: str,
    seed: int,
    generations: int,
    critic_updates_per_generation: int,
) -> _OnlineGenerationResult:
    anchor = teacher_replay(training, sequence_length=config.sequence_length, stride=10)
    historical: list[G1NeuralTorqueReplay] = []
    all_replays: list[G1NeuralTorqueReplay] = []
    all_collection: list[dict[str, Any]] = []
    all_critic_updates: list[G1NeuralTorqueUpdate] = []
    all_actor_updates: list[G1NeuralTorqueUpdate] = []
    all_trust_runs: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    current_artifact = plastic_parent_artifact
    final_replay = anchor
    final_selected: dict[str, Any] | None = None
    initial_parent_development: tuple[G1NeuralTorqueRolloutEvidence, ...] = ()
    accepted_generation_count = 0

    for generation in range(generations):
        fresh_replays, collection = _collect_online_generation(
            backend=backend,
            parameters=parameters,
            stable_artifact=stable_artifact,
            plastic_artifact=current_artifact,
            gate=gate,
            scenarios=training_scenarios,
            config=config,
            seed=seed + generation * 10_000,
            generation=generation,
        )
        stale = tuple(stale_neural_torque_replay(item) for item in historical)
        final_replay = G1NeuralTorqueReplay.combine(anchor, *stale, *fresh_replays)
        all_replays.extend(fresh_replays)
        all_collection.extend(collection)
        generation_critic_updates = tuple(
            learner.update(final_replay, update_actor=False)
            for _ in range(critic_updates_per_generation)
        )
        all_critic_updates.extend(generation_critic_updates)
        parent_development = tuple(
            _context_evidence(
                backend,
                scenario,
                parameters,
                stable=stable_artifact,
                plastic=current_artifact,
                gate=gate,
                stage=f"recovery_online_g{generation}_parent_development",
                strict=False,
            )
            for scenario in development_scenarios
        )
        if not initial_parent_development:
            initial_parent_development = parent_development
        fresh_transition_count = int(
            np.count_nonzero(
                (final_replay.partitions == 0) & (final_replay.policy_lags <= 1)
            )
        )
        stale_transition_count = int(np.count_nonzero(final_replay.policy_lags > 1))
        replay_hash = neural_torque_replay_hash(final_replay)
        if fresh_transition_count < config.batch_size:
            generation_records.append(
                {
                    "generation": generation,
                    "accepted": False,
                    "stop_reason": "insufficient_fresh_actor_transitions",
                    "fresh_actor_transition_count": fresh_transition_count,
                    "stale_or_anchor_transition_count": stale_transition_count,
                    "replay_hash": replay_hash,
                }
            )
            break

        parent_snapshot = learner.actor_snapshot()
        actor_update = learner.update(final_replay, update_actor=True)
        proposal_snapshot = learner.actor_snapshot()
        all_actor_updates.append(actor_update)
        actor_effect = _actor_effect(
            learner,
            parent_snapshot,
            proposal_snapshot,
            final_replay,
        )
        online_hash = canonical_hash(
            {
                "plastic_parent_dataset_hash": plastic_dataset_hash,
                "current_artifact_hash": current_artifact.artifact_hash,
                "generation": generation,
                "recovery_replay_hash": replay_hash,
            }
        )
        generation_trust_runs: list[dict[str, Any]] = []
        eligible: list[dict[str, Any]] = []
        for fraction in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0):
            learner.install_interpolated_actor(
                parent_snapshot,
                proposal_snapshot,
                fraction=fraction,
            )
            path = root / (
                f"recovery-online-g{generation}-trust-"
                f"{str(fraction).replace('.', 'p')}.bin"
            )
            _write_candidate(
                path,
                learner,
                body_hash=body_hash,
                parent_policy_hash=parent_policy_hash,
                dataset_hash=canonical_hash(
                    {
                        "online_dataset_hash": online_hash,
                        "actor_trust_fraction": fraction,
                    }
                ),
            )
            artifact = load_g1_neural_torque_artifact(path, expected_body_hash=body_hash)
            values = tuple(
                _context_evidence(
                    backend,
                    scenario,
                    parameters,
                    stable=stable_artifact,
                    plastic=artifact,
                    gate=gate,
                    stage=f"recovery_online_g{generation}_{fraction}_development",
                    strict=False,
                )
                for scenario in development_scenarios
            )
            accepted, reasons = _recovery_update_gate(
                parent=parent_development,
                candidate=values,
            )
            aggregate = _aggregate(values)
            meaningful = _meaningful_recovery_improvement(
                _aggregate(parent_development), aggregate
            )
            if accepted and not meaningful:
                reasons = (*reasons, "no_measurable_recovery_improvement")
                accepted = False
            value = {
                "generation": generation,
                "fraction": fraction,
                "artifact_path": str(path),
                "artifact_hash": artifact.artifact_hash,
                "accepted": accepted,
                "meaningful": meaningful,
                "reasons": list(reasons),
                "aggregate": aggregate,
                "rollouts": [item.to_dict() for item in values],
            }
            generation_trust_runs.append(value)
            all_trust_runs.append(value)
            if accepted:
                eligible.append(value)
        selected = (
            max(
                eligible,
                key=lambda item: (
                    float(item["aggregate"]["mean_score"]),
                    -float(item["aggregate"]["mean_support_slip_m"]),
                    -float(item["fraction"]),
                ),
            )
            if eligible
            else None
        )
        generation_records.append(
            {
                "generation": generation,
                "accepted": selected is not None,
                "fresh_actor_transition_count": fresh_transition_count,
                "stale_or_anchor_transition_count": stale_transition_count,
                "replay_hash": replay_hash,
                "collection_scenario_count": len(collection),
                "actor_update": actor_update.to_dict(),
                "actor_effect": actor_effect,
                "parent_aggregate": _aggregate(parent_development),
                "selected": selected,
                "trust_candidate_count": len(generation_trust_runs),
            }
        )
        if selected is None:
            learner.install_interpolated_actor(
                parent_snapshot,
                proposal_snapshot,
                fraction=0.0,
            )
            break
        current_artifact = load_g1_neural_torque_artifact(
            Path(str(selected["artifact_path"])),
            expected_body_hash=body_hash,
        )
        learner.install_actor_artifact(
            current_artifact,
            expected_body_hash=body_hash,
            expected_parent_policy_hash=parent_policy_hash,
        )
        historical.extend(fresh_replays)
        final_selected = selected
        accepted_generation_count += 1

    return _OnlineGenerationResult(
        selected_artifact=current_artifact,
        final_replay=final_replay,
        recovery_replays=tuple(all_replays),
        collection=tuple(all_collection),
        critic_updates=tuple(all_critic_updates),
        actor_updates=tuple(all_actor_updates),
        parent_development=initial_parent_development,
        trust_runs=tuple(all_trust_runs),
        generations=tuple(generation_records),
        selected=final_selected,
        accepted_generation_count=accepted_generation_count,
    )


def _collect_online_generation(
    *,
    backend: G1MuJoCoBackend,
    parameters: ShotParameters,
    stable_artifact: G1NeuralTorqueArtifact,
    plastic_artifact: G1NeuralTorqueArtifact,
    gate: G1StabilityPlasticityGateConfig,
    scenarios: tuple[Any, ...],
    config: G1NeuralTorqueLearnerConfig,
    seed: int,
    generation: int,
) -> tuple[tuple[G1NeuralTorqueReplay, ...], tuple[dict[str, Any], ...]]:
    replays: list[G1NeuralTorqueReplay] = []
    collection: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        exploration = G1TorqueExplorationConfig(
            noise_std_ratio=0.004,
            noise_clip_ratio=0.012,
            temporal_correlation=0.995,
            minimum_recovery_phase=gate.minimum_recovery_phase,
            minimum_pelvis_height_m=gate.minimum_pelvis_height_m,
            maximum_projected_gravity_z=gate.maximum_projected_gravity_z,
            seed=seed + 100 + scenario_index,
        )
        episode, policy = _run_context(
            backend,
            scenario,
            parameters,
            stable=stable_artifact,
            plastic=plastic_artifact,
            gate=gate,
            exploration=exploration,
        )
        receipt = policy.build_receipt()
        score = _recovery_score(episode.result)
        replay = recovery_online_replay(
            policy.plastic.episode(),
            trajectory=episode.trajectory,
            sequence_length=config.sequence_length,
            recovery_score=score,
            fell=episode.result.post_kick_fall,
            critical_failure=_critical(episode.result),
            projection_fallback_rate=(
                receipt.projection_fallback_count / max(1, receipt.inference_count)
            ),
            recovery_start_phase=(
                gate.minimum_anticipatory_phase or gate.minimum_recovery_phase
            ),
            actor_eligible_mask=policy.plastic_activation_mask(),
            fall_quarantine_sec=0.10,
            stride=10,
        )
        replays.append(replay)
        collection.append(
            {
                "generation": generation,
                "scenario_id": scenario.scenario_id,
                "source_artifact_hash": plastic_artifact.artifact_hash,
                "recovery_score": score,
                "transition_count": replay.count,
                "fresh_actor_transition_count": int(
                    np.count_nonzero(replay.partitions == 0)
                ),
                "critic_only_transition_count": int(
                    np.count_nonzero(replay.partitions == 2)
                ),
                "mean_fall_cost": float(np.mean(replay.fall_costs)),
                "mean_constraint_cost": float(np.mean(replay.constraint_costs)),
                "plastic_activation_fraction": receipt.plastic_activation_fraction,
                "anticipatory_blend_count": receipt.anticipatory_blend_count,
                "exploration_config_hash": receipt.exploration_config_hash,
                "exploration_attempt_count": receipt.exploration_attempt_count,
                "exploration_applied_count": receipt.exploration_applied_count,
                "exploration_rejection_count": receipt.exploration_rejection_count,
                "exploration_noise_rms_ratio": receipt.exploration_noise_rms_ratio,
                "exploration_noise_peak_ratio": receipt.exploration_noise_peak_ratio,
                "result": episode.result.summary_dict(),
            }
        )
    return tuple(replays), tuple(collection)


def _actor_effect(
    learner: G1ContinualTorqueActorCritic,
    parent: dict[str, np.ndarray],
    proposal: dict[str, np.ndarray],
    replay: G1NeuralTorqueReplay,
) -> dict[str, float]:
    parameter_square = 0.0
    parent_square = 0.0
    parameter_peak = 0.0
    changed = 0
    total = 0
    for name in sorted(parent):
        before = np.asarray(parent[name], dtype=np.float64)
        after = np.asarray(proposal[name], dtype=np.float64)
        delta = after - before
        parameter_square += float(np.sum(np.square(delta)))
        parent_square += float(np.sum(np.square(before)))
        parameter_peak = max(parameter_peak, float(np.max(np.abs(delta))))
        changed += int(np.count_nonzero(np.abs(delta) > 1e-8))
        total += delta.size
    eligible = np.flatnonzero((replay.partitions == 0) & (replay.policy_lags <= 1))
    if len(eligible) > 256:
        eligible = eligible[np.linspace(0, len(eligible) - 1, 256, dtype=np.int64)]
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    learner.install_interpolated_actor(parent, proposal, fraction=0.0)
    parent_actions = np.asarray(
        [learner.deterministic_action(replay.observations[index]) for index in eligible]
    )
    learner.install_interpolated_actor(parent, proposal, fraction=1.0)
    proposal_actions = np.asarray(
        [learner.deterministic_action(replay.observations[index]) for index in eligible]
    )
    action_ratio = (
        (proposal_actions - parent_actions) / limits if len(eligible) else np.zeros((0, 29))
    )
    return {
        "parameter_l2": math.sqrt(parameter_square),
        "parameter_relative_l2": math.sqrt(parameter_square / max(parent_square, 1e-24)),
        "parameter_peak_abs": parameter_peak,
        "parameter_changed_fraction_gt_1e_8": changed / max(1, total),
        "action_sequence_count": float(len(eligible)),
        "raw_action_delta_rms_ratio": (
            float(np.sqrt(np.mean(np.square(action_ratio)))) if len(eligible) else 0.0
        ),
        "raw_action_delta_peak_ratio": (
            float(np.max(np.abs(action_ratio))) if len(eligible) else 0.0
        ),
    }


def _run_context(
    backend: G1MuJoCoBackend,
    scenario: Any,
    parameters: ShotParameters,
    *,
    stable: G1NeuralTorqueArtifact,
    plastic: G1NeuralTorqueArtifact,
    gate: G1StabilityPlasticityGateConfig,
    exploration: G1TorqueExplorationConfig | None = None,
) -> tuple[GoalForgeEpisode, G1StabilityPlasticityTorquePolicy]:
    policy = G1StabilityPlasticityTorquePolicy(
        G1NeuralTorquePolicy(
            stable,
            expected_body_hash=backend.qualification.body_hash,
            expected_parent_policy_hash=backend.qualification.kick_prior_hash,
        ),
        G1NeuralTorquePolicy(
            plastic,
            expected_body_hash=backend.qualification.body_hash,
            expected_parent_policy_hash=backend.qualification.kick_prior_hash,
            exploration=exploration,
        ),
        config=gate,
    )
    return backend.run(scenario, parameters, torque_policy=policy), policy


def _context_evidence(
    backend: G1MuJoCoBackend,
    scenario: Any,
    parameters: ShotParameters,
    *,
    stable: G1NeuralTorqueArtifact,
    plastic: G1NeuralTorqueArtifact,
    gate: G1StabilityPlasticityGateConfig,
    stage: str,
    strict: bool,
) -> G1NeuralTorqueRolloutEvidence:
    episode, policy = _run_context(
        backend, scenario, parameters, stable=stable, plastic=plastic, gate=gate
    )
    strict_replay = False
    if strict:
        replay, replay_policy = _run_context(
            backend, scenario, parameters, stable=stable, plastic=plastic, gate=gate
        )
        strict_replay = bool(
            episode.result.summary_dict() == replay.result.summary_dict()
            and trajectory_digest(episode.trajectory) == trajectory_digest(replay.trajectory)
            and policy.build_receipt().to_dict() == replay_policy.build_receipt().to_dict()
        )
    return _rollout_evidence(
        stage=stage,
        episode=episode,
        policy=policy,  # type: ignore[arg-type]
        strict_replay=strict_replay,
    )


def _recovery_score(result: GoalForgeResult) -> float:
    value = 3.0 * float(not result.post_kick_fall)
    value -= 8.0 * float(result.post_kick_fall)
    value -= 5.0 * float(result.joint_limit_violation or result.torque_limit_violation)
    value -= 20.0 * min(result.support_foot_slip_m, 0.20)
    value -= 2.0 * result.torso_roll_peak_rad
    value -= 2.0 * result.torso_pitch_peak_rad
    value += min(result.post_kick_stability_time_sec, 2.0)
    return float(max(-20.0, min(20.0, value)))


def _meaningful_recovery_improvement(
    parent: dict[str, float],
    candidate: dict[str, float],
) -> bool:
    return bool(
        candidate["critical_failure_rate"] < parent["critical_failure_rate"]
        or candidate["mean_score"] >= parent["mean_score"] + 0.05
        or candidate["mean_support_slip_m"] <= parent["mean_support_slip_m"] * 0.98
        or candidate["mean_torso_roll_peak_rad"]
        <= parent["mean_torso_roll_peak_rad"] * 0.98
    )


def _recovery_update_gate(
    *,
    parent: tuple[G1NeuralTorqueRolloutEvidence, ...],
    candidate: tuple[G1NeuralTorqueRolloutEvidence, ...],
) -> tuple[bool, tuple[str, ...]]:
    accepted, base_reasons = _online_update_gate(parent=parent, candidate=candidate)
    if len(parent) != len(candidate) or not parent:
        return accepted, base_reasons
    reasons = list(base_reasons)
    parent_slip = float(np.mean([item.support_slip_m for item in parent]))
    candidate_slip = float(np.mean([item.support_slip_m for item in candidate]))
    maximum_slip = max(parent_slip * 1.15, parent_slip + 0.003)
    if candidate_slip > maximum_slip:
        reasons.append("matched_support_slip_regression")
    for label in ("torso_roll_peak_rad", "torso_pitch_peak_rad"):
        before = float(np.mean([getattr(item, label) for item in parent]))
        after = float(np.mean([getattr(item, label) for item in candidate]))
        maximum = max(before * 1.05, before + 0.02)
        if after > maximum:
            reasons.append(f"matched_{label}_regression")
    return not reasons, tuple(reasons)


def _anticipatory_development_scenarios() -> tuple[GoalForgeScenario, ...]:
    """Return disjoint difficult development cases for pre-contact balance."""

    ledger = SeedLedger(
        task_id="g1_anticipatory_recovery_development",
        secret=b"rosclaw-phase8-anticipatory-recovery-v1" * 2,
    )
    values: list[GoalForgeScenario] = []
    for index, generation in enumerate((4, 7)):
        generated = generate_goalforge_scenarios(
            ledger=ledger,
            partition=Partition.DEVELOPMENT,
            count=index + 1,
            generation=generation,
        )
        values.append(
            replace(
                generated[index],
                scenario_id=f"anticipatory-dev-{index:02d}-g{generation}",
            )
        )
    ledger.assert_disjoint()
    return tuple(values)


def _external_root(path: Path, source_checkout: Path) -> Path:
    root = path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("recovery online-RL evidence must be outside the source checkout")
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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = ["run_g1_recovery_online_rl_validation"]
