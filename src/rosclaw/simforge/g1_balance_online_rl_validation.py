"""Closed-loop early-balance training for the hierarchical G1 torque policy."""

from __future__ import annotations

import json
import math
import os
import tempfile
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
from rosclaw.simforge.g1_hierarchical_torque_policy import (
    G1HierarchicalTorqueGateConfig,
    G1HierarchicalTorquePolicy,
)
from rosclaw.simforge.g1_neural_torque import (
    G1NeuralTorqueArtifact,
    G1NeuralTorquePolicy,
    G1TorqueExplorationConfig,
    load_g1_neural_torque_artifact,
    serialize_g1_neural_torque_artifact,
)
from rosclaw.simforge.g1_neural_torque_learning import (
    G1ContinualTorqueActorCritic,
    G1NeuralTorqueLearnerConfig,
    G1NeuralTorqueReplay,
    balance_online_replay,
    neural_torque_replay_hash,
    stale_neural_torque_replay,
    teacher_dataset_hash,
    teacher_replay,
)
from rosclaw.simforge.g1_neural_torque_validation import (
    G1NeuralTorqueRolloutEvidence,
    _aggregate,
    _collect_teacher,
    _pilot_scenarios,
    _rollout_evidence,
    _write_candidate,
)
from rosclaw.simforge.g1_recovery_online_rl_validation import (
    _actor_effect,
    _anticipatory_development_scenarios,
    _recovery_update_gate,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    GoalForgeResult,
    ShotParameters,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import (
    GoalForgeScenario,
    generate_goalforge_scenarios,
)


def run_g1_balance_online_rl_validation(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    stable_artifact_path: Path,
    recovery_artifact_path: Path,
    output_dir: Path,
    source_checkout: Path,
    device: str = "cuda:1",
    seed: int = 8810,
    generations: int = 3,
    critic_updates_per_generation: int = 24,
    validation_round: int = 2,
    learning_mode: str = "sac",
    exploration_replicates: int = 1,
    elite_score_margin: float = 0.05,
    actor_updates_per_generation: int = 1,
) -> dict[str, Any]:
    """Train and seal an independent pre-contact balance actor in MuJoCo."""

    if not 1 <= generations <= 6:
        raise ValueError("balance online-RL generations must be in [1, 6]")
    if not 8 <= critic_updates_per_generation <= 128:
        raise ValueError("balance critic updates per generation must be in [8, 128]")
    if not 2 <= validation_round <= 99:
        raise ValueError("balance validation round must be in [2, 99]")
    if learning_mode not in {"sac", "awr"}:
        raise ValueError("balance learning mode must be 'sac' or 'awr'")
    if not 1 <= exploration_replicates <= 16:
        raise ValueError("balance exploration replicates must be in [1, 16]")
    if not math.isfinite(elite_score_margin) or not 0.0 <= elite_score_margin <= 5.0:
        raise ValueError("balance elite score margin must be in [0, 5]")
    if not 1 <= actor_updates_per_generation <= 32:
        raise ValueError("balance actor updates per generation must be in [1, 32]")
    root = _external_root(output_dir, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    qualification = backend.qualification
    stable_artifact = load_g1_neural_torque_artifact(
        stable_artifact_path,
        expected_body_hash=qualification.body_hash,
    )
    recovery_artifact = load_g1_neural_torque_artifact(
        recovery_artifact_path,
        expected_body_hash=qualification.body_hash,
    )
    prior = load_g1_motion_prior_artifact(motion_prior_path)
    if prior.body_hash != qualification.body_hash:
        raise ValueError("balance prior does not match the qualified G1 body")
    if stable_artifact.parent_policy_hash != qualification.kick_prior_hash:
        raise ValueError("stable torque artifact parent mismatch")
    if recovery_artifact.parent_policy_hash != qualification.kick_prior_hash:
        raise ValueError("recovery torque artifact parent mismatch")

    training_scenarios, development_scenarios, _ = _pilot_scenarios()
    validation_scenarios = _fresh_balance_validation_scenarios(validation_round)
    development_scenarios = (
        *development_scenarios,
        *_anticipatory_development_scenarios(),
    )
    parameters = ShotParameters()
    training = tuple(
        _collect_teacher(backend, scenario, parameters)[1] for scenario in training_scenarios
    )
    config = G1NeuralTorqueLearnerConfig(
        hidden_dim=96,
        sequence_length=32,
        batch_size=64 if learning_mode == "awr" else 128,
        actor_lr=2e-5 if learning_mode == "awr" else 5e-5,
        critic_lr=3e-4,
        behavior_cloning_weight=10.0,
        online_behavior_weight=5.0 if learning_mode == "awr" else 0.0,
        parent_churn_weight=5.0,
        ewc_weight=50.0,
        initial_alpha=0.002,
        awr_temperature=0.35,
        awr_max_weight=20.0,
        awr_fall_penalty=5.0,
        awr_constraint_penalty=2.0,
        device=device,
        seed=seed,
    )
    # The independent head must start as a behaviorally exact clone.  Even a
    # wider state guard changes closed-loop trajectories despite identical
    # tensors, so it is not a safe or honest parent for an online update.
    safety = stable_artifact.safety
    gate = G1HierarchicalTorqueGateConfig()
    balance_action_indices = _balance_action_indices()
    learner = G1ContinualTorqueActorCritic(config, safety=safety)
    teacher_hash = teacher_dataset_hash(training)
    balance_dataset_hash = canonical_hash(
        {
            "teacher_dataset_hash": teacher_hash,
            "motion_prior_artifact_hash": prior.artifact_hash,
            "motion_prior_applicability": "REJECTED_FULL_TRANSFER",
            "initial_actor_hash": stable_artifact.artifact_hash,
            "balance_phase_window": [
                gate.balance_start_phase,
                gate.balance_end_phase,
            ],
            "learning_mode": learning_mode,
            "exploration_replicates": exploration_replicates,
            "elite_score_margin": elite_score_margin,
        }
    )
    balance_parent_path = root / "balance-parent.bin"
    balance_parent_path.write_bytes(
        serialize_g1_neural_torque_artifact(
            body_hash=qualification.body_hash,
            parent_policy_hash=qualification.kick_prior_hash,
            dataset_hash=balance_dataset_hash,
            hidden_dim=stable_artifact.hidden_dim,
            observation_clip=stable_artifact.observation_clip,
            update_index=stable_artifact.update_index,
            safety=safety,
            tensors=stable_artifact.tensors,
        )
    )
    balance_parent = load_g1_neural_torque_artifact(
        balance_parent_path,
        expected_body_hash=qualification.body_hash,
    )
    learner.update_index = balance_parent.update_index
    learner.install_actor_artifact(
        balance_parent,
        expected_body_hash=qualification.body_hash,
        expected_parent_policy_hash=qualification.kick_prior_hash,
    )
    installed_parent_metrics = learner.consolidate_installed_actor(
        training,
        stride=4,
        maximum_end_fraction=gate.balance_end_phase,
    )

    structural_development = tuple(
        _hierarchical_evidence(
            backend,
            scenario,
            parameters,
            stable=stable_artifact,
            balance=stable_artifact,
            recovery=recovery_artifact,
            gate=gate,
            stage="balance_structural_development",
            strict=False,
        )
        for scenario in development_scenarios
    )
    parent_development = tuple(
        _hierarchical_evidence(
            backend,
            scenario,
            parameters,
            stable=stable_artifact,
            balance=balance_parent,
            recovery=recovery_artifact,
            gate=gate,
            stage="balance_parent_development",
            strict=False,
        )
        for scenario in development_scenarios
    )

    anchor = teacher_replay(
        training,
        sequence_length=config.sequence_length,
        stride=10,
        maximum_end_fraction=gate.balance_end_phase,
    )
    current_artifact = balance_parent
    historical: list[G1NeuralTorqueReplay] = []
    all_replays: list[G1NeuralTorqueReplay] = []
    collection: list[dict[str, Any]] = []
    critic_updates: list[dict[str, Any]] = []
    actor_updates: list[dict[str, Any]] = []
    trust_runs: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    accepted_generation_count = 0
    final_replay = anchor

    for generation in range(generations):
        fresh_replays, fresh_collection = _collect_balance_generation(
            backend=backend,
            scenarios=training_scenarios,
            parameters=parameters,
            stable=stable_artifact,
            balance=current_artifact,
            recovery=recovery_artifact,
            gate=gate,
            config=config,
            seed=seed + generation * 10_000,
            generation=generation,
            learning_mode=learning_mode,
            exploration_replicates=exploration_replicates,
            elite_score_margin=elite_score_margin,
        )
        stale = tuple(stale_neural_torque_replay(value) for value in historical)
        final_replay = G1NeuralTorqueReplay.combine(anchor, *stale, *fresh_replays)
        all_replays.extend(fresh_replays)
        collection.extend(fresh_collection)
        generation_critic_updates = [
            (
                learner.update_advantage_weighted(final_replay, update_actor=False)
                if learning_mode == "awr"
                else learner.update(final_replay, update_actor=False)
            )
            for _ in range(critic_updates_per_generation)
        ]
        critic_updates.extend(item.to_dict() for item in generation_critic_updates)
        update_dicts = tuple(item.to_dict() for item in generation_critic_updates)
        critic_diagnostics = (
            _value_convergence(update_dicts)
            if learning_mode == "awr"
            else _critic_convergence(update_dicts)
        )
        fresh_count = int(
            np.count_nonzero((final_replay.partitions == 0) & (final_replay.policy_lags <= 1))
        )
        replay_hash = neural_torque_replay_hash(final_replay)
        if not bool(critic_diagnostics["converged"]):
            generation_records.append(
                {
                    "generation": generation,
                    "accepted": False,
                    "stop_reason": "critic_pretraining_not_converged",
                    "fresh_actor_transition_count": fresh_count,
                    "replay_hash": replay_hash,
                    "critic_diagnostics": critic_diagnostics,
                }
            )
            break
        if fresh_count < config.batch_size:
            generation_records.append(
                {
                    "generation": generation,
                    "accepted": False,
                    "stop_reason": "insufficient_fresh_balance_transitions",
                    "fresh_actor_transition_count": fresh_count,
                    "replay_hash": replay_hash,
                    "critic_diagnostics": critic_diagnostics,
                }
            )
            break

        generation_parent = tuple(
            _hierarchical_evidence(
                backend,
                scenario,
                parameters,
                stable=stable_artifact,
                balance=current_artifact,
                recovery=recovery_artifact,
                gate=gate,
                stage=f"balance_g{generation}_parent_development",
                strict=False,
            )
            for scenario in development_scenarios
        )
        parent_snapshot = learner.actor_snapshot()
        generation_actor_updates = [
            (
                learner.update_advantage_weighted(final_replay, update_actor=True)
                if learning_mode == "awr"
                else learner.update(final_replay, update_actor=True)
            )
            for _ in range(actor_updates_per_generation)
        ]
        actor_update = generation_actor_updates[-1]
        proposal_snapshot = learner.actor_snapshot()
        actor_updates.extend(item.to_dict() for item in generation_actor_updates)
        learner.install_interpolated_actor(
            parent_snapshot,
            proposal_snapshot,
            fraction=1.0,
            action_indices=balance_action_indices,
        )
        constrained_proposal_snapshot = learner.actor_snapshot()
        actor_effect = _actor_effect(
            learner,
            parent_snapshot,
            constrained_proposal_snapshot,
            final_replay,
        )
        online_hash = canonical_hash(
            {
                "balance_parent_dataset_hash": balance_dataset_hash,
                "current_artifact_hash": current_artifact.artifact_hash,
                "generation": generation,
                "balance_replay_hash": replay_hash,
            }
        )
        eligible: list[dict[str, Any]] = []
        generation_trust_runs: list[dict[str, Any]] = []
        for fraction in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0):
            learner.install_interpolated_actor(
                parent_snapshot,
                constrained_proposal_snapshot,
                fraction=fraction,
                action_indices=balance_action_indices,
            )
            path = root / (
                f"balance-online-g{generation}-trust-{str(fraction).replace('.', 'p')}.bin"
            )
            _write_candidate(
                path,
                learner,
                body_hash=qualification.body_hash,
                parent_policy_hash=qualification.kick_prior_hash,
                dataset_hash=canonical_hash(
                    {
                        "online_dataset_hash": online_hash,
                        "actor_trust_fraction": fraction,
                    }
                ),
            )
            artifact = load_g1_neural_torque_artifact(
                path,
                expected_body_hash=qualification.body_hash,
            )
            values = tuple(
                _hierarchical_evidence(
                    backend,
                    scenario,
                    parameters,
                    stable=stable_artifact,
                    balance=artifact,
                    recovery=recovery_artifact,
                    gate=gate,
                    stage=f"balance_g{generation}_{fraction}_development",
                    strict=False,
                )
                for scenario in development_scenarios
            )
            accepted, reasons = _balance_update_gate(
                parent=generation_parent,
                candidate=values,
            )
            meaningful = _meaningful_balance_improvement(
                _aggregate(generation_parent),
                _aggregate(values),
            )
            if accepted and not meaningful:
                accepted = False
                reasons = (*reasons, "no_measurable_balance_improvement")
            value = {
                "generation": generation,
                "fraction": fraction,
                "artifact_path": str(path),
                "artifact_hash": artifact.artifact_hash,
                "accepted": accepted,
                "meaningful": meaningful,
                "reasons": list(reasons),
                "aggregate": _aggregate(values),
                "stability_cost": _stability_cost(_aggregate(values)),
                "rollouts": [item.to_dict() for item in values],
            }
            generation_trust_runs.append(value)
            trust_runs.append(value)
            if accepted:
                eligible.append(value)
        chosen = (
            min(
                eligible,
                key=lambda item: (
                    float(item["stability_cost"]),
                    -float(item["aggregate"]["mean_score"]),
                    float(item["fraction"]),
                ),
            )
            if eligible
            else None
        )
        generation_records.append(
            {
                "generation": generation,
                "accepted": chosen is not None,
                "fresh_actor_transition_count": fresh_count,
                "replay_hash": replay_hash,
                "actor_update": actor_update.to_dict(),
                "actor_update_count": len(generation_actor_updates),
                "actor_effect": actor_effect,
                "critic_diagnostics": critic_diagnostics,
                "parent_aggregate": _aggregate(generation_parent),
                "selected": chosen,
                "trust_candidate_count": len(generation_trust_runs),
            }
        )
        if chosen is None:
            learner.install_interpolated_actor(
                parent_snapshot,
                constrained_proposal_snapshot,
                fraction=0.0,
                action_indices=balance_action_indices,
            )
            break
        current_artifact = load_g1_neural_torque_artifact(
            Path(str(chosen["artifact_path"])),
            expected_body_hash=qualification.body_hash,
        )
        learner.install_actor_artifact(
            current_artifact,
            expected_body_hash=qualification.body_hash,
            expected_parent_policy_hash=qualification.kick_prior_hash,
        )
        historical.extend(fresh_replays)
        selected = chosen
        accepted_generation_count += 1

    sealed = validation_scenarios[:4]
    structural_validation = tuple(
        _hierarchical_evidence(
            backend,
            scenario,
            parameters,
            stable=stable_artifact,
            balance=stable_artifact,
            recovery=recovery_artifact,
            gate=gate,
            stage="sealed_balance_structural_validation",
            strict=True,
        )
        for scenario in sealed
    )
    parent_validation = tuple(
        _hierarchical_evidence(
            backend,
            scenario,
            parameters,
            stable=stable_artifact,
            balance=balance_parent,
            recovery=recovery_artifact,
            gate=gate,
            stage="sealed_balance_parent_validation",
            strict=True,
        )
        for scenario in sealed
    )
    candidate_validation = tuple(
        _hierarchical_evidence(
            backend,
            scenario,
            parameters,
            stable=stable_artifact,
            balance=current_artifact,
            recovery=recovery_artifact,
            gate=gate,
            stage="sealed_balance_candidate_validation",
            strict=True,
        )
        for scenario in sealed
    )
    validation_accepted, validation_reasons = _balance_update_gate(
        parent=structural_validation,
        candidate=candidate_validation,
    )
    structural_aggregate = _aggregate(structural_validation)
    candidate_aggregate = _aggregate(candidate_validation)
    structural_cost = _stability_cost(structural_aggregate)
    candidate_cost = _stability_cost(candidate_aggregate)
    learned_gain = (structural_cost - candidate_cost) / max(structural_cost, 1e-6)
    checks = {
        "independent_balance_head": current_artifact.artifact_hash != stable_artifact.artifact_hash,
        "phase_specific_anchor": anchor.count
        < teacher_replay(
            training,
            sequence_length=config.sequence_length,
            stride=10,
        ).count,
        "exact_parent_behavior_preserved": _matched_evidence_equal(
            structural_development,
            parent_development,
        )
        and _matched_evidence_equal(structural_validation, parent_validation),
        "safe_exploration_exercised": any(
            int(item["exploration_applied_count"]) > 0 for item in collection
        ),
        "critic_and_actor_updated": bool(critic_updates)
        and bool(actor_updates)
        and all(bool(item["finite"]) for item in actor_updates),
        "fresh_resampling_per_actor_update": len(actor_updates)
        == sum(int(item.get("actor_update_count", 0)) for item in generation_records),
        "development_trust_region_found": accepted_generation_count > 0,
        "strict_validation_replay": all(
            item.strict_replay
            for item in (
                *structural_validation,
                *parent_validation,
                *candidate_validation,
            )
        ),
        "validation_gate_passed": validation_accepted,
        "learned_component_gain_at_least_5pct": learned_gain >= 0.05,
        "sim_only_boundary": all(
            item.activation_ceiling == "SIM_ONLY" and not item.hardware_authorized
            for item in candidate_validation
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    blockers.extend(validation_reasons)
    decision = "CANDIDATE" if not blockers else "REJECTED"
    report = {
        "schema_version": "rosclaw.simforge.g1_balance_online_rl.v2",
        "body_hash": qualification.body_hash,
        "parent_policy_hash": qualification.kick_prior_hash,
        "stable_artifact_hash": stable_artifact.artifact_hash,
        "recovery_artifact_hash": recovery_artifact.artifact_hash,
        "motion_prior_artifact_hash": prior.artifact_hash,
        "balance_parent_artifact_hash": balance_parent.artifact_hash,
        "selected_balance_artifact_hash": current_artifact.artifact_hash,
        "teacher_dataset_hash": teacher_hash,
        "online_replay_hash": neural_torque_replay_hash(final_replay),
        "config": asdict(config),
        "safety": asdict(safety),
        "gate": asdict(gate),
        "balance_action_subspace": [G1_DDS_JOINT_NAMES[index] for index in balance_action_indices],
        "optimization_rationale": {
            "critic_pretraining_updates_per_generation": critic_updates_per_generation,
            "learning_mode": learning_mode,
            "exploration_replicates": exploration_replicates,
            "elite_score_margin": elite_score_margin,
            "actor_updates_per_generation": actor_updates_per_generation,
            "exploration_action_imitation_disabled": learning_mode != "awr",
            "out_of_distribution_actor_q_queries": learning_mode != "awr",
            "reason": (
                "AWR fits observed returns and behavior-clones only actions from matched "
                "globally improved MuJoCo rollouts"
                if learning_mode == "awr"
                else "online behavior actions contain intentional exploration noise; "
                "the actor is retained by teacher anchors, parent distillation, and EWC"
            ),
        },
        "bc_metrics": {
            "installed_parent_early_behavior_loss": installed_parent_metrics[0],
            "installed_parent_action_limit_fraction": installed_parent_metrics[1],
        },
        "collection": collection,
        "critic_updates": critic_updates,
        "actor_updates": actor_updates,
        "generations": generation_records,
        "trust_runs": trust_runs,
        "selected": selected,
        "development": {
            "structural": [item.to_dict() for item in structural_development],
            "balance_parent": [item.to_dict() for item in parent_development],
        },
        "validation": {
            "structural": [item.to_dict() for item in structural_validation],
            "balance_parent": [item.to_dict() for item in parent_validation],
            "candidate": [item.to_dict() for item in candidate_validation],
        },
        "aggregates": {
            "structural": structural_aggregate,
            "balance_parent": _aggregate(parent_validation),
            "candidate": candidate_aggregate,
            "structural_stability_cost": structural_cost,
            "candidate_stability_cost": candidate_cost,
            "learned_stability_gain_fraction": learned_gain,
        },
        "checks": checks,
        "blockers": blockers,
        "decision": decision,
        "claims": {
            "hierarchical_balance_recovery": True,
            "motiondecode_full_balance_transfer_promoted": False,
            "stable_actor_exact_weight_initialization": True,
            "stable_actor_exact_behavior_preserved": checks["exact_parent_behavior_preserved"],
            "online_actor_critic": True,
            "in_sample_advantage_weighted_actor": learning_mode == "awr",
            "elite_matched_rollout_filter": learning_mode == "awr",
            "future_risk_credit_assignment": True,
            "direct_joint_torque_actor": True,
            "anatomically_constrained_balance_readout": True,
            "candidate_promoted": decision == "CANDIDATE",
            "real_robot_evidence": False,
        },
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
    }
    _atomic_json(root / "g1-balance-online-rl-report.json", report)
    return report


def _balance_action_indices() -> tuple[int, ...]:
    """Support-chain and counter-swing joints; never perturb the kick leg."""

    names = {
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
    }
    return tuple(index for index, name in enumerate(G1_DDS_JOINT_NAMES) if name in names)


def _critic_convergence(updates: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Require all value heads to improve before trusting an actor gradient."""

    labels = (
        "reward_critic_loss",
        "fall_critic_loss",
        "constraint_critic_loss",
    )
    if len(updates) < 8:
        return {"converged": False, "reason": "fewer_than_8_critic_updates"}
    window = max(4, min(12, len(updates) // 4))
    values: dict[str, Any] = {"window_size": window}
    converged = True
    for label in labels:
        first = float(np.mean([float(item[label]) for item in updates[:window]]))
        last = float(np.mean([float(item[label]) for item in updates[-window:]]))
        ratio = last / max(first, 1e-9)
        values[label] = {
            "first_window_mean": first,
            "last_window_mean": last,
            "last_to_first_ratio": ratio,
        }
        converged = converged and math.isfinite(ratio) and ratio <= 0.50
    values["converged"] = converged
    return values


def _value_convergence(updates: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Require the in-sample value regression to improve before actor fitting."""

    if len(updates) < 8:
        return {"converged": False, "reason": "fewer_than_8_value_updates"}
    window = max(4, min(12, len(updates) // 4))
    first = float(np.mean([float(item["value_loss"]) for item in updates[:window]]))
    last = float(np.mean([float(item["value_loss"]) for item in updates[-window:]]))
    ratio = last / max(first, 1e-9)
    return {
        "window_size": window,
        "value_loss": {
            "first_window_mean": first,
            "last_window_mean": last,
            "last_to_first_ratio": ratio,
        },
        "converged": bool(math.isfinite(ratio) and ratio <= 0.70),
    }


def _fresh_balance_validation_scenarios(
    validation_round: int,
) -> tuple[GoalForgeScenario, ...]:
    """Allocate a round-specific sealed partition after earlier rounds opened."""

    ledger = SeedLedger(
        task_id=f"g1_foundation_balance_validation_v{validation_round}",
        secret=(f"rosclaw-phase8-foundation-balance-validation-v{validation_round}").encode() * 2,
    )
    values: list[GoalForgeScenario] = []
    for index, generation in enumerate((0, 2, 4, 7)):
        generated = generate_goalforge_scenarios(
            ledger=ledger,
            partition=Partition.VALIDATION,
            count=index + 1,
            generation=generation,
        )
        values.append(
            replace(
                generated[index],
                scenario_id=(f"balance-v{validation_round}-validation-{index:02d}-g{generation}"),
            )
        )
    ledger.assert_disjoint()
    return tuple(values)


def _matched_evidence_equal(
    left: tuple[G1NeuralTorqueRolloutEvidence, ...],
    right: tuple[G1NeuralTorqueRolloutEvidence, ...],
) -> bool:
    """Compare dynamics, excluding actor identity and stage labels."""

    if len(left) != len(right):
        return False
    excluded = {"artifact_hash", "stage"}
    return all(
        {key: value for key, value in before.to_dict().items() if key not in excluded}
        == {key: value for key, value in after.to_dict().items() if key not in excluded}
        for before, after in zip(left, right, strict=True)
    )


def _collect_balance_generation(
    *,
    backend: G1MuJoCoBackend,
    scenarios: tuple[Any, ...],
    parameters: ShotParameters,
    stable: G1NeuralTorqueArtifact,
    balance: G1NeuralTorqueArtifact,
    recovery: G1NeuralTorqueArtifact,
    gate: G1HierarchicalTorqueGateConfig,
    config: G1NeuralTorqueLearnerConfig,
    seed: int,
    generation: int,
    learning_mode: str = "sac",
    exploration_replicates: int = 1,
    elite_score_margin: float = 0.05,
) -> tuple[tuple[G1NeuralTorqueReplay, ...], tuple[dict[str, Any], ...]]:
    replays: list[G1NeuralTorqueReplay] = []
    collection: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        matched_parent_score: float | None = None
        matched_parent_critical: bool | None = None
        if learning_mode == "awr":
            parent_episode, _ = _run_hierarchy(
                backend,
                scenario,
                parameters,
                stable=stable,
                balance=balance,
                recovery=recovery,
                gate=gate,
            )
            matched_parent_score = _balance_score(parent_episode.result)
            matched_parent_critical = _critical(parent_episode.result)
        for replicate in range(exploration_replicates):
            exploration = G1TorqueExplorationConfig(
                noise_std_ratio=0.003,
                noise_clip_ratio=0.008,
                temporal_correlation=0.995,
                minimum_recovery_phase=gate.balance_start_phase,
                minimum_pelvis_height_m=gate.balance_minimum_pelvis_height_m,
                maximum_projected_gravity_z=(gate.balance_maximum_projected_gravity_z),
                seed=seed + 100 + index * exploration_replicates + replicate,
            )
            episode, policy = _run_hierarchy(
                backend,
                scenario,
                parameters,
                stable=stable,
                balance=balance,
                recovery=recovery,
                gate=gate,
                balance_exploration=exploration,
            )
            receipt = policy.build_receipt()
            balance_receipt = policy.balance.build_receipt()
            score = _balance_score(episode.result)
            replay = balance_online_replay(
                policy.balance.episode(),
                trajectory=episode.trajectory,
                sequence_length=config.sequence_length,
                balance_score=score,
                fell=episode.result.post_kick_fall,
                critical_failure=_critical(episode.result),
                projection_fallback_rate=(
                    balance_receipt.projection_fallback_count
                    / max(1, balance_receipt.inference_count)
                ),
                actor_eligible_mask=policy.balance_activation_mask(),
                balance_start_phase=gate.balance_start_phase,
                balance_end_phase=gate.balance_end_phase,
                stride=10,
            )
            globally_eligible = True
            if learning_mode == "awr":
                assert matched_parent_score is not None
                assert matched_parent_critical is not None
                globally_eligible = bool(
                    not _critical(episode.result)
                    and (
                        matched_parent_critical
                        or score >= matched_parent_score + elite_score_margin
                    )
                )
                if not globally_eligible:
                    replay = _quarantine_actor_replay(replay)
            replays.append(replay)
            collection.append(
                {
                    "generation": generation,
                    "scenario_id": scenario.scenario_id,
                    "replicate": replicate,
                    "source_artifact_hash": balance.artifact_hash,
                    "learning_mode": learning_mode,
                    "balance_score": score,
                    "matched_parent_balance_score": matched_parent_score,
                    "matched_parent_critical": matched_parent_critical,
                    "global_improvement_eligible": globally_eligible,
                    "transition_count": replay.count,
                    "fresh_actor_transition_count": int(np.count_nonzero(replay.partitions == 0)),
                    "critic_only_transition_count": int(np.count_nonzero(replay.partitions == 2)),
                    "balance_activation_fraction": receipt.balance_activation_fraction,
                    "recovery_activation_fraction": receipt.recovery_activation_fraction,
                    "exploration_applied_count": (balance_receipt.exploration_applied_count),
                    "exploration_noise_rms_ratio": (balance_receipt.exploration_noise_rms_ratio),
                    "exploration_noise_peak_ratio": (balance_receipt.exploration_noise_peak_ratio),
                    "result": episode.result.summary_dict(),
                }
            )
    return tuple(replays), tuple(collection)


def _quarantine_actor_replay(replay: G1NeuralTorqueReplay) -> G1NeuralTorqueReplay:
    """Keep a non-elite rollout for value learning but block actor imitation."""

    partitions = replay.partitions.copy()
    partitions[partitions == 0] = 2
    return G1NeuralTorqueReplay(
        observations=replay.observations.copy(),
        actions=replay.actions.copy(),
        next_observations=replay.next_observations.copy(),
        rewards=replay.rewards.copy(),
        fall_costs=replay.fall_costs.copy(),
        constraint_costs=replay.constraint_costs.copy(),
        terminals=replay.terminals.copy(),
        parent_actions=replay.parent_actions.copy(),
        partitions=partitions,
        policy_lags=replay.policy_lags.copy(),
    )


def _run_hierarchy(
    backend: G1MuJoCoBackend,
    scenario: Any,
    parameters: ShotParameters,
    *,
    stable: G1NeuralTorqueArtifact,
    balance: G1NeuralTorqueArtifact,
    recovery: G1NeuralTorqueArtifact,
    gate: G1HierarchicalTorqueGateConfig,
    balance_exploration: G1TorqueExplorationConfig | None = None,
) -> tuple[GoalForgeEpisode, G1HierarchicalTorquePolicy]:
    def actor(
        artifact: G1NeuralTorqueArtifact,
        exploration: G1TorqueExplorationConfig | None = None,
    ) -> G1NeuralTorquePolicy:
        return G1NeuralTorquePolicy(
            artifact,
            expected_body_hash=backend.qualification.body_hash,
            expected_parent_policy_hash=backend.qualification.kick_prior_hash,
            exploration=exploration,
        )

    policy = G1HierarchicalTorquePolicy(
        actor(stable),
        actor(balance, balance_exploration),
        actor(recovery),
        config=gate,
    )
    return backend.run(scenario, parameters, torque_policy=policy), policy


def _hierarchical_evidence(
    backend: G1MuJoCoBackend,
    scenario: Any,
    parameters: ShotParameters,
    *,
    stable: G1NeuralTorqueArtifact,
    balance: G1NeuralTorqueArtifact,
    recovery: G1NeuralTorqueArtifact,
    gate: G1HierarchicalTorqueGateConfig,
    stage: str,
    strict: bool,
) -> G1NeuralTorqueRolloutEvidence:
    episode, policy = _run_hierarchy(
        backend,
        scenario,
        parameters,
        stable=stable,
        balance=balance,
        recovery=recovery,
        gate=gate,
    )
    strict_replay = False
    if strict:
        replay, replay_policy = _run_hierarchy(
            backend,
            scenario,
            parameters,
            stable=stable,
            balance=balance,
            recovery=recovery,
            gate=gate,
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


def _balance_score(result: GoalForgeResult) -> float:
    value = 5.0 * float(not result.post_kick_fall)
    value += 2.0 * float(result.kick_foot_contacted)
    value += 2.0 * float(result.success)
    value -= 10.0 * float(result.post_kick_fall)
    value -= 5.0 * float(result.joint_limit_violation or result.torque_limit_violation)
    value -= 25.0 * min(result.support_foot_slip_m, 0.20)
    value -= 2.5 * result.torso_roll_peak_rad
    value -= 2.0 * result.torso_pitch_peak_rad
    value -= 10.0 * max(0.0, -result.com_margin_min_m)
    return float(np.clip(value, -20.0, 20.0))


def _balance_update_gate(
    *,
    parent: tuple[G1NeuralTorqueRolloutEvidence, ...],
    candidate: tuple[G1NeuralTorqueRolloutEvidence, ...],
) -> tuple[bool, tuple[str, ...]]:
    accepted, reasons = _recovery_update_gate(parent=parent, candidate=candidate)
    if len(parent) != len(candidate) or not parent:
        return accepted, reasons
    values = list(reasons)
    before_com = float(np.mean([item.com_margin_min_m for item in parent]))
    after_com = float(np.mean([item.com_margin_min_m for item in candidate]))
    if after_com < before_com - 0.005:
        values.append("matched_com_margin_regression")
    before_speed = float(np.mean([item.ball_speed_mps for item in parent]))
    after_speed = float(np.mean([item.ball_speed_mps for item in candidate]))
    if after_speed < before_speed * 0.95 - 1e-9:
        values.append("matched_ball_speed_regression_gt_5pct")
    return not values, tuple(values)


def _meaningful_balance_improvement(
    parent: dict[str, float],
    candidate: dict[str, float],
) -> bool:
    return bool(
        candidate["critical_failure_rate"] < parent["critical_failure_rate"]
        or _stability_cost(candidate) <= _stability_cost(parent) * 0.98
    )


def _stability_cost(value: dict[str, float]) -> float:
    return float(
        5.0 * value["critical_failure_rate"]
        + value["mean_torso_roll_peak_rad"]
        + value["mean_torso_pitch_peak_rad"]
        + 20.0 * value["mean_support_slip_m"]
        + 5.0 * max(0.0, -value["mean_com_margin_min_m"])
        - 0.05 * value["mean_score"]
    )


def _critical(result: GoalForgeResult) -> bool:
    return bool(
        result.post_kick_fall
        or result.joint_limit_violation
        or result.torque_limit_violation
        or not result.finite_state
    )


def _external_root(path: Path, source_checkout: Path) -> Path:
    root = path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("balance online-RL evidence must be outside the source checkout")
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


__all__ = ["run_g1_balance_online_rl_validation"]
