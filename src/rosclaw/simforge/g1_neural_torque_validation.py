"""Closed-loop G1 neural-torque distillation and online-RL pilot."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
    trajectory_digest,
)
from rosclaw.simforge.g1_neural_torque import (
    G1NeuralTorquePolicy,
    G1TeacherTorqueCollector,
    G1TeacherTorqueEpisode,
    G1TorquePolicyReceipt,
    G1TorqueSafetyConfig,
    load_g1_neural_torque_artifact,
)
from rosclaw.simforge.g1_neural_torque_learning import (
    G1ContinualTorqueActorCritic,
    G1NeuralTorqueLearnerConfig,
    G1NeuralTorqueReplay,
    G1NeuralTorqueUpdate,
    neural_torque_replay_hash,
    online_replay,
    teacher_dataset_hash,
    teacher_replay,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    GoalForgeResult,
    ShotParameters,
    hash_bytes,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import (
    GoalForgeScenario,
    generate_goalforge_scenarios,
)


@dataclass(frozen=True)
class G1NeuralTorqueRolloutEvidence:
    stage: str
    partition: str
    scenario_id: str
    scenario_commitment: str
    status: str
    success: bool
    contact: bool
    target_error_m: float | None
    ball_speed_mps: float
    support_slip_m: float
    com_margin_min_m: float
    torso_roll_peak_rad: float
    torso_pitch_peak_rad: float
    fall: bool
    joint_violation: bool
    torque_violation: bool
    finite: bool
    score: float
    direct_torque_inferences: int
    learned_output_fraction: float
    fallback_fraction: float
    artifact_hash: str | None
    activation_ceiling: str | None
    hardware_authorized: bool
    strict_replay: bool
    trace_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1NeuralTorquePilotReport:
    source_commit: str
    body_hash: str
    parent_policy_hash: str
    dataset_hash: str
    gpu_workers: tuple[dict[str, Any], ...]
    selected_seed: int
    teacher_pass_through_exact: bool
    dagger_metrics: tuple[dict[str, Any], ...]
    online_updates: tuple[dict[str, Any], ...]
    online_gate_decisions: tuple[dict[str, Any], ...]
    parent_rollouts: tuple[G1NeuralTorqueRolloutEvidence, ...]
    bc_rollouts: tuple[G1NeuralTorqueRolloutEvidence, ...]
    online_rl_rollouts: tuple[G1NeuralTorqueRolloutEvidence, ...]
    promotion_checks: dict[str, bool]
    decision: str
    blockers: tuple[str, ...]
    artifact_paths: dict[str, str]
    pipeline_failures: tuple[str, ...]
    schema_version: str = "rosclaw.simforge.g1_neural_torque_pilot.v3"

    @property
    def pipeline_passed(self) -> bool:
        return not self.pipeline_failures and all(
            self.promotion_checks[name]
            for name in (
                "four_physical_gpus_exercised",
                "teacher_collection_causally_transparent",
                "behavior_cloning_converged",
                "direct_torque_physics_executed",
                "online_actor_critic_updated",
                "sim_only_boundary_preserved",
                "decision_fail_closed",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "body_hash": self.body_hash,
            "parent_policy_hash": self.parent_policy_hash,
            "dataset_hash": self.dataset_hash,
            "gpu_workers": list(self.gpu_workers),
            "selected_seed": self.selected_seed,
            "teacher_pass_through_exact": self.teacher_pass_through_exact,
            "dagger_metrics": list(self.dagger_metrics),
            "online_updates": list(self.online_updates),
            "online_gate_decisions": list(self.online_gate_decisions),
            "parent_rollouts": [item.to_dict() for item in self.parent_rollouts],
            "bc_rollouts": [item.to_dict() for item in self.bc_rollouts],
            "online_rl_rollouts": [item.to_dict() for item in self.online_rl_rollouts],
            "aggregates": {
                "parent": _aggregate(self.parent_rollouts),
                "bc": _aggregate(self.bc_rollouts),
                "online_rl": _aggregate(self.online_rl_rollouts),
            },
            "promotion_checks": self.promotion_checks,
            "decision": self.decision,
            "blockers": list(self.blockers),
            "artifact_paths": self.artifact_paths,
            "pipeline_failures": list(self.pipeline_failures),
            "pipeline_passed": self.pipeline_passed,
            "claims": {
                "evidence_domain": "SIM_ONLY",
                "direct_joint_torque_actor": True,
                "online_actor_critic": bool(self.online_updates),
                "continual_retention": "anchor_replay+parent_distillation+EWC",
                "candidate_promoted": self.decision == "SIM_CANDIDATE",
                "hardware_authorized": False,
                "real_robot_evidence": False,
            },
        }


def run_g1_neural_torque_pilot(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    gpu_epochs: int = 12,
    dagger_generations: int = 3,
    online_updates: int = 8,
) -> G1NeuralTorquePilotReport:
    """Execute teacher collection, four-GPU BC, DAgger, SAC, and sealed replay."""

    if not 4 <= gpu_epochs <= 50:
        raise ValueError("neural torque GPU epochs must be in [4, 50]")
    if not 1 <= dagger_generations <= 6:
        raise ValueError("neural torque DAgger generations must be in [1, 6]")
    if not 1 <= online_updates <= 30:
        raise ValueError("neural torque online updates must be in [1, 30]")
    root = _new_external_root(output_dir, source_checkout)
    gpu_root = root / "gpu-workers"
    gpu_root.mkdir()
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    qualification = backend.qualification
    parameters = ShotParameters()
    training_scenarios, dagger_scenarios, sealed_scenarios = _pilot_scenarios()
    training_episodes: list[G1TeacherTorqueEpisode] = []
    training_results: list[GoalForgeEpisode] = []
    for scenario in training_scenarios:
        episode, teacher = _collect_teacher(backend, scenario, parameters)
        training_results.append(episode)
        training_episodes.append(teacher)
    validation_teachers = tuple(
        _collect_teacher(backend, scenario, parameters)[1] for scenario in dagger_scenarios
    )
    control = backend.run(training_scenarios[0], parameters)
    teacher_pass_through_exact = bool(
        control.result.summary_dict() == training_results[0].result.summary_dict()
        and trajectory_digest(control.trajectory)
        == trajectory_digest(training_results[0].trajectory)
    )
    dataset_path = root / "teacher-dataset.npz"
    _write_teacher_dataset(
        dataset_path,
        tuple(training_episodes),
        validation_teachers,
    )
    dataset_hash = teacher_dataset_hash(tuple(training_episodes))
    worker_values, worker_failures = _run_gpu_workers(
        checkout=source_checkout,
        dataset_path=dataset_path,
        output_root=gpu_root,
        body_hash=qualification.body_hash,
        parent_policy_hash=qualification.kick_prior_hash,
        epochs=gpu_epochs,
    )
    failures = list(worker_failures)
    if worker_values:
        selected = min(
            worker_values,
            key=lambda value: (float(value["final_validation_loss"]), int(value["learner_seed"])),
        )
        selected_seed = int(selected["learner_seed"])
    else:
        selected_seed = 8101
        failures.append("no four-GPU behavior-cloning worker completed")

    safety = G1TorqueSafetyConfig(
        torque_guard_scale=0.80,
        maximum_mechanical_power_w=4000.0,
        maximum_parent_deviation_ratio=0.05,
        maximum_projection_ratio=0.35,
        maximum_observation_z=5.0,
        minimum_upright_gravity_z=-0.97,
        minimum_pelvis_height_m=0.70,
        recovery_cooldown_steps=250,
        warmup_steps=100,
    )
    learner = G1ContinualTorqueActorCritic(
        G1NeuralTorqueLearnerConfig(
            hidden_dim=96,
            sequence_length=32,
            batch_size=256,
            actor_lr=5e-5,
            critic_lr=2e-4,
            behavior_cloning_weight=5.0,
            parent_churn_weight=2.0,
            ewc_weight=20.0,
            device="cuda:2",
            seed=selected_seed,
        ),
        safety=safety,
    )
    accumulated = list(training_episodes)
    initial_metrics = learner.pretrain_behavior(
        tuple(accumulated),
        validation=validation_teachers,
        epochs=gpu_epochs,
        stride=4,
    )
    dagger_metrics: list[dict[str, Any]] = [{"generation": 0, **initial_metrics[-1].to_dict()}]
    bc_development_rollouts: list[G1NeuralTorqueRolloutEvidence] = []
    for generation in range(1, dagger_generations + 1):
        artifact_path = root / f"dagger-generation-{generation - 1}.bin"
        _write_candidate(
            artifact_path,
            learner,
            body_hash=qualification.body_hash,
            parent_policy_hash=qualification.kick_prior_hash,
            dataset_hash=teacher_dataset_hash(tuple(accumulated)),
        )
        corrections: list[G1TeacherTorqueEpisode] = []
        for scenario in dagger_scenarios:
            episode, policy = _run_candidate(
                backend,
                scenario,
                parameters,
                artifact_path,
            )
            bc_development_rollouts.append(
                _rollout_evidence(
                    stage=f"dagger_generation_{generation - 1}",
                    episode=episode,
                    policy=policy,
                    strict_replay=False,
                )
            )
            trace = policy.episode()
            corrections.append(
                G1TeacherTorqueEpisode(
                    observations=trace.observations,
                    actions=trace.parent_actions,
                    parent_actions=trace.parent_actions,
                )
            )
        accumulated.extend(corrections)
        metrics = learner.pretrain_behavior(
            tuple(accumulated),
            validation=validation_teachers,
            epochs=6,
            stride=4,
        )
        dagger_metrics.append({"generation": generation, **metrics[-1].to_dict()})

    bc_artifact_path = root / "g1-neural-torque-bc.bin"
    _write_candidate(
        bc_artifact_path,
        learner,
        body_hash=qualification.body_hash,
        parent_policy_hash=qualification.kick_prior_hash,
        dataset_hash=teacher_dataset_hash(tuple(accumulated)),
    )
    recent_replays: list[G1NeuralTorqueReplay] = []
    online_scenarios = (*training_scenarios, *dagger_scenarios)
    bc_online_rollouts: list[G1NeuralTorqueRolloutEvidence] = []
    for scenario in online_scenarios:
        episode, policy = _run_candidate(
            backend,
            scenario,
            parameters,
            bc_artifact_path,
        )
        evidence = _rollout_evidence(
            stage="bc_online_collection",
            episode=episode,
            policy=policy,
            strict_replay=False,
        )
        bc_development_rollouts.append(evidence)
        bc_online_rollouts.append(evidence)
        receipt = _require_torque_receipt(episode)
        critical = _critical(episode.result)
        recent_replays.append(
            online_replay(
                policy.episode(),
                sequence_length=learner.config.sequence_length,
                task_score=_task_score(episode.result),
                fell=episode.result.post_kick_fall,
                critical_failure=critical,
                projection_fallback_rate=(
                    receipt.projection_fallback_count / max(1, receipt.inference_count)
                ),
                stride=10,
            )
        )
    anchor = teacher_replay(
        tuple(training_episodes),
        sequence_length=learner.config.sequence_length,
        stride=10,
    )
    replay = G1NeuralTorqueReplay.combine(anchor, *recent_replays)
    online_dataset_hash = canonical_hash(
        {
            "teacher_dataset_hash": teacher_dataset_hash(tuple(accumulated)),
            "online_replay_hash": neural_torque_replay_hash(replay),
        }
    )
    update_values: list[G1NeuralTorqueUpdate] = []
    online_gate_decisions: list[dict[str, Any]] = []
    try:
        critic_warmup_updates = max(16, min(64, online_updates * 4))
        update_values.extend(
            learner.update(replay, update_actor=False) for _ in range(critic_warmup_updates)
        )
        current_gate_rollouts = tuple(bc_online_rollouts)
        trust_region_fractions = (1.0, 0.50, 0.20, 0.10, 0.05, 0.035, 0.02, 0.01, 0.005)
        for attempt in range(online_updates):
            rollback = learner.checkpoint_bytes()
            actor_parent = learner.actor_snapshot()
            update = learner.update(replay)
            update_values.append(update)
            actor_proposal = learner.actor_snapshot()
            accepted_step = False
            for fraction in trust_region_fractions:
                learner.install_interpolated_actor(
                    actor_parent,
                    actor_proposal,
                    fraction=fraction,
                )
                fraction_label = str(fraction).replace(".", "p")
                attempt_path = root / (
                    f"online-actor-attempt-{attempt}-fraction-{fraction_label}.bin"
                )
                _write_candidate(
                    attempt_path,
                    learner,
                    body_hash=qualification.body_hash,
                    parent_policy_hash=qualification.kick_prior_hash,
                    dataset_hash=online_dataset_hash,
                )
                attempt_rollouts = tuple(
                    _rollout_evidence(
                        stage=f"online_actor_attempt_{attempt}",
                        episode=episode,
                        policy=policy,
                        strict_replay=False,
                    )
                    for episode, policy in (
                        _run_candidate(
                            backend,
                            scenario,
                            parameters,
                            attempt_path,
                        )
                            for scenario in online_scenarios
                    )
                )
                accepted, reasons = _online_update_gate(
                    parent=current_gate_rollouts,
                    candidate=attempt_rollouts,
                )
                online_gate_decisions.append(
                    {
                        "attempt": attempt,
                        "update_index": update.update_index,
                        "trust_region_fraction": fraction,
                        "accepted": accepted,
                        "reasons": list(reasons),
                        "parent": _aggregate(current_gate_rollouts),
                        "candidate": _aggregate(attempt_rollouts),
                        "artifact_path": str(attempt_path),
                    }
                )
                if accepted:
                    current_gate_rollouts = attempt_rollouts
                    accepted_step = True
                    break
            if not accepted_step:
                learner.restore_checkpoint(rollback)
                break
    except ValueError as exc:
        failures.append(f"online_actor_critic:{exc}")
    online_artifact_path = root / "g1-neural-torque-online-rl.bin"
    _write_candidate(
        online_artifact_path,
        learner,
        body_hash=qualification.body_hash,
        parent_policy_hash=qualification.kick_prior_hash,
        dataset_hash=online_dataset_hash,
    )
    checkpoint_path = root / "g1-neural-torque-learner.ckpt"
    checkpoint_path.write_bytes(learner.checkpoint_bytes())

    parent_rollouts = tuple(
        _evaluate_parent_with_replay(
            backend,
            scenario,
            parameters,
            stage="sealed_parent",
        )
        for scenario in sealed_scenarios
    )
    bc_rollouts = tuple(
        _evaluate_candidate_with_replay(
            backend,
            scenario,
            parameters,
            bc_artifact_path,
            stage="sealed_bc",
        )
        for scenario in sealed_scenarios
    )
    online_rollouts = tuple(
        _evaluate_candidate_with_replay(
            backend,
            scenario,
            parameters,
            online_artifact_path,
            stage="sealed_online_rl",
        )
        for scenario in sealed_scenarios
    )
    parent_aggregate = _aggregate(parent_rollouts)
    bc_aggregate = _aggregate(bc_rollouts)
    online_aggregate = _aggregate(online_rollouts)
    finite_workers = bool(
        len(worker_values) == 4
        and len({str(value["gpu_uuid"]) for value in worker_values}) == 4
        and all(bool(value["finite"]) for value in worker_values)
    )
    direct_fraction = float(online_aggregate["learned_output_fraction"])
    plasticity_gain = _relative_gain(
        float(bc_aggregate["mean_score"]),
        float(online_aggregate["mean_score"]),
    )
    parent_score = float(parent_aggregate["mean_score"])
    retention_floor = parent_score - 0.03 * max(abs(parent_score), 1.0)
    checks = {
        "four_physical_gpus_exercised": finite_workers,
        "teacher_collection_causally_transparent": teacher_pass_through_exact,
        "behavior_cloning_converged": bool(
            initial_metrics[-1].finite
            and len(worker_values) == 4
            and all(
                float(value["final_validation_loss"]) < float(value["initial_validation_loss"])
                for value in worker_values
            )
        ),
        "direct_torque_physics_executed": bool(
            direct_fraction > 0.0
            and all(item.direct_torque_inferences > 0 for item in online_rollouts)
        ),
        "online_actor_critic_updated": bool(
            any(item.actor_updated for item in update_values)
            and all(item.finite for item in update_values)
        ),
        "online_update_locally_accepted": any(
            bool(value["accepted"]) for value in online_gate_decisions
        ),
        "sim_only_boundary_preserved": bool(
            all(
                item.activation_ceiling == "SIM_ONLY" and not item.hardware_authorized
                for item in online_rollouts
            )
        ),
        "critical_safety_regression_zero": all(
            not (
                (
                    candidate.fall
                    or candidate.joint_violation
                    or candidate.torque_violation
                    or not candidate.finite
                )
                and not (
                    parent.fall
                    or parent.joint_violation
                    or parent.torque_violation
                    or not parent.finite
                )
            )
            for parent, candidate in zip(
                parent_rollouts,
                online_rollouts,
                strict=True,
            )
        ),
        "historical_retention_within_3pct": bool(
            float(online_aggregate["mean_score"]) >= retention_floor
        ),
        "online_plasticity_gain_at_least_5pct": plasticity_gain >= 0.05,
        "learned_output_fraction_at_least_75pct": direct_fraction >= 0.75,
        "strict_replay_all_sealed": all(item.strict_replay for item in online_rollouts),
    }
    promotion_names = (
        "critical_safety_regression_zero",
        "historical_retention_within_3pct",
        "online_update_locally_accepted",
        "online_plasticity_gain_at_least_5pct",
        "learned_output_fraction_at_least_75pct",
        "strict_replay_all_sealed",
    )
    decision = "SIM_CANDIDATE" if all(checks[name] for name in promotion_names) else "REJECTED"
    checks["decision_fail_closed"] = decision == "SIM_CANDIDATE" or not all(
        checks[name] for name in promotion_names
    )
    blockers = tuple(name for name in promotion_names if not checks[name])
    report = G1NeuralTorquePilotReport(
        source_commit=_git_commit(source_checkout),
        body_hash=qualification.body_hash,
        parent_policy_hash=qualification.kick_prior_hash,
        dataset_hash=dataset_hash,
        gpu_workers=tuple(worker_values),
        selected_seed=selected_seed,
        teacher_pass_through_exact=teacher_pass_through_exact,
        dagger_metrics=tuple(dagger_metrics),
        online_updates=tuple(value.to_dict() for value in update_values),
        online_gate_decisions=tuple(online_gate_decisions),
        parent_rollouts=parent_rollouts,
        bc_rollouts=bc_rollouts,
        online_rl_rollouts=online_rollouts,
        promotion_checks=checks,
        decision=decision,
        blockers=blockers,
        artifact_paths={
            "teacher_dataset": str(dataset_path),
            "bc_actor": str(bc_artifact_path),
            "online_rl_actor": str(online_artifact_path),
            "learner_checkpoint": str(checkpoint_path),
        },
        pipeline_failures=tuple(failures),
    )
    _atomic_json(root / "g1-neural-torque-pilot.json", report.to_dict())
    return report


def _collect_teacher(
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
) -> tuple[GoalForgeEpisode, G1TeacherTorqueEpisode]:
    collector = G1TeacherTorqueCollector()
    episode = backend.run(scenario, parameters, torque_policy=collector)
    return episode, collector.episode()


def _run_candidate(
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
    artifact_path: Path,
) -> tuple[GoalForgeEpisode, G1NeuralTorquePolicy]:
    artifact = load_g1_neural_torque_artifact(
        artifact_path,
        expected_body_hash=backend.qualification.body_hash,
    )
    policy = G1NeuralTorquePolicy(
        artifact,
        expected_body_hash=backend.qualification.body_hash,
        expected_parent_policy_hash=backend.qualification.kick_prior_hash,
    )
    return backend.run(scenario, parameters, torque_policy=policy), policy


def _evaluate_candidate_with_replay(
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
    artifact_path: Path,
    *,
    stage: str,
) -> G1NeuralTorqueRolloutEvidence:
    episode, policy = _run_candidate(backend, scenario, parameters, artifact_path)
    replay, replay_policy = _run_candidate(backend, scenario, parameters, artifact_path)
    strict = bool(
        episode.result.summary_dict() == replay.result.summary_dict()
        and trajectory_digest(episode.trajectory) == trajectory_digest(replay.trajectory)
        and policy.build_receipt().to_dict() == replay_policy.build_receipt().to_dict()
    )
    return _rollout_evidence(
        stage=stage,
        episode=episode,
        policy=policy,
        strict_replay=strict,
    )


def _evaluate_parent_with_replay(
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
    *,
    stage: str,
) -> G1NeuralTorqueRolloutEvidence:
    episode = backend.run(scenario, parameters)
    replay = backend.run(scenario, parameters)
    strict = bool(
        episode.result.summary_dict() == replay.result.summary_dict()
        and trajectory_digest(episode.trajectory) == trajectory_digest(replay.trajectory)
    )
    return _rollout_evidence(
        stage=stage,
        episode=episode,
        policy=None,
        strict_replay=strict,
    )


def _rollout_evidence(
    *,
    stage: str,
    episode: GoalForgeEpisode,
    policy: G1NeuralTorquePolicy | None,
    strict_replay: bool,
) -> G1NeuralTorqueRolloutEvidence:
    receipt = policy.build_receipt() if policy is not None else None
    count = receipt.inference_count if receipt is not None else 0
    return G1NeuralTorqueRolloutEvidence(
        stage=stage,
        partition=episode.scenario.partition.value,
        scenario_id=episode.scenario.scenario_id,
        scenario_commitment=episode.scenario.scenario_commitment,
        status=episode.result.status.value,
        success=episode.result.success,
        contact=episode.result.kick_foot_contacted,
        target_error_m=(
            episode.result.target_error_m if math.isfinite(episode.result.target_error_m) else None
        ),
        ball_speed_mps=episode.result.ball_speed_mps,
        support_slip_m=episode.result.support_foot_slip_m,
        com_margin_min_m=episode.result.com_margin_min_m,
        torso_roll_peak_rad=episode.result.torso_roll_peak_rad,
        torso_pitch_peak_rad=episode.result.torso_pitch_peak_rad,
        fall=episode.result.post_kick_fall,
        joint_violation=episode.result.joint_limit_violation,
        torque_violation=episode.result.torque_limit_violation,
        finite=episode.result.finite_state,
        score=_task_score(episode.result),
        direct_torque_inferences=count,
        learned_output_fraction=(receipt.learned_output_count / count if count else 0.0),
        fallback_fraction=(receipt.fallback_count / count if count else 0.0),
        artifact_hash=receipt.artifact_hash if receipt is not None else None,
        activation_ceiling=receipt.activation_ceiling if receipt is not None else None,
        hardware_authorized=receipt.hardware_authorized if receipt is not None else False,
        strict_replay=strict_replay,
        trace_hash=trajectory_digest(episode.trajectory),
    )


def _require_torque_receipt(episode: GoalForgeEpisode) -> G1TorquePolicyReceipt:
    if episode.torque_policy_receipt is None:
        raise ValueError("neural torque rollout lacks its SIM receipt")
    return episode.torque_policy_receipt


def _write_candidate(
    path: Path,
    learner: G1ContinualTorqueActorCritic,
    *,
    body_hash: str,
    parent_policy_hash: str,
    dataset_hash: str,
) -> None:
    payload = learner.artifact_bytes(
        body_hash=body_hash,
        parent_policy_hash=parent_policy_hash,
        dataset_hash=dataset_hash,
    )
    path.write_bytes(payload)
    load_g1_neural_torque_artifact(
        path,
        expected_hash=hash_bytes(payload),
        expected_body_hash=body_hash,
    )


def _write_teacher_dataset(
    path: Path,
    training: tuple[G1TeacherTorqueEpisode, ...],
    validation: tuple[G1TeacherTorqueEpisode, ...],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "training_count": np.asarray(len(training), dtype=np.int64),
        "validation_count": np.asarray(len(validation), dtype=np.int64),
    }
    for prefix, episodes in (("training", training), ("validation", validation)):
        for index, episode in enumerate(episodes):
            arrays[f"{prefix}_{index}_observations"] = episode.observations
            arrays[f"{prefix}_{index}_actions"] = episode.actions
            arrays[f"{prefix}_{index}_parent_actions"] = episode.parent_actions
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


def _run_gpu_workers(
    *,
    checkout: Path,
    dataset_path: Path,
    output_root: Path,
    body_hash: str,
    parent_policy_hash: str,
    epochs: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    worker = checkout / "scripts/simforge/g1_neural_torque_gpu_worker.py"
    processes: list[tuple[int, Path, subprocess.Popen[str]]] = []
    for gpu in range(4):
        output = output_root / f"gpu-{gpu}.json"
        artifact = output_root / f"gpu-{gpu}.bin"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(checkout / "src") + (
            os.pathsep + existing if existing else ""
        )
        process = subprocess.Popen(
            [
                sys.executable,
                str(worker),
                "--physical-gpu",
                str(gpu),
                "--dataset",
                str(dataset_path),
                "--output",
                str(output),
                "--artifact",
                str(artifact),
                "--body-hash",
                body_hash,
                "--parent-policy-hash",
                parent_policy_hash,
                "--seed",
                str(8101 + gpu),
                "--epochs",
                str(epochs),
            ],
            cwd=checkout,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((gpu, output, process))
    values: list[dict[str, Any]] = []
    failures: list[str] = []
    for gpu, output, process in processes:
        try:
            stdout, stderr = process.communicate(timeout=300.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            failures.append(f"gpu{gpu}:timeout:{stdout[-200:]}:{stderr[-600:]}")
            continue
        if process.returncode != 0 or not output.is_file():
            failures.append(f"gpu{gpu}:exit={process.returncode}:{stdout[-200:]}:{stderr[-800:]}")
            continue
        values.append(json.loads(output.read_text(encoding="utf-8")))
    return sorted(values, key=lambda value: int(value["physical_gpu"])), tuple(failures)


def _pilot_scenarios() -> tuple[
    tuple[GoalForgeScenario, ...],
    tuple[GoalForgeScenario, ...],
    tuple[GoalForgeScenario, ...],
]:
    ledger = SeedLedger(task_id="g1_neural_torque_pilot", secret=b"rosclaw-phase8-torque-v1" * 2)

    def build(
        partition: Partition,
        generations: tuple[int, ...],
        prefix: str,
        *,
        start_index: int = 0,
    ):
        result = []
        for local_index, generation in enumerate(generations):
            index = start_index + local_index
            generated = generate_goalforge_scenarios(
                ledger=ledger,
                partition=partition,
                count=index + 1,
                generation=generation,
            )
            scenario = generated[index]
            result.append(
                replace(
                    scenario,
                    scenario_id=f"{prefix}-{local_index:02d}-g{generation}",
                )
            )
        return tuple(result)

    training = build(Partition.DEVELOPMENT, (0, 2, 4, 7), "torque-train")
    dagger = build(
        Partition.DEVELOPMENT,
        (2, 0),
        "torque-dagger",
        start_index=4,
    )
    sealed = (
        *build(Partition.VALIDATION, (0, 2, 4, 7), "torque-validation"),
        *build(Partition.HOLDOUT, (0, 2, 9), "torque-holdout"),
    )
    ledger.assert_disjoint()
    return training, dagger, sealed


def _critical(result: GoalForgeResult) -> bool:
    return bool(
        result.post_kick_fall
        or result.joint_limit_violation
        or result.torque_limit_violation
        or not result.finite_state
    )


def _online_update_gate(
    *,
    parent: tuple[G1NeuralTorqueRolloutEvidence, ...],
    candidate: tuple[G1NeuralTorqueRolloutEvidence, ...],
) -> tuple[bool, tuple[str, ...]]:
    """Accept one online actor step only when matched physics does not regress."""

    if len(parent) != len(candidate) or not parent:
        return False, ("matched_rollout_count_mismatch",)
    reasons: list[str] = []
    for index, (before, after) in enumerate(zip(parent, candidate, strict=True)):
        before_critical = before.fall or before.joint_violation or before.torque_violation
        after_critical = after.fall or after.joint_violation or after.torque_violation
        if after_critical and not before_critical:
            reasons.append(f"new_critical_regression:{index}")
        if not after.finite:
            reasons.append(f"nonfinite:{index}")
        if not after.strict_replay and after.stage.startswith("sealed_"):
            reasons.append(f"strict_replay_failed:{index}")
    parent_value = _aggregate(parent)
    candidate_value = _aggregate(candidate)
    parent_score = float(parent_value["mean_score"])
    minimum_score = parent_score - 0.01 * max(abs(parent_score), 1.0)
    if float(candidate_value["mean_score"]) < minimum_score:
        reasons.append("matched_score_regression_gt_1pct")
    if float(candidate_value["success_rate"]) < float(parent_value["success_rate"]):
        reasons.append("matched_success_regression")
    if (
        float(candidate_value["learned_output_fraction"])
        < float(parent_value["learned_output_fraction"]) - 0.05
    ):
        reasons.append("learned_output_fraction_regression_gt_5pp")
    return not reasons, tuple(reasons)


def _task_score(result: GoalForgeResult) -> float:
    score = 10.0 * float(result.success) + 2.0 * float(result.kick_foot_contacted)
    if math.isfinite(result.target_error_m):
        score -= min(3.0, result.target_error_m)
    score -= 8.0 * float(result.post_kick_fall)
    score -= 5.0 * float(result.joint_limit_violation or result.torque_limit_violation)
    score -= 2.0 * max(0.0, result.support_foot_slip_m - 0.04) / 0.04
    score -= 2.0 * max(0.0, -result.com_margin_min_m) / 0.04
    return float(np.clip(score, -20.0, 20.0))


def _aggregate(values: tuple[G1NeuralTorqueRolloutEvidence, ...]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "success_rate": 0.0,
            "critical_failure_rate": 0.0,
            "mean_score": 0.0,
            "mean_ball_speed_mps": 0.0,
            "learned_output_fraction": 0.0,
        }
    return {
        "count": float(len(values)),
        "success_rate": sum(item.success for item in values) / len(values),
        "critical_failure_rate": sum(
            item.fall or item.joint_violation or item.torque_violation or not item.finite
            for item in values
        )
        / len(values),
        "mean_score": sum(item.score for item in values) / len(values),
        "mean_ball_speed_mps": sum(item.ball_speed_mps for item in values) / len(values),
        "mean_support_slip_m": sum(item.support_slip_m for item in values) / len(values),
        "mean_torso_roll_peak_rad": sum(item.torso_roll_peak_rad for item in values) / len(values),
        "learned_output_fraction": sum(item.learned_output_fraction for item in values)
        / len(values),
        "fallback_fraction": sum(item.fallback_fraction for item in values) / len(values),
    }


def _relative_gain(parent: float, candidate: float) -> float:
    return (candidate - parent) / max(abs(parent), 1e-6)


def _git_commit(checkout: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()


def _new_external_root(output: Path, checkout: Path) -> Path:
    root = output.expanduser().resolve()
    source = checkout.expanduser().resolve()
    if root == source or source in root.parents:
        raise ValueError("neural torque evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    return root


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


__all__ = [
    "G1NeuralTorquePilotReport",
    "G1NeuralTorqueRolloutEvidence",
    "run_g1_neural_torque_pilot",
]
