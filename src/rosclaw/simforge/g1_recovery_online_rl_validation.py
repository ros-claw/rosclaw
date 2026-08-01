"""One fail-closed online actor-critic generation for post-kick recovery."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    G1TorqueSafetyConfig,
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
from rosclaw.simforge.tasks.g1_goalforge.concepts import GoalForgeResult, ShotParameters


def run_g1_recovery_online_rl_validation(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    output_dir: Path,
    source_checkout: Path,
    device: str = "cuda:2",
    seed: int = 8700,
) -> dict[str, Any]:
    root = _external_root(output_dir, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    qualification = backend.qualification
    prior = load_g1_motion_prior_artifact(motion_prior_path)
    if prior.body_hash != qualification.body_hash:
        raise ValueError("recovery prior does not match the qualified G1 body")
    training_scenarios, development_scenarios, validation_scenarios = _pilot_scenarios()
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
        minimum_upright_gravity_z=-0.97,
        minimum_pelvis_height_m=0.70,
        recovery_cooldown_steps=250,
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
        minimum_end_fraction=0.60,
    )
    plastic_parent_path = root / "recovery-plastic-parent.bin"
    plastic_dataset_hash = canonical_hash(
        {
            "teacher_dataset_hash": teacher_hash,
            "motion_prior_artifact_hash": prior.artifact_hash,
            "recovery_end_fraction": 0.60,
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
        minimum_recovery_phase=0.55,
        minimum_pelvis_height_m=0.74,
        maximum_projected_gravity_z=-0.92,
        eligibility_warmup_steps=5,
    )
    recovery_replays: list[G1NeuralTorqueReplay] = []
    collection: list[dict[str, Any]] = []
    for scenario in training_scenarios:
        episode, policy = _run_context(
            backend,
            scenario,
            parameters,
            stable=stable_artifact,
            plastic=plastic_parent_artifact,
            gate=gate,
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
            recovery_start_phase=0.55,
            stride=10,
        )
        recovery_replays.append(replay)
        collection.append(
            {
                "scenario_id": scenario.scenario_id,
                "recovery_score": score,
                "transition_count": replay.count,
                "plastic_activation_fraction": receipt.plastic_activation_fraction,
                "result": episode.result.summary_dict(),
            }
        )
    anchor = teacher_replay(training, sequence_length=config.sequence_length, stride=10)
    replay = G1NeuralTorqueReplay.combine(anchor, *recovery_replays)
    updates = [plastic.update(replay, update_actor=False) for _ in range(16)]
    parent_snapshot = plastic.actor_snapshot()
    actor_update = plastic.update(replay, update_actor=True)
    proposal_snapshot = plastic.actor_snapshot()

    parent_development = tuple(
        _context_evidence(
            backend,
            scenario,
            parameters,
            stable=stable_artifact,
            plastic=plastic_parent_artifact,
            gate=gate,
            stage="recovery_online_parent_development",
            strict=False,
        )
        for scenario in development_scenarios
    )
    trust_runs: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    online_hash = canonical_hash(
        {
            "plastic_parent_dataset_hash": plastic_dataset_hash,
            "recovery_replay_hash": neural_torque_replay_hash(replay),
        }
    )
    for fraction in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0):
        plastic.install_interpolated_actor(parent_snapshot, proposal_snapshot, fraction=fraction)
        path = root / f"recovery-online-trust-{str(fraction).replace('.', 'p')}.bin"
        _write_candidate(
            path,
            plastic,
            body_hash=qualification.body_hash,
            parent_policy_hash=qualification.kick_prior_hash,
            dataset_hash=canonical_hash(
                {"online_dataset_hash": online_hash, "actor_trust_fraction": fraction}
            ),
        )
        artifact = load_g1_neural_torque_artifact(
            path, expected_body_hash=qualification.body_hash
        )
        values = tuple(
            _context_evidence(
                backend,
                scenario,
                parameters,
                stable=stable_artifact,
                plastic=artifact,
                gate=gate,
                stage=f"recovery_online_{fraction}_development",
                strict=False,
            )
            for scenario in development_scenarios
        )
        accepted, reasons = _online_update_gate(parent=parent_development, candidate=values)
        aggregate = _aggregate(values)
        meaningful = _meaningful_recovery_improvement(
            _aggregate(parent_development), aggregate
        )
        if accepted and not meaningful:
            reasons = (*reasons, "no_measurable_recovery_improvement")
            accepted = False
        value = {
            "fraction": fraction,
            "artifact_path": str(path),
            "artifact_hash": artifact.artifact_hash,
            "accepted": accepted,
            "reasons": list(reasons),
            "aggregate": aggregate,
            "rollouts": [item.to_dict() for item in values],
        }
        trust_runs.append(value)
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
    selected_artifact = (
        load_g1_neural_torque_artifact(
            Path(str(selected["artifact_path"])), expected_body_hash=qualification.body_hash
        )
        if selected is not None
        else plastic_parent_artifact
    )
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
    validation_accepted, validation_reasons = _online_update_gate(
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
        "dense_replay_aligned": all(item.count > 0 for item in recovery_replays),
        "critic_and_actor_updated": bool(updates) and actor_update.actor_updated,
        "development_trust_region_found": selected is not None,
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
        "schema_version": "rosclaw.simforge.g1_recovery_online_rl.v1",
        "body_hash": qualification.body_hash,
        "parent_policy_hash": qualification.kick_prior_hash,
        "motion_prior_artifact_hash": prior.artifact_hash,
        "teacher_dataset_hash": teacher_hash,
        "online_replay_hash": neural_torque_replay_hash(replay),
        "gate": asdict(gate),
        "bc_metrics": {
            "stable_final_validation_loss": stable_metrics[-1].validation_loss,
            "plastic_warmup_validation_loss": plastic_warmup[-1].validation_loss,
            "plastic_recovery_validation_loss": plastic_metrics[-1].validation_loss,
        },
        "collection": collection,
        "critic_updates": [item.to_dict() for item in updates],
        "actor_update": actor_update.to_dict(),
        "parent_development": [item.to_dict() for item in parent_development],
        "trust_runs": trust_runs,
        "selected": selected,
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
            "direct_joint_torque_actor": True,
            "candidate_promoted": False,
            "real_robot_evidence": False,
        },
    }
    _atomic_json(root / "g1-recovery-online-rl-report.json", report)
    return report


def _run_context(
    backend: G1MuJoCoBackend,
    scenario: Any,
    parameters: ShotParameters,
    *,
    stable: G1NeuralTorqueArtifact,
    plastic: G1NeuralTorqueArtifact,
    gate: G1StabilityPlasticityGateConfig,
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
