"""A/B qualification of MotionDecode representation transfer into torque BC."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.collective.sources.motiondecode.motion_prior import load_g1_motion_prior_artifact
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.backends.unitree_mujoco_backend import G1MuJoCoBackend
from rosclaw.simforge.g1_neural_torque import (
    G1NeuralTorqueArtifact,
    G1TeacherTorqueEpisode,
    G1TorqueSafetyConfig,
    load_g1_neural_torque_artifact,
    serialize_g1_neural_torque_artifact,
)
from rosclaw.simforge.g1_neural_torque_learning import (
    G1ContinualTorqueActorCritic,
    G1NeuralTorqueLearnerConfig,
    teacher_dataset_hash,
)
from rosclaw.simforge.g1_neural_torque_validation import (
    G1NeuralTorqueRolloutEvidence,
    _aggregate,
    _collect_teacher,
    _evaluate_candidate_with_replay,
    _online_update_gate,
    _pilot_scenarios,
    _write_candidate,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters


@dataclass(frozen=True)
class G1MotionPriorTransferReport:
    body_hash: str
    parent_policy_hash: str
    teacher_dataset_hash: str
    motion_prior_artifact_hash: str
    motion_prior_pack_hash: str
    training_runs: tuple[dict[str, Any], ...]
    trust_region_runs: tuple[dict[str, Any], ...]
    proposal_initialization_fraction: float
    selected_transfer_fraction: float
    baseline_rollouts: tuple[G1NeuralTorqueRolloutEvidence, ...]
    transfer_rollouts: tuple[G1NeuralTorqueRolloutEvidence, ...]
    checks: dict[str, bool]
    decision: str
    blockers: tuple[str, ...]
    evidence_domain: str = "SIM_ONLY"
    hardware_authorized: bool = False
    promotion_evidence_eligible: bool = False
    schema_version: str = "rosclaw.simforge.g1_motion_prior_transfer.v2"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["baseline_rollouts"] = [item.to_dict() for item in self.baseline_rollouts]
        value["transfer_rollouts"] = [item.to_dict() for item in self.transfer_rollouts]
        value["aggregates"] = {
            "baseline": _aggregate(self.baseline_rollouts),
            "transfer": _aggregate(self.transfer_rollouts),
        }
        value["claims"] = {
            "motion_prior_outputs_torque": False,
            "motion_prior_initializes_actor_gru": True,
            "direct_torque_actor_physics_executed": bool(self.transfer_rollouts),
            "online_rl_executed": False,
            "candidate_promoted": False,
            "real_robot_evidence": False,
        }
        return value


def run_g1_motion_prior_transfer_validation(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    output_dir: Path,
    source_checkout: Path,
    device: str = "cuda:2",
    epochs: int = 6,
    seed: int = 8500,
) -> G1MotionPriorTransferReport:
    if not 2 <= epochs <= 30:
        raise ValueError("motion-prior transfer epochs must be in [2, 30]")
    root = _external_root(output_dir, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    qualification = backend.qualification
    prior = load_g1_motion_prior_artifact(motion_prior_path)
    if prior.body_hash != qualification.body_hash:
        raise ValueError("motion-prior artifact does not match the qualified G1 body")
    training_scenarios, development_scenarios, validation_scenarios = _pilot_scenarios()
    parameters = ShotParameters()
    training = tuple(
        _collect_teacher(backend, scenario, parameters)[1] for scenario in training_scenarios
    )
    validation = tuple(
        _collect_teacher(backend, scenario, parameters)[1] for scenario in development_scenarios
    )
    teacher_hash = teacher_dataset_hash(training)
    config = G1NeuralTorqueLearnerConfig(
        hidden_dim=96,
        sequence_length=32,
        batch_size=256,
        actor_lr=5e-5,
        critic_lr=2e-4,
        behavior_cloning_weight=5.0,
        parent_churn_weight=2.0,
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
    runs: list[dict[str, Any]] = []
    artifacts: dict[float, Path] = {}
    for fraction in (0.0, 0.25, 0.50, 0.75, 1.0):
        learner = G1ContinualTorqueActorCritic(config, safety=safety)
        if fraction:
            learner.install_motion_prior(
                prior,
                expected_body_hash=qualification.body_hash,
                fraction=fraction,
            )
        metrics = learner.pretrain_behavior(
            training,
            validation=validation,
            epochs=epochs,
            stride=4,
        )
        dataset_hash = (
            teacher_hash
            if not fraction
            else canonical_hash(
                {
                    "teacher_dataset_hash": teacher_hash,
                    "motion_prior_artifact_hash": prior.artifact_hash,
                    "transfer_fraction": fraction,
                }
            )
        )
        label = str(fraction).replace(".", "p")
        artifact_path = root / f"torque-bc-transfer-{label}.bin"
        _write_candidate(
            artifact_path,
            learner,
            body_hash=qualification.body_hash,
            parent_policy_hash=qualification.kick_prior_hash,
            dataset_hash=dataset_hash,
        )
        artifacts[fraction] = artifact_path
        runs.append(
            {
                "fraction": fraction,
                "motion_prior_installed": bool(fraction),
                "motion_prior_artifact_hash": learner.motion_prior_artifact_hash,
                "initial_validation_loss": metrics[0].validation_loss,
                "final_training_loss": metrics[-1].training_loss,
                "final_validation_loss": metrics[-1].validation_loss,
                "finite": all(item.finite for item in metrics),
                "artifact_path": str(artifact_path),
            }
        )
    baseline = runs[0]
    baseline_development = tuple(
        _evaluate_candidate_with_replay(
            backend,
            scenario,
            parameters,
            artifacts[0.0],
            stage="motion_prior_baseline_development",
        )
        for scenario in development_scenarios
    )
    baseline["development_rollouts"] = [item.to_dict() for item in baseline_development]
    baseline["development_aggregate"] = _aggregate(baseline_development)
    eligible_runs: list[dict[str, Any]] = []
    for run in runs[1:]:
        fraction = float(run["fraction"])
        development = tuple(
            _evaluate_candidate_with_replay(
                backend,
                scenario,
                parameters,
                artifacts[fraction],
                stage=f"motion_prior_transfer_{fraction}_development",
            )
            for scenario in development_scenarios
        )
        accepted, reasons = _online_update_gate(
            parent=baseline_development,
            candidate=development,
        )
        teacher_improved = (
            float(run["final_validation_loss"])
            <= float(baseline["final_validation_loss"]) * 0.98
        )
        run["development_rollouts"] = [item.to_dict() for item in development]
        run["development_aggregate"] = _aggregate(development)
        run["development_gate_passed"] = accepted
        run["development_gate_reasons"] = list(reasons)
        run["teacher_improved_2pct"] = teacher_improved
        if accepted and teacher_improved:
            eligible_runs.append(run)
    proposal = min(
        eligible_runs or runs[1:],
        key=lambda item: (item["final_validation_loss"], item["fraction"]),
    )
    proposal_fraction = float(proposal["fraction"])
    baseline_artifact = load_g1_neural_torque_artifact(
        artifacts[0.0], expected_body_hash=qualification.body_hash
    )
    proposal_artifact = load_g1_neural_torque_artifact(
        artifacts[proposal_fraction], expected_body_hash=qualification.body_hash
    )
    trust_runs: list[dict[str, Any]] = []
    eligible_trust_runs: list[dict[str, Any]] = []
    for trust_fraction in (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
        trust_path = root / f"torque-bc-trust-{str(trust_fraction).replace('.', 'p')}.bin"
        _write_interpolated_artifact(
            trust_path,
            baseline=baseline_artifact,
            proposal=proposal_artifact,
            fraction=trust_fraction,
            dataset_hash=canonical_hash(
                {
                    "teacher_dataset_hash": teacher_hash,
                    "motion_prior_artifact_hash": prior.artifact_hash,
                    "proposal_initialization_fraction": proposal_fraction,
                    "post_bc_trust_fraction": trust_fraction,
                }
            ),
        )
        development = tuple(
            _evaluate_candidate_with_replay(
                backend,
                scenario,
                parameters,
                trust_path,
                stage=f"motion_prior_trust_{trust_fraction}_development",
            )
            for scenario in development_scenarios
        )
        accepted, reasons = _online_update_gate(
            parent=baseline_development,
            candidate=development,
        )
        artifact = load_g1_neural_torque_artifact(
            trust_path, expected_body_hash=qualification.body_hash
        )
        behavior_loss = _artifact_behavior_loss(artifact, validation, sequence_length=32, stride=4)
        teacher_improved = behavior_loss <= float(baseline["final_validation_loss"]) * 0.98
        value = {
            "trust_fraction": trust_fraction,
            "artifact_path": str(trust_path),
            "artifact_hash": artifact.artifact_hash,
            "validation_loss": behavior_loss,
            "teacher_improved_2pct": teacher_improved,
            "development_gate_passed": accepted,
            "development_gate_reasons": list(reasons),
            "development_aggregate": _aggregate(development),
            "development_rollouts": [item.to_dict() for item in development],
        }
        trust_runs.append(value)
        if accepted and teacher_improved:
            eligible_trust_runs.append(value)
    selected_trust = min(
        eligible_trust_runs or trust_runs,
        key=lambda item: (item["trust_fraction"], item["validation_loss"]),
    )
    selected_fraction = float(selected_trust["trust_fraction"])
    # Development chose the transfer fraction.  The following disjoint
    # validation scenarios check physics retention but are not promotion data.
    physics_scenarios = validation_scenarios[:4]
    baseline_rollouts = tuple(
        _evaluate_candidate_with_replay(
            backend,
            scenario,
            parameters,
            artifacts[0.0],
            stage="motion_prior_baseline_validation",
        )
        for scenario in physics_scenarios
    )
    transfer_rollouts = tuple(
        _evaluate_candidate_with_replay(
            backend,
            scenario,
            parameters,
        Path(str(selected_trust["artifact_path"])),
            stage="motion_prior_transfer_validation",
        )
        for scenario in physics_scenarios
    )
    baseline_aggregate = _aggregate(baseline_rollouts)
    transfer_aggregate = _aggregate(transfer_rollouts)
    new_critical = any(
        (
            after.fall
            or after.joint_violation
            or after.torque_violation
            or not after.finite
        )
        and not (
            before.fall
            or before.joint_violation
            or before.torque_violation
            or not before.finite
        )
        for before, after in zip(baseline_rollouts, transfer_rollouts, strict=True)
    )
    checks = {
        "development_trust_region_passed": selected_trust in eligible_trust_runs,
        "held_out_teacher_loss_improved_2pct": (
            float(selected_trust["validation_loss"])
            <= float(baseline["final_validation_loss"]) * 0.98
        ),
        "all_training_finite": all(bool(item["finite"]) for item in runs),
        "strict_physics_replay": all(
            item.strict_replay for item in (*baseline_rollouts, *transfer_rollouts)
        ),
        "no_new_critical_failure": not new_critical,
        "physics_score_retained_within_1pct": (
            transfer_aggregate["mean_score"]
            >= baseline_aggregate["mean_score"]
            - 0.01 * max(abs(baseline_aggregate["mean_score"]), 1.0)
        ),
        "learned_output_fraction_retained_within_5pp": (
            transfer_aggregate["learned_output_fraction"]
            >= baseline_aggregate["learned_output_fraction"] - 0.05
        ),
        "sim_only_boundary_preserved": all(
            item.activation_ceiling == "SIM_ONLY" and not item.hardware_authorized
            for item in transfer_rollouts
        ),
    }
    blockers = tuple(name for name, passed in checks.items() if not passed)
    decision = "TRANSFER_CANDIDATE" if not blockers else "REJECTED"
    report = G1MotionPriorTransferReport(
        body_hash=qualification.body_hash,
        parent_policy_hash=qualification.kick_prior_hash,
        teacher_dataset_hash=teacher_hash,
        motion_prior_artifact_hash=prior.artifact_hash,
        motion_prior_pack_hash=prior.pack_hash,
        training_runs=tuple(runs),
        trust_region_runs=tuple(trust_runs),
        proposal_initialization_fraction=proposal_fraction,
        selected_transfer_fraction=selected_fraction,
        baseline_rollouts=baseline_rollouts,
        transfer_rollouts=transfer_rollouts,
        checks=checks,
        decision=decision,
        blockers=blockers,
    )
    _atomic_json(root / "g1-motion-prior-transfer-report.json", report.to_dict())
    return report


def _write_interpolated_artifact(
    path: Path,
    *,
    baseline: G1NeuralTorqueArtifact,
    proposal: G1NeuralTorqueArtifact,
    fraction: float,
    dataset_hash: str,
) -> None:
    if baseline.body_hash != proposal.body_hash or baseline.hidden_dim != proposal.hidden_dim:
        raise ValueError("torque artifacts cannot be interpolated across body or architecture")
    tensors: dict[str, np.ndarray] = {}
    for name, baseline_value in baseline.tensors.items():
        proposal_value = proposal.tensors[name]
        if name.startswith("actor."):
            value = baseline_value.astype(np.float64) + fraction * (
                proposal_value.astype(np.float64) - baseline_value.astype(np.float64)
            )
        else:
            if not np.allclose(baseline_value, proposal_value, rtol=0.0, atol=1e-6):
                raise ValueError(f"torque artifact contract tensor differs: {name}")
            value = baseline_value
        tensors[name] = np.asarray(value, dtype=np.float32)
    payload = serialize_g1_neural_torque_artifact(
        body_hash=baseline.body_hash,
        parent_policy_hash=baseline.parent_policy_hash,
        dataset_hash=dataset_hash,
        hidden_dim=baseline.hidden_dim,
        observation_clip=baseline.observation_clip,
        update_index=0,
        safety=baseline.safety,
        tensors=tensors,
    )
    path.write_bytes(payload)
    load_g1_neural_torque_artifact(
        path,
        expected_hash="sha256:" + __import__("hashlib").sha256(payload).hexdigest(),
        expected_body_hash=baseline.body_hash,
    )


def _artifact_behavior_loss(
    artifact: G1NeuralTorqueArtifact,
    episodes: tuple[G1TeacherTorqueEpisode, ...],
    *,
    sequence_length: int,
    stride: int,
) -> float:
    tensors = artifact.tensors
    limits = tensors["action_limits"]
    losses: list[float] = []
    for episode in episodes:
        for end in range(sequence_length, len(episode.observations) + 1, stride):
            sequence = episode.observations[end - sequence_length : end]
            normalized = np.clip(
                (sequence - tensors["observation_mean"]) / tensors["observation_std"],
                -artifact.observation_clip,
                artifact.observation_clip,
            )
            hidden = np.zeros(artifact.hidden_dim, dtype=np.float32)
            for observation in normalized:
                input_gates = tensors["actor.gru.weight_ih_l0"] @ observation + tensors[
                    "actor.gru.bias_ih_l0"
                ]
                hidden_gates = tensors["actor.gru.weight_hh_l0"] @ hidden + tensors[
                    "actor.gru.bias_hh_l0"
                ]
                reset_i, update_i, new_i = np.split(input_gates, 3)
                reset_h, update_h, new_h = np.split(hidden_gates, 3)
                reset = 1.0 / (1.0 + np.exp(-(reset_i + reset_h)))
                update = 1.0 / (1.0 + np.exp(-(update_i + update_h)))
                new = np.tanh(new_i + reset * new_h)
                hidden = (1.0 - update) * new + update * hidden
            output = tensors["actor.head.weight"] @ hidden + tensors["actor.head.bias"]
            prediction = np.tanh(output[: len(limits)]) * limits
            target = np.clip(episode.actions[end - 1], -limits, limits)
            losses.append(float(np.mean(np.square((prediction - target) / limits))))
    if not losses:
        raise ValueError("torque artifact behavior evaluation has no sequences")
    return float(sum(losses) / len(losses))


def _external_root(path: Path, source_checkout: Path) -> Path:
    root = path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("motion-prior transfer evidence must be outside the source checkout")
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


__all__ = ["G1MotionPriorTransferReport", "run_g1_motion_prior_transfer_validation"]
