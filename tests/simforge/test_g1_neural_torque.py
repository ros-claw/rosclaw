from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from rosclaw.simforge.g1_neural_torque import (
    G1_NEURAL_TORQUE_OBSERVATIONS,
    G1NeuralTorquePolicy,
    G1TeacherTorqueCollector,
    G1TorqueControlFrame,
    G1TorqueSafetyConfig,
    G1TorqueSafetyProjector,
    build_g1_neural_torque_observation,
    load_g1_neural_torque_artifact,
    serialize_g1_neural_torque_artifact,
)
from rosclaw.simforge.g1_stability_plasticity_policy import (
    G1StabilityPlasticityGateConfig,
    G1StabilityPlasticityTorquePolicy,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _frame(**changes: object) -> G1TorqueControlFrame:
    values: dict[str, object] = {
        "joint_position": np.zeros(29),
        "joint_velocity": np.zeros(29),
        "joint_lower_limits": np.full(29, -2.0),
        "joint_upper_limits": np.full(29, 2.0),
        "torso_quaternion_wxyz": np.asarray((1.0, 0.0, 0.0, 0.0)),
        "pelvis_position": np.asarray((0.0, 0.0, 0.8)),
        "ball_position": np.asarray((1.0, 0.1, 0.115)),
        "ball_velocity": np.asarray((-0.2, 0.05, 0.0)),
        "target_y_m": 0.75,
        "target_z_m": 0.9,
        "policy_phase": 0.25,
        "left_contact": True,
        "right_contact": False,
    }
    values.update(changes)
    return G1TorqueControlFrame(**values)  # type: ignore[arg-type]


def _artifact_bytes(
    *,
    safety: G1TorqueSafetyConfig | None = None,
    output_bias: float = 0.0,
) -> bytes:
    resolved = safety or G1TorqueSafetyConfig(
        maximum_projection_ratio=1.0,
        maximum_parent_deviation_ratio=0.25,
        warmup_steps=0,
    )
    observation_dim = len(G1_NEURAL_TORQUE_OBSERVATIONS)
    hidden = 4
    action_dim = len(G1_DDS_JOINT_NAMES)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float32) * resolved.torque_guard_scale
    return serialize_g1_neural_torque_artifact(
        body_hash=_digest("body"),
        parent_policy_hash=_digest("parent"),
        dataset_hash=_digest("dataset"),
        hidden_dim=hidden,
        observation_clip=8.0,
        update_index=3,
        safety=resolved,
        tensors={
            "observation_mean": np.zeros(observation_dim, dtype=np.float32),
            "observation_std": np.ones(observation_dim, dtype=np.float32),
            "action_limits": limits,
            "actor.gru.weight_ih_l0": np.zeros((3 * hidden, observation_dim)),
            "actor.gru.weight_hh_l0": np.zeros((3 * hidden, hidden)),
            "actor.gru.bias_ih_l0": np.zeros(3 * hidden),
            "actor.gru.bias_hh_l0": np.zeros(3 * hidden),
            "actor.head.weight": np.zeros((2 * action_dim, hidden)),
            "actor.head.bias": np.concatenate(
                (
                    np.full(action_dim, output_bias),
                    np.zeros(action_dim),
                )
            ),
        },
    )


def test_observation_contract_contains_raw_body_ball_phase_and_previous_torque() -> None:
    previous = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=float) * 0.1

    observation = build_g1_neural_torque_observation(_frame(), previous)

    assert observation.shape == (len(G1_NEURAL_TORQUE_OBSERVATIONS),)
    assert len(observation) == 102
    assert observation[58:61] == pytest.approx((0.0, 0.0, -1.0))
    assert observation[-29:] == pytest.approx(np.full(29, 0.1))


def test_observation_rejects_nonfinite_proprioception() -> None:
    position = np.zeros(29)
    position[4] = np.nan

    with pytest.raises(ValueError, match="joint position"):
        build_g1_neural_torque_observation(
            _frame(joint_position=position),
            np.zeros(29),
        )


def test_safety_projector_limits_amplitude_rate_power_and_joint_boundary() -> None:
    safety = G1TorqueSafetyConfig(
        torque_guard_scale=0.70,
        max_delta_ratio_per_step=0.05,
        maximum_mechanical_power_w=100.0,
        maximum_projection_ratio=1.0,
    )
    projector = G1TorqueSafetyProjector(safety)
    frame = _frame(
        joint_velocity=np.full(29, 5.0),
    )

    result = projector.project(
        np.asarray(G1_HARD_TORQUE_LIMITS, dtype=float),
        parent=np.zeros(29),
        previous=np.zeros(29),
        frame=frame,
    )

    assert not result.used_parent
    assert result.maximum_limit_ratio <= 0.05 + 1e-9
    assert result.mechanical_power_w <= 100.0 + 1e-8
    assert result.projected_joint_count == 29

    position = np.zeros(29)
    position[0] = 1.99
    boundary = projector.project(
        np.asarray(G1_HARD_TORQUE_LIMITS, dtype=float),
        parent=np.zeros(29),
        previous=np.zeros(29),
        frame=_frame(joint_position=position),
    )
    assert boundary.used_parent
    assert boundary.reason == "joint_limit_guard"


def test_safety_projector_falls_back_to_parent_on_nonfinite_or_overprojection() -> None:
    projector = G1TorqueSafetyProjector(G1TorqueSafetyConfig(maximum_projection_ratio=0.0))
    parent = np.linspace(-1.0, 1.0, 29)

    nonfinite = projector.project(
        np.full(29, np.nan),
        parent=parent,
        previous=np.zeros(29),
        frame=_frame(),
    )
    overprojected = projector.project(
        np.asarray(G1_HARD_TORQUE_LIMITS, dtype=float),
        parent=parent,
        previous=np.zeros(29),
        frame=_frame(),
    )

    assert nonfinite.used_parent
    assert nonfinite.reason is not None and nonfinite.reason.startswith("invalid_proposal")
    assert overprojected.used_parent
    assert overprojected.reason == "projection_ratio_exceeded"
    assert nonfinite.torque == pytest.approx(parent)


def test_artifact_roundtrip_and_numpy_actor_emit_direct_bounded_torque(tmp_path: Path) -> None:
    payload = _artifact_bytes(output_bias=0.01)
    path = tmp_path / "g1-neural-torque.bin"
    path.write_bytes(payload)
    artifact_hash = "sha256:" + hashlib.sha256(payload).hexdigest()

    artifact = load_g1_neural_torque_artifact(
        path,
        expected_hash=artifact_hash,
        expected_body_hash=_digest("body"),
    )
    policy = G1NeuralTorquePolicy(
        artifact,
        expected_body_hash=_digest("body"),
        expected_parent_policy_hash=_digest("parent"),
    )
    torque = policy.command(_frame(), np.zeros(29))
    policy.note_applied(torque)
    receipt = policy.build_receipt()

    assert torque.shape == (29,)
    assert np.all(torque > 0.0)
    assert receipt.inference_count == 1
    assert receipt.learned_output_count == 1
    assert receipt.direct_torque_output
    assert receipt.activation_ceiling == "SIM_ONLY"
    assert receipt.hardware_authorized is False
    assert receipt.dds_opened is False
    assert artifact.tensors["actor.head.bias"].flags.writeable is False


def test_artifact_loader_rejects_tampering_and_non_sim_safety(tmp_path: Path) -> None:
    payload = _artifact_bytes()
    path = tmp_path / "candidate.bin"
    path.write_bytes(payload + b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_g1_neural_torque_artifact(
            path,
            expected_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        )
    with pytest.raises(ValueError, match="SIM_ONLY"):
        G1TorqueSafetyConfig(activation_ceiling="REAL")


def test_teacher_collector_is_exact_pass_through_at_physics_rate() -> None:
    collector = G1TeacherTorqueCollector()
    collector.reset()
    parent = np.linspace(-10.0, 10.0, 29)

    command = collector.command(_frame(), parent)
    collector.note_applied(command)
    episode = collector.episode()

    assert command == pytest.approx(parent)
    assert episode.observations.shape == (1, 102)
    assert episode.actions[0] == pytest.approx(parent)
    assert episode.parent_actions[0] == pytest.approx(parent)


def test_stability_plasticity_gate_activates_only_after_safe_recovery_context(
    tmp_path: Path,
) -> None:
    stable_path = tmp_path / "stable.bin"
    plastic_path = tmp_path / "plastic.bin"
    stable_path.write_bytes(_artifact_bytes(output_bias=0.01))
    plastic_path.write_bytes(_artifact_bytes(output_bias=0.02))
    stable = G1NeuralTorquePolicy(
        load_g1_neural_torque_artifact(stable_path),
        expected_body_hash=_digest("body"),
        expected_parent_policy_hash=_digest("parent"),
    )
    plastic = G1NeuralTorquePolicy(
        load_g1_neural_torque_artifact(plastic_path),
        expected_body_hash=_digest("body"),
        expected_parent_policy_hash=_digest("parent"),
    )
    policy = G1StabilityPlasticityTorquePolicy(
        stable,
        plastic,
        config=G1StabilityPlasticityGateConfig(
            minimum_recovery_phase=0.8,
            eligibility_warmup_steps=2,
        ),
    )
    parent = np.zeros(29)
    policy.reset()
    before = policy.command(_frame(policy_phase=0.5), parent)
    policy.note_applied(before)
    warmup = policy.command(_frame(policy_phase=0.9), parent)
    policy.note_applied(warmup)
    active = policy.command(_frame(policy_phase=0.9), parent)
    policy.note_applied(active)
    unstable = policy.command(
        _frame(policy_phase=0.9, left_contact=False, right_contact=False), parent
    )
    policy.note_applied(unstable)
    receipt = policy.build_receipt()

    assert active[0] > before[0]
    assert warmup == pytest.approx(before)
    assert unstable == pytest.approx(before)
    assert receipt.plastic_activation_count == 1
    assert receipt.plastic_activation_fraction == pytest.approx(0.25)
    assert receipt.phase_rejection_count == 1
    assert receipt.contact_rejection_count == 1
    assert receipt.activation_ceiling == "SIM_ONLY"


def test_neural_torque_reset_clears_episode_receipt_state(tmp_path: Path) -> None:
    artifact_path = tmp_path / "actor.bin"
    artifact_path.write_bytes(_artifact_bytes(output_bias=0.01))
    policy = G1NeuralTorquePolicy(
        load_g1_neural_torque_artifact(artifact_path),
        expected_body_hash=_digest("body"),
        expected_parent_policy_hash=_digest("parent"),
    )
    parent = np.zeros(29)
    command = policy.command(_frame(), parent)
    policy.note_applied(command)
    assert policy.build_receipt().inference_count == 1

    policy.reset()

    with pytest.raises(ValueError, match="incomplete neural torque receipt"):
        policy.build_receipt()
