from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from rosclaw.simforge.g1_contact_recovery_torque_policy import (
    G1ContactRecoveryGateConfig,
    G1ContactRecoveryTorquePolicy,
    G1DelayedRecoveryContextObserver,
    G1RecoveryContextSnapshot,
)
from rosclaw.simforge.g1_hierarchical_torque_policy import (
    G1HierarchicalTorqueGateConfig,
    G1HierarchicalTorquePolicy,
)
from rosclaw.simforge.g1_neural_torque import (
    G1_NEURAL_TORQUE_OBSERVATIONS,
    G1NeuralTorquePolicy,
    G1TeacherTorqueCollector,
    G1TeacherTorqueObserver,
    G1TorqueControlFrame,
    G1TorqueExplorationConfig,
    G1TorquePolicyReceipt,
    G1TorqueSafetyConfig,
    G1TorqueSafetyProjector,
    build_g1_neural_torque_observation,
    load_g1_neural_torque_artifact,
    serialize_g1_neural_torque_artifact,
)
from rosclaw.simforge.g1_neural_torque_overlay import (
    G1NeuralTorqueOverlayConfig,
    G1NeuralTorqueOverlayPolicy,
)
from rosclaw.simforge.g1_recovery_expert_router import build_g1_recovery_expert_router
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
        "base_linear_velocity": np.zeros(3),
        "base_angular_velocity": np.zeros(3),
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
    assert len(observation) == 108
    assert observation[58:61] == pytest.approx((0.0, 0.0, -1.0))
    assert observation[61:67] == pytest.approx(np.zeros(6))
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


def test_sim_exploration_is_seeded_context_gated_and_projected(tmp_path: Path) -> None:
    path = tmp_path / "exploration.bin"
    path.write_bytes(_artifact_bytes(output_bias=0.01))
    artifact = load_g1_neural_torque_artifact(path)
    exploration = G1TorqueExplorationConfig(
        noise_std_ratio=0.004,
        noise_clip_ratio=0.012,
        temporal_correlation=0.99,
        minimum_recovery_phase=0.5,
        seed=91,
    )

    def rollout(*, allow: bool) -> tuple[np.ndarray, G1TorquePolicyReceipt]:
        policy = G1NeuralTorquePolicy(
            artifact,
            expected_body_hash=_digest("body"),
            expected_parent_policy_hash=_digest("parent"),
            exploration=exploration,
        )
        commands = []
        for _ in range(20):
            command = policy.command(
                _frame(policy_phase=0.8),
                np.zeros(29),
                allow_exploration=allow,
            )
            policy.note_applied(command)
            commands.append(command)
        return np.asarray(commands), policy.build_receipt()

    first, first_receipt = rollout(allow=True)
    replay, replay_receipt = rollout(allow=True)
    control, control_receipt = rollout(allow=False)

    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, control)
    assert first_receipt.to_dict() == replay_receipt.to_dict()
    assert first_receipt.exploration_attempt_count == 20
    assert first_receipt.exploration_applied_count > 0
    assert first_receipt.exploration_rejection_count == 0
    assert 0.0 < first_receipt.exploration_noise_peak_ratio <= 0.012
    assert control_receipt.exploration_attempt_count == 0


def test_exploration_fails_closed_outside_sim_and_safe_context(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SIM_ONLY"):
        G1TorqueExplorationConfig(activation_ceiling="REAL")
    path = tmp_path / "context.bin"
    path.write_bytes(_artifact_bytes(output_bias=0.01))
    policy = G1NeuralTorquePolicy(
        load_g1_neural_torque_artifact(path),
        expected_body_hash=_digest("body"),
        expected_parent_policy_hash=_digest("parent"),
        exploration=G1TorqueExplorationConfig(seed=4),
    )
    for frame in (
        _frame(policy_phase=0.4),
        _frame(policy_phase=0.9, left_contact=False, right_contact=False),
        _frame(policy_phase=0.9, pelvis_position=np.asarray((0.0, 0.0, 0.6))),
    ):
        command = policy.command(frame, np.zeros(29))
        policy.note_applied(command)

    assert policy.build_receipt().exploration_attempt_count == 0


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
    assert episode.observations.shape == (1, 108)
    assert episode.actions[0] == pytest.approx(parent)
    assert episode.parent_actions[0] == pytest.approx(parent)


def test_teacher_observer_records_without_exposing_a_command_surface() -> None:
    observer = G1TeacherTorqueObserver()
    action = np.linspace(-8.0, 8.0, 29)

    observer.observe(_frame(), action)
    episode = observer.episode()

    assert not hasattr(observer, "command")
    assert episode.observations.shape == (1, len(G1_NEURAL_TORQUE_OBSERVATIONS))
    assert episode.actions[0] == pytest.approx(action)
    assert episode.parent_actions[0] == pytest.approx(action)
    observer.reset()
    with pytest.raises(ValueError, match="empty"):
        observer.episode()


def test_delayed_recovery_context_observer_is_read_only_and_contact_causal() -> None:
    observer = G1DelayedRecoveryContextObserver(delay_steps=2)
    parent = np.zeros(29)

    observer.observe(_frame(ball_contact_observed=False, ball_velocity=np.ones(3)), parent)
    observer.observe(_frame(ball_contact_observed=True, ball_velocity=np.ones(3)), parent)
    with pytest.raises(ValueError, match="did not reach"):
        observer.context()
    observer.observe(_frame(ball_contact_observed=True, ball_velocity=np.ones(3)), parent)
    context = observer.context()

    assert not hasattr(observer, "command")
    assert context.ball_speed_mps == pytest.approx(np.sqrt(3.0))
    assert context.policy_phase == pytest.approx(0.25)


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
            minimum_anticipatory_phase=0.5,
            anticipatory_blend_fraction=0.25,
            eligibility_warmup_steps=2,
        ),
    )
    parent = np.zeros(29)
    policy.reset()
    before = policy.command(_frame(policy_phase=0.4), parent)
    policy.note_applied(before)
    warmup = policy.command(_frame(policy_phase=0.9), parent)
    policy.note_applied(warmup)
    blended = policy.command(_frame(policy_phase=0.6), parent)
    policy.note_applied(blended)
    active = policy.command(_frame(policy_phase=0.9), parent)
    policy.note_applied(active)
    unstable = policy.command(
        _frame(policy_phase=0.9, left_contact=False, right_contact=False), parent
    )
    policy.note_applied(unstable)
    receipt = policy.build_receipt()

    assert active[0] > before[0]
    assert before[0] < blended[0] < active[0]
    assert warmup == pytest.approx(before)
    assert unstable == pytest.approx(before)
    assert receipt.plastic_activation_count == 2
    assert receipt.plastic_activation_fraction == pytest.approx(0.4)
    assert receipt.anticipatory_blend_count == 1
    assert receipt.phase_rejection_count == 1
    assert receipt.contact_rejection_count == 1
    assert receipt.activation_ceiling == "SIM_ONLY"
    assert policy.plastic_activation_mask().tolist() == [False, False, True, True, False]


def test_contact_recovery_overlay_is_exact_until_causal_ball_contact(
    tmp_path: Path,
) -> None:
    stable_path = tmp_path / "stable.bin"
    retained_path = tmp_path / "retained.bin"
    candidate_path = tmp_path / "candidate.bin"
    stable_path.write_bytes(_artifact_bytes(output_bias=0.01))
    retained_path.write_bytes(_artifact_bytes(output_bias=0.02))
    candidate_path.write_bytes(_artifact_bytes(output_bias=0.03))

    def actor(path: Path) -> G1NeuralTorquePolicy:
        return G1NeuralTorquePolicy(
            load_g1_neural_torque_artifact(path),
            expected_body_hash=_digest("body"),
            expected_parent_policy_hash=_digest("parent"),
        )

    baseline = G1StabilityPlasticityTorquePolicy(
        actor(stable_path),
        actor(retained_path),
        config=G1StabilityPlasticityGateConfig(
            minimum_recovery_phase=0.2,
            eligibility_warmup_steps=1,
        ),
    )
    policy = G1ContactRecoveryTorquePolicy(
        baseline,
        actor(candidate_path),
        config=G1ContactRecoveryGateConfig(
            minimum_policy_phase=0.4,
            minimum_ball_speed_mps=0.5,
            eligibility_warmup_steps=2,
        ),
    )
    parent = np.zeros(29)
    frames = (
        _frame(policy_phase=0.3, ball_velocity=np.zeros(3)),
        # A moving ball is not causal contact evidence until the backend's
        # physics contact latch is true.
        _frame(policy_phase=0.5, ball_velocity=np.asarray((1.0, 0.0, 0.0))),
        _frame(
            policy_phase=0.5,
            ball_velocity=np.asarray((1.0, 0.0, 0.0)),
            ball_contact_observed=True,
        ),
        _frame(policy_phase=0.5, ball_velocity=np.zeros(3)),
        _frame(
            policy_phase=0.5,
            ball_velocity=np.zeros(3),
            left_contact=False,
            right_contact=False,
        ),
    )
    commands = []
    for frame in frames:
        command = policy.command(frame, parent)
        policy.note_applied(command)
        commands.append(command)
    receipt = policy.build_receipt()

    assert commands[1] == pytest.approx(commands[0])
    assert commands[2] == pytest.approx(commands[0])
    assert commands[3][0] > commands[0][0]
    assert commands[4][0] < commands[0][0]
    assert policy.candidate_activation_mask().tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert receipt.ball_contact_latched
    assert receipt.contact_context is not None
    assert receipt.contact_context_hash == receipt.contact_context.context_hash
    assert receipt.contact_context.ball_speed_mps == pytest.approx(1.0)
    assert receipt.contact_context.left_contact
    assert receipt.candidate_activation_count == 1
    assert receipt.contact_rejection_count == 1
    assert receipt.support_rejection_count == 1
    assert receipt.activation_ceiling == "SIM_ONLY"
    assert not receipt.hardware_authorized


def test_neural_torque_overlay_ramps_only_after_true_contact(tmp_path: Path) -> None:
    path = tmp_path / "teacher-distilled.bin"
    path.write_bytes(_artifact_bytes(output_bias=0.03))
    actor = G1NeuralTorquePolicy(
        load_g1_neural_torque_artifact(path),
        expected_body_hash=_digest("body"),
        expected_parent_policy_hash=_digest("parent"),
    )
    policy = G1NeuralTorqueOverlayPolicy(
        actor,
        config=G1NeuralTorqueOverlayConfig(
            trust_fraction=0.20,
            eligibility_warmup_steps=2,
            ramp_steps=2,
        ),
    )
    parent = np.zeros(29)
    frames = (
        _frame(policy_phase=0.5, ball_velocity=np.asarray((1.0, 0.0, 0.0))),
        _frame(
            policy_phase=0.5,
            ball_velocity=np.asarray((1.0, 0.0, 0.0)),
            ball_contact_observed=True,
        ),
        _frame(policy_phase=0.5, ball_velocity=np.zeros(3)),
        _frame(policy_phase=0.5, ball_velocity=np.zeros(3)),
        _frame(
            policy_phase=0.5,
            ball_velocity=np.zeros(3),
            left_contact=False,
            right_contact=False,
        ),
    )
    commands = []
    for frame in frames:
        command = policy.command(frame, parent)
        policy.note_applied(command)
        commands.append(command)
    receipt = policy.build_receipt()

    assert commands[0] == pytest.approx(parent)
    assert commands[1] == pytest.approx(parent)
    assert 0.0 < commands[2][0] < commands[3][0]
    assert commands[4] == pytest.approx(parent)
    assert receipt.ball_contact_latched
    assert receipt.activation_count == 2
    assert receipt.activation_fraction == pytest.approx(0.4)
    assert receipt.maximum_applied_trust_fraction == pytest.approx(0.20)
    assert 0.0 < receipt.maximum_residual_ratio < 0.25
    assert receipt.hardware_authorized is False
    assert receipt.dds_opened is False


def test_neural_torque_overlay_records_proposal_before_trust_blending(tmp_path: Path) -> None:
    path = tmp_path / "candidate.bin"
    path.write_bytes(_artifact_bytes(output_bias=0.03))
    candidate = G1NeuralTorquePolicy(
        load_g1_neural_torque_artifact(path),
        expected_body_hash=_digest("body"),
        expected_parent_policy_hash=_digest("parent"),
    )
    overlay = G1NeuralTorqueOverlayPolicy(
        candidate,
        config=G1NeuralTorqueOverlayConfig(
            trust_fraction=0.20,
            minimum_policy_phase=0.20,
            eligibility_warmup_steps=1,
            ramp_steps=1,
        ),
    )

    applied = overlay.command(
        _frame(policy_phase=0.9, ball_contact_observed=True, ball_velocity=np.ones(3)),
        np.zeros(29),
    )
    proposal = candidate.pending_projection.torque.copy()
    overlay.note_applied(applied)
    trace = overlay.policy_episode()

    assert trace.activation_mask.tolist() == [True]
    assert trace.trust_fractions.tolist() == pytest.approx([0.20])
    assert trace.policy_episode.actions[0] == pytest.approx(proposal)
    assert trace.applied_actions[0] == pytest.approx(applied)
    assert np.max(np.abs(trace.policy_episode.actions[0])) > np.max(np.abs(applied))
    assert trace.applied_actions.flags.writeable is False
    assert trace.trust_fractions.flags.writeable is False
    assert trace.activation_mask.flags.writeable is False


def test_neural_torque_overlay_waits_for_observable_applicability_route(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.bin"
    path.write_bytes(_artifact_bytes(output_bias=0.03))
    artifact = load_g1_neural_torque_artifact(path)
    frame = _frame(policy_phase=0.9, ball_contact_observed=True, ball_velocity=np.ones(3))
    speed = float(np.linalg.norm(frame.ball_velocity))
    context = G1RecoveryContextSnapshot.from_frame(frame, ball_speed_mps=speed)
    router = build_g1_recovery_expert_router(
        body_hash=artifact.body_hash,
        parent_policy_hash=artifact.parent_policy_hash,
        source_evidence_hash=_digest("observable-context-evidence"),
        contexts=(context,),
        selected_expert_ids=("candidate",),
        measured_gains=(0.10,),
        task_preserved=(True,),
        expert_artifact_hashes={"candidate": artifact.artifact_hash},
    )
    candidate = G1NeuralTorquePolicy(
        artifact,
        expected_body_hash=_digest("body"),
        expected_parent_policy_hash=_digest("parent"),
    )
    overlay = G1NeuralTorqueOverlayPolicy(
        candidate,
        config=G1NeuralTorqueOverlayConfig(
            trust_fraction=0.20,
            minimum_policy_phase=0.20,
            eligibility_warmup_steps=1,
            ramp_steps=1,
        ),
        applicability_router=router,
        applicability_expert_id="candidate",
        applicability_delay_steps=2,
    )
    parent = np.zeros(29)

    waiting = overlay.command(frame, parent)
    overlay.note_applied(waiting)
    active = overlay.command(frame, parent)
    overlay.note_applied(active)
    receipt = overlay.build_receipt()

    assert waiting == pytest.approx(parent)
    assert np.max(np.abs(active)) > 0.0
    assert receipt.activation_count == 1
    assert receipt.context_rejection_count == 1
    assert receipt.applicability_artifact_hash == router.artifact_hash
    assert receipt.applicability_route is not None
    assert receipt.applicability_route.eligible
    assert receipt.applicability_context_hash == context.context_hash


def test_hierarchical_policy_routes_disjoint_balance_and_recovery_heads(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / name for name in ("stable.bin", "balance.bin", "recovery.bin")]
    for path, bias in zip(paths, (0.01, 0.02, 0.03), strict=True):
        path.write_bytes(_artifact_bytes(output_bias=bias))

    def actor(path: Path) -> G1NeuralTorquePolicy:
        return G1NeuralTorquePolicy(
            load_g1_neural_torque_artifact(path),
            expected_body_hash=_digest("body"),
            expected_parent_policy_hash=_digest("parent"),
        )

    policy = G1HierarchicalTorquePolicy(
        actor(paths[0]),
        actor(paths[1]),
        actor(paths[2]),
        config=G1HierarchicalTorqueGateConfig(
            balance_start_phase=0.02,
            balance_end_phase=0.42,
            recovery_start_phase=0.45,
            balance_warmup_steps=2,
            recovery_warmup_steps=2,
        ),
    )
    parent = np.zeros(29)
    frames = (
        _frame(policy_phase=0.01),
        _frame(policy_phase=0.03),
        _frame(policy_phase=0.03),
        _frame(policy_phase=0.43),
        _frame(policy_phase=0.46),
        _frame(policy_phase=0.46),
        _frame(policy_phase=0.47, left_contact=False, right_contact=False),
    )
    commands = []
    for frame in frames:
        command = policy.command(frame, parent)
        policy.note_applied(command)
        commands.append(command)
    receipt = policy.build_receipt()

    assert commands[2][0] > commands[0][0]
    assert commands[5][0] > commands[2][0]
    assert commands[1] == pytest.approx(commands[0])
    assert commands[3] == pytest.approx(commands[0])
    assert commands[4] == pytest.approx(commands[0])
    assert commands[6] == pytest.approx(commands[0])
    assert policy.balance_activation_mask().tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert policy.recovery_activation_mask().tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    assert receipt.balance_activation_count == 1
    assert receipt.recovery_activation_count == 1
    assert receipt.contact_rejection_count == 1
    assert receipt.activation_ceiling == "SIM_ONLY"
    assert not receipt.hardware_authorized


def test_hierarchical_policy_contract_rejects_overlap_and_hardware() -> None:
    with pytest.raises(ValueError, match="phase windows"):
        G1HierarchicalTorqueGateConfig(
            balance_end_phase=0.5,
            recovery_start_phase=0.4,
        )
    with pytest.raises(ValueError, match="SIM_ONLY"):
        G1HierarchicalTorqueGateConfig(activation_ceiling="REAL")


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
