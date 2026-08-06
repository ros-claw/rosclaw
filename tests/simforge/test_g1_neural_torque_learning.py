from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from rosclaw.collective.sources.motiondecode.motion_prior import (
    G1_MOTION_PRIOR_FEATURES,
    G1MotionPriorArtifact,
)
from rosclaw.simforge.g1_neural_torque import (
    G1_NEURAL_TORQUE_OBSERVATIONS,
    G1TeacherTorqueEpisode,
    load_g1_neural_torque_artifact,
)
from rosclaw.simforge.g1_neural_torque_learning import (
    G1ContinualTorqueActorCritic,
    G1NeuralTorqueLearnerConfig,
    G1NeuralTorqueReplay,
    G1RecoveryPhaseReturn,
    _discounted_returns,
    _quantized_export,
    balance_online_replay,
    online_replay,
    overlay_recovery_online_replay,
    recovery_online_replay,
    stale_neural_torque_replay,
    teacher_dataset_hash,
    teacher_replay,
)
from rosclaw.simforge.g1_neural_torque_overlay import G1NeuralTorqueOverlayEpisode
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _episode(seed: int, *, count: int = 96) -> G1TeacherTorqueEpisode:
    rng = np.random.default_rng(seed)
    observation_dim = len(G1_NEURAL_TORQUE_OBSERVATIONS)
    time = np.linspace(0.0, 2.0 * np.pi, count, dtype=np.float32)
    observations = rng.normal(0.0, 0.05, (count, observation_dim)).astype(np.float32)
    observations[:, 0] = np.sin(time)
    observations[:, 29] = np.cos(time)
    observations[:, 60] = -1.0
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float32)
    actions = np.zeros((count, 29), dtype=np.float32)
    actions[:, 0] = 0.08 * limits[0] * np.sin(time)
    actions[:, 1] = 0.04 * limits[1] * np.cos(time)
    return G1TeacherTorqueEpisode(observations, actions, actions)


def _learner(seed: int = 3) -> G1ContinualTorqueActorCritic:
    return G1ContinualTorqueActorCritic(
        G1NeuralTorqueLearnerConfig(
            hidden_dim=12,
            sequence_length=8,
            batch_size=8,
            device="cpu",
            seed=seed,
        )
    )


def _motion_prior(hidden_dim: int = 12) -> G1MotionPriorArtifact:
    rng = np.random.default_rng(91)
    feature_count = len(G1_MOTION_PRIOR_FEATURES)
    tensors = {
        "gru.weight_ih_l0": rng.normal(0.0, 0.01, (3 * hidden_dim, feature_count)).astype(
            np.float32
        ),
        "gru.weight_hh_l0": rng.normal(0.0, 0.01, (3 * hidden_dim, hidden_dim)).astype(np.float32),
        "gru.bias_ih_l0": np.zeros(3 * hidden_dim, dtype=np.float32),
        "gru.bias_hh_l0": np.zeros(3 * hidden_dim, dtype=np.float32),
        "head.weight": np.full((feature_count, hidden_dim), 999.0, dtype=np.float32),
        "head.bias": np.full(feature_count, 999.0, dtype=np.float32),
        "observation_mean": np.zeros(feature_count, dtype=np.float32),
        "observation_std": np.ones(feature_count, dtype=np.float32),
    }
    return G1MotionPriorArtifact(
        artifact_hash=_digest("motion-prior"),
        weights_hash=_digest("motion-prior-weights"),
        pack_hash=_digest("motion-prior-pack"),
        pilot_report_hash=_digest("motion-prior-pilot"),
        body_hash=_digest("body"),
        feature_names=G1_MOTION_PRIOR_FEATURES,
        hidden_dim=hidden_dim,
        sequence_length=8,
        observation_mean=tensors["observation_mean"],
        observation_std=tensors["observation_std"],
        tensors=tensors,
        source_truth_level="T4",
        action_semantics="ABSENT",
        activation_ceiling="SIM_ONLY_REPRESENTATION_INITIALIZATION",
    )


def test_behavior_cloning_learns_nonzero_direct_torque_and_exports_safe_actor(
    tmp_path,
) -> None:
    learner = _learner()
    training = (_episode(1), _episode(2))
    validation = (_episode(4),)

    metrics = learner.pretrain_behavior(
        training,
        validation=validation,
        epochs=8,
        stride=2,
    )
    payload = learner.artifact_bytes(
        body_hash=_digest("body"),
        parent_policy_hash=_digest("parent"),
        dataset_hash=teacher_dataset_hash(training),
    )
    path = tmp_path / "actor.bin"
    path.write_bytes(payload)
    artifact = load_g1_neural_torque_artifact(path)
    sequence = validation[0].observations[:8]
    action = learner.deterministic_action(sequence)

    assert metrics[-1].training_loss < metrics[0].training_loss
    assert metrics[-1].finite
    assert np.linalg.norm(action) > 0.0
    assert np.all(np.abs(action) <= np.asarray(artifact.action_limits) + 1e-6)
    assert artifact.update_index == 0
    assert artifact.safety.activation_ceiling == "SIM_ONLY"

    early = _learner(seed=8)
    early_metrics = early.pretrain_behavior(
        training,
        validation=validation,
        epochs=1,
        stride=2,
        maximum_end_fraction=0.20,
    )
    assert early_metrics[-1].finite
    with pytest.raises(ValueError, match="maximum end fraction"):
        early.pretrain_behavior(
            training,
            epochs=1,
            minimum_end_fraction=0.3,
            maximum_end_fraction=0.2,
        )


def test_motion_prior_initializes_only_matching_sim_proprioceptive_gru() -> None:
    learner = _learner(seed=17)
    before = learner.actor_snapshot()
    prior = _motion_prior()
    learner.install_motion_prior(prior, expected_body_hash=_digest("body"))
    learner.pretrain_behavior((_episode(1), _episode(2)), epochs=1, stride=2)
    after = learner.actor_snapshot()

    assert learner.motion_prior_artifact_hash == prior.artifact_hash
    feature_count = len(G1_MOTION_PRIOR_FEATURES)
    assert not np.array_equal(
        after["gru.weight_ih_l0"][:, :feature_count],
        before["gru.weight_ih_l0"][:, :feature_count],
    )
    assert not np.allclose(after["head.weight"], 999.0)
    with pytest.raises(ValueError, match="before torque consolidation"):
        learner.install_motion_prior(prior, expected_body_hash=_digest("body"))


def test_motion_prior_rejects_body_mismatch() -> None:
    learner = _learner()
    with pytest.raises(ValueError, match="body hash"):
        learner.install_motion_prior(_motion_prior(), expected_body_hash=_digest("other-body"))

    incomplete = replace(
        _motion_prior(),
        feature_names=G1_MOTION_PRIOR_FEATURES[:-1],
    )
    with pytest.raises(ValueError, match="feature order"):
        learner.install_motion_prior(incomplete, expected_body_hash=_digest("body"))


def test_recovery_replay_aligns_500hz_policy_with_dense_50hz_physics() -> None:
    episode = _episode(31, count=100)
    trajectory = {
        "policy_phase": np.full(10, 0.7),
        "support_foot_slip": np.linspace(0.0, 0.05, 10),
        "com_y_relative": np.linspace(0.0, 0.13, 10),
        "left_foot_contact": np.ones(10),
        "right_foot_contact": np.ones(10),
    }
    replay = recovery_online_replay(
        episode,
        trajectory=trajectory,
        sequence_length=8,
        recovery_score=2.0,
        fell=False,
        critical_failure=False,
        projection_fallback_rate=0.0,
        stride=10,
    )

    assert replay.count == 10
    assert replay.partitions.tolist() == [0] * 8 + [2, 2]
    assert replay.rewards[-1, 0] > replay.rewards[-2, 0]
    assert replay.constraint_costs[-1, 0] == 1.0
    assert replay.fall_costs[-1, 0] == 0.0
    assert replay.terminals[-1, 0] == 1.0
    eligibility = np.ones(100, dtype=np.bool_)
    eligibility[7] = False
    masked = recovery_online_replay(
        episode,
        trajectory=trajectory,
        sequence_length=8,
        recovery_score=2.0,
        fell=False,
        critical_failure=False,
        projection_fallback_rate=0.0,
        actor_eligible_mask=eligibility,
        stride=10,
    )
    assert masked.partitions.tolist() == [2, *([0] * 7), 2, 2]

    critical_but_upright = recovery_online_replay(
        episode,
        trajectory={
            **trajectory,
            "support_foot_slip": np.zeros(10),
            "com_y_relative": np.zeros(10),
        },
        sequence_length=8,
        recovery_score=-2.0,
        fell=False,
        critical_failure=True,
        projection_fallback_rate=0.0,
        actor_eligible_mask=np.ones(100, dtype=np.bool_),
        stride=10,
    )
    assert np.all(critical_but_upright.partitions == 0)
    assert critical_but_upright.constraint_costs[-1, 0] == 1.0

    fallen = recovery_online_replay(
        episode,
        trajectory={
            **trajectory,
            "support_foot_slip": np.zeros(10),
            "com_y_relative": np.zeros(10),
        },
        sequence_length=8,
        recovery_score=-10.0,
        fell=True,
        critical_failure=True,
        projection_fallback_rate=0.0,
        actor_eligible_mask=np.ones(100, dtype=np.bool_),
        fall_quarantine_sec=0.10,
        stride=10,
    )
    assert np.count_nonzero(fallen.partitions == 2) == 5
    assert fallen.fall_costs[-1, 0] == 1.0
    with pytest.raises(ValueError, match="eligibility mask"):
        recovery_online_replay(
            episode,
            trajectory=trajectory,
            sequence_length=8,
            recovery_score=0.0,
            fell=False,
            critical_failure=False,
            projection_fallback_rate=0.0,
            actor_eligible_mask=np.ones(99, dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="misaligned"):
        recovery_online_replay(
            episode,
            trajectory={**trajectory, "policy_phase": np.full(9, 0.7)},
            sequence_length=8,
            recovery_score=0.0,
            fell=False,
            critical_failure=False,
            projection_fallback_rate=0.0,
        )
    with pytest.raises(ValueError, match="non-numeric"):
        recovery_online_replay(
            episode,
            trajectory={**trajectory, "policy_phase": np.full(10, "bad")},
            sequence_length=8,
            recovery_score=0.0,
            fell=False,
            critical_failure=False,
            projection_fallback_rate=0.0,
        )


def test_recovery_replay_preserves_phase_specific_and_task_credit() -> None:
    episode = _episode(63, count=100)
    trajectory = {
        "policy_phase": np.full(10, 0.8),
        "support_foot_slip": np.zeros(10),
        "com_y_relative": np.zeros(10),
        "left_foot_contact": np.ones(10),
        "right_foot_contact": np.ones(10),
    }
    baseline = recovery_online_replay(
        episode,
        trajectory=trajectory,
        sequence_length=8,
        recovery_score=0.0,
        fell=False,
        critical_failure=False,
        projection_fallback_rate=0.0,
        stride=10,
    )
    segmented = recovery_online_replay(
        episode,
        trajectory=trajectory,
        sequence_length=8,
        recovery_score=0.0,
        fell=False,
        critical_failure=False,
        projection_fallback_rate=0.0,
        phase_return=G1RecoveryPhaseReturn(
            impulse_acceptance=1.0,
            momentum_unloading=2.0,
            terminal_settling=3.0,
            task_retention=4.0,
        ),
        stride=10,
    )

    delta = segmented.rewards[:, 0] - baseline.rewards[:, 0]
    assert np.flatnonzero(delta).tolist() == [3, 6, 9]
    assert delta[[3, 6, 9]] == pytest.approx([0.25, 0.50, 1.75])
    with pytest.raises(ValueError, match="finite and bounded"):
        G1RecoveryPhaseReturn(terminal_settling=float("nan"))


def test_overlay_recovery_replay_credits_proposal_only_when_trust_was_applied() -> None:
    source = _episode(57, count=100)
    proposals = source.actions.copy()
    proposals[:, 0] += 2.0
    parent = np.zeros_like(proposals)
    applied = 0.2 * proposals
    active = np.ones(100, dtype=np.bool_)
    active[7] = False
    trace = G1NeuralTorqueOverlayEpisode(
        policy_episode=G1TeacherTorqueEpisode(source.observations, proposals, parent),
        applied_actions=applied,
        trust_fractions=np.where(active, 0.2, 0.0).astype(np.float32),
        activation_mask=active,
    )
    trajectory = {
        "policy_phase": np.full(10, 0.8),
        "support_foot_slip": np.zeros(10),
        "com_y_relative": np.zeros(10),
        "left_foot_contact": np.ones(10),
        "right_foot_contact": np.ones(10),
    }

    replay = overlay_recovery_online_replay(
        trace,
        trajectory=trajectory,
        sequence_length=8,
        recovery_score=1.0,
        fell=False,
        critical_failure=False,
        projection_fallback_rate=0.0,
        stride=10,
    )

    assert replay.partitions[0] == 2
    assert replay.partitions[1] == 0
    assert replay.actions[0] == pytest.approx(proposals[7])
    assert replay.actions[0] != pytest.approx(applied[7])
    assert replay.parent_actions[0] == pytest.approx(parent[7])


def test_balance_replay_uses_disjoint_phase_mask_and_future_risk() -> None:
    episode = _episode(41, count=100)
    phase = np.linspace(0.01, 0.50, 10)
    pelvis_pose = np.zeros((10, 7), dtype=np.float64)
    pelvis_pose[:, 2] = 0.80
    pelvis_pose[:, 3] = 1.0
    trajectory = {
        "policy_phase": phase,
        "support_foot_slip": np.zeros(10),
        "com_y_relative": np.zeros(10),
        "left_foot_contact": np.ones(10),
        "right_foot_contact": np.zeros(10),
        "pelvis_pose": pelvis_pose,
    }
    eligible = np.ones(100, dtype=np.bool_)
    eligible[17] = False
    replay = balance_online_replay(
        episode,
        trajectory=trajectory,
        sequence_length=8,
        balance_score=2.0,
        fell=False,
        critical_failure=False,
        projection_fallback_rate=0.0,
        actor_eligible_mask=eligible,
        stride=10,
    )

    assert replay.count == 3
    assert replay.partitions.tolist() == [2, 0, 0]
    assert replay.terminals[-1, 0] == 1.0
    assert replay.rewards[-1, 0] > replay.rewards[-2, 0]

    risky_observations = episode.observations.copy()
    risky_observations[45:65, 58] = 0.55
    risky_observations[45:65, 60] = 0.0
    risky = balance_online_replay(
        G1TeacherTorqueEpisode(
            risky_observations,
            episode.actions,
            episode.parent_actions,
        ),
        trajectory=trajectory,
        sequence_length=8,
        balance_score=-5.0,
        fell=True,
        critical_failure=True,
        projection_fallback_rate=0.0,
        actor_eligible_mask=np.ones(100, dtype=np.bool_),
        stride=10,
    )
    assert np.count_nonzero(risky.partitions == 2) > 0
    assert np.max(risky.fall_costs) == 1.0
    assert np.max(risky.constraint_costs) == 1.0
    with pytest.raises(ValueError, match="pelvis trajectory"):
        balance_online_replay(
            episode,
            trajectory={**trajectory, "pelvis_pose": pelvis_pose[:-1]},
            sequence_length=8,
            balance_score=0.0,
            fell=False,
            critical_failure=False,
            projection_fallback_rate=0.0,
            actor_eligible_mask=np.ones(100, dtype=np.bool_),
        )


def test_actor_export_is_canonical_and_trust_region_is_bounded(tmp_path: Path) -> None:
    learner = _learner(seed=13)
    training = (_episode(1), _episode(2))
    learner.pretrain_behavior(training, epochs=2, stride=2)
    parent = learner.actor_snapshot()
    proposal = {name: value.copy() for name, value in parent.items()}
    first = next(iter(proposal))
    proposal[first].flat[0] += 1e-4

    assert np.array_equal(
        _quantized_export(np.asarray((0.12345641,), dtype=np.float64)),
        _quantized_export(np.asarray((0.12345642,), dtype=np.float64)),
    )
    learner.install_interpolated_actor(parent, proposal, fraction=0.0)
    restored = learner.actor_snapshot()
    assert all(np.array_equal(restored[name], parent[name]) for name in parent)
    with pytest.raises(ValueError, match="fraction"):
        learner.install_interpolated_actor(parent, proposal, fraction=1.01)
    payload = learner.artifact_bytes(
        body_hash=_digest("body"),
        parent_policy_hash=_digest("parent"),
        dataset_hash=_digest("dataset"),
    )
    path = tmp_path / "quantized.bin"
    path.write_bytes(payload)
    artifact = load_g1_neural_torque_artifact(path)
    learner.install_interpolated_actor(parent, proposal, fraction=1.0)
    learner.install_actor_artifact(
        artifact,
        expected_body_hash=_digest("body"),
        expected_parent_policy_hash=_digest("parent"),
    )
    installed = learner.actor_snapshot()
    consolidation = learner.consolidate_installed_actor(
        training,
        maximum_end_fraction=0.20,
    )
    assert all(
        np.array_equal(installed[name], learner.actor_snapshot()[name]) for name in installed
    )
    assert all(np.isfinite(consolidation))
    assert (
        learner.artifact_bytes(
            body_hash=_digest("body"),
            parent_policy_hash=_digest("parent"),
            dataset_hash=_digest("dataset"),
        )
        == payload
    )
    with pytest.raises(ValueError, match="observation contract"):
        learner.install_actor_artifact(
            replace(artifact, observation_names=artifact.observation_names[:-1]),
            expected_body_hash=_digest("body"),
            expected_parent_policy_hash=_digest("parent"),
        )
    invalid_tensors = dict(artifact.tensors)
    invalid_tensors["observation_std"] = np.zeros_like(invalid_tensors["observation_std"])
    with pytest.raises(ValueError, match="normalization"):
        learner.install_actor_artifact(
            replace(artifact, tensors=invalid_tensors),
            expected_body_hash=_digest("body"),
            expected_parent_policy_hash=_digest("parent"),
        )


def test_actor_trust_region_can_limit_updates_to_anatomical_readout() -> None:
    learner = _learner(seed=14)
    parent = learner.actor_snapshot()
    proposal = {name: value + np.asarray(1e-3, dtype=value.dtype) for name, value in parent.items()}

    learner.install_interpolated_actor(
        parent,
        proposal,
        fraction=1.0,
        action_indices=(0, 13),
    )
    installed = learner.actor_snapshot()

    assert np.array_equal(installed["gru.weight_ih_l0"], parent["gru.weight_ih_l0"])
    assert np.array_equal(installed["head.weight"][1], parent["head.weight"][1])
    assert np.array_equal(installed["head.weight"][29:], parent["head.weight"][29:])
    assert not np.array_equal(installed["head.weight"][0], parent["head.weight"][0])
    assert not np.array_equal(installed["head.bias"][13], parent["head.bias"][13])
    with pytest.raises(ValueError, match="action subspace"):
        learner.install_interpolated_actor(
            parent,
            proposal,
            fraction=1.0,
            action_indices=(29,),
        )


def test_continual_actor_critic_uses_stale_and_boundary_only_for_critics() -> None:
    learner = _learner(seed=5)
    anchors = (_episode(1), _episode(2))
    learner.pretrain_behavior(anchors, epochs=2, stride=2)
    anchor_replay = teacher_replay(anchors, sequence_length=8, stride=4)
    fresh_replay = online_replay(
        _episode(7),
        sequence_length=8,
        task_score=2.0,
        fell=False,
        critical_failure=False,
        projection_fallback_rate=0.0,
        stride=4,
    )
    stale_replay = online_replay(
        _episode(8),
        sequence_length=8,
        task_score=0.0,
        fell=False,
        critical_failure=False,
        projection_fallback_rate=0.0,
        policy_lag=2,
        stride=4,
    )
    boundary_replay = online_replay(
        _episode(9),
        sequence_length=8,
        task_score=-5.0,
        fell=True,
        critical_failure=True,
        projection_fallback_rate=0.4,
        stride=4,
    )
    replay = G1NeuralTorqueReplay.combine(
        anchor_replay,
        fresh_replay,
        stale_replay,
        boundary_replay,
    )

    update = learner.update(replay)

    assert update.finite
    assert update.actor_transition_count == fresh_replay.count
    assert update.stale_actor_transition_count == stale_replay.count
    assert update.anchor_transition_count == anchor_replay.count
    assert update.critic_transition_count == replay.count
    assert update.anchor_loss >= 0.0
    assert update.ewc_loss >= 0.0

    aged = stale_neural_torque_replay(fresh_replay)
    assert np.all(aged.policy_lags == 2)
    assert np.all(fresh_replay.policy_lags == 0)
    stale_only = G1NeuralTorqueReplay.combine(anchor_replay, aged)
    with pytest.raises(ValueError, match="fresh online transitions"):
        learner.update(stale_only)
    assert learner.update(stale_only, update_actor=False).finite

    early_anchor = teacher_replay(
        anchors,
        sequence_length=8,
        stride=4,
        maximum_end_fraction=0.20,
    )
    assert early_anchor.count < anchor_replay.count
    assert np.all(early_anchor.partitions == 1)
    with pytest.raises(ValueError, match="end-fraction window"):
        teacher_replay(
            anchors,
            sequence_length=8,
            minimum_end_fraction=0.4,
            maximum_end_fraction=0.2,
        )


def test_online_update_requires_fresh_plasticity_and_historical_stability() -> None:
    learner = _learner()
    anchors = (_episode(1),)
    learner.pretrain_behavior(anchors, epochs=1, stride=2)
    anchor_replay = teacher_replay(anchors, sequence_length=8, stride=4)
    boundary = online_replay(
        _episode(3),
        sequence_length=8,
        task_score=-3.0,
        fell=True,
        critical_failure=True,
        projection_fallback_rate=1.0,
        stride=4,
    )

    with pytest.raises(ValueError, match="fresh online transitions"):
        learner.update(G1NeuralTorqueReplay.combine(anchor_replay, boundary))
    critic_only = learner.update(
        G1NeuralTorqueReplay.combine(anchor_replay, boundary),
        update_actor=False,
    )
    assert critic_only.finite
    assert not critic_only.actor_updated
    assert critic_only.actor_transition_count == 0
    with pytest.raises(ValueError, match="historical anchors"):
        learner.update(
            online_replay(
                _episode(4),
                sequence_length=8,
                task_score=1.0,
                fell=False,
                critical_failure=False,
                projection_fallback_rate=0.0,
                stride=4,
            )
        )


def test_discounted_returns_stop_at_each_trajectory_boundary() -> None:
    rewards = np.asarray([[1.0], [2.0], [10.0], [20.0]], dtype=np.float32)
    terminals = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)

    values = _discounted_returns(rewards, terminals, gamma=0.5)

    assert values[:, 0] == pytest.approx((2.0, 2.0, 20.0, 20.0))


def test_advantage_weighted_update_is_in_sample_and_restorable() -> None:
    learner = _learner(seed=31)
    anchors = (_episode(1), _episode(2))
    learner.pretrain_behavior(anchors, epochs=2, stride=2)
    replay = G1NeuralTorqueReplay.combine(
        teacher_replay(anchors, sequence_length=8, stride=4),
        online_replay(
            _episode(5),
            sequence_length=8,
            task_score=2.0,
            fell=False,
            critical_failure=False,
            projection_fallback_rate=0.0,
            stride=4,
        ),
    )
    before = learner.actor_snapshot()
    for _ in range(4):
        value_only = learner.update_advantage_weighted(replay, update_actor=False)
    update = learner.update_advantage_weighted(replay)
    after = learner.actor_snapshot()

    assert value_only.learning_mode == "AWR_IN_SAMPLE"
    assert value_only.value_loss >= 0.0
    assert update.finite
    assert update.actor_updated
    assert update.advantage_weight_mean == pytest.approx(1.0)
    assert 0.0 < update.advantage_weight_max <= learner.config.awr_max_weight
    assert any(not np.array_equal(before[name], after[name]) for name in before)

    checkpoint = learner.checkpoint_bytes()
    restored = _learner(seed=31)
    restored.restore_checkpoint(checkpoint)
    assert restored.update_advantage_weighted(replay, update_actor=False).finite


def test_full_checkpoint_restores_actor_critics_optimizers_and_ewc_state() -> None:
    learner = _learner(seed=11)
    anchors = (_episode(1), _episode(2))
    learner.pretrain_behavior(anchors, epochs=2, stride=2)
    replay = G1NeuralTorqueReplay.combine(
        teacher_replay(anchors, sequence_length=8, stride=4),
        online_replay(
            _episode(5),
            sequence_length=8,
            task_score=1.0,
            fell=False,
            critical_failure=False,
            projection_fallback_rate=0.0,
            stride=4,
        ),
    )
    learner.update(replay)
    checkpoint = learner.checkpoint_bytes()
    expected = learner.artifact_bytes(
        body_hash=_digest("body"),
        parent_policy_hash=_digest("parent"),
        dataset_hash=teacher_dataset_hash(anchors),
    )

    recovered = _learner(seed=11)
    recovered.restore_checkpoint(checkpoint)
    actual = recovered.artifact_bytes(
        body_hash=_digest("body"),
        parent_policy_hash=_digest("parent"),
        dataset_hash=teacher_dataset_hash(anchors),
    )

    assert recovered.update_index == learner.update_index
    assert actual == expected
    assert recovered.update(replay).finite
