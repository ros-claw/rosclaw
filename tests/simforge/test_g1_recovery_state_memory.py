from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from rosclaw.simforge.g1_cerebellar_recovery import (
    G1CerebellarRecoveryConfig,
    G1CerebellarRecoveryController,
)
from rosclaw.simforge.g1_contextual_recovery import G1ContextualRecoveryPrimitive
from rosclaw.simforge.g1_recovery_state_memory import (
    G1_RECOVERY_STATE_FEATURES,
    G1_RECOVERY_STATE_OBSERVATIONS,
    G1RecoveryStateArtifact,
    G1RecoveryStatePolicy,
    load_g1_recovery_state_artifact,
)

_HASH = "sha256:" + "1" * 64
_MOTION_HASH = "sha256:" + "2" * 64
_REGIME_HASH = "sha256:" + "6" * 64


def _primitive() -> G1ContextualRecoveryPrimitive:
    return G1ContextualRecoveryPrimitive(
        start_policy_frame=300,
        blend_frames=80,
        settling_start_policy_frame=380,
        settling_blend_frames=60,
        settling_standing_pose_blend=0.42,
        settling_waist_pitch_bias_rad=0.10,
        target_smoothing_alpha=0.54,
    )


def _artifact() -> G1RecoveryStateArtifact:
    width = 2 * len(G1_RECOVERY_STATE_FEATURES)
    return G1RecoveryStateArtifact(
        body_hash=_HASH,
        motion_hash=_MOTION_HASH,
        baseline_recovery_config_hash="sha256:" + "3" * 64,
        fallback_recovery_config_hash="sha256:" + "4" * 64,
        training_dataset_hash="sha256:" + "5" * 64,
        observation_mean=(0.0,) * len(G1_RECOVERY_STATE_OBSERVATIONS),
        observation_scale=(1.0,) * len(G1_RECOVERY_STATE_OBSERVATIONS),
        descriptor_feature_names=G1_RECOVERY_STATE_FEATURES,
        descriptor_prototypes=(
            (0.0,) * width,
            (0.01,) * width,
            (1.0,) * width,
        ),
        prototype_primitive_indices=(0, 0, -1),
        prototype_composite_advantages=(0.12, 0.09, -0.20),
        prototype_component_minimums=(0.02, 0.01, -0.50),
        primitives=(_primitive(),),
        selection_window_frames=5,
        neighbor_count=2,
        maximum_neighbor_distance=0.20,
        minimum_primitive_consensus=1.0,
        minimum_advantage_lower_bound=0.05,
        minimum_component_lower_bound=-0.02,
        maximum_feature_z=8.0,
        training_episode_count=12,
        training_seed=17,
    )


def _observation(value: float = 0.0) -> dict[str, float]:
    return dict.fromkeys(G1_RECOVERY_STATE_OBSERVATIONS, value)


def _controller() -> G1CerebellarRecoveryController:
    baseline = G1CerebellarRecoveryConfig(
        start_policy_frame=300,
        blend_frames=100,
        standing_pose_blend=0.20,
        roll_posture_bias_rad=0.0,
        settling_start_policy_frame=400,
        settling_blend_frames=100,
        settling_standing_pose_blend=0.42,
        settling_roll_posture_bias_rad=0.0,
        settling_waist_pitch_bias_rad=0.11,
        target_smoothing_alpha=0.54,
        target_smoothing_start_policy_frame=300,
    )
    fallback = G1CerebellarRecoveryConfig()
    common = {
        "body_hash": _HASH,
        "motion_hash": _MOTION_HASH,
        "regime_commitment": _REGIME_HASH,
        "regime_eligible": True,
        "regime_reasons": (),
        "standing_pose": np.zeros(29),
    }
    baseline_hash = G1CerebellarRecoveryController(config=baseline, **common).config_hash
    fallback_hash = G1CerebellarRecoveryController(config=fallback, **common).config_hash
    artifact = replace(
        _artifact(),
        baseline_recovery_config_hash=baseline_hash,
        fallback_recovery_config_hash=fallback_hash,
    )
    return G1CerebellarRecoveryController(
        config=baseline,
        fallback_config=fallback,
        recovery_state_artifact=artifact,
        **common,
    )


def test_recovery_state_policy_waits_for_window_then_routes_conservative_neighbor() -> None:
    policy = G1RecoveryStatePolicy(_artifact())

    pending = [policy.select(_observation()) for _ in range(4)]
    selected = policy.select(_observation())
    replay = policy.select(_observation(7.0))
    receipt = policy.build_receipt()

    assert all(not item.ready for item in pending)
    assert selected.ready and not selected.out_of_distribution
    assert selected.primitive_index == 0
    assert selected.neighbor_count == 2
    assert selected.primitive_consensus == 1.0
    assert selected.advantage_lower_bound == pytest.approx(0.09)
    assert replay == selected
    assert receipt.pending_count == 4
    assert receipt.selection_count == 1
    assert receipt.fallback_count == 0
    assert receipt.descriptor_hash is not None


def test_recovery_state_policy_treats_negative_neighbors_as_abstention_evidence() -> None:
    artifact = replace(
        _artifact(),
        prototype_primitive_indices=(-1, -1, 0),
        prototype_composite_advantages=(-0.10, -0.05, 0.20),
        prototype_component_minimums=(-0.20, -0.10, 0.10),
    )
    policy = G1RecoveryStatePolicy(artifact)

    selections = [policy.select(_observation()) for _ in range(5)]
    receipt = policy.build_receipt()

    assert selections[-1].ready
    assert selections[-1].out_of_distribution
    assert selections[-1].primitive_index is None
    assert selections[-1].fallback_reason == "nearest_exemplar_abstains"
    assert receipt.fallback_count == 1


def test_recovery_state_policy_requires_neighbors_to_agree_on_same_primitive() -> None:
    second = replace(_primitive(), settling_standing_pose_blend=0.38)
    artifact = replace(
        _artifact(),
        primitives=(_primitive(), second),
        prototype_primitive_indices=(0, 1, -1),
    )
    policy = G1RecoveryStatePolicy(artifact)

    selections = [policy.select(_observation()) for _ in range(5)]

    assert selections[-1].ready and selections[-1].out_of_distribution
    assert selections[-1].primitive_index is None
    assert selections[-1].primitive_consensus == pytest.approx(0.5)
    assert selections[-1].fallback_reason == "primitive_consensus_below_gate"


def test_recovery_state_v3_negative_evidence_vetoes_a_nearby_positive_island() -> None:
    width = 2 * len(G1_RECOVERY_STATE_FEATURES)
    artifact = replace(
        _artifact(),
        schema_version="rosclaw.g1_goalforge.recovery_state_artifact.v3",
        negative_veto_distance_ratio=2.0,
        descriptor_prototypes=(
            (0.05,) * width,
            (0.05,) * width,
            (0.09,) * width,
        ),
    )
    policy = G1RecoveryStatePolicy(artifact)

    selections = [policy.select(_observation()) for _ in range(5)]
    receipt = policy.build_receipt()

    assert selections[-1].ready and selections[-1].out_of_distribution
    assert selections[-1].primitive_index is None
    assert selections[-1].fallback_reason == "negative_evidence_veto"
    assert selections[-1].nearest_distance == pytest.approx(0.05)
    assert selections[-1].nearest_negative_distance == pytest.approx(0.09)
    assert receipt.nearest_negative_distance == pytest.approx(0.09)


def test_recovery_state_v3_allows_a_positive_island_with_negative_margin() -> None:
    width = 2 * len(G1_RECOVERY_STATE_FEATURES)
    artifact = replace(
        _artifact(),
        schema_version="rosclaw.g1_goalforge.recovery_state_artifact.v3",
        negative_veto_distance_ratio=2.0,
        descriptor_prototypes=(
            (0.05,) * width,
            (0.05,) * width,
            (0.12,) * width,
        ),
    )
    policy = G1RecoveryStatePolicy(artifact)

    selections = [policy.select(_observation()) for _ in range(5)]

    assert selections[-1].ready and not selections[-1].out_of_distribution
    assert selections[-1].primitive_index == 0
    assert selections[-1].nearest_negative_distance == pytest.approx(0.12)


def test_recovery_state_policy_fails_closed_on_nonfinite_observation() -> None:
    policy = G1RecoveryStatePolicy(_artifact())
    observation = _observation()
    observation["pelvis_velocity_x_m_s"] = np.nan

    selection = policy.select(observation)

    assert selection.ready and selection.out_of_distribution
    assert selection.fallback_reason == "nonfinite_observation"


def test_recovery_state_policy_fails_closed_on_non_numeric_observation() -> None:
    policy = G1RecoveryStatePolicy(_artifact())
    observation = _observation()
    observation["pelvis_velocity_x_m_s"] = "not-a-number"  # type: ignore[assignment]

    selection = policy.select(observation)

    assert selection.ready and selection.out_of_distribution
    assert selection.fallback_reason == "invalid_observation"


def test_recovery_state_artifact_roundtrips_as_content_addressed_json(tmp_path) -> None:
    artifact = _artifact()
    path = tmp_path / "recovery-state.json"
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")

    loaded = load_g1_recovery_state_artifact(path)

    assert loaded == artifact
    assert loaded.artifact_hash == artifact.artifact_hash


def test_recovery_state_v3_artifact_roundtrips_with_negative_veto(tmp_path) -> None:
    artifact = replace(
        _artifact(),
        schema_version="rosclaw.g1_goalforge.recovery_state_artifact.v3",
        negative_veto_distance_ratio=2.0,
    )
    path = tmp_path / "recovery-state-v3.json"
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")

    loaded = load_g1_recovery_state_artifact(path)

    assert loaded == artifact
    assert loaded.artifact_hash == artifact.artifact_hash
    assert loaded.to_dict()["negative_veto_distance_ratio"] == 2.0


def test_recovery_state_artifact_rejects_invalid_routes_and_hardware_ceiling() -> None:
    with pytest.raises(ValueError, match="route"):
        replace(_artifact(), prototype_primitive_indices=(2, 0, -1))
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(_artifact(), activation_ceiling="REAL")
    with pytest.raises(ValueError, match="negative veto"):
        replace(
            _artifact(),
            schema_version="rosclaw.g1_goalforge.recovery_state_artifact.v3",
            negative_veto_distance_ratio=0.5,
        )


def test_controller_collects_temporal_state_before_latching_primitive() -> None:
    controller = _controller()

    effects = [
        controller.adapt_target(
            target=np.ones(29),
            policy_frame=300 + frame,
            timestamp_sec=6.0 + frame * 0.02,
            ball_contact_detected=True,
            left_support=True,
            right_support=True,
            muscle_memory_observation=_observation(),
        )
        for frame in range(5)
    ]
    receipt = controller.build_receipt(strict_replay=True)

    assert all(effect.recovery_state_pending for effect in effects[:4])
    assert not effects[-1].recovery_state_pending
    assert effects[-1].recovery_state_active
    assert effects[-1].recovery_state_primitive_index == 0
    assert receipt.recovery_state_receipt is not None
    assert receipt.recovery_state_receipt["pending_count"] == 4
    assert receipt.recovery_state_receipt["selection_count"] == 1
    assert receipt.recovery_state_receipt["activation_ceiling"] == "SIM_ONLY"


def test_controller_can_select_before_motion_gate_without_early_actuation() -> None:
    controller = _controller()

    effects = [
        controller.adapt_target(
            target=np.ones(29),
            policy_frame=250 + frame,
            timestamp_sec=5.0 + frame * 0.02,
            ball_contact_detected=True,
            left_support=True,
            right_support=True,
            muscle_memory_observation=_observation(),
        )
        for frame in range(5)
    ]
    after_gate = controller.adapt_target(
        target=np.ones(29),
        policy_frame=350,
        timestamp_sec=7.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
        muscle_memory_observation=_observation(),
    )

    assert all(not effect.active for effect in effects)
    assert effects[-1].recovery_state_primitive_index == 0
    assert not effects[-1].recovery_state_active
    assert after_gate.active
    assert after_gate.recovery_state_primitive_index == 0


def test_controller_temporal_state_failure_latches_retained_parent() -> None:
    controller = _controller()
    fallback_effects = []
    for frame in range(5):
        observation = _observation()
        if frame == 0:
            observation["pelvis_velocity_x_m_s"] = float("nan")
        fallback_effects.append(
            controller.adapt_target(
                target=np.ones(29),
                policy_frame=520 + frame,
                timestamp_sec=10.4 + frame * 0.02,
                ball_contact_detected=True,
                left_support=True,
                right_support=True,
                muscle_memory_observation=observation,
            )
        )
    receipt = controller.build_receipt(strict_replay=True)

    assert all(effect.recovery_state_out_of_distribution for effect in fallback_effects)
    assert all(effect.recovery_state_primitive_index is None for effect in fallback_effects)
    assert receipt.recovery_state_receipt is not None
    assert receipt.recovery_state_receipt["fallback_reason"] == "nonfinite_observation"
    assert receipt.recovery_state_receipt["fallback_count"] == 1
