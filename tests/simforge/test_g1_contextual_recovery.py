from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw.simforge.g1_cerebellar_recovery import (
    G1CerebellarRecoveryConfig,
    G1CerebellarRecoveryController,
)
from rosclaw.simforge.g1_contextual_recovery import (
    G1_CONTEXTUAL_RECOVERY_FEATURES,
    G1ContextualRecoveryArtifact,
    G1ContextualRecoveryPolicy,
    G1ContextualRecoveryPrimitive,
    load_g1_contextual_recovery_artifact,
)
from rosclaw.simforge.g1_contextual_recovery_training import (
    _bootstrap_lower_95,
    _contextual_route_prototype,
    _primitive_library,
)
from rosclaw.simforge.g1_muscle_memory import G1_MUSCLE_MEMORY_OBSERVATIONS

_BODY_HASH = "sha256:" + "1" * 64
_MOTION_HASH = "sha256:" + "2" * 64
_REGIME_HASH = "sha256:" + "3" * 64
_DATASET_HASH = "sha256:" + "4" * 64


def _observation(**changes: float) -> dict[str, float]:
    value = dict.fromkeys(G1_MUSCLE_MEMORY_OBSERVATIONS, 0.0)
    value.update(changes)
    return value


def _configs() -> tuple[G1CerebellarRecoveryConfig, G1CerebellarRecoveryConfig]:
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
    return baseline, G1CerebellarRecoveryConfig()


def test_contextual_prototype_matches_latched_runtime_selection_frame() -> None:
    count = len(G1_MUSCLE_MEMORY_OBSERVATIONS)
    rows = np.zeros((3, count), dtype=np.float64)
    impulse = G1_MUSCLE_MEMORY_OBSERVATIONS.index("contact_impulse_ns")
    right_support = G1_MUSCLE_MEMORY_OBSERVATIONS.index("right_support")
    rows[:, impulse] = 1.0
    rows[0, right_support] = 1.0
    rows[1] = np.arange(count, dtype=np.float64) + 1.0
    rows[1, impulse] = 1.0
    rows[1, right_support] = 0.0
    rows[2] = 99.0
    episode = SimpleNamespace(
        trajectory={
            "recovery_proprioception": rows,
            "recovery_active": np.asarray((False, True, True)),
        }
    )

    prototype = _contextual_route_prototype(
        episode,
        observation_mean=(0.0,) * count,
        observation_scale=(1.0,) * count,
    )

    assert prototype == pytest.approx(rows[1])


def _controller_without_artifact(
    config: G1CerebellarRecoveryConfig,
) -> G1CerebellarRecoveryController:
    return G1CerebellarRecoveryController(
        body_hash=_BODY_HASH,
        motion_hash=_MOTION_HASH,
        regime_commitment=_REGIME_HASH,
        regime_eligible=True,
        regime_reasons=(),
        standing_pose=np.zeros(29),
        config=config,
    )


def _artifact() -> G1ContextualRecoveryArtifact:
    baseline, fallback = _configs()
    baseline_hash = _controller_without_artifact(baseline).config_hash
    fallback_hash = _controller_without_artifact(fallback).config_hash
    feature_count = len(G1_CONTEXTUAL_RECOVERY_FEATURES)
    return G1ContextualRecoveryArtifact(
        body_hash=_BODY_HASH,
        motion_hash=_MOTION_HASH,
        baseline_recovery_config_hash=baseline_hash,
        fallback_recovery_config_hash=fallback_hash,
        training_dataset_hash=_DATASET_HASH,
        observation_mean=(0.0,) * len(G1_MUSCLE_MEMORY_OBSERVATIONS),
        observation_scale=(1.0,) * len(G1_MUSCLE_MEMORY_OBSERVATIONS),
        regime_feature_names=G1_CONTEXTUAL_RECOVERY_FEATURES,
        regime_prototypes=(
            (0.0,) * feature_count,
            (0.5,) * feature_count,
        ),
        primitives=(
            G1ContextualRecoveryPrimitive(
                start_policy_frame=300,
                blend_frames=100,
                settling_start_policy_frame=400,
                settling_blend_frames=80,
                settling_standing_pose_blend=0.38,
                settling_waist_pitch_bias_rad=0.08,
                target_smoothing_alpha=0.56,
            ),
            G1ContextualRecoveryPrimitive(
                start_policy_frame=300,
                blend_frames=80,
                settling_start_policy_frame=380,
                settling_blend_frames=60,
                settling_standing_pose_blend=0.42,
                settling_waist_pitch_bias_rad=0.12,
                target_smoothing_alpha=0.54,
            ),
        ),
        maximum_regime_distance=0.30,
        maximum_feature_z=8.0,
        training_episode_count=12,
        training_seed=20260802,
    )


def _controller(artifact: G1ContextualRecoveryArtifact) -> G1CerebellarRecoveryController:
    baseline, fallback = _configs()
    return G1CerebellarRecoveryController(
        body_hash=_BODY_HASH,
        motion_hash=_MOTION_HASH,
        regime_commitment=_REGIME_HASH,
        regime_eligible=True,
        regime_reasons=(),
        standing_pose=np.zeros(29),
        config=baseline,
        contextual_recovery_artifact=artifact,
        fallback_config=fallback,
    )


def test_contextual_artifact_roundtrips_as_content_addressed_safe_json(tmp_path) -> None:
    artifact = _artifact()
    path = tmp_path / "contextual-recovery.json"
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")

    loaded = load_g1_contextual_recovery_artifact(path)

    assert loaded == artifact
    assert loaded.artifact_hash == artifact.artifact_hash
    assert loaded.activation_ceiling == "SIM_ONLY"


def test_contextual_artifact_rejects_non_object_primitive() -> None:
    payload = _artifact().to_dict()
    payload["primitives"][0] = "not-an-object"

    with pytest.raises(ValueError, match="primitives must contain objects"):
        G1ContextualRecoveryArtifact.from_dict(payload)


def test_contextual_contract_rejects_schema_and_boolean_threshold_spoofing() -> None:
    with pytest.raises(ValueError, match="unsupported contextual recovery primitive schema"):
        replace(
            _artifact().primitives[0],
            schema_version="rosclaw.g1_goalforge.contextual_recovery_primitive.v999",
        )
    with pytest.raises(ValueError, match="thresholds must be finite numbers"):
        replace(_artifact(), maximum_regime_distance=True)


def test_contextual_policy_latches_nearest_prototype_once() -> None:
    policy = G1ContextualRecoveryPolicy(_artifact())

    first = policy.select(_observation())
    second = policy.select(
        _observation(
            pelvis_velocity_x_m_s=0.5,
            pelvis_velocity_y_m_s=0.5,
            pelvis_velocity_z_m_s=0.5,
            torso_pitch_rad=0.5,
            torso_angular_velocity_y_rad_s=0.5,
            left_ground_force_scale=0.5,
            right_ground_force_scale=0.5,
            contact_impulse_ns=0.5,
        )
    )

    assert first.primitive_index == 0
    assert second == first
    assert policy.build_receipt().selection_count == 1


def test_contextual_policy_routes_ood_state_to_retained_parent() -> None:
    policy = G1ContextualRecoveryPolicy(_artifact())

    selection = policy.select(_observation(torso_roll_rad=20.0))
    receipt = policy.build_receipt()

    assert selection.out_of_distribution
    assert selection.primitive_index is None
    assert receipt.fallback_count == 1
    assert not receipt.hardware_command_sent


@pytest.mark.parametrize(
    "observation",
    [
        {},
        _observation(torso_roll_rad=float("nan")),
        _observation(torso_roll_rad=float("inf")),
    ],
)
def test_contextual_policy_fails_closed_on_missing_or_nonfinite_observation(
    observation: dict[str, float],
) -> None:
    policy = G1ContextualRecoveryPolicy(_artifact())

    selection = policy.select(observation)
    receipt = policy.build_receipt()

    assert selection.primitive_index is None
    assert selection.out_of_distribution
    assert np.isfinite(selection.nearest_distance)
    assert receipt.selection_count == 1
    assert receipt.fallback_count == 1


def test_controller_applies_selected_primitive_after_contact_and_landing() -> None:
    artifact = _artifact()
    controller = _controller(artifact)

    effect = controller.adapt_target(
        target=np.ones(29),
        policy_frame=350,
        timestamp_sec=7.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
        muscle_memory_observation=_observation(),
    )
    receipt = controller.build_receipt(strict_replay=True)

    assert effect.active
    assert effect.contextual_recovery_active
    assert not effect.contextual_recovery_out_of_distribution
    assert effect.contextual_recovery_primitive_index == 0
    assert effect.blend_fraction == pytest.approx(0.5)
    assert receipt.contextual_recovery_receipt is not None
    assert receipt.contextual_recovery_receipt["selected_primitive_index"] == 0
    assert receipt.contextual_recovery_receipt["activation_ceiling"] == "SIM_ONLY"


def test_controller_waits_for_causal_recovery_frame_before_context_selection() -> None:
    controller = _controller(_artifact())

    before = controller.adapt_target(
        target=np.ones(29),
        policy_frame=250,
        timestamp_sec=5.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
        muscle_memory_observation=_observation(),
    )
    before_receipt = controller.build_receipt(strict_replay=True)
    after = controller.adapt_target(
        target=np.ones(29),
        policy_frame=350,
        timestamp_sec=7.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
        muscle_memory_observation=_observation(),
    )
    after_receipt = controller.build_receipt(strict_replay=True)

    assert not before.active
    assert before.contextual_recovery_primitive_index is None
    assert before_receipt.contextual_recovery_receipt is not None
    assert before_receipt.contextual_recovery_receipt["selection_count"] == 0
    assert after.contextual_recovery_primitive_index == 0
    assert after_receipt.contextual_recovery_receipt is not None
    assert after_receipt.contextual_recovery_receipt["selection_count"] == 1


def test_controller_ood_fallback_is_bound_and_deterministic() -> None:
    controller = _controller(_artifact())

    first = controller.adapt_target(
        target=np.ones(29),
        policy_frame=520,
        timestamp_sec=10.4,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
        muscle_memory_observation=_observation(torso_roll_rad=20.0),
    )
    second = controller.adapt_target(
        target=np.ones(29),
        policy_frame=521,
        timestamp_sec=10.42,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
        muscle_memory_observation=_observation(),
    )
    receipt = controller.build_receipt(strict_replay=True)

    assert first.contextual_recovery_out_of_distribution
    assert second.contextual_recovery_out_of_distribution
    assert first.contextual_recovery_primitive_index is None
    assert second.contextual_recovery_primitive_index is None
    assert receipt.contextual_recovery_receipt is not None
    assert receipt.contextual_recovery_receipt["fallback_count"] == 1


def test_contextual_artifact_compatibility_is_fail_closed() -> None:
    baseline, fallback = _configs()
    artifact = replace(
        _artifact(),
        fallback_recovery_config_hash="sha256:" + "f" * 64,
    )

    with pytest.raises(ValueError, match="fallback config hash mismatch"):
        G1CerebellarRecoveryController(
            body_hash=_BODY_HASH,
            motion_hash=_MOTION_HASH,
            regime_commitment=_REGIME_HASH,
            regime_eligible=True,
            regime_reasons=(),
            standing_pose=np.zeros(29),
            config=baseline,
            contextual_recovery_artifact=artifact,
            fallback_config=fallback,
        )


def test_contextual_training_library_and_bootstrap_are_bounded_and_deterministic() -> None:
    library = _primitive_library()
    first = _bootstrap_lower_95((0.02, 0.07, 0.08), seed=17)
    second = _bootstrap_lower_95((0.02, 0.07, 0.08), seed=17)

    assert len(library) == len({primitive.primitive_hash for primitive in library})
    assert first == second
    assert first > 0.0
