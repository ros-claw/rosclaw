import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from test_g1_neural_torque import _artifact_bytes, _digest, _frame

from rosclaw.control.parent_residual import compose_parent_residual, parent_context
from rosclaw.simforge.g1_neural_torque import (
    G1TorqueSafetyConfig,
    load_g1_neural_torque_artifact,
)
from rosclaw.simforge.g1_parent_residual_torque import (
    G1ParentResidualTorquePolicy,
    load_parent_residual_head,
    serialize_parent_residual_head,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS


@pytest.fixture
def encoder(tmp_path):
    path = tmp_path / "encoder.bin"
    path.write_bytes(
        _artifact_bytes(
            safety=G1TorqueSafetyConfig(
                torque_guard_scale=0.85, warmup_steps=0, maximum_projection_ratio=1.0
            )
        )
    )
    return load_g1_neural_torque_artifact(path)


def head(tmp_path, encoder, **kwargs):
    payload = serialize_parent_residual_head(
        encoder=encoder,
        parent_policy_hash=_digest("new-parent"),
        dataset_hash=_digest("data"),
        **kwargs,
    )
    path = tmp_path / "residual.json"
    path.write_bytes(payload)
    return load_parent_residual_head(
        path,
        encoder=encoder,
        expected_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        expected_body_hash=encoder.body_hash,
        expected_parent_policy_hash=_digest("new-parent"),
    )


def policy(artifact, encoder):
    return G1ParentResidualTorquePolicy(
        artifact,
        encoder=encoder,
        expected_body_hash=encoder.body_hash,
        expected_parent_policy_hash=_digest("new-parent"),
    )


def test_zero_is_exact_parent_even_above_candidate_power_or_slew(tmp_path, encoder):
    original = {k: v.copy() for k, v in encoder.tensors.items()}
    p = policy(head(tmp_path, encoder), encoder)
    parent = np.asarray(G1_HARD_TORQUE_LIMITS) * 0.8
    parent[0] = -0.0
    frame = _frame(joint_velocity=np.full(29, 5.0))
    for _ in range(4):
        command = p.command(frame, parent)
        assert command.tobytes() == parent.tobytes()
        assert p.pending_projection.reason == "zero_residual_parent"
        p.note_applied(command)
    ep = p.residual_episode()
    assert ep["observations"].shape == (4, 131)
    assert ep["actor_residual_valid"].all()
    assert not ep["actor_residuals"].any()
    assert p.residual_receipt()["hardware_authorized"] is False
    assert all(np.array_equal(v, encoder.tensors[k]) for k, v in original.items())


def test_current_parent_has_real_head_authority_and_exact_causal_columns(tmp_path, encoder):
    weights = np.zeros((29, encoder.hidden_dim + 29), np.float32)
    weights[0, encoder.hidden_dim] = 1.0
    p = policy(head(tmp_path, encoder, head_weight=weights), encoder)
    for value in (0.0, 0.123456789):
        parent = np.zeros(29)
        parent[0] = value
        command = p.command(_frame(), parent)
        if value:
            assert command[0] > parent[0]
            assert abs(command[0] - parent[0]) <= G1_HARD_TORQUE_LIMITS[0] * 0.01
        p.note_applied(command)
    ep = p.residual_episode()
    np.testing.assert_array_equal(
        ep["observations"][-1, -29:],
        (parent / np.asarray(G1_HARD_TORQUE_LIMITS)).astype(np.float32),
    )
    ep["observations"].fill(99)
    assert np.max(p.residual_episode()["observations"]) < 99


def test_fifo_mask_and_reset(tmp_path, encoder):
    encoder = replace(encoder, safety=replace(encoder.safety, warmup_steps=1))
    p = policy(head(tmp_path, encoder), encoder)
    parent = np.zeros(29)
    out = p.command(_frame(), parent)
    with pytest.raises(RuntimeError):
        p.command(_frame(), parent)
    p.note_applied(out)
    out = p.command(_frame(), parent)
    p.note_applied(out)
    assert p.residual_episode()["actor_residual_valid"].tolist() == [False, True]
    with pytest.raises(RuntimeError):
        p.note_applied(out)
    p.reset()
    with pytest.raises(ValueError):
        p.residual_episode()


@pytest.mark.parametrize(
    "key,value",
    [
        ("action_semantics", "DIRECT_JOINT_TORQUE"),
        ("activation_ceiling", "REAL"),
        ("residual_limit_ratio", 0.03),
        ("residual_limit_ratio", True),
        ("residual_limit_ratio", float("nan")),
        ("update_index", True),
        ("encoder_artifact_hash", "sha256:" + "0" * 64),
        ("head_bias", [0.0] * 28),
        ("head_bias", [float("inf")] * 29),
        ("observation_names", ["pretend102"]),
    ],
)
def test_strict_new_semantics_and_shapes(tmp_path, encoder, key, value):
    payload = json.loads(
        serialize_parent_residual_head(
            encoder=encoder, parent_policy_hash=_digest("new-parent"), dataset_hash=_digest("data")
        )
    )
    payload[key] = value
    raw = json.dumps(payload).encode()
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_parent_residual_head(
            path,
            encoder=encoder,
            expected_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
            expected_body_hash=encoder.body_hash,
            expected_parent_policy_hash=_digest("new-parent"),
        )


def test_old_loader_rejects_new_head_and_bindings_are_not_optional(tmp_path, encoder):
    artifact = head(tmp_path, encoder)
    with pytest.raises(ValueError):
        load_g1_neural_torque_artifact(tmp_path / "residual.json")
    with pytest.raises(ValueError):
        G1ParentResidualTorquePolicy(
            artifact,
            encoder=encoder,
            expected_body_hash=encoder.body_hash,
            expected_parent_policy_hash=encoder.parent_policy_hash,
        )


@pytest.mark.parametrize("parent", [np.full(29, np.nan), np.ones(28), np.full(29, 1e9)])
def test_invalid_parent_is_not_a_zero_safe_label(tmp_path, encoder, parent):
    p = policy(head(tmp_path, encoder), encoder)
    with pytest.raises(ValueError):
        p.command(_frame(), parent)


def test_generic_context_and_residual_are_owned_and_reject_over_budget():
    parent = np.array([-0.0, 2.0], np.float64)
    original = parent.copy()
    out = compose_parent_residual(parent, np.zeros(2), np.full(2, 10.0), maximum_ratio=0.01)
    assert out.tobytes() == parent.tobytes()
    out[1] = 8
    np.testing.assert_array_equal(parent, original)
    np.testing.assert_array_equal(
        parent_context(np.array([1.0]), parent, np.array([10.0, 10.0])),
        np.array([1.0, 0.0, 0.2], np.float32),
    )
    with pytest.raises(ValueError):
        compose_parent_residual(parent, np.full(2, 0.101), np.full(2, 10.0), maximum_ratio=0.01)
