"""Unit tests for P4 observation/action contracts."""

from __future__ import annotations

import pytest

from rosclaw.integrations.lerobot.contracts import (
    ActionProposalV2,
    ObservationContract,
    ObservationFeature,
    ObservationFeatureSnapshot,
    ObservationSnapshot,
    build_action_proposal_v2,
    infer_action_names,
    infer_action_representation,
    infer_action_shape,
    infer_action_units,
    validate_action_values,
    validate_observation_snapshot,
)


def test_observation_contract_round_trip() -> None:
    contract = ObservationContract(
        id="test_contract",
        policy={"name": "act"},
        body={"robot": "aloha"},
        features={
            "observation.state": ObservationFeature(
                required=True,
                source={"names": ["joint_0", "joint_1"]},
                shape=[2],
                unit="radian",
            ),
            "observation.images.front": ObservationFeature(
                required=False,
                shape=[3, 480, 640],
                transport="artifact_ref",
            ),
        },
    )
    data = contract.to_dict()
    restored = ObservationContract.from_dict(data)
    assert restored.id == contract.id
    assert restored.get_state_names() == ["joint_0", "joint_1"]
    assert restored.features["observation.images.front"].required is False


def test_validate_missing_required_feature_blocks() -> None:
    contract = ObservationContract(
        id="test",
        policy={},
        body={},
        features={
            "observation.state": ObservationFeature(required=True, shape=[2])
        },
    )
    snapshot = ObservationSnapshot(snapshot_id="s1")
    result = validate_observation_snapshot(contract, snapshot)
    assert result.status == "blocked"
    assert result.errors[0]["feature"] == "observation.state"


def test_validate_stale_required_feature_blocks() -> None:
    contract = ObservationContract(
        id="test",
        policy={},
        body={},
        features={
            "observation.state": ObservationFeature(required=True, max_age_ms=100)
        },
    )
    snapshot = ObservationSnapshot(
        snapshot_id="s1",
        captured_at_monotonic_ns=0,
        features={
            "observation.state": ObservationFeatureSnapshot(
                valid=True, values=[0.0, 0.0], captured_at_monotonic_ns=0
            )
        },
    )
    result = validate_observation_snapshot(contract, snapshot, now_monotonic_ns=int(0.2 * 1e9))
    assert result.status == "blocked"
    assert result.errors[0]["code"] == "observation_stale"


def test_validate_optional_missing_is_not_blocked() -> None:
    contract = ObservationContract(
        id="test",
        policy={},
        body={},
        features={
            "observation.images.front": ObservationFeature(required=False)
        },
    )
    snapshot = ObservationSnapshot(snapshot_id="s1")
    result = validate_observation_snapshot(contract, snapshot)
    assert result.status == "ok"
    assert "observation.images.front" in result.missing_optional_features


def test_validate_optional_invalid_is_not_blocked() -> None:
    contract = ObservationContract(
        id="test",
        policy={},
        body={},
        features={
            "observation.images.front": ObservationFeature(required=False)
        },
    )
    snapshot = ObservationSnapshot(
        snapshot_id="s1",
        features={
            "observation.images.front": ObservationFeatureSnapshot(valid=False)
        },
    )
    result = validate_observation_snapshot(contract, snapshot)
    assert result.status == "ok"
    assert "observation.images.front" in result.missing_optional_features


def test_observation_snapshot_round_trip() -> None:
    snapshot = ObservationSnapshot(
        snapshot_id="s1",
        session_id="session_1",
        body_id="body_1",
        captured_at_monotonic_ns=12345,
        task="pick cube",
        features={
            "observation.state": ObservationFeatureSnapshot(
                valid=True,
                values=[0.1, 0.2],
                names=["joint_0", "joint_1"],
                captured_at_monotonic_ns=12345,
            )
        },
    )
    restored = ObservationSnapshot.from_dict(snapshot.to_dict())
    assert restored.snapshot_id == "s1"
    assert restored.features["observation.state"].values == [0.1, 0.2]


def test_action_proposal_v2_chunk_detection() -> None:
    proposal = ActionProposalV2(
        proposal_id="p1",
        session_id="s1",
        step_index=0,
        policy_path="local/test",
        policy_hash=None,
        runtime_id=None,
        processor_hash=None,
        representation="joint_position",
        reference_frame="base",
        values=[[0.0] * 7 for _ in range(50)],
        shape=[50, 7],
        dtype="float32",
        names=[f"joint_{i}" for i in range(7)],
        units="radian",
        semantic_source="explicit_policy_contract",
        authoritative=True,
        chunk={"is_chunk": True, "length": 50},
        safety={"executable": False},
    )
    data = proposal.to_dict()
    assert data["schema_version"] == "rosclaw.action_proposal.v2"
    assert data["action"]["shape"] == [50, 7]
    assert data["chunk"]["is_chunk"] is True
    assert data["chunk"]["length"] == 50
    assert data["safety"]["executable"] is False


def test_build_action_proposal_v2_infers_names_units() -> None:
    processed = {"type": "raw", "values": [0.1, 0.2, 0.3], "shape": [3], "dtype": "float32"}
    metadata = {
        "output_features": {
            "action": {
                "representation": "joint_position",
                "unit": "radian",
                "names": ["j0", "j1", "j2"],
            }
        }
    }
    proposal = build_action_proposal_v2(
        proposal_id="p1",
        session_id="s1",
        step_index=0,
        policy_path="local/test",
        policy_metadata=metadata,
        processed_action=processed,
    )
    assert proposal.representation == "joint_position"
    assert proposal.units == "radian"
    assert proposal.names == ["j0", "j1", "j2"]
    assert proposal.chunk.is_chunk is False
    assert proposal.chunk.length == 1


def test_build_action_proposal_v2_chunk_from_shape() -> None:
    processed = {"values": [[0.0] * 6 for _ in range(20)], "shape": [20, 6], "dtype": "float32"}
    metadata = {"output_features": {"action": {"shape": [20, 6]}}}
    proposal = build_action_proposal_v2(
        proposal_id="p1",
        session_id=None,
        step_index=5,
        policy_path="local/test",
        policy_metadata=metadata,
        processed_action=processed,
    )
    assert proposal.chunk.is_chunk is True
    assert proposal.chunk.length == 20
    assert proposal.shape == [20, 6]


def test_infer_action_names_priority() -> None:
    metadata = {
        "output_features": {"action": {"names": ["a", "b"]}},
        "extra": {"action_names": ["c", "d"]},
    }
    assert infer_action_names(metadata) == ["a", "b"]
    metadata2 = {"extra": {"action_names": ["c", "d"]}}
    assert infer_action_names(metadata2) == ["c", "d"]
    assert infer_action_names({}, manifest_action_space=["x", "y"]) == ["x", "y"]
    assert infer_action_names({}, action_dim=3) == []


def test_infer_action_representation_and_units() -> None:
    metadata = {
        "output_features": {"action": {"representation": "joint_delta", "unit": "degree"}}
    }
    assert infer_action_representation(metadata) == "joint_delta"
    assert infer_action_units(metadata) == "degree"
    assert infer_action_representation({"extra": {"action_representation": "cartesian_pose"}}) == "cartesian_pose"
    assert infer_action_units({"extra": {"action_unit": "meter"}}) == "meter"


def test_infer_action_shape() -> None:
    assert infer_action_shape({"output_features": {"action": {"shape": [100, 14]}}}) == [100, 14]
    assert infer_action_shape({}) is None


def test_validate_action_values_rejects_nan_inf() -> None:
    ok, error = validate_action_values([0.0, 1.0, 2.0])
    assert ok is True
    assert error is None

    ok, error = validate_action_values([0.0, float("nan")])
    assert ok is False
    assert "NaN/Inf" in (error or "")

    ok, error = validate_action_values([float("inf")])
    assert ok is False
    assert "NaN/Inf" in (error or "")


def test_validate_action_values_rejects_non_numeric() -> None:
    ok, error = validate_action_values([0.0, "oops"])
    assert ok is False
    assert "not numeric" in (error or "")


def test_build_action_proposal_v2_preserves_raw_model_output() -> None:
    processed = {"values": [0.5], "shape": [1], "dtype": "float32"}
    raw = {"values": [0.4], "shape": [1], "dtype": "float32"}
    proposal = build_action_proposal_v2(
        proposal_id="p1",
        session_id=None,
        step_index=0,
        policy_path="local/test",
        policy_metadata={},
        processed_action=processed,
        raw_action=raw,
    )
    assert proposal.raw_model_output == [0.4]


def test_action_proposal_from_dict_handles_missing_action_block() -> None:
    data = {
        "proposal_id": "p1",
        "policy": {"path": "local/test"},
        "representation": "joint_position",
        "safety": {"executable": False},
    }
    proposal = ActionProposalV2.from_dict(data)
    assert proposal.values == []
    assert proposal.shape == []
    assert proposal.names == []


def test_build_action_proposal_v2_squeezes_batch_one() -> None:
    processed = {
        "values": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
        "shape": [1, 7],
        "dtype": "float32",
    }
    metadata = {
        "output_features": {
            "action": {
                "representation": "joint_position",
                "unit": "radian",
                "names": ["j0", "j1", "j2", "j3", "j4", "j5", "j6"],
            }
        }
    }
    proposal = build_action_proposal_v2(
        proposal_id="p1",
        session_id=None,
        step_index=0,
        policy_path="local/test",
        policy_metadata=metadata,
        processed_action=processed,
    )
    assert proposal.shape == [7]
    assert proposal.values == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert proposal.chunk.is_chunk is False


def test_build_action_proposal_v2_blocks_batch_action_not_supported() -> None:
    processed = {
        "values": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "shape": [2, 3],
        "dtype": "float32",
    }
    metadata = {
        "output_features": {
            "action": {
                "representation": "joint_position",
                "unit": "radian",
                "names": ["j0", "j1", "j2"],
                "shape": [3],
            }
        }
    }
    proposal = build_action_proposal_v2(
        proposal_id="p1",
        session_id=None,
        step_index=0,
        policy_path="local/test",
        policy_metadata=metadata,
        processed_action=processed,
    )
    assert proposal.safety.get("error_code") == "batch_action_not_supported"
    assert proposal.safety.get("executable") is False


def test_build_action_proposal_v2_blocks_unknown_representation() -> None:
    processed = {"values": [0.1] * 14, "shape": [14], "dtype": "float32"}
    proposal = build_action_proposal_v2(
        proposal_id="p1",
        session_id=None,
        step_index=0,
        policy_path="local/test",
        policy_metadata={},
        processed_action=processed,
    )
    assert proposal.representation == "unknown"
    assert proposal.units == "unknown"
    assert proposal.names == []
    assert proposal.safety.get("error_code") == "unknown_action_semantics"
    assert proposal.safety.get("executable") is False


def test_build_action_proposal_v2_semantic_source_and_authoritative() -> None:
    explicit_metadata = {
        "output_features": {
            "action": {
                "representation": "joint_position",
                "unit": "radian",
                "names": ["j0"],
            }
        }
    }
    proposal = build_action_proposal_v2(
        proposal_id="p1",
        session_id=None,
        step_index=0,
        policy_path="local/test",
        policy_metadata=explicit_metadata,
        processed_action={"values": [0.0], "shape": [1], "dtype": "float32"},
    )
    assert proposal.semantic_source == "explicit_policy_contract"
    assert proposal.authoritative is True

    inferred_metadata = {"extra": {"action_representation": "joint_delta", "action_unit": "degree"}}
    proposal = build_action_proposal_v2(
        proposal_id="p2",
        session_id=None,
        step_index=0,
        policy_path="local/test",
        policy_metadata=inferred_metadata,
        processed_action={"values": [0.0], "shape": [1], "dtype": "float32"},
    )
    assert proposal.semantic_source == "inferred"
    assert proposal.authoritative is False

    proposal = build_action_proposal_v2(
        proposal_id="p3",
        session_id=None,
        step_index=0,
        policy_path="local/test",
        policy_metadata={},
        processed_action={"values": [0.0], "shape": [1], "dtype": "float32"},
    )
    assert proposal.semantic_source == "unknown"
    assert proposal.authoritative is False
