"""Unit tests for observation/action adapters and policy manifest."""

from __future__ import annotations

import pytest

from rosclaw.integrations.lerobot.action_adapter import adapt_action_to_proposal
from rosclaw.integrations.lerobot.contracts import ObservationContract, ObservationFeature
from rosclaw.integrations.lerobot.observation_adapter import adapt_observation_for_worker
from rosclaw.integrations.lerobot.policy_manifest import load_policy_manifest
from rosclaw.integrations.lerobot.worker_schema import WorkerAction


def test_observation_adapter_state_task() -> None:
    out = adapt_observation_for_worker(
        {
            "observation": {
                "task": "pick cube",
                "state": [0.1, 0.2, 0.3],
                "state_names": ["joint_a", "joint_b", "joint_c"],
            }
        }
    )
    assert out["task"] == "pick cube"
    assert out["observation.state"] == [0.1, 0.2, 0.3]


def test_observation_adapter_flat_keys(tmp_path) -> None:
    img = tmp_path / "front.jpg"
    img.write_bytes(b"")
    out = adapt_observation_for_worker(
        {
            "task": "place cube",
            "observation.state": [1.0, 2.0],
            "observation.images.front": str(img),
        }
    )
    assert out["task"] == "place cube"
    assert out["observation.state"] == [1.0, 2.0]
    assert out["observation.images.front"] == str(img)


def test_observation_adapter_resolves_relative_images_from_base_dir(tmp_path) -> None:
    img = tmp_path / "front.jpg"
    img.write_bytes(b"")
    out = adapt_observation_for_worker(
        {
            "_base_dir": str(tmp_path),
            "observation": {
                "state": [1.0, 2.0],
                "images": {"front": "front.jpg"},
            },
        }
    )
    assert out["observation.images.front"] == str(img)


def test_observation_adapter_missing_image_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        adapt_observation_for_worker(
            {"observation.images.front": "/does/not/exist.jpg"}
        )


def test_observation_adapter_dict_state_requires_contract() -> None:
    with pytest.raises(ValueError, match="ObservationContract"):
        adapt_observation_for_worker(
            {
                "observation": {
                    "state": {"joint_a": 0.1, "joint_b": 0.2},
                }
            }
        )


def test_observation_adapter_dict_state_with_contract() -> None:
    contract = ObservationContract(
        id="test",
        policy={},
        body={},
        features={
            "observation.state": ObservationFeature(
                source={"names": ["joint_b", "joint_a"]}
            )
        },
    )
    out = adapt_observation_for_worker(
        {
            "observation": {
                "state": {"joint_a": 0.1, "joint_b": 0.2},
            }
        },
        contract=contract,
    )
    # Order comes from contract, not dict insertion order.
    assert out["observation.state"] == [0.2, 0.1]


def test_observation_adapter_list_state_without_names_fails() -> None:
    with pytest.raises(ValueError, match="state_names"):
        adapt_observation_for_worker(
            {
                "observation": {
                    "state": [0.1, 0.2],
                }
            }
        )


def test_action_adapter_from_worker_action() -> None:
    action = WorkerAction(values=[0.1, 0.2, 0.3], shape=[3], dtype="float32")
    proposal = adapt_action_to_proposal(
        action,
        policy_path="local/test",
        policy_metadata={"extra": {"action_unit": "radian"}},
        manifest_action_space=["a", "b", "c"],
    )
    assert proposal["action"]["values"] == [0.1, 0.2, 0.3]
    assert proposal["action"]["names"] == ["a", "b", "c"]
    assert proposal["action"]["units"] == "radian"
    assert proposal["safety"]["executable"] is False
    assert proposal["safety"]["requires_sandbox"] is True


def test_action_adapter_from_dict() -> None:
    proposal = adapt_action_to_proposal(
        {"type": "raw", "values": [0.5, 0.6], "shape": [2], "dtype": "float32"},
        policy_path="local/test",
        manifest_action_space=["x", "y"],
    )
    assert proposal["action"]["values"] == [0.5, 0.6]
    assert proposal["safety"]["executable"] is False


def test_action_adapter_none() -> None:
    proposal = adapt_action_to_proposal(None)
    assert proposal["action"]["values"] == []
    assert proposal["safety"]["executable"] is False
    assert proposal["safety"]["requires_sandbox"] is True


def test_action_adapter_blocks_nan() -> None:
    proposal = adapt_action_to_proposal(
        {"values": [0.0, float("nan")], "shape": [2], "dtype": "float32"},
        policy_path="local/test",
    )
    assert proposal["safety"].get("error_code") == "invalid_policy_action"


def test_load_policy_manifest(minimal_policy_dir) -> None:
    manifest = load_policy_manifest(minimal_policy_dir)
    assert manifest["policy_type"] == "act"
    assert manifest["config_found"] is True
    assert "input_features" in manifest
    assert "output_features" in manifest
    assert "raw_config_keys" in manifest


def test_load_policy_manifest_missing_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_policy_manifest(tmp_path / "missing")
