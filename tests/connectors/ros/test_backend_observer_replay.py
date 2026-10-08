"""Original-source join with explicit pose decoder doubles; no live ROS/physics."""

import importlib
import json
from pathlib import Path

import pytest

from tests.connectors.ros.test_backend_source_gate import Fixture, retained, sources


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    replay = importlib.import_module("backend_observer_replay")
    probe = importlib.import_module("closed_backend_probe")

    def decode(raw, *args):
        return json.loads(raw)["sim"], json.loads(raw)["pose"]

    monkeypatch.setattr(replay, "decode_ros_pose", decode)
    monkeypatch.setattr(probe, "decode_ros_pose", decode)
    return replay


def engine(module):
    _, robot, _, probe = sources()
    return module.BackendObserverReplay(
        robot,
        probe,
        robot_pose_frame="synthetic_robot_world",
        probe_pose_frame="synthetic_probe_world",
    )


def test_initial_pending_original_packet_is_retained_and_cannot_open(module):
    e = engine(module)
    packet, _, _, _ = sources()
    payload = {
        **retained(json.dumps(packet).encode(), "gazebo_ecm_contact_sensor_data_json"),
        "received_monotonic_sec": 100,
        "received_unix_ns": packet["captured_at_unix_ns"],
    }
    assert e.apply("backend_robot_components", payload) == {"startup_pose_pending": True}
    result = e.apply("backend_observation_sample", {"received_monotonic_sec": 100.01})
    assert result["actor_envelope"]["sequence"] == 0
    assert not result["actor_envelope"]["live_source_constraint_satisfied"]
    assert not result["snapshot"]["backend_health_admitted"]


def test_measured_cycle_projection_still_requires_daemon_lease_and_body_admission(module):
    e = engine(module)
    f = Fixture(importlib.import_module("backend_source_gate"))
    f.cycle()
    e.gate = f.gate
    result = e.apply("backend_observation_sample", {"received_monotonic_sec": 101.81})
    assert result["actor_envelope"]["live_source_constraint_satisfied"]
    assert not result["actor_envelope"]["authorization"]
    assert result["snapshot"]["actual_world_and_body_admission"] == "NOT_VERIFIED"
    assert result["snapshot"]["physical_acceptance"] == "NOT_VERIFIED"
    second = e.apply("backend_observation_sample", {"received_monotonic_sec": 101.82})
    assert second["actor_envelope"]["sequence"] == 1


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("unknown", {"received_monotonic_sec": 100}),
        ("backend_observation_sample", {"received_monotonic_sec": float("nan")}),
        ("backend_observation_sample", {"received_monotonic_sec": True}),
        ("backend_robot_pose", {"received_monotonic_sec": 100}),
        ("backend_probe_lift_ack", {"received_monotonic_sec": 100}),
    ],
)
def test_source_rejection_is_latched_and_negative_projection_cannot_reopen(module, kind, payload):
    e = engine(module)
    with pytest.raises(ValueError):
        e.apply(kind, payload)
    assert e.gate.fault
    with pytest.raises(ValueError, match="remains latched"):
        e.apply("backend_observation_sample", {"received_monotonic_sec": 101})
    result = e.failed_sample(101.01)
    assert not result["actor_envelope"]["live_source_constraint_satisfied"]
    assert result["actor_envelope"]["source_fault"]


def test_independent_original_pose_is_decoded_before_contact_admission(module):
    e = engine(module)
    raw = json.dumps({"sim": 1, "pose": [0, 0, 0, 1, 0, 0, 0]}).encode()
    result = e.apply(
        "backend_robot_pose",
        {**retained(raw, "tf2_msgs/msg/TFMessage"), "received_monotonic_sec": 100},
    )
    assert result == {"sim_time_sec": 1}
    assert e.gate.robot.world_pose
    with pytest.raises(ValueError, match="regressed"):
        e.apply(
            "backend_robot_pose",
            {**retained(raw, "tf2_msgs/msg/TFMessage"), "received_monotonic_sec": 99},
        )


def test_owned_policy_source_refuses_symlink_and_unowned_path(module, monkeypatch, tmp_path):
    observer = importlib.import_module("backend_observer")
    path = tmp_path / "policy.json"
    path.write_text('{"approved":true}')
    assert observer.owned_json(path, tmp_path)[1] == {"approved": True}
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="owned"):
        observer.owned_json(link, tmp_path)
    with pytest.raises(ValueError, match="owned"):
        observer.owned_json(path, tmp_path / "other")


def test_retained_original_component_byte_hash_is_not_inferred(module):
    observer = importlib.import_module("backend_observer")
    raw = b'{"original":"bytes"}'
    payload = observer.retained(raw, "gazebo_ecm_contact_sensor_data_json", 100, 123)
    source = importlib.import_module("closed_native_contact_evidence")
    assert source.original_ros_bytes(payload, "gazebo_ecm_contact_sensor_data_json") == raw
    with pytest.raises(ValueError):
        observer.retained(b"x" * 262145, "oversized", 100, 123)
