"""Required-wire negatives on synthetic original events, no service or World."""

import importlib
import json

import pytest

from tests.connectors.ros import test_probe_scene_geometry as spatial_contract
from tests.connectors.ros.test_backend_source_gate import Fixture, retained, sources


@pytest.fixture
def engine(monkeypatch):
    _, _, _, _, declaration, binding, _ = spatial_contract.fixture.__wrapped__(monkeypatch)
    _, robot_policy, _, probe_policy = sources()
    m = importlib.import_module("backend_observer_replay")
    result = m.BackendObserverReplay(
        robot_policy,
        probe_policy,
        robot_pose_frame="synthetic_world",
        probe_pose_frame="synthetic_world",
        scene_binding=binding,
        probe_declaration=declaration,
        instrument_service_binary_sha256="a" * 64,
    )
    f = Fixture(importlib.import_module("backend_source_gate"))
    f.gate = result.gate

    def synthetic_pose(raw, *_):
        source = json.loads(raw)
        return source["sim"], source["pose"]

    monkeypatch.setattr(
        importlib.import_module("closed_backend_probe"), "decode_ros_pose", synthetic_pose
    )
    for _ in range(12):
        f.frame()
    return result, f


@pytest.mark.parametrize(
    "fault",
    ["missing_wire", "wrong_binary", "source_only_reply", "wrong_request", "wrong_completion"],
)
def test_required_original_sdk_wire_cannot_be_bypassed_by_a_text_ack_or_source_fixture(
    engine, fault
):
    e, f = engine
    request = importlib.import_module("probe_lift_evidence").lift_request(f.spec)
    unix = f.origin + 603_000_000
    p = {
        "request": retained(request, "gz.msgs.Pose_protobuf_text"),
        "response": retained(b"data: true\n", "gz.msgs.Boolean_protobuf_text"),
        "returncode": 0,
        "received_monotonic_sec": 100.603,
        "received_unix_ns": unix,
        "acknowledged_at_unix_ns": unix,
        "service_binary_sha256": "b" * 64
        if fault == "wrong_binary"
        else None
        if fault == "missing_wire"
        else "a" * 64,
    }
    record = {
        "request_text_hex": request.hex(),
        "request_protobuf_hex": "deadbeef",
        "response_protobuf_hex": "1001",
        "response_text_hex": b"data: true\n".hex(),
        "transport_returned": True,
        "service_result": True,
        "response_data": True,
        "service_executed": fault != "source_only_reply",
        "authorization": False,
        "acknowledged_at_unix_ns": unix,
    }
    p["original_service_record"] = retained(
        json.dumps(record).encode(), "owned_gazebo_instrument_RPC_original_json"
    )
    if fault == "wrong_request":
        p["request"] = retained(b"wrong_robot", "gz.msgs.Pose_protobuf_text")
    if fault == "wrong_completion":
        p["acknowledged_at_unix_ns"] += 1
    with pytest.raises(ValueError):
        e.apply("backend_probe_lift_ack", p)
    assert e.gate.fault and e.gate.probe.tracker.phase == "READY_FOR_LIFT"
    assert e.gate.probe.tracker.cycles == 0


def test_original_wire_policy_cannot_be_installed_without_independent_spatial_sources(engine):
    e, _ = engine
    _, robot_policy, _, probe_policy = sources()
    with pytest.raises(ValueError, match="requires spatial"):
        importlib.import_module("backend_observer_replay").BackendObserverReplay(
            robot_policy,
            probe_policy,
            robot_pose_frame="synthetic_world",
            probe_pose_frame="synthetic_world",
            instrument_service_binary_sha256="a" * 64,
        )
    assert len(e.gate.policy_hash) == 64
