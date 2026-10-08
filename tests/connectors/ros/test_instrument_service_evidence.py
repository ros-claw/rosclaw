"""Explicit synthetic RPC records, never actual service or physical evidence."""

import importlib
import json
from pathlib import Path

import pytest

from tests.connectors.ros.test_backend_probe_evidence import specification


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("instrument_service_evidence")


def synthetic_record(module):
    policy = specification()
    return (
        policy,
        {
            "request_text_hex": importlib.import_module("probe_lift_evidence")
            .lift_request(policy)
            .hex(),
            "request_protobuf_hex": "deadbeef",  # Deliberate synthetic invalid Pose wire; not SDK-verified.
            "response_protobuf_hex": "1001",
            "response_text_hex": b"data: true\n".hex(),
            "transport_returned": True,
            "service_result": True,
            "response_data": True,
            "acknowledged_at_unix_ns": 1791504000000000000,
            "service_executed": True,
            "authorization": False,
        },
    )


def test_structural_parse_preserves_originals_without_claiming_source_authentication_or_sdk_validation(
    module,
):
    policy, source = synthetic_record(module)
    record, sources, ack = module.parse_original_service_record(json.dumps(source).encode(), policy)
    assert record == source and sources["request_protobuf_hex"] == b"\xde\xad\xbe\xef"
    assert (
        ack["service_success"] and ack["source"] == "owned_gazebo_world_set_pose_ack_not_pose_proof"
    )
    assert "original_source_sdk_projection_verified" not in ack


@pytest.mark.parametrize(
    "fault",
    [
        "transport_returned",
        "service_result",
        "response_data",
        "service_executed",
        "authorization",
        "wrong_target",
        "wire_false",
        "wire_extra",
        "wire_nonhex",
        "completion",
        "duplicate",
        "missing",
        "array",
    ],
)
def test_source_only_failure_mismatched_wire_or_missing_original_is_refused(module, fault):
    policy, source = synthetic_record(module)
    if fault in {"transport_returned", "service_result", "response_data", "service_executed"}:
        source[fault] = False
    elif fault == "authorization":
        source[fault] = True
    elif fault == "wrong_target":
        source["request_text_hex"] = b"wrong_robot".hex()
    elif fault == "wire_false":
        source["response_protobuf_hex"] = "1000"
    elif fault == "wire_extra":
        source["response_protobuf_hex"] = "10011801"
    elif fault == "wire_nonhex":
        source["request_protobuf_hex"] = "DEADBEEF"
    elif fault == "completion":
        source["acknowledged_at_unix_ns"] = True
    elif fault == "missing":
        source.pop("response_protobuf_hex")
    raw = json.dumps(source).encode()
    if fault == "duplicate":
        raw = raw[:-1] + b',"authorization":false}'
    if fault == "array":
        raw = b"[]"
    with pytest.raises(ValueError):
        module.parse_original_service_record(raw, policy)
