"""Synthetic scene correspondence contracts; no services or SIM are invoked."""

import importlib
from pathlib import Path

import pytest

from tests.connectors.ros.test_backend_probe_evidence import specification


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("probe_lift_evidence")


def test_ack_is_derived_from_original_exact_instrument_request_and_response(module):
    policy = specification()
    raw = module.lift_request(policy)
    ack = module.derive_lift_ack(
        policy,
        request_bytes=raw,
        response_bytes=b"data: true\n",
        returncode=0,
        acknowledged_at_unix_ns=100,
    )
    assert ack["model_name"] == policy["native_policy"]["contact_policy"]["model_name"]
    assert ack["target_xyz"] == [6, 0, 10] and ack["service_success"] is True
    spec = module.lift_service_spec(
        policy, world_name="declared_world", robot_model_name="actual_robot", wall_timeout_sec=1
    )
    assert spec["argv"][3] == "/world/declared_world/set_pose"
    assert spec["request_sha256"] == ack["request_sha256"]
    assert spec["authorization"] is False and spec["source_ownership_admitted"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_bytes", b'name: "actual_robot"'),
        ("response_bytes", b"data: false\n"),
        ("response_bytes", b"data: true\nerror: ignored"),
        ("response_bytes", b"data: true\ndata: true"),
        ("response_bytes", b""),
        ("response_bytes", b" " * 4097),
        ("returncode", True),
        ("returncode", 1),
        ("acknowledged_at_unix_ns", True),
        ("acknowledged_at_unix_ns", 0),
    ],
)
def test_wrong_target_failed_or_ambiguous_original_reply_cannot_make_ack(module, field, value):
    policy = specification()
    args = {
        "request_bytes": module.lift_request(policy),
        "response_bytes": b"data: true",
        "returncode": 0,
        "acknowledged_at_unix_ns": 100,
    }
    args[field] = value
    with pytest.raises(ValueError):
        module.derive_lift_ack(policy, **args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_name", 'world" /cmd_vel'),
        ("robot_model_name", ""),
        ("wall_timeout_sec", True),
        ("wall_timeout_sec", 0),
        ("wall_timeout_sec", 3),
        ("wall_timeout_sec", float("nan")),
    ],
)
def test_unbounded_or_ambiguous_scene_specification_is_refused(module, field, value):
    args = {
        "world_name": "declared_world",
        "robot_model_name": "actual_robot",
        "wall_timeout_sec": 1,
    }
    args[field] = value
    with pytest.raises(ValueError):
        module.lift_service_spec(specification(), **args)


def test_instrument_cannot_alias_active_robot(module):
    policy = specification()
    with pytest.raises(ValueError, match="active robot"):
        module.lift_service_spec(
            policy,
            world_name="declared_world",
            robot_model_name=policy["native_policy"]["contact_policy"]["model_name"],
            wall_timeout_sec=1,
        )
