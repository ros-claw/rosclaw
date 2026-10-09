"""Deterministic native-producer cache regressions; no Node or DDS connection.

Run with ROS-host Python after sourcing ROS, to exercise actual message types
and probe code. This is a regression test, not a live observation acceptance.
"""

import json
import sys
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2"))
from probe import GetParameters, GetState, ReadOnlyProbe  # noqa: E402


def main():
    now = time.monotonic()
    old = (datetime.now(UTC) - timedelta(seconds=6)).isoformat()
    fixture = SimpleNamespace(
        parameters={"/amcl": {"use_sim_time": True}},
        parameter_received={"/amcl": now - 6},
        parameter_captured_at={"/amcl": old},
        lifecycle_states={"/amcl": {"name": "/amcl", "state": "ACTIVE", "captured_at": old}},
        clock_readings=deque([(now - 6, 1.0), (now - 5, 2.0)]),
        clock_values=deque([1.0, 2.0]),
        get_topic_names_and_types=lambda: [],
        get_node_names_and_namespaces=lambda: [],
        get_service_names_and_types=lambda: [],
        get_fully_qualified_name=lambda: "/probe_test",
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)),
        get_parameter=lambda _: SimpleNamespace(value=True),
        edges={},
        localization_observations={},
        subscriptions_by_topic={},
        packages=[],
        errors=[],
        pending={},
        clients_by_name={},
    )
    snapshot = ReadOnlyProbe.snapshot(fixture)
    assert snapshot["observations"]["node_parameters"] == {}
    assert snapshot["observations"]["parameter_captured_at"]["/amcl"] == old
    assert snapshot["lifecycle"][0]["state"] == "UNKNOWN"
    assert snapshot["lifecycle"][0]["last_observed_state"] == "ACTIVE"
    assert snapshot["lifecycle"][0]["captured_at"] == old
    assert not snapshot["completeness"]["lifecycle"]
    assert not snapshot["completeness"]["time"]
    assert snapshot["observations"]["clock_advancing"] is False

    # A delayed old callback must neither erase a newer RPC nor refresh cache.
    callbacks = []
    old_future = SimpleNamespace(add_done_callback=callbacks.append)
    client = SimpleNamespace(service_is_ready=lambda: True, call_async=lambda _: old_future)
    name = "/amcl/get_state"
    fixture.clients_by_name[name] = client
    ReadOnlyProbe.read_rpc(fixture, name, GetState, GetState.Request())
    newer_future = object()
    fixture.pending[name] = (newer_future, now)
    callbacks[0](old_future)
    assert fixture.pending[name][0] is newer_future
    assert fixture.lifecycle_states["/amcl"]["captured_at"] == old

    # Fresh reads restore knowledge while preserving original capture times.
    fixture.parameter_received["/amcl"] = time.monotonic()
    fresh = datetime.now(UTC).isoformat()
    fixture.lifecycle_states["/amcl"]["captured_at"] = fresh
    with patch("probe.get_action_server_names_and_types_by_node", return_value=[]):
        snapshot = ReadOnlyProbe.snapshot(fixture)
    assert snapshot["observations"]["node_use_sim_time"] == {"/amcl": True}
    assert snapshot["lifecycle"][0]["state"] == "ACTIVE"
    assert snapshot["lifecycle"][0]["captured_at"] == fresh
    # Real ROS GoalStatusArray decoding; status is a transition observation,
    # never an independent physical-stop measurement.
    from action_msgs.msg import GoalStatus, GoalStatusArray

    fixture.samples = {}
    fixture.action_observations = {}
    status = GoalStatus()
    status.goal_info.goal_id.uuid = list(range(16))
    status.status = 6
    message = GoalStatusArray(status_list=[status])
    ReadOnlyProbe.observe(fixture, "/robot/navigate_to_pose/_action/status", message)
    recorded = fixture.action_observations["/robot/navigate_to_pose"]
    assert recorded["goals"] == [{"goal_uuid": list(range(16)), "status": 6}]
    assert recorded["source"] == "/robot/navigate_to_pose/_action/status"
    snapshot = ReadOnlyProbe.snapshot(fixture)
    assert snapshot["observations"]["action_statuses"] == fixture.action_observations
    json.dumps(snapshot, allow_nan=False)  # ROS UUID elements may be numpy.uint8.
    # Actual ROS GetParameters decoding, including plugin lists and integers.
    parameter_callbacks = []
    values = [
        SimpleNamespace(type=9, string_array_value=["map", "lidar", "padding"]),
        SimpleNamespace(type=2, integer_value=1),
        SimpleNamespace(type=1, bool_value=False),
    ]
    parameter_future = SimpleNamespace(
        add_done_callback=parameter_callbacks.append,
        cancelled=lambda: False,
        exception=lambda: None,
        result=lambda: SimpleNamespace(values=values),
    )
    parameter_name = "/costmap/get_parameters"
    fixture.clients_by_name[parameter_name] = SimpleNamespace(
        service_is_ready=lambda: True,
        call_async=lambda _: parameter_future,
    )
    request = GetParameters.Request()
    request.names = ["plugins", "lidar.combination_method", "map.use_maximum"]
    ReadOnlyProbe.read_rpc(fixture, parameter_name, GetParameters, request)
    parameter_callbacks[0](parameter_future)
    assert fixture.parameters["/costmap"] == {
        "plugins": ["map", "lidar", "padding"],
        "lidar.combination_method": 1,
        "map.use_maximum": False,
    }
    print(
        json.dumps(
            {
                "status": "PASS",
                "evidence_class": "deterministic_regression",
                "dds_connections": 0,
                "cases": [
                    "expired_parameters",
                    "expired_lifecycle",
                    "historical_clock",
                    "late_rpc_identity",
                    "fresh_read_recovery",
                    "action_status_transition_observation",
                    "plugin_parameter_types",
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
