"""Offline rosapi wire fixtures; no ROS client, provider, or socket calls.

Shapes follow pinned GetActionServers (both versions) and ROS2 ActionType.
ROS1 has actionlib's five-topic name discovery but no ActionType service.
"""

from __future__ import annotations

from typing import Any

import pytest

from rosclaw.connectors.ros.compiler import CapabilityManifestCompiler
from rosclaw.connectors.ros.discovery.graph import RosGraphDiscovery, RosGraphSnapshot
from rosclaw.connectors.ros.discovery.rosapi_resolver import RosApiProfile, RosVersion
from rosclaw.connectors.ros.transport.base import RosTransportResult


class FixtureTransport:
    """Strict scripted boundary: unexpected calls fail, never reach a provider."""

    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_service(self, service: str, args: dict[str, Any]):
        self.calls.append((service, args))
        assert self.responses, f"Unexpected service call: {service}"
        return self.responses.pop(0)


def response(**values: Any) -> RosTransportResult:
    return RosTransportResult(ok=True, data={"op": "service_response", "values": values})


def discovery(transport, version=RosVersion.ROS2, prefix="/rosapi"):
    return RosGraphDiscovery(
        transport,
        RosApiProfile(
            version=version,
            distro="fixture-only",
            service_prefix=prefix,
            type_prefix="rosapi_msgs/srv" if version == RosVersion.ROS2 else "rosapi",
        ),
    )


def snapshot(actions):
    return RosGraphSnapshot(
        ros_version="ros2",
        distro="fixture-only",
        endpoint="offline://action-metadata",
        topics=[],
        services=[],
        actions=actions,
        nodes=[],
        params=[],
        captured_at="fixture",
    )


@pytest.mark.parametrize("prefix", ["/rosapi", "/rosapi_node", "/robot/introspection"])
@pytest.mark.parametrize(
    "raw", ["example_interfaces/action/Fibonacci", " example_interfaces/Fibonacci "]
)
def test_ros2_official_wire_shape_normalizes_and_compiles(prefix, raw):
    transport = FixtureTransport(response(action_servers=["/compute_sequence"]), response(type=raw))
    graph = discovery(transport, prefix=prefix)
    actions = graph.list_actions()
    assert [(a.name, a.action_type, a.risk_hint) for a in actions] == [
        ("/compute_sequence", "example_interfaces/action/Fibonacci", "medium")
    ]
    assert transport.calls == [
        (f"{prefix}/action_servers", {}),
        (f"{prefix}/action_type", {"action": "/compute_sequence"}),
    ]
    assert transport.responses == []
    manifest = CapabilityManifestCompiler().compile(snapshot(actions))
    assert manifest.capabilities[0].interface.msg_type == "example_interfaces/action/Fibonacci"
    assert manifest.endpoint["execution_eligible"] is False
    assert actions[0].goal_schema == actions[0].feedback_schema == actions[0].result_schema == {}


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        123,
        True,
        ["one/action/First", "two/action/Second"],
        {"type": "one/action/First"},
        "Fibonacci",
        "pkg/msg/Fibonacci",
        "pkg/srv/Fibonacci",
        "pkg/action/",
        "/pkg/action/Fibonacci",
        "pkg//Fibonacci",
        "pkg/action/Foo Bar",
        "pkg/action/A,pkg/action/B",
        "pkg/action/A\npkg/action/B",
    ],
)
def test_invalid_or_ambiguous_type_is_retained_unknown_and_compiler_rejects(raw, caplog):
    transport = FixtureTransport(response(action_servers=["/compute"]), response(type=raw))
    actions = discovery(transport).list_actions()
    assert len(actions) == 1
    assert actions[0].name == "/compute"
    assert actions[0].action_type == ""
    assert "ROS_ACTION_TYPE_UNRESOLVED: /compute" in caplog.text
    with pytest.raises(ValueError, match="ROS_INTERFACE_TYPE_INVALID: action"):
        CapabilityManifestCompiler().compile(snapshot(actions))


@pytest.mark.parametrize(
    "reply",
    [
        RosTransportResult(ok=False, error="service not found"),
        RosTransportResult(ok=False, error="timeout"),
        RosTransportResult(
            ok=True, error="transport error", data={"values": {"type": "p/action/T"}}
        ),
        RosTransportResult(ok=True, data=None),
        RosTransportResult(ok=True, data={"values": {}}),
        RosTransportResult(ok=True, data={"values": ["p/action/T"]}),
        RosTransportResult(ok=True, data={"type": "p/action/T"}),
        {"result": False, "values": {"type": "p/action/T"}},
    ],
)
def test_unsupported_missing_failed_envelope_never_supplies_type(reply):
    transport = FixtureTransport(reply)
    graph = discovery(transport)
    assert graph.get_action_details("/compute") is None
    assert transport.calls == [("/rosapi/action_type", {"action": "/compute"})]


def test_ros1_official_names_remain_unknown_without_nonexistent_service(caplog):
    # Pinned ROS1 proxy discovers these namespaces from cancel, feedback,
    # goal, result and status topics. Its GetActionServers contains names only.
    transport = FixtureTransport(
        response(action_servers=["/move_base", "/arm/follow_joint_trajectory"])
    )
    graph = discovery(transport, RosVersion.ROS1)
    actions = graph.list_actions()
    assert [a.name for a in actions] == ["/move_base", "/arm/follow_joint_trajectory"]
    assert all(a.action_type == "" and a.risk_hint == "medium" for a in actions)
    assert graph.get_action_type("/move_base") == ""
    assert graph.get_action_details("/move_base") is None
    assert transport.calls == [("/rosapi/action_servers", {})]
    assert "ROS_ACTION_TYPE_UNRESOLVED" in caplog.text
    with pytest.raises(ValueError, match="ROS_INTERFACE_TYPE_INVALID"):
        CapabilityManifestCompiler().compile(snapshot(actions))


@pytest.mark.parametrize("field", ["actions", "action_servers"])
def test_legacy_name_alias_and_order_preserved(field):
    transport = FixtureTransport(
        {"values": {field: ["/z", "/a"]}},
        {"values": {"type": "pkg/action/Z"}},
        {"values": {"type": "pkg/action/A"}},
    )
    actions = discovery(transport).list_actions()
    assert [(a.name, a.action_type) for a in actions] == [
        ("/z", "pkg/action/Z"),
        ("/a", "pkg/action/A"),
    ]
    assert [args for _, args in transport.calls] == [{}, {"action": "/z"}, {"action": "/a"}]


@pytest.mark.parametrize(
    "values",
    [
        {"action_servers": "/compute"},
        {"action_servers": [None]},
        {"action_servers": [{"name": "/compute"}]},
        {"action_servers": [""]},
        {"action_servers": [" /compute"]},
        {"action_servers": None},
        {"action_servers": [], "actions": ["/hidden"]},
    ],
)
def test_malformed_or_conflicting_name_arrays_raise_not_silent_success(values):
    transport = FixtureTransport({"values": values})
    with pytest.raises(ValueError, match="ROS_ACTION_METADATA_INVALID"):
        discovery(transport).list_actions()
    assert transport.calls == [("/rosapi/action_servers", {})]


@pytest.mark.parametrize(
    "reply",
    [
        response(action_servers=[]),
        RosTransportResult(ok=False, error="unavailable"),
        {"result": False, "values": {"action_servers": ["/not_success"]}},
    ],
)
def test_empty_or_failed_listing_does_not_issue_type_calls(reply):
    transport = FixtureTransport(reply)
    assert discovery(transport).list_actions() == []
    assert transport.calls == [("/rosapi/action_servers", {})]


def test_type_lookup_not_cached_across_graph_changes_and_details_shape():
    transport = FixtureTransport(response(type="pkg/action/First"), response(type=""))
    graph = discovery(transport)
    info = graph.get_action_details("/changing")
    assert info is not None
    assert (info.name, info.action_type) == ("/changing", "pkg/action/First")
    assert graph.get_action_type("/changing") == ""
    assert len(transport.calls) == 2


@pytest.mark.parametrize("action", [None, "", "   ", 42, " /compute"])
def test_invalid_lookup_argument_does_not_call_transport(action):
    transport = FixtureTransport()
    assert discovery(transport).get_action_type(action) == ""
    assert transport.calls == []


def test_unknown_ros_version_never_assumes_ros2_service():
    transport = FixtureTransport()
    assert discovery(transport, RosVersion.UNKNOWN).get_action_type("/compute") == ""
    assert transport.calls == []
