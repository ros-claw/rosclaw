"""Tests for ROS graph discovery."""

from __future__ import annotations

from rosclaw.connectors.ros.discovery import (
    RosApiProfile,
    RosGraphDiscovery,
    RosVersion,
)
from rosclaw.connectors.ros.transport import MockTransport


def _make_profile(version: RosVersion = RosVersion.ROS2, distro: str = "humble") -> RosApiProfile:
    return RosApiProfile(
        version=version,
        distro=distro,
        service_prefix="/rosapi",
        type_prefix="rosapi_msgs/srv",
    )


def test_list_topics_with_types_and_risk():
    transport = MockTransport()
    transport.queue_json(
        {"op": "service_response", "values": {
            "topics": ["/turtle1/cmd_vel", "/turtle1/pose", "/camera/image_raw"],
            "types": ["geometry_msgs/msg/Twist", "turtlesim/msg/Pose", "sensor_msgs/msg/Image"],
        }}
    )
    discovery = RosGraphDiscovery(transport, profile=_make_profile())
    topics = discovery.list_topics()

    assert len(topics) == 3
    by_name = {t.name: t for t in topics}
    assert by_name["/turtle1/cmd_vel"].msg_type == "geometry_msgs/msg/Twist"
    assert by_name["/turtle1/cmd_vel"].is_command
    assert by_name["/turtle1/cmd_vel"].risk_hint == "high"
    assert by_name["/turtle1/pose"].risk_hint is None
    assert by_name["/camera/image_raw"].is_sensor
    assert by_name["/camera/image_raw"].risk_hint == "low"


def test_get_topic_details_aggregates_publishers_and_subscribers():
    transport = MockTransport()
    transport.queue_json({"op": "service_response", "values": {"type": "std_msgs/msg/String"}})
    transport.queue_json({"op": "service_response", "values": {"publishers": ["/node_a"]}})
    transport.queue_json({"op": "service_response", "values": {"subscribers": ["/node_b"]}})

    discovery = RosGraphDiscovery(transport, profile=_make_profile())
    info = discovery.get_topic_details("/chatter")
    assert info is not None
    assert info.msg_type == "std_msgs/msg/String"
    assert info.publishers == ["/node_a"]
    assert info.subscribers == ["/node_b"]


def test_list_services_classifies_risk():
    transport = MockTransport()
    transport.queue_json({"op": "service_response", "values": {"services": ["/go2/move", "/go2/stand_up", "/rosout/get_loggers"]}})
    transport.queue_json({"op": "service_response", "values": {"type": "go2_interfaces/srv/Move"}})
    transport.queue_json({"op": "service_response", "values": {"type": "std_srvs/srv/Trigger"}})
    transport.queue_json({"op": "service_response", "values": {"type": "roscpp/srv/GetLoggers"}})

    discovery = RosGraphDiscovery(transport, profile=_make_profile())
    services = discovery.list_services()
    assert len(services) == 3
    by_name = {s.name: s for s in services}
    assert by_name["/go2/move"].risk_hint == "medium"
    assert by_name["/go2/stand_up"].risk_hint == "medium"
    assert by_name["/rosout/get_loggers"].risk_hint == "low"


def test_list_nodes():
    transport = MockTransport()
    transport.queue_json({"op": "service_response", "values": {"nodes": ["/turtlesim", "/rosbridge_websocket"]}})
    discovery = RosGraphDiscovery(transport, profile=_make_profile())
    nodes = discovery.list_nodes()
    assert [n["name"] for n in nodes] == ["/turtlesim", "/rosbridge_websocket"]


def test_list_params():
    transport = MockTransport()
    transport.queue_json({"op": "service_response", "values": {"params": ["/rosdistro", "/use_sim_time"]}})
    discovery = RosGraphDiscovery(transport, profile=_make_profile())
    params = discovery.list_params()
    assert "/rosdistro" in params


def test_discover_returns_snapshot():
    transport = MockTransport()
    profile = _make_profile()
    # resolver already provided, so no rosapi version calls needed.
    transport.queue_json({"op": "service_response", "values": {"topics": ["/turtle1/cmd_vel"], "types": ["geometry_msgs/msg/Twist"]}})
    transport.queue_json({"op": "service_response", "values": {"services": []}})
    transport.queue_json({"op": "service_response", "values": {"actions": []}})
    transport.queue_json({"op": "service_response", "values": {"nodes": []}})
    transport.queue_json({"op": "service_response", "values": {"params": []}})

    discovery = RosGraphDiscovery(transport, profile=profile)
    snapshot = discovery.discover()
    assert snapshot.ros_version == "ros2"
    assert snapshot.distro == "humble"
    assert len(snapshot.topics) == 1
    assert snapshot.topics[0].name == "/turtle1/cmd_vel"


def test_get_message_details_parses_json():
    transport = MockTransport()
    transport.queue_json({"op": "service_response", "values": {"message": '{"fields": {"linear": {"x": "float64"}}}'}})
    discovery = RosGraphDiscovery(transport, profile=_make_profile())
    details = discovery.get_message_details("geometry_msgs/msg/Twist")
    assert "linear" in details.get("fields", {})


def test_get_message_details_invalid_json_returns_error():
    transport = MockTransport()
    transport.queue_json({"op": "service_response", "values": {"message": "not-json"}})
    discovery = RosGraphDiscovery(transport, profile=_make_profile())
    details = discovery.get_message_details("geometry_msgs/msg/Twist")
    assert "error" in details


def test_normalizer_ros1():
    from rosclaw.connectors.ros.discovery import normalize_msg_type
    assert normalize_msg_type("geometry_msgs/Twist", RosVersion.ROS1) == "geometry_msgs/msg/Twist"


def test_normalizer_ros2():
    from rosclaw.connectors.ros.discovery import normalize_msg_type
    assert normalize_msg_type("geometry_msgs/msg/Twist", RosVersion.ROS2) == "geometry_msgs/msg/Twist"


def test_service_type_caching():
    transport = MockTransport()
    transport.queue_json({"op": "service_response", "values": {"type": "std_srvs/srv/Empty"}})
    discovery = RosGraphDiscovery(transport, profile=_make_profile())
    t1 = discovery.get_service_type("/reset")
    t2 = discovery.get_service_type("/reset")
    assert t1 == t2 == "std_srvs/srv/Empty"
    # Only one transport request because of cache.
    assert len([r for r in transport.requests if r.get("service") == "/rosapi/service_type"]) == 1
