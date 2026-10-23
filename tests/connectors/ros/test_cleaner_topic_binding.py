"""Typed topic cleaner bindings must not be diagnosed as SetBool services."""

from datetime import UTC, datetime

import pytest

from rosclaw.connectors.ros.intelligence.system_model import RosSystemModel, Signal
from rosclaw.connectors.ros.resolver.capability_resolver import resolve_capabilities


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("none", "AVAILABLE"),
        ("missing_state_binding", "UNKNOWN"),
        ("unobserved_command", "UNKNOWN"),
        ("wrong_command_type", "BLOCKED"),
        ("wrong_state_type", "BLOCKED"),
        ("stale_state", "BLOCKED"),
        ("wrong_boolean", "BLOCKED"),
        ("missing_body", "BLOCKED"),
    ],
)
def test_topic_cleaner_binding(fault, expected):
    now = datetime.now(UTC)
    binding = {
        "name": "/cleaner/enable",
        "msg_type": "std_msgs/msg/Bool",
        "data": True,
        "state_topic": "/cleaner/state",
        "state_type": "std_msgs/msg/Bool",
    }
    model = RosSystemModel(
        robot_id="topic-cleaner",
        snapshot_id="typed-fixture",
        captured_at=now,
        body={
            "effective_body_hash": "body",
            "ros_capability_bindings": {"cleaning.enable": binding},
        },
        graph={
            "services": [],
            "topics": [
                {"name": "/cleaner/enable", "msg_type": "std_msgs/msg/Bool"},
                {"name": "/cleaner/state", "msg_type": "std_msgs/msg/Bool"},
            ],
        },
        signals=[
            Signal(
                topic="/cleaner/state",
                source="fixture",
                captured_at=now,
                publisher_count=1,
                last_message_age_ms=10,
            )
        ],
    )
    if fault == "missing_state_binding":
        binding.pop("state_topic")
        binding.pop("state_type")
    elif fault == "unobserved_command":
        model.graph["topics"] = model.graph["topics"][1:]
    elif fault == "wrong_command_type":
        model.graph["topics"][0]["msg_type"] = "std_msgs/msg/String"
    elif fault == "wrong_state_type":
        model.graph["topics"][1]["msg_type"] = "std_msgs/msg/String"
    elif fault == "stale_state":
        model.signals[0].last_message_age_ms = 2000
    elif fault == "wrong_boolean":
        binding["data"] = "true"
    elif fault == "missing_body":
        model.body.pop("effective_body_hash")
    row = next(
        r for r in resolve_capabilities(model, now=now) if r["semantic_id"] == "cleaning.enable"
    )
    assert row["status"] == expected, row
    assert row["usable_for_real_execution"] is False
    assert row["execution_entry"] == "request_action"


@pytest.mark.parametrize(
    "fault,blocked",
    [
        ("none", False),
        ("stale_scan", True),
        ("missing_tf", True),
        ("other_publisher", True),
        ("explicit_rate", True),
        ("old_tf_snapshot", True),
    ],
)
def test_event_pose_never_masks_sensor_tf_or_explicit_rate_faults(fault, blocked):
    from datetime import timedelta

    from rosclaw.connectors.ros.diagnosis import diagnose
    from rosclaw.connectors.ros.intelligence.system_model import Transform

    now = datetime.now(UTC)
    model = RosSystemModel(
        robot_id="slam-fixture",
        snapshot_id="event",
        captured_at=now,
        graph={
            "topics": [
                {
                    "name": "/pose",
                    "msg_type": "geometry_msgs/msg/PoseWithCovarianceStamped",
                    "publishers": ["/slam_toolbox"],
                },
                {
                    "name": "/scan",
                    "msg_type": "sensor_msgs/msg/LaserScan",
                    "publishers": ["/lidar"],
                },
            ]
        },
        observations={
            "node_parameters": {
                "/slam_toolbox": {
                    "map_frame": "map",
                    "odom_frame": "odom",
                    "transform_publish_period": 0.02,
                }
            }
        },
        transforms=[
            Transform(source="/tf", captured_at=now, parent=p, child=c, age_ms=10)
            for p, c in [("map", "odom"), ("odom", "base_link")]
        ],
        signals=[
            Signal(
                topic=t,
                source="fixture",
                captured_at=now,
                publisher_count=1,
                last_message_age_ms=age,
            )
            for t, age in [("/pose", 40000), ("/scan", 10)]
        ],
    )
    if fault == "stale_scan":
        model.signals[1].last_message_age_ms = 5000
    elif fault == "missing_tf":
        model.transforms = []
    elif fault == "other_publisher":
        model.graph["topics"][0]["publishers"] = ["/unknown_localizer"]
    elif fault == "explicit_rate":
        model.body["minimum_rates"] = {"/pose": 1}
    elif fault == "old_tf_snapshot":
        model.transforms[0].captured_at = now - timedelta(seconds=10)
    result = diagnose(model, now=now)
    assert any(i["issue_code"] == "ROS_TOPIC_002" for i in result["issues"]) is blocked
    if not blocked:
        row = next(
            r
            for r in resolve_capabilities(model, now=now)
            if r["semantic_id"] == "localization.pose_estimate"
        )
        assert row["status"] == "UNKNOWN"  # No old pose is promoted to a current pose.
        assert any(
            i["issue_code"] == "ROS_POSE_EVENT_001" and i["severity"] == "warning"
            for i in result["issues"]
        )


@pytest.mark.parametrize(
    "bound,stale_state,blocked", [(True, False, False), (False, False, True), (True, True, True)]
)
def test_bound_command_age_is_not_sensor_age(bound, stale_state, blocked):
    from rosclaw.connectors.ros.diagnosis import diagnose

    now = datetime.now(UTC)
    body = (
        {
            "effective_body_hash": "body",
            "ros_capability_bindings": {
                "cleaning.enable": {
                    "name": "/cleaner/enable",
                    "msg_type": "std_msgs/msg/Bool",
                    "data": True,
                }
            },
        }
        if bound
        else {}
    )
    model = RosSystemModel(
        robot_id="cleaner",
        snapshot_id="command",
        captured_at=now,
        body=body,
        graph={
            "topics": [
                {"name": t, "msg_type": "std_msgs/msg/Bool"}
                for t in ["/cleaner/enable", "/cleaner/state"]
            ]
        },
        signals=[
            Signal(
                topic=t,
                source="fixture",
                captured_at=now,
                publisher_count=1,
                last_message_age_ms=age,
            )
            for t, age in [
                ("/cleaner/enable", 40000),
                ("/cleaner/state", 5000 if stale_state else 10),
            ]
        ],
    )
    result = diagnose(model, now=now)
    assert any(i["issue_code"] == "ROS_TOPIC_002" for i in result["issues"]) is blocked
