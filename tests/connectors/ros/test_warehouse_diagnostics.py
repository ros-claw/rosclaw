"""Warehouse-derived regressions: configuration evidence never authorizes repair."""

from datetime import UTC, datetime, timedelta

import pytest

from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.intelligence import RosSystemModel

NOW = datetime(2026, 10, 9, tzinfo=UTC)


def model(plugins=None, maximum=False, age=0):
    return RosSystemModel(
        robot_id="sim",
        snapshot_id="fixture",
        captured_at=NOW,
        graph={"topics": [{"name": "/scan", "msg_type": "sensor_msgs/msg/PointCloud2"}]},
        observations={
            "parameter_captured_at": {
                "/robot/local_costmap/local_costmap": (NOW - timedelta(seconds=age)).isoformat()
            },
            "node_parameters": {
                "/robot/local_costmap/local_costmap": {
                    "plugins": plugins or ["lidar", "padding", "map"],
                    "lidar.plugin": "nav2_costmap_2d::VoxelLayer",
                    "padding.plugin": "nav2_costmap_2d::InflationLayer",
                    "map.plugin": "nav2_costmap_2d::StaticLayer",
                    "map.use_maximum": maximum,
                }
            },
        },
    )


def issues(m):
    return {i["issue_code"]: i for i in diagnose(m, now=NOW)["issues"]}


def test_late_static_layer_warns_with_sources_without_claiming_loss():
    result = issues(model())["NAV2_COSTMAP_003"]
    assert result["severity"] == "warning"
    assert result["official_sources"]
    assert result["runtime_mutation_required"] is False
    assert "does not prove" in result["hypotheses"][0]
    assert result["evidence"][0]["observation"]["earlier_obstacle_layers"] == ["lidar"]


@pytest.mark.parametrize(
    "kwargs",
    [{"plugins": ["map", "lidar", "padding"]}, {"maximum": True}, {"maximum": None}, {"age": 6}],
)
def test_no_overwrite_claim_when_order_combination_or_freshness_differs(kwargs):
    assert "NAV2_COSTMAP_003" not in issues(model(**kwargs))


def test_plugin_instance_names_do_not_imply_types():
    m = model()
    m.observations["node_parameters"]["/robot/local_costmap/local_costmap"]["map.plugin"] = (
        "custom::SafeMerge"
    )
    assert "NAV2_COSTMAP_003" not in issues(m)


def test_disabled_static_layer_does_not_warn():
    m = model()
    m.observations["node_parameters"]["/robot/local_costmap/local_costmap"]["map.enabled"] = False
    assert "NAV2_COSTMAP_003" not in issues(m)


def test_inflation_before_obstacles_warns():
    assert "NAV2_COSTMAP_004" in issues(model(plugins=["padding", "lidar", "map"]))


def test_explicit_body_topic_type_mismatch_only():
    m = model()
    m.completeness["graph"] = True
    assert "ROS_TOPIC_005" not in issues(m)
    m.body["required_topic_types"] = {"/scan": "sensor_msgs/msg/LaserScan"}
    assert "ROS_TOPIC_005" in issues(m)
    m.completeness["graph"] = False
    assert "ROS_TOPIC_005" not in issues(m)
