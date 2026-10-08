"""Exercise pure ROS-host capture helpers without ROS dependencies or writes."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def helpers():
    path = Path(__file__).resolve().parents[3] / "integrations/ros_probe/ros2/probe.py"
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("readonly_parameter_names", "latched_observation")
    ]
    scope = {"DurabilityPolicy": SimpleNamespace(TRANSIENT_LOCAL="TRANSIENT")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def test_renamed_observation_topic_keys_are_read_without_capturing_unrelated_parameters():
    capture = helpers()["readonly_parameter_names"]
    assert capture(
        [
            "use_sim_time",
            "observation_sources",
            "ranger.topic",
            "front.range.topic",
            "private_token",
        ]
    ) == ["use_sim_time", "observation_sources", "ranger.topic", "front.range.topic"]


@pytest.mark.parametrize(
    "names",
    [
        ["scan.topic", "scan.topic"],
        ["x" + str(i) + ".topic" for i in range(65)],
        [True],
        ["unrelated"] * 4097,
    ],
)
def test_ambiguous_or_unbounded_parameter_capture_fails_closed(names):
    with pytest.raises(ValueError):
        helpers()["readonly_parameter_names"](names)


def test_renamed_latched_map_uses_actual_type_and_publisher_durability():
    helper = helpers()["latched_observation"]
    publisher = SimpleNamespace(qos_profile=SimpleNamespace(durability="TRANSIENT"))
    assert helper("/other/robot/floor", ["nav_msgs/msg/OccupancyGrid"], [publisher])
    assert not helper("/map", ["sensor_msgs/msg/LaserScan"], [publisher])
    publisher.qos_profile.durability = "VOLATILE"
    assert not helper("/other/robot/floor", ["nav_msgs/msg/OccupancyGrid"], [publisher])
    assert not helper("/other/robot/floor", ["nav_msgs/msg/OccupancyGrid"], [])
