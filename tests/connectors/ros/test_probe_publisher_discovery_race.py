"""Exercise the actual optional SDK discovery method without importing ROS."""

import ast
from pathlib import Path
from types import SimpleNamespace


def test_subscriber_first_map_discovery_waits_for_actual_publisher_qos():
    path = Path(__file__).resolve().parents[3] / "integrations/ros_probe/ros2/probe.py"
    tree = ast.parse(path.read_text())
    method = next(
        n
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == "ReadOnlyProbe"
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "discover_reads"
    )
    latched = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "latched_observation"
    )
    namespace = {
        "QoSProfile": lambda **kwargs: SimpleNamespace(**kwargs),
        "ReliabilityPolicy": SimpleNamespace(BEST_EFFORT="BEST_EFFORT"),
        "DurabilityPolicy": SimpleNamespace(TRANSIENT_LOCAL="TRANSIENT_LOCAL"),
        "READ_TYPES": {"nav_msgs/msg/OccupancyGrid"},
        "get_message": lambda value: value,
    }
    exec(compile(ast.Module(body=[latched, method], type_ignores=[]), str(path), "exec"), namespace)
    publications, subscriptions = [], []
    observer = SimpleNamespace(
        subscriptions_by_topic={},
        errors=[],
        observe=lambda *_: None,
        get_topic_names_and_types=lambda: [("/renamed_floor", ["nav_msgs/msg/OccupancyGrid"])],
        get_publishers_info_by_topic=lambda _: publications,
        get_service_names_and_types=lambda: [],
        create_subscription=lambda *args: subscriptions.append(args) or "actual-subscription",
    )
    namespace["discover_reads"](observer)
    assert not subscriptions and not observer.subscriptions_by_topic
    publications.append(SimpleNamespace(qos_profile=SimpleNamespace(durability="TRANSIENT_LOCAL")))
    namespace["discover_reads"](observer)
    assert len(subscriptions) == 1
    assert subscriptions[0][-1].durability == "TRANSIENT_LOCAL"
    namespace["discover_reads"](observer)
    assert len(subscriptions) == 1 and not observer.errors
