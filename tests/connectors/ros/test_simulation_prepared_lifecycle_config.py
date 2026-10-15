"""Original staged manager lists, rather than overriding a bad list in launch."""

import importlib
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def configure(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_prepared_lifecycle_config").managed_lifecycle_nodes


def test_staged_missing_manager_node_remains_missing_until_actual_staging_repair(configure):
    available = ["map_server", "planner_server", "collision_monitor"]
    namespace = "/independent/robot"
    config = {
        namespace: {
            "lifecycle_manager_navigation": {
                "ros__parameters": {"node_names": ["map_server", "collision_monitor"]}
            }
        }
    }
    assert configure(yaml.safe_dump(config), namespace, available) == [
        "map_server",
        "collision_monitor",
    ]
    config[namespace]["lifecycle_manager_navigation"]["ros__parameters"]["node_names"] = available
    assert configure(yaml.safe_dump(config), namespace, available) == available


def test_default_manager_list_preserves_existing_prepared_behavior(configure):
    assert configure("/declared: {}", "/declared", ["map_server", "planner_server"]) == [
        "map_server",
        "planner_server",
    ]
    assert configure("{}", "", ["map_server"]) == ["map_server"]


@pytest.mark.parametrize(
    "invalid", [[], ["unknown"], ["map_server", "map_server"], "map_server", [1]]
)
def test_unbounded_or_unknown_manager_scope_is_rejected_before_launch(configure, invalid):
    config = {"lifecycle_manager_navigation": {"ros__parameters": {"node_names": invalid}}}
    with pytest.raises(ValueError, match="unique declared existing"):
        configure(yaml.safe_dump(config), "", ["map_server"])
