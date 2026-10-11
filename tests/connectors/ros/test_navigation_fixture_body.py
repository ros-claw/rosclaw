"""Operator fixture compilation only: no ROS, simulator, or execution Runtime."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding


@pytest.mark.parametrize("model", ["turtlebot3_waffle", "turtlebot3_burger"])
def test_known_navigation_fixture_compiles_without_cleaner(tmp_path, model):
    source = (
        Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance/fixture_body.py"
    )
    spec = importlib.util.spec_from_file_location("fixture_body_navigation_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    urdf = tmp_path / "vendor.urdf"
    urdf.write_text(f'<robot name="{model}"><link name="base_footprint"/></robot>')
    home = tmp_path / "home"
    body_hash = module.configure_fixture_body(
        home,
        {"body_id": "known-fixture", "physical_radius_m": 0.25, "cleaning_polygon": []},
        urdf,
        navigation_run_id="fresh-known-fixture",
    )
    effective = BodyResolver(workspace=home).get_effective_body(recompile_if_stale=False)
    binding = SimulationNavigationBinding.from_body(effective)
    assert binding.body_snapshot_hash == body_hash
    assert binding.run_id == "fresh-known-fixture"
    bindings = effective.provider_interfaces["ros_capability_bindings"]
    assert set(bindings) == {
        "navigation.navigate_to_pose",
        "safety.simulation_lease",
        "safety.simulation_observation",
    }
    assert (
        bindings["safety.simulation_observation"]["urdf_sha256"]
        == hashlib.sha256(urdf.read_bytes()).hexdigest()
    )
    assert "coverage.execute" not in effective.capabilities.get("enabled", [])
    assert "coverage.execute" not in effective.capabilities.get("degraded", [])
    assert "cleaning" not in str(bindings)
