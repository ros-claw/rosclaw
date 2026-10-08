"""Simulation registration and typed external executor contracts."""

from rosclaw.eurdf.registry import RobotRegistry
from rosclaw.sim.robot_kit import kit_for_body, kit_server_spec


def test_microduck_profile_declares_simulation_only():
    profile = RobotRegistry().get("microduck")
    assert profile is not None
    assert profile.embodiment.dof == 14
    assert len(profile.embodiment.joints) == 14
    assert len(profile.embodiment.actuators) == 14
    assert profile.safety.environment["real_robot_execution_allowed"] is False
    assert profile.identity["evidence_domain"] == "simulation"


def test_microduck_action_is_not_an_observation_and_has_its_own_output_schema():
    kit = kit_for_body("microduck-lavender")
    assert kit is not None and kit.mode == "SIMULATION"
    spec = kit_server_spec(kit)
    assert spec["sim_executor"] and spec["effect_domain"] == "SIMULATION_STATE_ONLY"
    assert "microduck.start_game" in spec["action_tools"]
    assert "microduck.start_game" not in spec["observation_tools"]
    assert spec["timeout_ms"] == 600000
    assert "microduck.get_game_status" in spec["output_schemas"]
    assert not any(n.startswith("ur5e.") for n in spec["output_schemas"])
    assert spec["required_body_types"] == ["microduck-lavender"]


def test_existing_ur5e_output_contract_is_preserved():
    spec = kit_server_spec(kit_for_body("sim/ur5e"))
    assert "ur5e.get_joint_state" in spec["output_schemas"]
    assert "microduck.get_game_status" not in spec["output_schemas"]
