"""Tests for MCP Drivers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rosclaw.mcp_drivers.base import DriverState, TrajectoryCommand
from rosclaw.mcp_drivers.mujoco_sim_driver import MuJoCoSimDriver
from rosclaw.mcp_drivers.ros2_driver import ROS2Driver
from rosclaw.mcp_drivers.serial_driver import SerialDriver


def test_driver_state_default():
    state = DriverState()
    assert state.connected is False
    assert state.joint_positions == []
    assert state.error_code == 0
    assert state.is_ready() is False


def test_driver_state_ready():
    state = DriverState(connected=True, error_code=0)
    assert state.is_ready() is True


def test_trajectory_command():
    cmd = TrajectoryCommand(
        waypoints=[[0.0] * 6, [1.0] * 6],
        times=[0.0, 2.0],
    )
    assert len(cmd.waypoints) == 2


def test_ros2_driver_fixture_mode():
    driver = ROS2Driver("test_bot", fixture_mode=True)
    driver.initialize()
    driver.start()
    assert driver.is_connected()
    assert driver.get_joint_positions() == [0.0] * 6
    assert driver.move_joints([0.1] * 6, duration=1.0) is True
    driver.stop()
    assert not driver.is_connected()


def test_ros2_driver_does_not_enter_fixture_implicitly() -> None:
    driver = ROS2Driver("test_bot")

    with (
        patch.dict("sys.modules", {"rclpy": None}),
        pytest.raises(RuntimeError, match="ROS2 initialization failed"),
    ):
        driver.initialize()

    assert driver.is_connected() is False


def test_ros2_driver_dof_mismatch():
    driver = ROS2Driver("test_bot", joint_dof=6, fixture_mode=True)
    driver.initialize()
    driver.start()
    # _validate_joint_positions raises ValueError before connected check
    try:
        driver.move_joints([0.1] * 5)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    driver.stop()


def test_mujoco_driver_fixture_mode():
    driver = MuJoCoSimDriver("test_bot", model_path="/nonexistent.xml", fixture_mode=True)
    driver._do_initialize()
    assert driver.is_connected()
    assert driver.get_joint_positions() == [0.0] * 6
    driver._do_stop()


def test_mujoco_driver_loads_real_model_and_reports_simulated_stop() -> None:
    model_path = Path(__file__).parents[1] / "e-urdf-zoo" / "ur5e" / "robot.mjcf.xml"
    driver = MuJoCoSimDriver("sim_ur5e", model_path=str(model_path))
    driver.initialize()
    driver.start()
    try:
        receipt = driver.emergency_stop()
    finally:
        driver.stop()

    assert receipt["acknowledged"] is True
    assert receipt["execution_mode"] == "SIMULATION"
    assert receipt["trust_level"] == "SIMULATED"
    assert driver.state.usable_for_real_execution is False


def test_serial_driver_fixture_mode():
    driver = SerialDriver("test_bot", port="/dev/ttyFAKE", fixture_mode=True)
    driver._do_initialize()
    assert driver.is_connected()
    assert driver.get_joint_positions() == [0.0] * 6
    assert driver.set_gripper(0.5) is True
    assert driver.state.execution_mode == "FIXTURE"
    assert driver.state.trust_level == "SYNTHETIC"
    assert driver.state.usable_for_real_execution is False
    driver._do_stop()


def test_serial_driver_emergency_stop():
    driver = SerialDriver("test_bot", fixture_mode=True)
    driver._do_initialize()
    driver.emergency_stop()
    assert driver.state.error_code == 99
    driver._do_stop()
