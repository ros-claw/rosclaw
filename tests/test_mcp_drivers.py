"""Tests for MCP Drivers."""


from rosclaw.mcp_drivers.base import DriverState, TrajectoryCommand
from rosclaw.mcp_drivers.ros2_driver import ROS2Driver
from rosclaw.mcp_drivers.mujoco_sim_driver import MuJoCoSimDriver
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


def test_ros2_driver_mock_mode():
    driver = ROS2Driver("test_bot")
    driver.initialize()
    driver.start()
    assert driver.is_connected()
    assert driver.get_joint_positions() == [0.0] * 6
    assert driver.move_joints([0.1] * 6, duration=1.0) is True
    driver.stop()
    assert not driver.is_connected()


def test_ros2_driver_dof_mismatch():
    driver = ROS2Driver("test_bot", joint_dof=6)
    driver.initialize()
    driver.start()
    # _validate_joint_positions raises ValueError before connected check
    try:
        driver.move_joints([0.1] * 5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    driver.stop()


def test_mujoco_driver_mock_mode():
    driver = MuJoCoSimDriver("test_bot", model_path="/nonexistent.xml")
    driver._do_initialize()
    assert driver.is_connected()
    assert driver.get_joint_positions() == [0.0] * 6
    driver._do_stop()


def test_serial_driver_mock_mode():
    driver = SerialDriver("test_bot", port="/dev/ttyFAKE")
    driver._do_initialize()
    assert driver.is_connected()
    assert driver.get_joint_positions() == [0.0] * 6
    assert driver.set_gripper(0.5) is True
    driver._do_stop()


def test_serial_driver_emergency_stop():
    driver = SerialDriver("test_bot")
    driver._do_initialize()
    driver.emergency_stop()
    assert driver.state.error_code == 99
    driver._do_stop()
