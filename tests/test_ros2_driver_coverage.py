"""Coverage tests for mcp_drivers/ros2_driver.py."""

from unittest.mock import MagicMock

import pytest

from rosclaw.mcp_drivers.ros2_driver import ROS2Driver
from rosclaw.mcp_drivers.base import TrajectoryCommand


class TestROS2DriverInit:
    def test_init_default(self):
        driver = ROS2Driver("ur5", joint_dof=6)
        assert driver.robot_id == "ur5"
        assert driver.joint_dof == 6
        assert driver._node_name == "rosclaw_driver"
        assert driver._rclpy is None
        assert driver._node is None

    def test_init_custom_node_name(self):
        driver = ROS2Driver("panda", joint_dof=7, node_name="custom_driver")
        assert driver._node_name == "custom_driver"


class TestROS2DriverLifecycle:
    def test_initialize_mock_mode(self):
        # rclpy is mocked by test_mcp_server.py; driver initializes in "ros2" mode
        driver = ROS2Driver("ur5")
        driver.initialize()
        driver.start()
        assert driver._driver_state.connected is True
        driver.stop()

    def test_stop_without_init(self):
        driver = ROS2Driver("ur5")
        # Stop without initialize — should not crash
        driver.stop()
        assert driver._driver_state.connected is False

    def test_stop_clears_node(self):
        driver = ROS2Driver("ur5")
        driver.initialize()
        driver.start()
        driver.stop()
        # _node is destroyed and set to None
        assert driver._node is None


class TestROS2DriverJointState:
    def test_on_joint_state(self):
        driver = ROS2Driver("ur5", joint_dof=6)
        msg = MagicMock()
        msg.position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        msg.velocity = [0.01] * 6
        msg.effort = [0.5] * 6

        driver._on_joint_state(msg)
        assert driver._latest_joint_state["positions"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        assert driver._driver_state.joint_positions == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def test_on_joint_state_truncates(self):
        driver = ROS2Driver("ur5", joint_dof=3)
        msg = MagicMock()
        msg.position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        msg.velocity = [0.01] * 6
        msg.effort = [0.5] * 6

        driver._on_joint_state(msg)
        assert driver._driver_state.joint_positions == [0.1, 0.2, 0.3]


class TestROS2DriverGetters:
    def test_get_joint_positions_no_data(self):
        driver = ROS2Driver("ur5", joint_dof=6)
        assert driver.get_joint_positions() == [0.0] * 6

    def test_get_joint_positions_with_data(self):
        driver = ROS2Driver("ur5", joint_dof=6)
        driver._latest_joint_state = {
            "positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "velocities": [0.01] * 6,
            "efforts": [0.5] * 6,
        }
        assert driver.get_joint_positions() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def test_get_joint_velocities_no_data(self):
        driver = ROS2Driver("ur5", joint_dof=6)
        assert driver.get_joint_velocities() == [0.0] * 6

    def test_get_joint_velocities_with_data(self):
        driver = ROS2Driver("ur5", joint_dof=6)
        driver._latest_joint_state = {
            "positions": [0.1] * 6,
            "velocities": [0.2] * 6,
            "efforts": [0.3] * 6,
        }
        assert driver.get_joint_velocities() == [0.2] * 6

    def test_get_joint_torques_no_data(self):
        driver = ROS2Driver("ur5", joint_dof=6)
        assert driver.get_joint_torques() == [0.0] * 6

    def test_get_joint_torques_with_data(self):
        driver = ROS2Driver("ur5", joint_dof=6)
        driver._latest_joint_state = {
            "positions": [0.1] * 6,
            "velocities": [0.2] * 6,
            "efforts": [0.3] * 6,
        }
        assert driver.get_joint_torques() == [0.3] * 6

    def test_get_positions_truncates(self):
        driver = ROS2Driver("ur5", joint_dof=3)
        driver._latest_joint_state = {
            "positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "velocities": [0.01] * 6,
            "efforts": [0.5] * 6,
        }
        assert driver.get_joint_positions() == [0.1, 0.2, 0.3]


class TestROS2DriverMoveJoints:
    def test_move_joints_not_connected(self):
        driver = ROS2Driver("ur5")
        driver.initialize()
        driver.start()
        driver._driver_state.connected = False
        result = driver.move_joints([0.0] * 6)
        assert result is False

    def test_move_joints_mock_mode(self):
        driver = ROS2Driver("ur5")
        driver.initialize()
        result = driver.move_joints([0.5] * 6)
        assert result is True
        assert driver._driver_state.joint_positions == [0.5] * 6

    def test_move_joints_wrong_size(self):
        driver = ROS2Driver("ur5", joint_dof=6)
        driver.initialize()
        with pytest.raises(ValueError, match="Expected 6 joint positions"):
            driver.move_joints([0.0] * 3)

    def test_move_joints_negative_duration(self):
        driver = ROS2Driver("ur5", joint_dof=6)
        driver.initialize()
        with pytest.raises(ValueError, match="Duration must be positive"):
            driver.move_joints([0.0] * 6, duration=-1.0)

    def test_move_joints_ros2_mode(self):
        driver = ROS2Driver("ur5")
        driver.initialize()
        driver._rclpy = MagicMock()  # Simulate ROS2 mode
        driver._pub_joint_cmd = MagicMock()
        result = driver.move_joints([0.5] * 6, duration=2.0)
        assert result is True
        driver._pub_joint_cmd.publish.assert_called_once()


class TestROS2DriverExecuteTrajectory:
    def test_execute_trajectory_not_connected(self):
        driver = ROS2Driver("ur5")
        driver._driver_state.connected = False
        traj = TrajectoryCommand(waypoints=[[0.0] * 6], times=[1.0])
        result = driver.execute_trajectory(traj)
        assert result is False

    def test_execute_trajectory_mock_mode(self):
        driver = ROS2Driver("ur5")
        driver.initialize()
        driver.start()
        driver._rclpy = None  # Force mock mode
        traj = TrajectoryCommand(waypoints=[[0.1] * 6, [0.2] * 6], times=[1.0, 2.0])
        result = driver.execute_trajectory(traj)
        assert result is True
        assert driver._driver_state.joint_positions == [0.2] * 6

    def test_execute_trajectory_empty_waypoints(self):
        driver = ROS2Driver("ur5")
        driver.initialize()
        driver.start()
        traj = TrajectoryCommand(waypoints=[], times=[])
        with pytest.raises(ValueError, match="at least one waypoint"):
            driver.execute_trajectory(traj)

    def test_execute_trajectory_ros2_mode(self):
        driver = ROS2Driver("ur5")
        driver.initialize()
        driver.start()
        driver._rclpy = MagicMock()
        driver._pub_joint_cmd = MagicMock()
        traj = TrajectoryCommand(waypoints=[[0.1] * 6, [0.2] * 6], times=[1.0, 2.0])
        result = driver.execute_trajectory(traj)
        assert result is True
        assert driver._pub_joint_cmd.publish.call_count == 1

    def test_execute_trajectory_mismatched_lengths(self):
        driver = ROS2Driver("ur5")
        driver.initialize()
        driver.start()
        traj = TrajectoryCommand(waypoints=[[0.0] * 6], times=[1.0, 2.0])
        with pytest.raises(ValueError, match="Waypoint count"):
            driver.execute_trajectory(traj)


class TestROS2DriverGripper:
    def test_set_gripper(self):
        driver = ROS2Driver("ur5")
        result = driver.set_gripper(0.5, force=0.3)
        assert result is True
        assert driver._driver_state.gripper_state == 0.5


class TestROS2DriverEmergencyStop:
    def test_emergency_stop(self):
        driver = ROS2Driver("ur5")
        driver.initialize()
        driver.emergency_stop()
        assert driver._driver_state.error_code == 99
        assert "Emergency stop" in driver._driver_state.error_message


class TestROS2DriverGetState:
    def test_get_state(self):
        driver = ROS2Driver("ur5")
        state = driver.get_state()
        assert state.connected is False
        assert state.error_code == 0

    def test_get_state_after_init(self):
        driver = ROS2Driver("ur5")
        driver.initialize()
        state = driver.get_state()
        assert state.connected is True
