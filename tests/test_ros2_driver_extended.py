"""Extended tests for ROS2Driver."""

from rosclaw.mcp_drivers.base import TrajectoryCommand
from rosclaw.mcp_drivers.ros2_driver import ROS2Driver


class TestROS2DriverExtended:
    def test_on_joint_state_updates(self):
        driver = ROS2Driver("test_bot", joint_dof=3)
        driver.initialize()
        driver.start()

        class FakeMsg:
            position = [1.0, 2.0, 3.0]
            velocity = [0.1, 0.2, 0.3]
            effort = [10.0, 20.0, 30.0]

        driver._on_joint_state(FakeMsg())
        assert driver.get_joint_positions() == [1.0, 2.0, 3.0]
        assert driver.get_joint_velocities() == [0.1, 0.2, 0.3]
        assert driver.get_joint_torques() == [10.0, 20.0, 30.0]
        driver.stop()

    def test_get_joint_positions_fallback(self):
        driver = ROS2Driver("test_bot")
        driver.initialize()
        driver.start()
        assert driver.get_joint_positions() == [0.0] * 6
        driver.stop()

    def test_get_joint_velocities_fallback(self):
        driver = ROS2Driver("test_bot")
        driver.initialize()
        driver.start()
        assert driver.get_joint_velocities() == [0.0] * 6
        driver.stop()

    def test_get_joint_torques_fallback(self):
        driver = ROS2Driver("test_bot")
        driver.initialize()
        driver.start()
        assert driver.get_joint_torques() == [0.0] * 6
        driver.stop()

    def test_move_joints_mock_mode(self):
        driver = ROS2Driver("test_bot")
        driver.initialize()
        driver.start()
        assert driver.move_joints([0.1] * 6, duration=1.0) is True
        assert driver._driver_state.joint_positions == [0.1] * 6
        driver.stop()

    def test_execute_trajectory_returns_true_when_connected(self):
        # When rclpy is available and connected, execute_trajectory publishes and returns True
        driver = ROS2Driver("test_bot")
        driver.initialize()
        driver.start()
        traj = TrajectoryCommand(
            waypoints=[[0.0] * 6, [0.1] * 6],
            times=[1.0, 2.0],
        )
        assert driver.execute_trajectory(traj) is True
        driver.stop()

    def test_execute_trajectory_not_connected(self):
        driver = ROS2Driver("test_bot")
        traj = TrajectoryCommand(waypoints=[[0.0] * 6], times=[1.0])
        assert driver.execute_trajectory(traj) is False

    def test_set_gripper(self):
        driver = ROS2Driver("test_bot")
        driver.initialize()
        driver.start()
        assert driver.set_gripper(0.75, force=0.3) is True
        assert driver._driver_state.gripper_state == 0.75
        driver.stop()

    def test_emergency_stop(self):
        driver = ROS2Driver("test_bot")
        driver.initialize()
        driver.start()
        driver._latest_joint_state = {"positions": [0.5] * 6, "velocities": [0.0] * 6, "efforts": [0.0] * 6}
        driver.emergency_stop()
        assert driver._driver_state.error_code == 99
        assert "Emergency stop" in driver._driver_state.error_message
        driver.stop()

    def test_get_state(self):
        driver = ROS2Driver("test_bot")
        driver.initialize()
        driver.start()
        state = driver.get_state()
        assert state is not None
        driver.stop()
