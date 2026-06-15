"""ROS2-native tests for mcp_drivers/ros2_driver.py - runs in real rclpy environment.

Run with:
    source /opt/ros/humble/setup.bash
    export LD_LIBRARY_PATH="/tmp/ros2-local/opt/ros/humble/lib:$LD_LIBRARY_PATH"
    export PYTHONPATH="/tmp/ros2-local/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"
    /tmp/ros2-venv/bin/pytest tests/test_ros2_driver_ros2.py -v -p no:xdist
"""

import sys

import pytest

# Skip entire module if not on Python 3.10 (rclpy ABI mismatch)
if sys.version_info[:2] != (3, 10):
    pytest.skip(
        f"ROS2 tests require Python 3.10 (found {sys.version_info.major}.{sys.version_info.minor})",
        allow_module_level=True,
    )

# Clean up sys.modules mocks from test_mcp_server.py so real ROS2 imports work.
# Must run before ANY rosclaw or rclpy imports.
for _mod in list(sys.modules.keys()):
    if _mod.startswith(("rclpy.", "rosclaw.", "geometry_msgs", "sensor_msgs",
                        "std_msgs", "trajectory_msgs", "control_msgs",
                        "builtin_interfaces", "unique_identifier_msgs",
                        "action_msgs", "rcl_interfaces")):
        sys.modules.pop(_mod, None)
# Also remove top-level rclpy itself
sys.modules.pop("rclpy", None)

import rclpy  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from rosclaw.mcp_drivers.base import TrajectoryCommand  # noqa: E402
from rosclaw.mcp_drivers.ros2_driver import ROS2Driver  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def ros_module_context():
    """Initialize rclpy once for the entire module and clean up after."""
    if not rclpy.ok():
        rclpy.init(args=None)
    yield
    # Do NOT call rclpy.shutdown() here - it would break other test modules
    # that may share the context. Individual tests destroy their own nodes.


@pytest.fixture
def ros_driver():
    """Create a ROS2Driver with real rclpy backend and clean up."""
    driver = ROS2Driver("ur5", joint_dof=6, node_name="test_ros2_driver")
    driver.initialize()
    yield driver
    driver.stop()


class TestROS2DriverInitReal:
    def test_init_creates_ros_node(self, ros_driver):
        assert ros_driver._node is not None
        assert ros_driver._node.get_name() == "test_ros2_driver"

    def test_init_creates_publisher(self, ros_driver):
        assert ros_driver._pub_joint_cmd is not None
        # Publisher topic name should match the configured topic
        assert "joint_trajectory" in ros_driver._pub_joint_cmd.topic

    def test_init_creates_subscription(self, ros_driver):
        assert ros_driver._sub_joint_state is not None
        assert ros_driver._sub_joint_state.topic == "/joint_states"

    def test_init_sets_connected(self, ros_driver):
        assert ros_driver._driver_state.connected is True

    def test_rclpy_module_loaded(self, ros_driver):
        assert ros_driver._rclpy is not None
        assert ros_driver._rclpy.ok() is True


class TestROS2DriverStopReal:
    def test_stop_destroys_node(self, ros_driver):
        assert ros_driver._node is not None
        ros_driver.stop()
        assert ros_driver._node is None
        assert ros_driver._driver_state.connected is False

    def test_stop_without_init(self):
        driver = ROS2Driver("ur5")
        # Stop without initialize should not crash
        driver.stop()


class TestROS2DriverJointStateCallbackReal:
    def test_on_joint_state_updates_state(self, ros_driver):
        msg = JointState()
        msg.position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        msg.velocity = [0.01] * 6
        msg.effort = [0.5] * 6

        ros_driver._on_joint_state(msg)

        assert ros_driver._latest_joint_state["positions"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        assert ros_driver._driver_state.joint_positions == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        assert ros_driver._driver_state.joint_velocities == [0.01] * 6
        assert ros_driver._driver_state.joint_torques == [0.5] * 6

    def test_on_joint_state_truncates_dof(self, ros_driver):
        driver3 = ROS2Driver("panda", joint_dof=3)
        driver3.initialize()
        msg = JointState()
        msg.position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        msg.velocity = [0.01] * 6
        msg.effort = [0.5] * 6

        driver3._on_joint_state(msg)

        assert driver3._driver_state.joint_positions == [0.1, 0.2, 0.3]
        assert driver3._driver_state.joint_velocities == [0.01] * 3
        driver3.stop()


class TestROS2DriverMoveJointsReal:
    def test_move_joints_publishes_trajectory(self, ros_driver):
        result = ros_driver.move_joints([0.5] * 6, duration=2.0)
        assert result is True
        assert ros_driver._driver_state.joint_positions == [0.5] * 6

    def test_move_joints_invalid_size_raises(self, ros_driver):
        with pytest.raises(ValueError, match="Expected 6 joint positions"):
            ros_driver.move_joints([0.0] * 3)

    def test_move_joints_negative_duration_raises(self, ros_driver):
        with pytest.raises(ValueError, match="Duration must be positive"):
            ros_driver.move_joints([0.0] * 6, duration=-1.0)

    def test_move_joints_not_connected(self, ros_driver):
        ros_driver._driver_state.connected = False
        result = ros_driver.move_joints([0.0] * 6)
        assert result is False


class TestROS2DriverExecuteTrajectoryReal:
    def test_execute_trajectory_publishes(self, ros_driver):
        traj = TrajectoryCommand(
            waypoints=[[0.1] * 6, [0.2] * 6],
            times=[1.0, 2.0],
        )
        result = ros_driver.execute_trajectory(traj)
        assert result is True
        # In real ROS2 mode, execute_trajectory publishes but does not
        # update internal state (that comes from /joint_states callback)

    def test_execute_empty_waypoints_raises(self, ros_driver):
        traj = TrajectoryCommand(waypoints=[], times=[])
        with pytest.raises(ValueError, match="at least one waypoint"):
            ros_driver.execute_trajectory(traj)

    def test_execute_mismatched_lengths_raises(self, ros_driver):
        traj = TrajectoryCommand(waypoints=[[0.0] * 6], times=[1.0, 2.0])
        with pytest.raises(ValueError, match="Waypoint count"):
            ros_driver.execute_trajectory(traj)


class TestROS2DriverGettersReal:
    def test_get_joint_positions_with_data(self, ros_driver):
        ros_driver._latest_joint_state = {
            "positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "velocities": [0.01] * 6,
            "efforts": [0.5] * 6,
        }
        assert ros_driver.get_joint_positions() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def test_get_joint_positions_no_data(self, ros_driver):
        ros_driver._latest_joint_state = None
        assert ros_driver.get_joint_positions() == [0.0] * 6

    def test_get_joint_velocities_with_data(self, ros_driver):
        ros_driver._latest_joint_state = {
            "positions": [0.1] * 6,
            "velocities": [0.2] * 6,
            "efforts": [0.3] * 6,
        }
        assert ros_driver.get_joint_velocities() == [0.2] * 6

    def test_get_joint_torques_with_data(self, ros_driver):
        ros_driver._latest_joint_state = {
            "positions": [0.1] * 6,
            "velocities": [0.2] * 6,
            "efforts": [0.3] * 6,
        }
        assert ros_driver.get_joint_torques() == [0.3] * 6


class TestROS2DriverGripperReal:
    def test_set_gripper(self, ros_driver):
        result = ros_driver.set_gripper(0.75, force=0.4)
        assert result is True
        assert ros_driver._driver_state.gripper_state == 0.75


class TestROS2DriverEmergencyStopReal:
    def test_emergency_stop_sets_error(self, ros_driver):
        ros_driver.emergency_stop()
        assert ros_driver._driver_state.error_code == 99
        assert "Emergency stop" in ros_driver._driver_state.error_message


class TestROS2DriverGetStateReal:
    def test_get_state_returns_driver_state(self, ros_driver):
        state = ros_driver.get_state()
        assert state.connected is True
