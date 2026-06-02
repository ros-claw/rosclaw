"""ROS2-native tests for ur5_server.py - runs in real rclpy environment.

These tests require a ROS2 environment with rclpy and message packages installed.
Run with:
    source /opt/ros/humble/setup.bash
    export LD_LIBRARY_PATH="/tmp/ros2-local/opt/ros/humble/lib:$LD_LIBRARY_PATH"
    export PYTHONPATH="/tmp/ros2-local/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"
    /tmp/ros2-venv/bin/pytest tests/test_ur5_server_ros2.py -v -p no:xdist
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

import asyncio
from pathlib import Path
from unittest.mock import patch

import numpy as np

import rclpy
from sensor_msgs.msg import JointState

from rosclaw.mcp.ur5_server import (
    RCLPY_AVAILABLE,
    ROS_IMPORTS_OK,
    RobotState,
    UR5MCPServer,
    UR5ROSNode,
    main,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def ros_context():
    """Initialize and shutdown rclpy for each test."""
    needs_shutdown = False
    if not rclpy.ok():
        rclpy.init()
        needs_shutdown = True
    yield
    if needs_shutdown and rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def ros_node(ros_context):
    """Create a UR5ROSNode and clean it up."""
    node = UR5ROSNode(robot_ip="127.0.0.1", namespace="test_ns")
    yield node
    node.destroy_node()


# ------------------------------------------------------------------
# Import flags
# ------------------------------------------------------------------


class TestImportFlags:
    def test_rclpy_available(self):
        assert RCLPY_AVAILABLE is True

    def test_ros_imports_ok(self):
        assert ROS_IMPORTS_OK is True


# ------------------------------------------------------------------
# RobotState
# ------------------------------------------------------------------


class TestRobotState:
    def test_default_state(self):
        state = RobotState(
            joint_positions=[0.0] * 6,
            joint_velocities=[0.0] * 6,
            joint_efforts=[0.0] * 6,
            joint_names=UR5ROSNode.JOINT_NAMES,
        )
        assert state.is_connected is False
        assert state.last_update_time == 0.0
        assert len(state.joint_positions) == 6

    def test_state_with_pose(self):
        from geometry_msgs.msg import Pose

        state = RobotState(
            joint_positions=[0.1] * 6,
            joint_velocities=[0.0] * 6,
            joint_efforts=[0.0] * 6,
            joint_names=UR5ROSNode.JOINT_NAMES,
            end_effector_pose=Pose(),
            is_connected=True,
            last_update_time=123.0,
        )
        assert state.is_connected is True
        assert state.last_update_time == 123.0


# ------------------------------------------------------------------
# UR5ROSNode
# ------------------------------------------------------------------


class TestUR5ROSNodeInit:
    def test_node_creation(self, ros_context):
        node = UR5ROSNode(robot_ip="192.168.1.50", namespace="ur_test")
        assert node.robot_ip == "192.168.1.50"
        assert node.namespace == "ur_test"
        assert node.get_name() == "rosclaw_ur5_node"
        node.destroy_node()

    def test_default_namespace(self, ros_context):
        node = UR5ROSNode(robot_ip="127.0.0.1")
        assert node.namespace == "ur"
        node.destroy_node()

    def test_initial_state(self, ros_context):
        node = UR5ROSNode(robot_ip="127.0.0.1")
        assert node.state.joint_positions == [0.0] * 6
        assert node.state.joint_velocities == [0.0] * 6
        assert node.state.joint_efforts == [0.0] * 6
        assert node.state.joint_names == UR5ROSNode.JOINT_NAMES
        node.destroy_node()

    def test_callback_group_created(self, ros_context):
        node = UR5ROSNode(robot_ip="127.0.0.1")
        assert node.callback_group is not None
        node.destroy_node()

    def test_subscriber_created(self, ros_context):
        node = UR5ROSNode(robot_ip="127.0.0.1")
        assert node.joint_state_sub is not None
        node.destroy_node()

    def test_publishers_created(self, ros_context):
        node = UR5ROSNode(robot_ip="127.0.0.1")
        assert node.joint_trajectory_pub is not None
        assert node.velocity_pub is not None
        node.destroy_node()

    def test_action_client_created(self, ros_context):
        node = UR5ROSNode(robot_ip="127.0.0.1")
        assert node.trajectory_client is not None
        node.destroy_node()


class TestUR5ROSNodeJointStateCallback:
    def test_callback_updates_state(self, ros_node):
        msg = JointState()
        msg.name = UR5ROSNode.JOINT_NAMES
        msg.position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        msg.velocity = [0.01] * 6
        msg.effort = [0.5] * 6

        ros_node._joint_state_callback(msg)

        assert ros_node.state.joint_positions == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        assert ros_node.state.joint_velocities == [0.01] * 6
        assert ros_node.state.joint_efforts == [0.5] * 6
        assert ros_node.state.is_connected is True
        assert ros_node.state.last_update_time > 0

    def test_callback_partial_joints(self, ros_node):
        msg = JointState()
        msg.name = ["shoulder_pan_joint", "elbow_joint"]
        msg.position = [0.5, 0.6]
        msg.velocity = [0.1, 0.2]
        msg.effort = [1.0, 2.0]

        ros_node._joint_state_callback(msg)

        assert ros_node.state.joint_positions[0] == 0.5  # shoulder_pan
        assert ros_node.state.joint_positions[1] == 0.0  # missing -> 0.0
        assert ros_node.state.joint_positions[2] == 0.6  # elbow

    def test_callback_empty_message(self, ros_node):
        msg = JointState()
        msg.name = []

        ros_node._joint_state_callback(msg)

        assert ros_node.state.joint_positions == [0.0] * 6
        assert ros_node.state.is_connected is True

    def test_callback_none_fields(self, ros_node):
        msg = JointState()
        msg.name = UR5ROSNode.JOINT_NAMES
        # position/velocity/effort are None by default in some contexts
        ros_node._joint_state_callback(msg)
        assert ros_node.state.is_connected is True

    def test_callback_error_handling(self, ros_node):
        # ROS2 logger bypasses standard Python logging; just verify no crash
        msg = "not a joint state"
        ros_node._joint_state_callback(msg)  # should not raise


class TestUR5ROSNodeGetPositions:
    def test_get_current_joint_positions(self, ros_node):
        ros_node.state.joint_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        positions = ros_node.get_current_joint_positions()
        assert positions == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def test_get_positions_returns_copy(self, ros_node):
        ros_node.state.joint_positions = [0.1] * 6
        positions = ros_node.get_current_joint_positions()
        positions[0] = 99.0
        assert ros_node.state.joint_positions[0] == 0.1


class TestUR5ROSNodeValidateLimits:
    def test_valid_positions(self, ros_node):
        valid, msg = ros_node.validate_joint_limits([0.0] * 6)
        assert valid is True
        assert msg == ""

    def test_position_below_min(self, ros_node):
        valid, msg = ros_node.validate_joint_limits([-10.0, 0, 0, 0, 0, 0])
        assert valid is False
        assert "shoulder_pan_joint" in msg
        assert "outside limits" in msg

    def test_position_above_max(self, ros_node):
        valid, msg = ros_node.validate_joint_limits([10.0, 0, 0, 0, 0, 0])
        assert valid is False
        assert "shoulder_pan_joint" in msg

    def test_elbow_limit(self, ros_node):
        # Elbow has tighter limits (-pi, pi)
        valid, msg = ros_node.validate_joint_limits([0, 0, 4.0, 0, 0, 0])
        assert valid is False
        assert "elbow_joint" in msg

    def test_all_joints_valid(self, ros_node):
        positions = [1.0, -1.0, 2.0, -2.0, 1.5, -1.5]
        valid, msg = ros_node.validate_joint_limits(positions)
        assert valid is True


class TestUR5ROSNodeEmergencyStop:
    def test_emergency_stop_publishes(self, ros_node):
        # Just verify it doesn't crash
        ros_node.emergency_stop()

    def test_emergency_stop_twist_message(self, ros_node):
        ros_node.emergency_stop()
        # The Twist message has zero velocity by default


class TestUR5ROSNodeExecuteTrajectory:
    @pytest.mark.asyncio
    async def test_execute_server_not_ready(self, ros_node):
        # Action server not available - should fail fast
        success, msg = await ros_node.execute_joint_trajectory(
            [[0.0] * 6],
            [1.0],
        )
        assert success is False
        assert "not available" in msg


# ------------------------------------------------------------------
# UR5MCPServer
# ------------------------------------------------------------------


class TestUR5MCPServerInit:
    def test_server_creation_with_firewall(self, ros_context):
        model_path = str(
            Path(__file__).parent.parent
            / "src"  # noqa: W503
            / "rosclaw"  # noqa: W503
            / "specs"  # noqa: W503
            / "ur5e.xml"  # noqa: W503
        )
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path=model_path,
        )
        assert server.robot_ip == "127.0.0.1"
        assert server.firewall is not None
        assert server.ros_node is not None
        assert server.server is not None
        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def test_server_creation_without_firewall(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )
        assert server.firewall is None
        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def test_find_default_model_exists(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )
        path = server._find_default_model()
        # Should return the last fallback path
        assert "ur5e.xml" in path
        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class TestUR5MCPServerInterpolate:
    def test_interpolate_two_points(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )
        traj = server._interpolate_trajectory([0.0] * 6, [1.0] * 6, 3)
        assert len(traj) == 3
        assert traj[0] == [0.0] * 6
        assert traj[2] == [1.0] * 6
        assert traj[1] == [0.5] * 6
        server.ros_node.destroy_node()

    def test_interpolate_single_point(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )
        traj = server._interpolate_trajectory([0.0] * 6, [1.0] * 6, 1)
        assert len(traj) == 1
        assert traj[0] == [0.0] * 6
        server.ros_node.destroy_node()

    def test_interpolate_different_values(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )
        traj = server._interpolate_trajectory([0.0, 1.0], [2.0, 5.0], 3)
        assert traj[0] == [0.0, 1.0]
        assert traj[1] == [1.0, 3.0]
        assert traj[2] == [2.0, 5.0]
        server.ros_node.destroy_node()


class TestUR5MCPServerHandleGetJointStates:
    @pytest.mark.asyncio
    async def test_handle_get_joint_states(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )
        server.ros_node.state.joint_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        server.ros_node.state.is_connected = True

        result = await server._handle_get_joint_states()
        assert len(result) == 1
        text = result[0].text
        assert "joint_positions" in text
        assert "0.1" in text
        assert "is_connected" in text

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class TestUR5MCPServerHandleGetLimits:
    @pytest.mark.asyncio
    async def test_handle_get_limits_with_firewall(self, ros_context):
        model_path = str(
            Path(__file__).parent.parent
            / "src"  # noqa: W503
            / "rosclaw"  # noqa: W503
            / "specs"  # noqa: W503
            / "ur5e.xml"  # noqa: W503
        )
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path=model_path,
        )

        result = await server._handle_get_limits()
        assert len(result) == 1
        text = result[0].text
        assert "joint_limits" in text
        assert "torque_limits" in text
        assert "firewall_enabled" in text
        assert "true" in text.lower() or "True" in text

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    @pytest.mark.asyncio
    async def test_handle_get_limits_without_firewall(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )

        result = await server._handle_get_limits()
        assert len(result) == 1
        text = result[0].text
        assert "firewall_enabled" in text
        assert "false" in text.lower() or "None" in text

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class TestUR5MCPServerHandleValidateTrajectory:
    @pytest.mark.asyncio
    async def test_validate_trajectory_no_firewall(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )

        result = await server._handle_validate_trajectory({
            "waypoints": [[0.0] * 6],
        })
        assert len(result) == 1
        assert "firewall not available" in result[0].text

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    @pytest.mark.asyncio
    async def test_validate_safe_trajectory(self, ros_context):
        model_path = str(
            Path(__file__).parent.parent
            / "src"  # noqa: W503
            / "rosclaw"  # noqa: W503
            / "specs"  # noqa: W503
            / "ur5e.xml"  # noqa: W503
        )
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path=model_path,
        )

        result = await server._handle_validate_trajectory({
            "waypoints": [[0.0] * 6, [0.1] * 6],
        })
        assert len(result) == 1
        text = result[0].text
        response = __import__("json").loads(text)
        assert "is_safe" in response

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class TestUR5MCPServerHandleEmergencyStop:
    @pytest.mark.asyncio
    async def test_handle_emergency_stop(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )

        result = await server._handle_emergency_stop()
        assert len(result) == 1
        assert "EMERGENCY STOP" in result[0].text

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class TestUR5MCPServerHandleMoveJoints:
    @pytest.mark.asyncio
    async def test_move_joints_wrong_count(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )

        result = await server._handle_move_joints({
            "joint_positions": [0.0] * 3,
        })
        assert "Expected 6" in result[0].text

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    @pytest.mark.asyncio
    async def test_move_joints_limit_violation(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )

        result = await server._handle_move_joints({
            "joint_positions": [100.0] * 6,
        })
        assert "Joint limit violation" in result[0].text

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    @pytest.mark.asyncio
    async def test_move_joints_valid_no_execution(self, ros_context):
        # Valid positions but action server not available
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )

        result = await server._handle_move_joints({
            "joint_positions": [0.0] * 6,
            "validate": False,
        })
        # Should attempt to execute but server not available
        assert "not available" in result[0].text or "failed" in result[0].text.lower()

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class TestUR5MCPServerHandleExecuteTrajectory:
    @pytest.mark.asyncio
    async def test_execute_mismatched_lengths(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )

        result = await server._handle_execute_trajectory({
            "waypoints": [[0.0] * 6],
            "times": [1.0, 2.0],
        })
        assert "same length" in result[0].text

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    @pytest.mark.asyncio
    async def test_execute_wrong_waypoint_size(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )

        result = await server._handle_execute_trajectory({
            "waypoints": [[0.0] * 3],
            "times": [1.0],
        })
        assert "should have 6 values" in result[0].text

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    @pytest.mark.asyncio
    async def test_execute_limit_violation(self, ros_context):
        server = UR5MCPServer(
            robot_ip="127.0.0.1",
            firewall_model_path="/nonexistent/model.xml",
        )

        result = await server._handle_execute_trajectory({
            "waypoints": [[100.0] * 6],
            "times": [1.0],
        })
        assert "Error in waypoint" in result[0].text

        server.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ------------------------------------------------------------------
# main()
# ------------------------------------------------------------------


class TestMain:
    @patch("rosclaw.mcp.ur5_server.UR5MCPServer")
    @patch("rosclaw.mcp.ur5_server.asyncio.run")
    def test_main_default_args(self, mock_run, mock_server_cls):
        with patch.object(sys, "argv", ["ur5_server"]):
            main()

        mock_server_cls.assert_called_once()
        args = mock_server_cls.call_args
        assert args.kwargs["robot_ip"] == "192.168.1.100"

    @patch("rosclaw.mcp.ur5_server.UR5MCPServer")
    @patch("rosclaw.mcp.ur5_server.asyncio.run")
    def test_main_custom_ip(self, mock_run, mock_server_cls):
        with patch.object(sys, "argv", ["ur5_server", "--robot-ip", "10.0.0.1"]):
            main()

        args = mock_server_cls.call_args
        assert args.kwargs["robot_ip"] == "10.0.0.1"

    @patch("rosclaw.mcp.ur5_server.UR5MCPServer")
    @patch("rosclaw.mcp.ur5_server.asyncio.run")
    def test_main_custom_model(self, mock_run, mock_server_cls):
        with patch.object(sys, "argv", [
            "ur5_server",
            "--firewall-model",
            "/path/to/model.xml",
        ]):
            main()

        args = mock_server_cls.call_args
        assert args.kwargs["firewall_model_path"] == "/path/to/model.xml"


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------


class TestConstants:
    def test_joint_names_count(self):
        assert len(UR5ROSNode.JOINT_NAMES) == 6

    def test_joint_limits_keys(self):
        assert set(UR5ROSNode.JOINT_LIMITS.keys()) == set(UR5ROSNode.JOINT_NAMES)

    def test_torque_limits_keys(self):
        assert set(UR5ROSNode.TORQUE_LIMITS.keys()) == set(UR5ROSNode.JOINT_NAMES)

    def test_joint_limits_ranges(self):
        for name, (min_val, max_val) in UR5ROSNode.JOINT_LIMITS.items():
            assert min_val < max_val

    def test_torque_limits_positive(self):
        for name, limit in UR5ROSNode.TORQUE_LIMITS.items():
            assert limit > 0
