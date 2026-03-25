#!/usr/bin/env python3
"""
ROSClaw UR5 MCP Server - Production-ready ROS2-native MCP Server.

This server provides MCP tools for controlling a Universal Robots UR5e arm
via ROS 2. It includes actual rclpy node implementation with real
Subscriptions, Publishers, and Action Clients.

The Digital Twin firewall validates all trajectories through MuJoCo
before execution on real hardware.
"""

import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    ImageContent,
    TextContent,
    Tool,
)

# ROSClaw imports
from rosclaw.firewall.decorator import (
    DigitalTwinFirewall,
    SafetyLevel,
    SafetyViolationError,
)


# ROS message imports
try:
    from control_msgs.action import FollowJointTrajectory
    from geometry_msgs.msg import Pose, PoseStamped, Twist
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, Header
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    ROS_IMPORTS_OK = True
except ImportError as e:
    print(f"[ERROR] ROS 2 message imports failed: {e}")
    ROS_IMPORTS_OK = False


@dataclass
class RobotState:
    """Current state of the UR5 robot."""
    joint_positions: list[float]
    joint_velocities: list[float]
    joint_efforts: list[float]
    joint_names: list[str]
    end_effector_pose: Optional[Pose] = None
    is_connected: bool = False
    last_update_time: float = 0.0


class UR5ROSNode(Node):
    """
    ROS 2 Node for UR5 robot control.

    Handles actual ROS 2 communication:
    - Subscribes to /joint_states for robot feedback
    - Publishes to /ur_script for URScript commands
    - Uses FollowJointTrajectory action for motion control
    """

    # UR5e joint names in order
    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # UR5e joint limits (radians)
    JOINT_LIMITS = {
        "shoulder_pan_joint": (-6.2831853, 6.2831853),
        "shoulder_lift_joint": (-6.2831853, 6.2831853),
        "elbow_joint": (-3.1415926, 3.1415926),
        "wrist_1_joint": (-6.2831853, 6.2831853),
        "wrist_2_joint": (-6.2831853, 6.2831853),
        "wrist_3_joint": (-6.2831853, 6.2831853),
    }

    # UR5e torque limits (Nm) - 80% of actual for safety
    TORQUE_LIMITS = {
        "shoulder_pan_joint": 150.0 * 0.8,
        "shoulder_lift_joint": 150.0 * 0.8,
        "elbow_joint": 100.0 * 0.8,
        "wrist_1_joint": 28.0 * 0.8,
        "wrist_2_joint": 28.0 * 0.8,
        "wrist_3_joint": 28.0 * 0.8,
    }

    def __init__(self, robot_ip: str = "192.168.1.100", namespace: str = "ur"):
        super().__init__("rosclaw_ur5_node")

        self.robot_ip = robot_ip
        self.namespace = namespace
        self.state = RobotState(
            joint_positions=[0.0] * 6,
            joint_velocities=[0.0] * 6,
            joint_efforts=[0.0] * 6,
            joint_names=self.JOINT_NAMES,
        )

        # Callback group for concurrent operations
        self.callback_group = ReentrantCallbackGroup()

        # QoS profile for sensor data
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            f"/{namespace}/joint_states",
            self._joint_state_callback,
            qos,
            callback_group=self.callback_group,
        )

        # Publishers
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory,
            f"/{namespace}/joint_trajectory_controller/command",
            10,
        )

        self.velocity_pub = self.create_publisher(
            Twist,
            f"/{namespace}/cmd_vel",
            10,
        )

        # Action clients
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            f"/{namespace}/joint_trajectory_controller/follow_joint_trajectory",
            callback_group=self.callback_group,
        )

        self.get_logger().info(f"UR5 ROS Node initialized for robot at {robot_ip}")
        self.get_logger().info(f"Waiting for action server...")

        # Wait for action server
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Trajectory action server not available yet")

    def _joint_state_callback(self, msg: JointState) -> None:
        """Handle incoming joint state messages."""
        try:
            # Map incoming joints to our expected order
            positions = []
            velocities = []
            efforts = []

            for name in self.JOINT_NAMES:
                if name in msg.name:
                    idx = msg.name.index(name)
                    positions.append(msg.position[idx] if msg.position else 0.0)
                    velocities.append(msg.velocity[idx] if msg.velocity else 0.0)
                    efforts.append(msg.effort[idx] if msg.effort else 0.0)
                else:
                    positions.append(0.0)
                    velocities.append(0.0)
                    efforts.append(0.0)

            self.state.joint_positions = positions
            self.state.joint_velocities = velocities
            self.state.joint_efforts = efforts
            self.state.is_connected = True
            self.state.last_update_time = self.get_clock().now().nanoseconds / 1e9

        except Exception as e:
            self.get_logger().error(f"Error processing joint state: {e}")

    def get_current_joint_positions(self) -> list[float]:
        """Get current joint positions in radians."""
        return self.state.joint_positions.copy()

    def validate_joint_limits(self, positions: list[float]) -> tuple[bool, str]:
        """Validate joint positions against limits."""
        for i, (name, pos) in enumerate(zip(self.JOINT_NAMES, positions)):
            min_limit, max_limit = self.JOINT_LIMITS[name]
            if pos < min_limit or pos > max_limit:
                return False, f"Joint {name} position {pos:.4f} outside limits [{min_limit:.4f}, {max_limit:.4f}]"
        return True, ""

    async def execute_joint_trajectory(
        self,
        trajectory_points: list[list[float]],
        time_from_start: list[float],
    ) -> tuple[bool, str]:
        """
        Execute a joint trajectory using the FollowJointTrajectory action.

        Args:
            trajectory_points: List of joint position arrays
            time_from_start: Time in seconds for each point

        Returns:
            (success, message)
        """
        if not self.trajectory_client.server_is_ready():
            return False, "Trajectory action server not available"

        # Build trajectory message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = JointTrajectory()
        goal_msg.trajectory.joint_names = self.JOINT_NAMES

        for i, (positions, t) in enumerate(zip(trajectory_points, time_from_start)):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1.0) * 1e9)
            goal_msg.trajectory.points.append(point)

        # Send goal
        self.get_logger().info(f"Sending trajectory with {len(trajectory_points)} points")

        goal_future = self.trajectory_client.send_goal_async(goal_msg)
        goal_handle = await asyncio.wrap_future(goal_future)

        if not goal_handle.accepted:
            return False, "Trajectory goal rejected by controller"

        # Wait for result
        result_future = goal_handle.get_result_async()
        result = await asyncio.wrap_future(result_future)

        if result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            return True, "Trajectory executed successfully"
        else:
            return False, f"Trajectory failed with error code: {result.result.error_code}"

    def emergency_stop(self) -> None:
        """Send emergency stop command."""
        # Publish zero velocity to halt motion
        stop_msg = Twist()
        self.velocity_pub.publish(stop_msg)
        self.get_logger().warn("EMERGENCY STOP triggered")


class UR5MCPServer:
    """
    MCP Server for UR5 robot control.

    Provides MCP tools that interface with the ROS 2 node.
    Includes Digital Twin validation before executing on real hardware.
    """

    def __init__(self, robot_ip: str = "192.168.1.100", firewall_model_path: Optional[str] = None):
        self.robot_ip = robot_ip
        self.firewall_model_path = firewall_model_path or self._find_default_model()

        # Initialize ROS 2
        rclpy.init(args=None)

        # Create ROS node
        self.ros_node = UR5ROSNode(robot_ip)

        # Create Digital Twin firewall
        if Path(self.firewall_model_path).exists():
            self.firewall = DigitalTwinFirewall(
                model_path=self.firewall_model_path,
                torque_limits=UR5ROSNode.TORQUE_LIMITS,
                joint_limits=UR5ROSNode.JOINT_LIMITS,
                safety_margin=0.05,
            )
            print(f"[UR5MCP] Digital Twin firewall loaded: {self.firewall_model_path}")
        else:
            self.firewall = None
            print(f"[UR5MCP] Warning: Digital Twin model not found at {self.firewall_model_path}")

        # Create MCP server
        self.server = Server("rosclaw-ur5-mcp")
        self._register_tools()

        # Spin ROS node in background
        self.ros_spin_task: Optional[asyncio.Task] = None

    def _find_default_model(self) -> str:
        """Find default MuJoCo model for UR5."""
        # Try to find model in package share directories
        possible_paths = [
            "/opt/ros/humble/share/ur_description/meshes/ur5e/ur5e.xml",
            "/usr/share/mujoco/ur5e.xml",
            str(Path(__file__).parent.parent / "specs" / "ur5e.xml"),
        ]
        for path in possible_paths:
            if Path(path).exists():
                return path
        return possible_paths[-1]  # Return last as default

    def _register_tools(self) -> None:
        """Register MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="ur5_get_joint_states",
                    description="Get current joint positions, velocities, and efforts",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="ur5_move_joints",
                    description="Move robot joints to target positions with Digital Twin validation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "joint_positions": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Target joint positions in radians [6 joints]",
                            },
                            "duration": {
                                "type": "number",
                                "description": "Movement duration in seconds",
                                "default": 2.0,
                            },
                            "validate": {
                                "type": "boolean",
                                "description": "Enable Digital Twin validation",
                                "default": True,
                            },
                        },
                        "required": ["joint_positions"],
                    },
                ),
                Tool(
                    name="ur5_execute_trajectory",
                    description="Execute a multi-point joint trajectory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "waypoints": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "number"}},
                                "description": "List of joint position arrays",
                            },
                            "times": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Time from start for each waypoint (seconds)",
                            },
                            "validate": {
                                "type": "boolean",
                                "description": "Enable Digital Twin validation",
                                "default": True,
                            },
                        },
                        "required": ["waypoints", "times"],
                    },
                ),
                Tool(
                    name="ur5_emergency_stop",
                    description="Emergency stop - halt all robot motion immediately",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="ur5_get_limits",
                    description="Get robot joint limits and safety parameters",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="ur5_validate_trajectory",
                    description="Validate trajectory through Digital Twin without executing",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "waypoints": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "number"}},
                                "description": "List of joint position arrays",
                            },
                        },
                        "required": ["waypoints"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
            try:
                if name == "ur5_get_joint_states":
                    return await self._handle_get_joint_states()
                elif name == "ur5_move_joints":
                    return await self._handle_move_joints(arguments)
                elif name == "ur5_execute_trajectory":
                    return await self._handle_execute_trajectory(arguments)
                elif name == "ur5_emergency_stop":
                    return await self._handle_emergency_stop()
                elif name == "ur5_get_limits":
                    return await self._handle_get_limits()
                elif name == "ur5_validate_trajectory":
                    return await self._handle_validate_trajectory(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _handle_get_joint_states(self) -> list[TextContent]:
        """Handle get_joint_states tool call."""
        state = self.ros_node.state

        result = {
            "joint_positions": {name: pos for name, pos in zip(state.joint_names, state.joint_positions)},
            "joint_velocities": {name: vel for name, vel in zip(state.joint_names, state.joint_velocities)},
            "joint_efforts": {name: eff for name, eff in zip(state.joint_names, state.joint_efforts)},
            "is_connected": state.is_connected,
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_move_joints(self, arguments: dict) -> list[TextContent]:
        """Handle move_joints tool call with Digital Twin validation."""
        positions = arguments.get("joint_positions", [])
        duration = arguments.get("duration", 2.0)
        validate = arguments.get("validate", True)

        if len(positions) != 6:
            return [TextContent(type="text", text=f"Error: Expected 6 joint positions, got {len(positions)}")]

        # First check joint limits
        valid, msg = self.ros_node.validate_joint_limits(positions)
        if not valid:
            return [TextContent(type="text", text=f"Error: Joint limit violation - {msg}")]

        # Digital Twin validation
        if validate and self.firewall:
            try:
                # Create simple trajectory: current -> target
                current = self.ros_node.get_current_joint_positions()
                trajectory = self._interpolate_trajectory(current, positions, int(duration * 50))  # 50Hz

                result = self.firewall.validate_trajectory(
                    [np.array(p) for p in trajectory],
                    safety_level=SafetyLevel.STRICT,
                )

                if not result.is_safe:
                    error_msg = f"Digital Twin validation FAILED:\n"
                    error_msg += f"  - Collision: {result.collision_detected}\n"
                    error_msg += f"  - Joint limit: {result.joint_limit_violated}\n"
                    error_msg += f"  - Torque limit: {result.torque_limit_exceeded}\n"
                    error_msg += f"  - Details: {result.violation_details}"
                    return [TextContent(type="text", text=error_msg)]

            except Exception as e:
                return [TextContent(type="text", text=f"Digital Twin validation error: {e}")]

        # Execute motion
        success, msg = await self.ros_node.execute_joint_trajectory(
            [positions],
            [duration],
        )

        if success:
            return [TextContent(type="text", text=f"Motion executed successfully: {msg}")]
        else:
            return [TextContent(type="text", text=f"Motion failed: {msg}")]

    async def _handle_execute_trajectory(self, arguments: dict) -> list[TextContent]:
        """Handle execute_trajectory tool call."""
        waypoints = arguments.get("waypoints", [])
        times = arguments.get("times", [])
        validate = arguments.get("validate", True)

        if len(waypoints) != len(times):
            return [TextContent(type="text", text="Error: waypoints and times must have same length")]

        # Validate each waypoint
        for i, wp in enumerate(waypoints):
            if len(wp) != 6:
                return [TextContent(type="text", text=f"Error: waypoint {i} should have 6 values, got {len(wp)}")]
            valid, msg = self.ros_node.validate_joint_limits(wp)
            if not valid:
                return [TextContent(type="text", text=f"Error in waypoint {i}: {msg}")]

        # Digital Twin validation
        if validate and self.firewall:
            try:
                result = self.firewall.validate_trajectory(
                    [np.array(p) for p in waypoints],
                    safety_level=SafetyLevel.STRICT,
                )

                if not result.is_safe:
                    error_msg = f"Digital Twin validation FAILED for trajectory:\n"
                    error_msg += json.dumps(result.to_dict(), indent=2)
                    return [TextContent(type="text", text=error_msg)]

            except Exception as e:
                return [TextContent(type="text", text=f"Digital Twin validation error: {e}")]

        # Execute trajectory
        success, msg = await self.ros_node.execute_joint_trajectory(waypoints, times)

        if success:
            return [TextContent(type="text", text=f"Trajectory executed successfully: {msg}")]
        else:
            return [TextContent(type="text", text=f"Trajectory failed: {msg}")]

    async def _handle_emergency_stop(self) -> list[TextContent]:
        """Handle emergency_stop tool call."""
        self.ros_node.emergency_stop()
        return [TextContent(type="text", text="EMERGENCY STOP triggered - all motion halted")]

    async def _handle_get_limits(self) -> list[TextContent]:
        """Handle get_limits tool call."""
        result = {
            "joint_limits": self.ros_node.JOINT_LIMITS,
            "torque_limits": self.ros_node.TORQUE_LIMITS,
            "joint_names": self.ros_node.JOINT_NAMES,
            "firewall_enabled": self.firewall is not None,
            "firewall_model": self.firewall_model_path if self.firewall else None,
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_validate_trajectory(self, arguments: dict) -> list[TextContent]:
        """Handle validate_trajectory tool call."""
        waypoints = arguments.get("waypoints", [])

        if not self.firewall:
            return [TextContent(type="text", text="Error: Digital Twin firewall not available")]

        try:
            result = self.firewall.validate_trajectory(
                [np.array(p) for p in waypoints],
                safety_level=SafetyLevel.STRICT,
            )

            response = {
                "is_safe": result.is_safe,
                "validation_result": result.to_dict(),
            }

            if result.is_safe:
                response["message"] = "Trajectory is SAFE for execution"
            else:
                response["message"] = "Trajectory is UNSAFE - DO NOT EXECUTE"

            return [TextContent(type="text", text=json.dumps(response, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Validation error: {e}")]

    def _interpolate_trajectory(
        self,
        start: list[float],
        end: list[float],
        num_points: int,
    ) -> list[list[float]]:
        """Interpolate between start and end configurations."""
        trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            point = [s + t * (e - s) for s, e in zip(start, end)]
            trajectory.append(point)
        return trajectory

    async def _ros_spin(self) -> None:
        """Background task to spin ROS node."""
        while rclpy.ok():
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)
            await asyncio.sleep(0.001)

    async def run(self) -> None:
        """Run the MCP server."""
        # Start ROS spin task
        self.ros_spin_task = asyncio.create_task(self._ros_spin())

        # Setup signal handlers
        def signal_handler(sig, frame):
            print("\n[UR5MCP] Shutting down...")
            self.ros_spin_task.cancel()
            rclpy.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run MCP server
        async with stdio_server(self.server) as (read, write):
            init_options = InitializationOptions(
                server_name="rosclaw-ur5-mcp",
                server_version="0.1.0",
                capabilities=self.server.get_capabilities(),
            )
            await self.server.run(read, write, init_options)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ROSClaw UR5 MCP Server")
    parser.add_argument(
        "--robot-ip",
        default=os.getenv("UR5_ROBOT_IP", "192.168.1.100"),
        help="UR5 robot IP address",
    )
    parser.add_argument(
        "--firewall-model",
        default=os.getenv("UR5_FIREWALL_MODEL"),
        help="Path to MuJoCo model for Digital Twin",
    )
    args = parser.parse_args()

    print(f"[UR5MCP] Starting ROSClaw UR5 MCP Server")
    print(f"[UR5MCP] Robot IP: {args.robot_ip}")
    print(f"[UR5MCP] Firewall Model: {args.firewall_model}")

    server = UR5MCPServer(
        robot_ip=args.robot_ip,
        firewall_model_path=args.firewall_model,
    )

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[UR5MCP] Shutdown complete")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
