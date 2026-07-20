#!/usr/bin/env python3
"""Legacy ROS 2 MCP surface for observing and validating a UR5 robot.

The ROS node is observation-only. Direct motion is disabled, and emergency
requests use the independent rosclawd boundary.
"""

import asyncio
import json
import os
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    # Stub classes for import-time compatibility when ROS 2 is not installed

    class Node:
        def __init__(self, *args, **kwargs):
            pass

    class ReentrantCallbackGroup:
        pass

    class QoSProfile:
        def __init__(self, *args, **kwargs):
            pass

    class ReliabilityPolicy:
        BEST_EFFORT = None


# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    ImageContent,
    TextContent,
    Tool,
)

# ROSClaw imports
from rosclaw.daemon.client import DaemonClient, DaemonClientError
from rosclaw.firewall.decorator import (
    DigitalTwinFirewall,
    SafetyLevel,
)

# ROS message imports
ROS_IMPORT_ERROR: str | None = None
try:
    from geometry_msgs.msg import Pose
    from sensor_msgs.msg import JointState

    ROS_IMPORTS_OK = True
except ImportError as e:
    ROS_IMPORTS_OK = False
    ROS_IMPORT_ERROR = str(e)
    # Stubs for type annotations when ROS 2 is not installed

    class Pose:
        pass

    class JointState:
        pass


@dataclass
class RobotState:
    """Current state of the UR5 robot."""

    joint_positions: list[float]
    joint_velocities: list[float]
    joint_efforts: list[float]
    joint_names: list[str]
    end_effector_pose: Pose | None = None
    is_connected: bool = False
    last_update_time: float = 0.0


class UR5ROSNode(Node):
    """
    Observation-only ROS 2 Node for UR5 state.

    Handles read-only ROS 2 communication:
    - Subscribes to /joint_states for robot feedback

    Command publishers and action clients deliberately do not exist in this
    Agent-facing compatibility process.
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
        self._state_lock = threading.Lock()
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

        self.get_logger().info(f"UR5 observation-only ROS Node initialized for robot at {robot_ip}")

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

            with self._state_lock:
                self.state.joint_positions = positions
                self.state.joint_velocities = velocities
                self.state.joint_efforts = efforts
                self.state.is_connected = True
                self.state.last_update_time = self.get_clock().now().nanoseconds / 1e9

        except Exception as e:
            self.get_logger().error(f"Error processing joint state: {e}")

    def get_current_joint_positions(self) -> list[float]:
        """Get current joint positions in radians."""
        with self._state_lock:
            return self.state.joint_positions.copy()

    def validate_joint_limits(self, positions: list[float]) -> tuple[bool, str]:
        """Validate joint positions against limits."""
        for _i, (name, pos) in enumerate(zip(self.JOINT_NAMES, positions, strict=False)):
            min_limit, max_limit = self.JOINT_LIMITS[name]
            if pos < min_limit or pos > max_limit:
                return (
                    False,
                    f"Joint {name} position {pos:.4f} outside limits [{min_limit:.4f}, {max_limit:.4f}]",
                )
        return True, ""

    async def execute_joint_trajectory(
        self,
        trajectory_points: list[list[float]],
        time_from_start: list[float],
    ) -> tuple[bool, str]:
        """Fail closed; this observation node owns no command primitive."""
        del trajectory_points, time_from_start
        return False, "Direct trajectory execution is unavailable; use rosclawd request_action"

    def emergency_stop(self) -> dict[str, object]:
        """Fail closed; emergency requests belong to the rosclawd client."""
        return {
            "request_dispatched": False,
            "driver_acknowledged": False,
            "physical_stop_observed": False,
            "stopped": False,
            "error_code": "ROSCLAWD_REQUIRED",
            "evidence": [],
        }


class UR5MCPServer:
    """Legacy UR5 MCP server with direct physical execution fail-closed."""

    def __init__(
        self,
        robot_ip: str = "192.168.1.100",
        firewall_model_path: str | None = None,
        daemon_client: DaemonClient | None = None,
    ):
        if not RCLPY_AVAILABLE:
            raise RuntimeError(
                "ROS 2 rclpy is not installed. Install ROS 2 to use the UR5 MCP server."
            )
        if not ROS_IMPORTS_OK:
            raise RuntimeError(
                f"ROS 2 message imports failed: {ROS_IMPORT_ERROR}. "
                f"Install the missing ROS 2 message packages to use the UR5 MCP server."
            )

        self.robot_ip = robot_ip
        self.firewall_model_path = firewall_model_path or self._find_default_model()
        self._daemon_client = daemon_client or DaemonClient()

        # Initialize ROS 2
        if not rclpy.ok():
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
        self.ros_spin_task: asyncio.Task | None = None

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
                    description=(
                        "Legacy move interface; direct execution is blocked. Use canonical "
                        "request_action through rosclawd."
                    ),
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
                    description=(
                        "Legacy trajectory interface; direct execution is blocked. Use canonical "
                        "request_action through rosclawd."
                    ),
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
                    description=(
                        "Request emergency stop through rosclawd; physical stop remains "
                        "unverified without independent observation"
                    ),
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
        with self.ros_node._state_lock:
            state = self.ros_node.state

            result = {
                "joint_positions": dict(
                    zip(state.joint_names, state.joint_positions, strict=False)
                ),
                "joint_velocities": dict(
                    zip(state.joint_names, state.joint_velocities, strict=False)
                ),
                "joint_efforts": dict(zip(state.joint_names, state.joint_efforts, strict=False)),
                "is_connected": state.is_connected,
            }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_move_joints(self, arguments: dict) -> list[TextContent]:
        """Validate input, then block the legacy direct execution path."""
        positions = arguments.get("joint_positions", [])

        if len(positions) != 6:
            return [
                TextContent(
                    type="text", text=f"Error: Expected 6 joint positions, got {len(positions)}"
                )
            ]

        # First check joint limits
        valid, msg = self.ros_node.validate_joint_limits(positions)
        if not valid:
            return [TextContent(type="text", text=f"Error: Joint limit violation - {msg}")]

        return self._gateway_required("ur5_move_joints")

    async def _handle_execute_trajectory(self, arguments: dict) -> list[TextContent]:
        """Validate input, then block the legacy direct execution path."""
        waypoints = arguments.get("waypoints", [])
        times = arguments.get("times", [])

        if len(waypoints) != len(times):
            return [
                TextContent(type="text", text="Error: waypoints and times must have same length")
            ]

        # Validate each waypoint
        for i, wp in enumerate(waypoints):
            if len(wp) != 6:
                return [
                    TextContent(
                        type="text", text=f"Error: waypoint {i} should have 6 values, got {len(wp)}"
                    )
                ]
            valid, msg = self.ros_node.validate_joint_limits(wp)
            if not valid:
                return [TextContent(type="text", text=f"Error in waypoint {i}: {msg}")]

        return self._gateway_required("ur5_execute_trajectory")

    @staticmethod
    def _gateway_required(action: str) -> list[TextContent]:
        response = {
            "status": "blocked",
            "error_code": "ROSCLAWD_REQUEST_ACTION_REQUIRED",
            "action": action,
            "no_command_dispatched": True,
            "required_entrypoint": "MCP request_action -> rosclawd",
            "message": (
                "Legacy direct physical execution is disabled. Use canonical MCP "
                "request_action backed by rosclawd and a verified REAL executor."
            ),
        }
        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    async def _handle_emergency_stop(self) -> list[TextContent]:
        """Request E-Stop through rosclawd and preserve evidence semantics."""
        try:
            evidence = await asyncio.to_thread(
                self._daemon_client.emergency_stop,
                "Legacy UR5 MCP emergency stop",
                source="rosclaw-ur5-mcp",
            )
            response = {
                **evidence,
                "status": str(evidence.get("final_status", "UNVERIFIED")).lower(),
                "execution_mode": str(evidence.get("execution_mode", "REAL")),
                "trust": str(evidence.get("trust_level", "UNVERIFIED")),
            }
        except DaemonClientError as exc:
            response = {
                "status": "failed",
                "error_code": exc.code,
                "error": exc.message,
                "request_dispatched": False,
                "driver_acknowledged": False,
                "physical_stop_observed": False,
                "stopped": False,
                "execution_mode": "UNKNOWN",
                "trust": "UNAVAILABLE",
                "usable_for_real_execution": False,
                "message": "Activate the certified physical E-stop immediately.",
            }
        except Exception as exc:  # noqa: BLE001
            response = {
                "status": "failed",
                "error_code": "ROSCLAWD_REQUEST_FAILED",
                "error": str(exc),
                "request_dispatched": False,
                "driver_acknowledged": False,
                "physical_stop_observed": False,
                "stopped": False,
                "execution_mode": "UNKNOWN",
                "trust": "UNAVAILABLE",
                "usable_for_real_execution": False,
                "message": "Activate the certified physical E-stop immediately.",
            }
        return [TextContent(type="text", text=json.dumps(response, indent=2))]

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
            point = [s + t * (e - s) for s, e in zip(start, end, strict=False)]
            trajectory.append(point)
        return trajectory

    async def _ros_spin(self) -> None:
        """Background task to spin ROS node."""
        while RCLPY_AVAILABLE and rclpy.ok():
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
            if RCLPY_AVAILABLE:
                rclpy.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run MCP server
        async with stdio_server() as (read, write):
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

    print("[UR5MCP] Starting ROSClaw UR5 MCP Server")
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
        if RCLPY_AVAILABLE and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
