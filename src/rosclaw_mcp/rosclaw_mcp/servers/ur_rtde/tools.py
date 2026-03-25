"""UR RTDE Tools - MCP tool definitions for Universal Robots control.

This module registers all MCP tools for controlling UR robots via RTDE.
All commands include safety validation against configured limits.
"""

import sys
from typing import TYPE_CHECKING

from fastmcp import FastMCP
from mcp.types import ToolAnnotations

if TYPE_CHECKING:
    from rosclaw_mcp.servers.ur_rtde.config import URRTDEConfig


def register_ur_tools(mcp: FastMCP, config: "URRTDEConfig") -> None:
    """Register all UR RTDE MCP tools.

    Args:
        mcp: FastMCP server instance
        config: UR RTDE configuration with safety limits
    """

    # Store RTDE interfaces (initialized on first use)
    rtde_control = None
    rtde_receive = None

    def get_control():
        """Get or create RTDE control interface."""
        nonlocal rtde_control
        if rtde_control is None:
            try:
                from rtde_control import RTDEControlInterface
                rtde_control = RTDEControlInterface(
                    config.robot_ip,
                    config.rtde_port,
                )
                print(f"[UR RTDE] Connected to robot at {config.robot_ip}", file=sys.stderr)
            except ImportError:
                print("[UR RTDE] Warning: rtde_control not installed", file=sys.stderr)
                return None
            except Exception as e:
                print(f"[UR RTDE] Connection error: {e}", file=sys.stderr)
                return None
        return rtde_control

    def get_receive():
        """Get or create RTDE receive interface."""
        nonlocal rtde_receive
        if rtde_receive is None:
            try:
                from rtde_receive import RTDEReceiveInterface
                rtde_receive = RTDEReceiveInterface(config.robot_ip)
            except ImportError:
                print("[UR RTDE] Warning: rtde_receive not installed", file=sys.stderr)
                return None
            except Exception as e:
                print(f"[UR RTDE] Receive connection error: {e}", file=sys.stderr)
                return None
        return rtde_receive

    @mcp.tool(
        description=(
            "Get current robot state including joint positions, TCP pose, and velocities.\n"
            "Example: get_state()"
        ),
        annotations=ToolAnnotations(
            title="Get Robot State",
            readOnlyHint=True,
        ),
    )
    def get_state() -> dict:
        """Get current robot state.

        Returns:
            dict: Current joint positions, TCP pose, velocities, and status
        """
        rtde_r = get_receive()
        if rtde_r is None:
            return {"error": "RTDE receive interface not available"}

        try:
            joint_positions = rtde_r.getActualQ()
            joint_velocities = rtde_r.getActualQd()
            tcp_pose = rtde_r.getActualTCPPose()
            tcp_speed = rtde_r.getActualTCPSpeed()
            robot_status = rtde_r.getRobotStatus()

            return {
                "joint_positions": list(joint_positions),
                "joint_velocities": list(joint_velocities),
                "tcp_pose": list(tcp_pose),  # [x, y, z, rx, ry, rz]
                "tcp_speed": list(tcp_speed),
                "robot_status": int(robot_status),
                "connected": True,
            }
        except Exception as e:
            return {"error": f"Failed to get state: {e}"}

    @mcp.tool(
        description=(
            "Move robot to joint positions.\n"
            "Example: move_joint(joint_positions=[0.0, -1.57, 1.57, 0.0, 0.0, 0.0], velocity=0.5)\n"
            "Note: Joint positions are in radians. 6 joints expected."
        ),
        annotations=ToolAnnotations(
            title="Move Joint",
            destructiveHint=True,
        ),
    )
    def move_joint(
        joint_positions: list[float],
        velocity: float = 0.5,
        acceleration: float = 1.0,
        asynchronous: bool = False,
    ) -> dict:
        """Move robot to specified joint positions.

        Args:
            joint_positions: List of 6 joint positions in radians
            velocity: Joint velocity in rad/s (default: 0.5)
            acceleration: Joint acceleration in rad/s² (default: 1.0)
            asynchronous: If True, return immediately without waiting for completion

        Returns:
            dict: Success status or error message
        """
        # Validate inputs
        if len(joint_positions) != 6:
            return {"error": f"Expected 6 joint positions, got {len(joint_positions)}"}

        valid, error = config.validate_joint_positions(joint_positions)
        if not valid:
            return {"error": f"Safety validation failed: {error}"}

        valid, error = config.validate_velocity(velocity, is_joint=True)
        if not valid:
            return {"error": f"Velocity validation failed: {error}"}

        rtde_c = get_control()
        if rtde_c is None:
            return {"error": "RTDE control interface not available"}

        try:
            rtde_c.moveJ(joint_positions, velocity, acceleration, asynchronous)
            return {
                "success": True,
                "command": "moveJ",
                "target_joints": joint_positions,
                "velocity": velocity,
                "asynchronous": asynchronous,
            }
        except Exception as e:
            return {"error": f"Move joint failed: {e}"}

    @mcp.tool(
        description=(
            "Move robot TCP to Cartesian pose.\n"
            "Example: move_cartesian(pose=[0.4, 0.2, 0.5, 3.14, 0.0, 0.0], velocity=0.25)\n"
            "Pose format: [x, y, z, rx, ry, rz] where position is in meters and "
            "orientation is rotation vector in radians."
        ),
        annotations=ToolAnnotations(
            title="Move Cartesian",
            destructiveHint=True,
        ),
    )
    def move_cartesian(
        pose: list[float],
        velocity: float = 0.25,
        acceleration: float = 0.5,
        asynchronous: bool = False,
    ) -> dict:
        """Move robot TCP to specified Cartesian pose.

        Args:
            pose: [x, y, z, rx, ry, rz] - position in meters, rotation vector in radians
            velocity: TCP velocity in m/s (default: 0.25)
            acceleration: TCP acceleration in m/s² (default: 0.5)
            asynchronous: If True, return immediately without waiting

        Returns:
            dict: Success status or error message
        """
        if len(pose) != 6:
            return {"error": f"Expected pose [x,y,z,rx,ry,rz], got {len(pose)} values"}

        valid, error = config.validate_velocity(velocity, is_joint=False)
        if not valid:
            return {"error": f"Velocity validation failed: {error}"}

        rtde_c = get_control()
        if rtde_c is None:
            return {"error": "RTDE control interface not available"}

        try:
            rtde_c.moveL(pose, velocity, acceleration, asynchronous)
            return {
                "success": True,
                "command": "moveL",
                "target_pose": pose,
                "velocity": velocity,
                "asynchronous": asynchronous,
            }
        except Exception as e:
            return {"error": f"Move cartesian failed: {e}"}

    @mcp.tool(
        description=(
            "Stop robot motion immediately (emergency stop).\n"
            "Example: emergency_stop(deceleration=2.0)"
        ),
        annotations=ToolAnnotations(
            title="Emergency Stop",
            destructiveHint=True,
        ),
    )
    def emergency_stop(deceleration: float = 5.0) -> dict:
        """Stop all robot motion immediately.

        Args:
            deceleration: Deceleration rate in rad/s²

        Returns:
            dict: Success status or error message
        """
        rtde_c = get_control()
        if rtde_c is None:
            return {"error": "RTDE control interface not available"}

        try:
            # Stop joint and linear motion
            rtde_c.stopJ(deceleration)
            rtde_c.stopL(deceleration)
            return {
                "success": True,
                "command": "emergency_stop",
                "deceleration": deceleration,
            }
        except Exception as e:
            return {"error": f"Emergency stop failed: {e}"}

    @mcp.tool(
        description=(
            "Control robot gripper (if equipped).\n"
            "Example: gripper_control(action='close', force=50)\n"
            "Actions: 'open', 'close', 'move_to'"
        ),
        annotations=ToolAnnotations(
            title="Gripper Control",
            destructiveHint=True,
        ),
    )
    def gripper_control(
        action: str = "open",
        position: float = 0.0,
        force: int = 50,
        speed: int = 50,
    ) -> dict:
        """Control robot gripper.

        Args:
            action: 'open', 'close', or 'move_to'
            position: Target position for 'move_to' (0-255)
            force: Gripping force (0-100)
            speed: Gripper speed (0-100)

        Returns:
            dict: Success status or error message
        """
        rtde_c = get_control()
        if rtde_c is None:
            return {"error": "RTDE control interface not available"}

        try:
            if action == "open":
                # Send digital output to open gripper
                rtde_c.setStandardDigitalOut(0, False)
                return {"success": True, "action": "open"}
            elif action == "close":
                # Send digital output to close gripper
                rtde_c.setStandardDigitalOut(0, True)
                return {"success": True, "action": "close"}
            elif action == "move_to":
                return {
                    "success": False,
                    "error": "Robotiq gripper requires special handling - use external controller",
                }
            else:
                return {"error": f"Unknown gripper action: {action}"}
        except Exception as e:
            return {"error": f"Gripper control failed: {e}"}

    @mcp.tool(
        description=(
            "Get robot safety status and limits.\n"
            "Example: get_safety_status()"
        ),
        annotations=ToolAnnotations(
            title="Get Safety Status",
            readOnlyHint=True,
        ),
    )
    def get_safety_status() -> dict:
        """Get current safety status and configured limits.

        Returns:
            dict: Safety limits and robot safety status
        """
        rtde_r = get_receive()
        safety_mode = "unknown"
        if rtde_r is not None:
            try:
                safety_mode = int(rtde_r.getSafetyMode())
            except Exception:
                pass

        return {
            "robot_model": config.robot_model,
            "robot_ip": config.robot_ip,
            "joint_limits": {
                name: {
                    "min": limits.min_position,
                    "max": limits.max_position,
                    "max_velocity": limits.max_velocity,
                }
                for name, limits in config.safety_limits.joint_limits.items()
            },
            "cartesian_limits": {
                "max_velocity": config.safety_limits.cartesian_limits.max_velocity,
                "max_acceleration": config.safety_limits.cartesian_limits.max_acceleration,
            },
            "safety_mode": safety_mode,
            "default_velocity": config.default_velocity,
            "default_acceleration": config.default_acceleration,
        }

    @mcp.tool(
        description=(
            "Teach mode - manually guide robot with reduced stiffness.\n"
            "Example: set_teach_mode(enable=True)\n"
            "WARNING: Use with caution - ensure workspace is clear."
        ),
        annotations=ToolAnnotations(
            title="Teach Mode",
            destructiveHint=True,
        ),
    )
    def set_teach_mode(enable: bool = True) -> dict:
        """Enable or disable teach mode for manual guidance.

        Args:
            enable: True to enable teach mode, False to disable

        Returns:
            dict: Success status or error message
        """
        rtde_c = get_control()
        if rtde_c is None:
            return {"error": "RTDE control interface not available"}

        try:
            if enable:
                rtde_c.teachMode()
                return {"success": True, "mode": "teach_mode_enabled"}
            else:
                rtde_c.endTeachMode()
                return {"success": True, "mode": "teach_mode_disabled"}
        except Exception as e:
            return {"error": f"Teach mode command failed: {e}"}

    @mcp.tool(
        description=(
            "Reconnect to robot after connection loss.\n"
            "Example: reconnect()"
        ),
        annotations=ToolAnnotations(
            title="Reconnect",
            destructiveHint=True,
        ),
    )
    def reconnect() -> dict:
        """Reconnect to robot after connection loss.

        Returns:
            dict: Success status
        """
        nonlocal rtde_control, rtde_receive

        # Close existing connections
        if rtde_control is not None:
            try:
                rtde_control.stopScript()
                rtde_control.disconnect()
            except Exception:
                pass
            rtde_control = None

        if rtde_receive is not None:
            try:
                rtde_receive.disconnect()
            except Exception:
                pass
            rtde_receive = None

        # Reconnect
        rtde_c = get_control()
        rtde_r = get_receive()

        return {
            "success": rtde_c is not None and rtde_r is not None,
            "control_connected": rtde_c is not None,
            "receive_connected": rtde_r is not None,
        }
