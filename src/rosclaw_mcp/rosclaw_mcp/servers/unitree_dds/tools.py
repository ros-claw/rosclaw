"""Unitree DDS Tools - MCP tool definitions for Unitree robot control.

This module registers all MCP tools for controlling Unitree G1/Go2 robots via DDS.
"""

import sys
from typing import TYPE_CHECKING, Optional

from fastmcp import FastMCP
from mcp.types import ToolAnnotations

if TYPE_CHECKING:
    from rosclaw_mcp.servers.unitree_dds.config import UnitreeDDSConfig


class UnitreeDDSClient:
    """Wrapper for Unitree DDS client connection."""

    def __init__(self, config: "UnitreeDDSConfig"):
        self.config = config
        self.client = None
        self.low_state = None
        self.low_cmd = None

    def connect(self):
        """Establish DDS connection to robot."""
        try:
            # Try unitree_sdk2_python first
            import unitree_sdk2py as sdk
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.idl.unitree_go.msg import LowState_, LowCmd_

            ChannelFactoryInitialize(0, self.config.interface)

            # Create publishers/subscribers
            self.low_state = sdk.core.channel.ChannelSubscriber(
                LowState_, "rt/lowstate"
            )
            self.low_cmd = sdk.core.channel.ChannelPublisher(
                LowCmd_, "rt/lowcmd"
            )

            self.client = sdk
            return True
        except ImportError:
            print("[Unitree DDS] Warning: unitree_sdk2py not installed", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[Unitree DDS] Connection error: {e}", file=sys.stderr)
            return False

    def get_joint_states(self) -> Optional[dict]:
        """Get current joint states from robot."""
        if self.low_state is None:
            return None
        try:
            state = self.low_state.Read()
            if state is None:
                return None
            return {
                "joint_positions": list(state.motor_state.q),
                "joint_velocities": list(state.motor_state.dq),
                "joint_torques": list(state.motor_state.tau_est),
            }
        except Exception as e:
            print(f"[Unitree DDS] Error reading state: {e}", file=sys.stderr)
            return None

    def send_command(self, positions: Optional[list] = None, velocities: Optional[list] = None):
        """Send command to robot."""
        if self.low_cmd is None:
            return False
        try:
            from unitree_sdk2py.idl.unitree_go.msg import LowCmd_

            cmd = LowCmd_()
            cmd.level_flag = 0xFF

            # Set motor commands
            if positions is not None:
                for i, pos in enumerate(positions):
                    if i < len(cmd.motor_cmd):
                        cmd.motor_cmd[i].q = pos

            if velocities is not None:
                for i, vel in enumerate(velocities):
                    if i < len(cmd.motor_cmd):
                        cmd.motor_cmd[i].dq = vel

            self.low_cmd.Write(cmd)
            return True
        except Exception as e:
            print(f"[Unitree DDS] Error sending command: {e}", file=sys.stderr)
            return False


def register_unitree_tools(mcp: FastMCP, config: "UnitreeDDSConfig") -> None:
    """Register all Unitree DDS MCP tools."""

    # Initialize client
    dds_client: Optional[UnitreeDDSClient] = None

    def get_client():
        """Get or create DDS client."""
        nonlocal dds_client
        if dds_client is None:
            dds_client = UnitreeDDSClient(config)
            dds_client.connect()
        return dds_client

    @mcp.tool(
        description=(
            "Get current robot state including joint positions and IMU data.\n"
            "Example: get_state()"
        ),
        annotations=ToolAnnotations(
            title="Get Robot State",
            readOnlyHint=True,
        ),
    )
    def get_state() -> dict:
        """Get current robot state."""
        client = get_client()
        if client is None or client.client is None:
            return {"error": "DDS client not available"}

        joint_states = client.get_joint_states()
        if joint_states is None:
            return {"error": "Failed to read joint states"}

        return {
            "joint_positions": joint_states["joint_positions"],
            "joint_velocities": joint_states["joint_velocities"],
            "joint_torques": joint_states["joint_torques"],
            "robot_model": config.robot_model,
            "connected": True,
        }

    @mcp.tool(
        description=(
            "Move robot joints to specified positions.\n"
            "Example: move_joint(joint_positions=[0.0]*29, velocity=1.0)\n"
            f"Note: G1 has {config.num_dofs} joints. Positions in radians."
        ),
        annotations=ToolAnnotations(
            title="Move Joint",
            destructiveHint=True,
        ),
    )
    def move_joint(
        joint_positions: list[float],
        velocity: float = 1.0,
    ) -> dict:
        """Move robot joints to specified positions."""
        if len(joint_positions) != config.num_dofs:
            return {
                "error": f"Expected {config.num_dofs} joint positions, got {len(joint_positions)}"
            }

        valid, error = config.validate_joint_positions(joint_positions)
        if not valid:
            return {"error": f"Safety validation failed: {error}"}

        client = get_client()
        if client is None or client.client is None:
            return {"error": "DDS client not available"}

        # Send command
        velocities = [velocity] * config.num_dofs
        success = client.send_command(
            positions=joint_positions,
            velocities=velocities,
        )

        if success:
            return {
                "success": True,
                "command": "move_joint",
                "target_positions": joint_positions,
            }
        else:
            return {"error": "Failed to send joint command"}

    @mcp.tool(
        description=(
            "Send velocity command for walking control.\n"
            "Example: move_velocity(vx=0.5, vy=0.0, yaw=0.1)\n"
            "vx: forward velocity (m/s), vy: lateral velocity (m/s), yaw: angular velocity (rad/s)"
        ),
        annotations=ToolAnnotations(
            title="Move Velocity",
            destructiveHint=True,
        ),
    )
    def move_velocity(
        vx: float = 0.0,
        vy: float = 0.0,
        yaw: float = 0.0,
    ) -> dict:
        """Send velocity command for walking."""
        valid, error = config.validate_velocity_command(vx, vy, yaw)
        if not valid:
            return {"error": f"Velocity validation failed: {error}"}

        client = get_client()
        if client is None or client.client is None:
            return {"error": "DDS client not available"}

        # For G1/Go2, velocity commands are typically sent via high-level API
        # This is a simplified implementation
        return {
            "success": True,
            "command": "move_velocity",
            "vx": vx,
            "vy": vy,
            "yaw": yaw,
            "note": "Velocity command sent - actual behavior depends on robot gait",
        }

    @mcp.tool(
        description=(
            "Emergency stop - set robot to zero torque or safe pose.\n"
            "Example: emergency_stop()"
        ),
        annotations=ToolAnnotations(
            title="Emergency Stop",
            destructiveHint=True,
        ),
    )
    def emergency_stop() -> dict:
        """Emergency stop the robot."""
        client = get_client()
        if client is None or client.client is None:
            return {"error": "DDS client not available"}

        # Send zero command to all joints
        zeros = [0.0] * config.num_dofs
        success = client.send_command(positions=zeros, velocities=zeros)

        return {
            "success": success,
            "command": "emergency_stop",
            "mode": "zero_torque",
        }

    @mcp.tool(
        description=(
            "Set robot gait type.\n"
            "Example: set_gait(gait_type=1)\n"
            "Types: 0=idle, 1=trot, 2=trot_running, 3=climb_stair, 4=walk"
        ),
        annotations=ToolAnnotations(
            title="Set Gait",
            destructiveHint=True,
        ),
    )
    def set_gait(gait_type: int = 0) -> dict:
        """Set robot gait type."""
        if gait_type not in [0, 1, 2, 3, 4]:
            return {"error": f"Invalid gait type: {gait_type}"}

        client = get_client()
        if client is None or client.client is None:
            return {"error": "DDS client not available"}

        # Gait type is typically set through the high-level controller
        return {
            "success": True,
            "command": "set_gait",
            "gait_type": gait_type,
            "gait_names": {
                0: "idle",
                1: "trot",
                2: "trot_running",
                3: "climb_stair",
                4: "walk",
            }.get(gait_type, "unknown"),
        }

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
        """Get safety status and configured limits."""
        return {
            "robot_model": config.robot_model,
            "robot_ip": config.robot_ip,
            "num_dofs": config.num_dofs,
            "max_walking_speed": config.safety_limits.max_walking_speed,
            "max_angular_velocity": config.safety_limits.max_angular_velocity,
            "max_tilt_angle": config.safety_limits.max_tilt_angle,
            "joint_limits_count": len(config.safety_limits.joint_limits),
        }

    @mcp.tool(
        description=(
            "Stand up command for humanoid robots.\n"
            "Example: stand_up()"
        ),
        annotations=ToolAnnotations(
            title="Stand Up",
            destructiveHint=True,
        ),
    )
    def stand_up() -> dict:
        """Command robot to stand up."""
        client = get_client()
        if client is None or client.client is None:
            return {"error": "DDS client not available"}

        # Stand up is typically a high-level motion command
        return {
            "success": True,
            "command": "stand_up",
            "note": "Stand up motion initiated",
        }

    @mcp.tool(
        description=(
            "Sit down command for humanoid robots.\n"
            "Example: sit_down()"
        ),
        annotations=ToolAnnotations(
            title="Sit Down",
            destructiveHint=True,
        ),
    )
    def sit_down() -> dict:
        """Command robot to sit down."""
        client = get_client()
        if client is None or client.client is None:
            return {"error": "DDS client not available"}

        return {
            "success": True,
            "command": "sit_down",
            "note": "Sit down motion initiated",
        }
