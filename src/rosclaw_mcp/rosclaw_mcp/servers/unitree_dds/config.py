"""Unitree DDS Configuration - Settings and safety limits for Unitree robots.

This module defines configuration models for Unitree humanoid robot control.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JointLimits:
    """Joint limits for a single robot joint."""

    min_position: float  # radians
    max_position: float  # radians
    max_velocity: float  # rad/s
    max_effort: Optional[float] = None  # Nm


@dataclass
class SafetyLimits:
    """Safety limits for Unitree robot motion."""

    # Joint limits by joint name
    joint_limits: dict[str, JointLimits] = field(default_factory=dict)

    # Maximum velocities for different modes
    max_walking_speed: float = 1.0  # m/s
    max_angular_velocity: float = 1.0  # rad/s

    # Balance safety
    max_tilt_angle: float = 0.5  # rad (~30 degrees)

    def __post_init__(self):
        """Initialize default joint limits for G1 if not provided."""
        if not self.joint_limits:
            # G1 default joint limits
            self.joint_limits = {
                # Left leg
                "left_hip_yaw": JointLimits(-0.87, 0.87, 10.0),
                "left_hip_roll": JointLimits(-0.52, 0.52, 10.0),
                "left_hip_pitch": JointLimits(-1.57, 1.57, 10.0),
                "left_knee": JointLimits(-0.26, 2.09, 10.0),
                "left_ankle": JointLimits(-0.61, 0.61, 10.0),
                # Right leg
                "right_hip_yaw": JointLimits(-0.87, 0.87, 10.0),
                "right_hip_roll": JointLimits(-0.52, 0.52, 10.0),
                "right_hip_pitch": JointLimits(-1.57, 1.57, 10.0),
                "right_knee": JointLimits(-0.26, 2.09, 10.0),
                "right_ankle": JointLimits(-0.61, 0.61, 10.0),
                # Waist
                "waist_yaw": JointLimits(-1.22, 1.22, 5.0),
                # Left arm
                "left_shoulder_pitch": JointLimits(-3.14, 3.14, 10.0),
                "left_shoulder_roll": JointLimits(-0.52, 3.14, 10.0),
                "left_elbow": JointLimits(-2.61, 0.0, 10.0),
                # Right arm
                "right_shoulder_pitch": JointLimits(-3.14, 3.14, 10.0),
                "right_shoulder_roll": JointLimits(-3.14, 0.52, 10.0),
                "right_elbow": JointLimits(0.0, 2.61, 10.0),
            }


@dataclass
class UnitreeDDSConfig:
    """Configuration for Unitree DDS MCP server."""

    # Robot connection
    robot_ip: str = "192.168.123.161"
    robot_model: str = "g1"  # g1, go2, h1, b2
    interface: str = "eth0"

    # DDS domain ID
    domain_id: int = 0

    # Connection timeout
    connection_timeout: float = 5.0

    # Safety limits
    safety_limits: SafetyLimits = field(default_factory=SafetyLimits)

    # Control parameters
    default_velocity: float = 1.0
    default_gait_type: int = 0  # 0: idle, 1: trot, 2: trot running, 3: climb stair

    # G1 specific: number of joints
    num_dofs: int = 29  # G1 has 29 DoF

    def validate_joint_positions(self, joint_positions: list[float]) -> tuple[bool, str]:
        """Validate joint positions against safety limits."""
        if len(joint_positions) != len(self.safety_limits.joint_limits):
            return False, (
                f"Expected {len(self.safety_limits.joint_limits)} joint positions, "
                f"got {len(joint_positions)}"
            )

        joint_names = list(self.safety_limits.joint_limits.keys())
        for i, (name, pos) in enumerate(zip(joint_names, joint_positions)):
            limits = self.safety_limits.joint_limits[name]
            if pos < limits.min_position or pos > limits.max_position:
                return False, (
                    f"Joint {i} ({name}) position {pos:.4f} rad exceeds limits "
                    f"[{limits.min_position:.4f}, {limits.max_position:.4f}]"
                )

        return True, ""

    def validate_velocity_command(self, vx: float, vy: float, yaw: float) -> tuple[bool, str]:
        """Validate velocity command against safety limits."""
        speed = (vx**2 + vy**2) ** 0.5
        if speed > self.safety_limits.max_walking_speed:
            return False, f"Walking speed {speed:.2f} exceeds max {self.safety_limits.max_walking_speed}"

        if abs(yaw) > self.safety_limits.max_angular_velocity:
            return False, f"Angular velocity {yaw:.2f} exceeds max {self.safety_limits.max_angular_velocity}"

        return True, ""
