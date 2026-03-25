"""UR RTDE Configuration - Settings and safety limits for Universal Robots.

This module defines configuration models for UR robot control including
safety limits that must be validated before any motion commands.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JointLimits:
    """Joint limits for a single UR robot joint."""

    min_position: float  # radians
    max_position: float  # radians
    max_velocity: float  # rad/s
    max_acceleration: Optional[float] = None  # rad/s²
    max_effort: Optional[float] = None  # Nm


@dataclass
class CartesianLimits:
    """Cartesian motion limits for TCP movement."""

    max_velocity: float = 1.0  # m/s
    max_acceleration: float = 1.5  # m/s²
    position_tolerance: float = 0.001  # m
    orientation_tolerance: float = 0.01  # rad


@dataclass
class SafetyLimits:
    """Safety limits for UR robot motion.

    All motion commands are validated against these limits before execution.
    Violations result in command rejection with an error message.
    """

    # Joint limits for each joint (6-DOF for UR robots)
    joint_limits: dict[str, JointLimits] = field(default_factory=lambda: {
        "shoulder_pan_joint": JointLimits(
            min_position=-6.28, max_position=6.28,
            max_velocity=2.09, max_effort=330.0
        ),
        "shoulder_lift_joint": JointLimits(
            min_position=-6.28, max_position=6.28,
            max_velocity=2.09, max_effort=330.0
        ),
        "elbow_joint": JointLimits(
            min_position=-3.14, max_position=3.14,
            max_velocity=3.14, max_effort=150.0
        ),
        "wrist_1_joint": JointLimits(
            min_position=-6.28, max_position=6.28,
            max_velocity=3.14, max_effort=56.0
        ),
        "wrist_2_joint": JointLimits(
            min_position=-6.28, max_position=6.28,
            max_velocity=3.14, max_effort=56.0
        ),
        "wrist_3_joint": JointLimits(
            min_position=-6.28, max_position=6.28,
            max_velocity=3.14, max_effort=56.0
        ),
    })

    # Cartesian limits
    cartesian_limits: CartesianLimits = field(default_factory=CartesianLimits)

    # Force/torque safety limits (for force mode)
    max_force: float = 100.0  # N
    max_torque: float = 20.0  # Nm

    # Emergency stop sensitivity
    emergency_stop_deceleration: float = 5.0  # rad/s²


@dataclass
class URRTDEConfig:
    """Configuration for UR RTDE MCP server.

    This configuration defines all parameters needed to connect to and
    control a Universal Robots arm via the RTDE protocol.
    """

    # Robot connection
    robot_ip: str = "192.168.1.100"
    robot_model: str = "ur5e"

    # RTDE port (default 30004 for UR)
    rtde_port: int = 30004

    # Connection timeout
    connection_timeout: float = 5.0

    # Safety limits (MUST be enforced)
    safety_limits: dict = field(default_factory=dict)

    # Control parameters
    default_velocity: float = 0.5  # rad/s or m/s
    default_acceleration: float = 1.0  # rad/s² or m/s²
    default_blend_radius: float = 0.0  # m (0 = no blending)

    # Tool configuration
    tcp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    payload_mass: float = 0.0  # kg
    payload_cog: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        """Initialize safety limits from dict if provided."""
        if isinstance(self.safety_limits, dict) and self.safety_limits:
            self.safety_limits = SafetyLimits(**self.safety_limits)
        elif not self.safety_limits:
            self.safety_limits = SafetyLimits()

    def validate_joint_positions(self, joint_positions: list[float]) -> tuple[bool, str]:
        """Validate joint positions against safety limits.

        Args:
            joint_positions: List of 6 joint positions in radians

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(joint_positions) != 6:
            return False, f"Expected 6 joint positions, got {len(joint_positions)}"

        joint_names = list(self.safety_limits.joint_limits.keys())
        for i, (name, pos) in enumerate(zip(joint_names, joint_positions)):
            limits = self.safety_limits.joint_limits[name]
            if pos < limits.min_position or pos > limits.max_position:
                return False, (
                    f"Joint {i} ({name}) position {pos:.4f} rad exceeds limits "
                    f"[{limits.min_position:.4f}, {limits.max_position:.4f}]"
                )

        return True, ""

    def validate_velocity(self, velocity: float, is_joint: bool = True) -> tuple[bool, str]:
        """Validate velocity command against safety limits.

        Args:
            velocity: Velocity value (rad/s for joints, m/s for cartesian)
            is_joint: True if joint velocity, False if cartesian

        Returns:
            Tuple of (is_valid, error_message)
        """
        if velocity <= 0:
            return False, f"Velocity must be positive, got {velocity}"

        if is_joint:
            max_vel = max(l.max_velocity for l in self.safety_limits.joint_limits.values())
            if velocity > max_vel:
                return False, f"Joint velocity {velocity} exceeds max {max_vel}"
        else:
            if velocity > self.safety_limits.cartesian_limits.max_velocity:
                return False, (
                    f"Cartesian velocity {velocity} exceeds max "
                    f"{self.safety_limits.cartesian_limits.max_velocity}"
                )

        return True, ""
