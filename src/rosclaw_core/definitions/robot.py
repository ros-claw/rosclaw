"""Robot manifest definitions for ROSClaw V4.

Defines the schema for robot hardware specifications, capabilities,
and runtime configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol


class JointType(Enum):
    """Types of robot joints."""
    REVOLUTE = auto()
    PRISMATIC = auto()
    CONTINUOUS = auto()
    FIXED = auto()


class SensorType(Enum):
    """Types of robot sensors."""
    CAMERA = auto()
    LIDAR = auto()
    IMU = auto()
    FORCE_TORQUE = auto()
    ENCODER = auto()


@dataclass
class JointLimits:
    """Physical limits for a joint."""
    min_position: float = -3.14159
    max_position: float = 3.14159
    max_velocity: float = 1.0
    max_effort: float = 10.0

    def validate_position(self, position: float) -> bool:
        """Check if position is within limits."""
        return self.min_position <= position <= self.max_position


@dataclass
class JointConfig:
    """Configuration for a single joint/actuator."""
    name: str
    type: JointType
    motor_id: int
    limits: JointLimits = field(default_factory=JointLimits)
    gear_ratio: float = 1.0
    offset: float = 0.0


@dataclass
class SensorConfig:
    """Configuration for a sensor."""
    name: str
    type: SensorType
    topic: str
    frame_id: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotCapabilities:
    """Hardware capabilities of a robot."""
    dof: int = 6  # Degrees of freedom
    has_gripper: bool = False
    max_payload_kg: float = 0.5
    reach_m: float = 0.3
    precision_mm: float = 1.0
    max_speed_m_s: float = 0.1

    # Supported operations
    supports_teleop: bool = True
    supports_recording: bool = True
    supports_policy: bool = True


@dataclass
class RobotManifest:
    """Complete robot hardware specification.

    This manifest defines the physical robot configuration including
    joints, sensors, safety limits, and operational capabilities.

    Attributes:
        name: Unique robot identifier
        version: Manifest version
        hardware_type: Hardware platform (e.g., "so101", "ur5", "g1")
        joints: List of joint configurations
        sensors: List of sensor configurations
        capabilities: Hardware capabilities
        calibration_dir: Path to calibration files
        safety_limits: Additional safety constraints
        ros_namespace: ROS topic namespace
    """

    name: str
    version: str = "1.0.0"
    hardware_type: str = "so101"

    # Hardware configuration
    joints: list[JointConfig] = field(default_factory=list)
    sensors: list[SensorConfig] = field(default_factory=list)
    capabilities: RobotCapabilities = field(default_factory=RobotCapabilities)

    # Paths and runtime config
    calibration_dir: Path = field(default_factory=lambda: Path("~/.rosclaw/calibration"))
    ros_namespace: str = "/robot"

    # Safety limits (enforced at runtime)
    safety_limits: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate manifest after creation."""
        self.calibration_dir = Path(self.calibration_dir).expanduser()

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration files exist."""
        return self.calibration_dir.exists() and any(self.calibration_dir.iterdir())

    @property
    def follower_joints(self) -> list[JointConfig]:
        """Get joints configured as followers."""
        return [j for j in self.joints if "follower" in j.name.lower()]

    @property
    def leader_joints(self) -> list[JointConfig]:
        """Get joints configured as leaders."""
        return [j for j in self.joints if "leader" in j.name.lower()]

    def get_joint(self, name: str) -> JointConfig | None:
        """Get joint configuration by name."""
        for joint in self.joints:
            if joint.name == name:
                return joint
        return None

    def get_sensor(self, name: str) -> SensorConfig | None:
        """Get sensor configuration by name."""
        for sensor in self.sensors:
            if sensor.name == name:
                return sensor
        return None

    def validate_state(self, positions: dict[str, float]) -> list[str]:
        """Validate joint positions against limits.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        for joint_name, position in positions.items():
            joint = self.get_joint(joint_name)
            if joint and not joint.limits.validate_position(position):
                errors.append(
                    f"Joint '{joint_name}' position {position} exceeds limits "
                    f"[{joint.limits.min_position}, {joint.limits.max_position}]"
                )
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "hardware_type": self.hardware_type,
            "joints": [
                {
                    "name": j.name,
                    "type": j.type.name,
                    "motor_id": j.motor_id,
                    "limits": {
                        "min_position": j.limits.min_position,
                        "max_position": j.limits.max_position,
                        "max_velocity": j.limits.max_velocity,
                        "max_effort": j.limits.max_effort,
                    },
                    "gear_ratio": j.gear_ratio,
                    "offset": j.offset,
                }
                for j in self.joints
            ],
            "sensors": [
                {
                    "name": s.name,
                    "type": s.type.name,
                    "topic": s.topic,
                    "frame_id": s.frame_id,
                    "parameters": s.parameters,
                }
                for s in self.sensors
            ],
            "capabilities": {
                "dof": self.capabilities.dof,
                "has_gripper": self.capabilities.has_gripper,
                "max_payload_kg": self.capabilities.max_payload_kg,
                "supports_teleop": self.capabilities.supports_teleop,
                "supports_recording": self.capabilities.supports_recording,
                "supports_policy": self.capabilities.supports_policy,
            },
            "calibration_dir": str(self.calibration_dir),
            "ros_namespace": self.ros_namespace,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotManifest:
        """Create manifest from dictionary."""
        joints = [
            JointConfig(
                name=j["name"],
                type=JointType[j["type"]],
                motor_id=j["motor_id"],
                limits=JointLimits(**j["limits"]),
                gear_ratio=j.get("gear_ratio", 1.0),
                offset=j.get("offset", 0.0),
            )
            for j in data.get("joints", [])
        ]

        sensors = [
            SensorConfig(
                name=s["name"],
                type=SensorType[s["type"]],
                topic=s["topic"],
                frame_id=s["frame_id"],
                parameters=s.get("parameters", {}),
            )
            for s in data.get("sensors", [])
        ]

        caps_data = data.get("capabilities", {})
        capabilities = RobotCapabilities(
            dof=caps_data.get("dof", 6),
            has_gripper=caps_data.get("has_gripper", False),
            max_payload_kg=caps_data.get("max_payload_kg", 0.5),
            supports_teleop=caps_data.get("supports_teleop", True),
            supports_recording=caps_data.get("supports_recording", True),
            supports_policy=caps_data.get("supports_policy", True),
        )

        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            hardware_type=data.get("hardware_type", "so101"),
            joints=joints,
            sensors=sensors,
            capabilities=capabilities,
            calibration_dir=Path(data.get("calibration_dir", "~/.rosclaw/calibration")),
            ros_namespace=data.get("ros_namespace", "/robot"),
        )
