"""FT Sensor Configuration - Settings for force/torque sensor.

This module defines configuration for FT sensor access.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FTSensorConfig:
    """Configuration for FT Sensor MCP server."""

    # Sensor type
    sensor_type: str = "ati"  # ati, robotiq, wacoh, optoforce, mock
    connection_type: str = "ros2"  # ros2, ethercat, tcp, mock

    # ROS2 configuration
    ros_topic: str = "/ft_sensor/data"
    ros_node_name: str = "ft_sensor_mcp"

    # TCP/ethernet configuration
    ip_address: str = "192.168.1.1"
    port: int = 5000

    # Calibration
    force_offset: list[float] = None
    torque_offset: list[float] = None
    force_scale: float = 1.0
    torque_scale: float = 1.0

    # Safety limits (N and Nm)
    max_force: float = 100.0  # N
    max_torque: float = 10.0  # Nm

    # Data rate
    publish_rate: float = 100.0  # Hz

    def __post_init__(self):
        """Initialize defaults."""
        if self.force_offset is None:
            self.force_offset = [0.0, 0.0, 0.0]
        if self.torque_offset is None:
            self.torque_offset = [0.0, 0.0, 0.0]

        if self.sensor_type not in ["ati", "robotiq", "wacoh", "optoforce", "mock"]:
            raise ValueError(f"Invalid sensor type: {self.sensor_type}")

        if self.connection_type not in ["ros2", "ethercat", "tcp", "mock"]:
            raise ValueError(f"Invalid connection type: {self.connection_type}")

    def validate_force(self, force: list[float]) -> tuple[bool, str]:
        """Validate force reading against safety limits."""
        import math
        magnitude = math.sqrt(sum(f**2 for f in force))
        if magnitude > self.max_force:
            return False, f"Force magnitude {magnitude:.2f}N exceeds limit {self.max_force}N"
        return True, ""

    def validate_torque(self, torque: list[float]) -> tuple[bool, str]:
        """Validate torque reading against safety limits."""
        import math
        magnitude = math.sqrt(sum(t**2 for t in torque))
        if magnitude > self.max_torque:
            return False, f"Torque magnitude {magnitude:.2f}Nm exceeds limit {self.max_torque}Nm"
        return True, ""
