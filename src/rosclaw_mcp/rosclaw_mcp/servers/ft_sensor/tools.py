"""FT Sensor Tools - MCP tool definitions for force/torque sensor.

This module registers all MCP tools for force/torque sensor access.
"""

import math
import sys
from typing import TYPE_CHECKING, Optional

from fastmcp import FastMCP
from mcp.types import ToolAnnotations

if TYPE_CHECKING:
    from rosclaw_mcp.servers.ft_sensor.config import FTSensorConfig


class FTSensorInterface:
    """FT Sensor interface supporting multiple backends."""

    def __init__(self, config: "FTSensorConfig"):
        self.config = config
        self.node = None
        self.subscription = None
        self.latest_data = None
        self.is_mock = config.connection_type == "mock"

    def connect(self) -> bool:
        """Connect to FT sensor."""
        try:
            if self.config.connection_type == "ros2":
                import rclpy
                from geometry_msgs.msg import WrenchStamped

                rclpy.init()
                self.node = rclpy.create_node(self.config.ros_node_name)

                def callback(msg):
                    self.latest_data = {
                        "force": [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
                        "torque": [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
                        "timestamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    }

                self.subscription = self.node.create_subscription(
                    WrenchStamped,
                    self.config.ros_topic,
                    callback,
                    10,
                )

                # Spin once to start receiving
                import threading
                self.spin_thread = threading.Thread(target=self._spin)
                self.spin_thread.daemon = True
                self.spin_thread.start()

                return True

            elif self.config.connection_type == "tcp":
                import socket

                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.config.ip_address, self.config.port))
                return True

            elif self.config.connection_type == "mock":
                return True

        except ImportError as e:
            print(f"[FT Sensor] Import error: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[FT Sensor] Connection error: {e}", file=sys.stderr)
            return False

    def _spin(self):
        """ROS2 spin thread."""
        import rclpy
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def disconnect(self):
        """Disconnect from sensor."""
        if self.config.connection_type == "ros2" and self.node:
            import rclpy
            self.node.destroy_node()
            rclpy.shutdown()
        elif self.config.connection_type == "tcp" and hasattr(self, "socket"):
            self.socket.close()

    def get_data(self) -> Optional[dict]:
        """Get current force/torque data."""
        if self.is_mock:
            import time
            import random
            return {
                "force": [
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-20, 20),
                ],
                "torque": [
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                ],
                "timestamp": time.time(),
                "mock": True,
            }

        if self.config.connection_type == "ros2":
            return self.latest_data

        elif self.config.connection_type == "tcp":
            try:
                # Read data from socket (format depends on sensor)
                data = self.socket.recv(1024)
                # Parse based on sensor protocol
                return self._parse_tcp_data(data)
            except Exception as e:
                print(f"[FT Sensor] Read error: {e}", file=sys.stderr)
                return None

        return None

    def _parse_tcp_data(self, data: bytes) -> Optional[dict]:
        """Parse TCP data based on sensor type."""
        # Simplified parsing - actual implementation depends on sensor protocol
        try:
            # Assume CSV format: fx,fy,fz,tx,ty,tz
            values = [float(v) for v in data.decode().strip().split(",")]
            if len(values) >= 6:
                return {
                    "force": values[0:3],
                    "torque": values[3:6],
                    "timestamp": None,
                }
        except Exception:
            pass
        return None


def register_ft_tools(mcp: FastMCP, config: "FTSensorConfig") -> None:
    """Register all FT Sensor MCP tools."""

    sensor: Optional[FTSensorInterface] = None

    def get_sensor():
        """Get or create sensor interface."""
        nonlocal sensor
        if sensor is None:
            sensor = FTSensorInterface(config)
            if not sensor.connect():
                return None
        return sensor

    @mcp.tool(
        description=(
            "Read current force/torque values.\n"
            "Example: read_force_torque()\n"
            "Returns force [Fx, Fy, Fz] in Newtons and torque [Tx, Ty, Tz] in Newton-meters."
        ),
        annotations=ToolAnnotations(
            title="Read Force/Torque",
            readOnlyHint=True,
        ),
    )
    def read_force_torque() -> dict:
        """Read current force/torque values."""
        ft = get_sensor()
        if ft is None:
            return {"error": "FT sensor not available"}

        data = ft.get_data()
        if data is None:
            return {"error": "No data available from sensor"}

        # Apply calibration offsets and scales
        force = [
            (f - offset) * config.force_scale
            for f, offset in zip(data["force"], config.force_offset)
        ]
        torque = [
            (t - offset) * config.torque_scale
            for t, offset in zip(data["torque"], config.torque_offset)
        ]

        # Validate against safety limits
        force_valid, force_error = config.validate_force(force)
        torque_valid, torque_error = config.validate_torque(torque)

        result = {
            "success": True,
            "force": {
                "x": round(force[0], 4),
                "y": round(force[1], 4),
                "z": round(force[2], 4),
                "magnitude": round(math.sqrt(sum(f**2 for f in force)), 4),
            },
            "torque": {
                "x": round(torque[0], 4),
                "y": round(torque[1], 4),
                "z": round(torque[2], 4),
                "magnitude": round(math.sqrt(sum(t**2 for t in torque)), 4),
            },
            "mock": data.get("mock", False),
        }

        if not force_valid:
            result["force_warning"] = force_error
        if not torque_valid:
            result["torque_warning"] = torque_error

        return result

    @mcp.tool(
        description=(
            "Get force magnitude (total force).\n"
            "Example: get_force_magnitude()\n"
            "Returns total force magnitude in Newtons."
        ),
        annotations=ToolAnnotations(
            title="Get Force Magnitude",
            readOnlyHint=True,
        ),
    )
    def get_force_magnitude() -> dict:
        """Get total force magnitude."""
        ft = get_sensor()
        if ft is None:
            return {"error": "FT sensor not available"}

        data = ft.get_data()
        if data is None:
            return {"error": "No data available"}

        force = data["force"]
        magnitude = math.sqrt(sum(f**2 for f in force))

        return {
            "success": True,
            "magnitude": round(magnitude, 4),
            "force_vector": [round(f, 4) for f in force],
            "mock": data.get("mock", False),
        }

    @mcp.tool(
        description=(
            "Zero/tare the sensor (set current reading as zero).\n"
            "Example: zero_sensor()\n"
            "Call this when no external forces are applied."
        ),
        annotations=ToolAnnotations(
            title="Zero Sensor",
            destructiveHint=True,
        ),
    )
    def zero_sensor() -> dict:
        """Zero the sensor by setting current reading as offset."""
        ft = get_sensor()
        if ft is None:
            return {"error": "FT sensor not available"}

        data = ft.get_data()
        if data is None:
            return {"error": "No data available"}

        # Update offsets
        config.force_offset = list(data["force"])
        config.torque_offset = list(data["torque"])

        return {
            "success": True,
            "new_force_offset": config.force_offset,
            "new_torque_offset": config.torque_offset,
        }

    @mcp.tool(
        description=(
            "Get sensor configuration and status.\n"
            "Example: get_sensor_info()"
        ),
        annotations=ToolAnnotations(
            title="Get Sensor Info",
            readOnlyHint=True,
        ),
    )
    def get_sensor_info() -> dict:
        """Get sensor configuration and status."""
        ft = get_sensor()
        connected = ft is not None

        return {
            "sensor_type": config.sensor_type,
            "connection_type": config.connection_type,
            "ros_topic": config.ros_topic if config.connection_type == "ros2" else None,
            "ip_address": config.ip_address if config.connection_type in ["tcp", "ethercat"] else None,
            "max_force_limit": config.max_force,
            "max_torque_limit": config.max_torque,
            "force_scale": config.force_scale,
            "torque_scale": config.torque_scale,
            "connected": connected,
            "is_mock": config.connection_type == "mock",
        }

    @mcp.tool(
        description=(
            "Monitor force/torque for contact detection.\n"
            "Example: detect_contact(force_threshold=5.0, torque_threshold=0.5)\n"
            "Returns True if contact is detected based on thresholds."
        ),
        annotations=ToolAnnotations(
            title="Detect Contact",
            readOnlyHint=True,
        ),
    )
    def detect_contact(
        force_threshold: float = 5.0,
        torque_threshold: float = 0.5,
    ) -> dict:
        """Detect contact based on force/torque thresholds."""
        ft = get_sensor()
        if ft is None:
            return {"error": "FT sensor not available"}

        data = ft.get_data()
        if data is None:
            return {"error": "No data available"}

        force = data["force"]
        torque = data["torque"]

        force_mag = math.sqrt(sum(f**2 for f in force))
        torque_mag = math.sqrt(sum(t**2 for t in torque))

        contact_detected = (
            force_mag > force_threshold or
            torque_mag > torque_threshold
        )

        return {
            "success": True,
            "contact_detected": contact_detected,
            "force_magnitude": round(force_mag, 4),
            "torque_magnitude": round(torque_mag, 4),
            "force_threshold": force_threshold,
            "torque_threshold": torque_threshold,
            "mock": data.get("mock", False),
        }

    @mcp.tool(
        description=(
            "Reconnect to sensor.\n"
            "Example: reconnect_sensor()"
        ),
        annotations=ToolAnnotations(
            title="Reconnect Sensor",
            destructiveHint=True,
        ),
    )
    def reconnect_sensor() -> dict:
        """Reconnect to sensor."""
        nonlocal sensor

        if sensor is not None:
            sensor.disconnect()
            sensor = None

        ft = get_sensor()
        return {
            "success": ft is not None,
            "connected": ft is not None,
        }
