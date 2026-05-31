"""Serial Driver - Generic serial/CAN device robot control."""

import struct
import time
from typing import Any, Optional

from rosclaw.mcp_drivers.base import BaseDriver, DriverState, TrajectoryCommand


class SerialDriver(BaseDriver):
    """
    Hardware driver for serial-port or CAN-bus connected robots.

    Communicates via simple binary protocol.
    """

    def __init__(
        self,
        robot_id: str,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        joint_dof: int = 6,
    ):
        super().__init__(robot_id, joint_dof)
        self._port = port
        self._baudrate = baudrate
        self._serial: Optional[Any] = None

    def _do_initialize(self) -> None:
        try:
            import serial
        except ImportError:
            print("[SerialDriver] pyserial not available, running in mock mode")
            self._driver_state.connected = True
            return

        try:
            self._serial = serial.Serial(self._port, self._baudrate, timeout=1.0)
            self._driver_state.connected = True
            print(f"[SerialDriver] Serial port opened: {self._port} @ {self._baudrate}")
        except Exception as e:
            print(f"[SerialDriver] Could not open {self._port}: {e}, running in mock mode")
            self._driver_state.connected = True

    def _do_stop(self) -> None:
        if self._serial and hasattr(self._serial, "is_open") and self._serial.is_open:
            self._serial.close()
        self._driver_state.connected = False

    def _build_move_packet(self, positions: list[float], duration: float) -> bytes:
        fmt = f"<B{self.joint_dof}ff"
        return struct.pack(fmt, 0x01, *positions, duration)

    def _build_gripper_packet(self, position: float, force: float) -> bytes:
        return struct.pack("<Bff", 0x02, position, force)

    def _build_stop_packet(self) -> bytes:
        return struct.pack("<B", 0x03)

    def _build_state_request(self) -> bytes:
        return struct.pack("<B", 0x10)

    def _parse_state_response(self, data: bytes) -> dict:
        if len(data) < 1 + self.joint_dof * 12:
            return {}
        fmt = f"<B{self.joint_dof}f{self.joint_dof}f{self.joint_dof}f"
        parts = struct.unpack(fmt, data[:struct.calcsize(fmt)])
        return {
            "positions": list(parts[1:1 + self.joint_dof]),
            "velocities": list(parts[1 + self.joint_dof:1 + 2 * self.joint_dof]),
            "torques": list(parts[1 + 2 * self.joint_dof:1 + 3 * self.joint_dof]),
        }

    def get_joint_positions(self) -> list[float]:
        state = self._read_state()
        return state.get("positions", [0.0] * self.joint_dof)

    def get_joint_velocities(self) -> list[float]:
        state = self._read_state()
        return state.get("velocities", [0.0] * self.joint_dof)

    def get_joint_torques(self) -> list[float]:
        state = self._read_state()
        return state.get("torques", [0.0] * self.joint_dof)

    def _read_state(self) -> dict:
        if self._serial is None:
            return {}
        self._serial.write(self._build_state_request())
        time.sleep(0.01)
        if self._serial.in_waiting:
            data = self._serial.read(self._serial.in_waiting)
            return self._parse_state_response(data)
        return {}

    def move_joints(self, positions: list[float], duration: float = 2.0) -> bool:
        self._ensure_ready("move_joints")
        self._validate_joint_positions(positions)
        self._validate_duration(duration)
        if not self._driver_state.connected:
            return False

        if self._serial is None:
            self._driver_state.joint_positions = list(positions)
            return True

        packet = self._build_move_packet(positions, duration)
        self._serial.write(packet)
        return True

    def execute_trajectory(self, trajectory: TrajectoryCommand) -> bool:
        if not self._driver_state.connected:
            return False
        self._ensure_ready("execute_trajectory")
        self._validate_trajectory(trajectory)
        for wp, t in zip(trajectory.waypoints, trajectory.times):
            if not self.move_joints(wp, duration=t):
                return False
        return True

    def set_gripper(self, position: float, force: float = 0.5) -> bool:
        self._driver_state.gripper_state = position
        if self._serial is not None:
            packet = self._build_gripper_packet(position, force)
            self._serial.write(packet)
        return True

    def emergency_stop(self) -> None:
        self._driver_state.error_code = 99
        self._driver_state.error_message = "Emergency stop triggered"
        if self._serial is not None:
            self._serial.write(self._build_stop_packet())

    def get_state(self) -> DriverState:
        state = self._read_state()
        if state:
            self._driver_state.joint_positions = state.get("positions", [])
            self._driver_state.joint_velocities = state.get("velocities", [])
            self._driver_state.joint_torques = state.get("torques", [])
        return self._driver_state
