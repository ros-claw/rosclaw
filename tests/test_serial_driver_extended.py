"""Extended tests for SerialDriver packet building and state parsing."""

import struct

import pytest

from rosclaw.mcp_drivers.base import TrajectoryCommand
from rosclaw.mcp_drivers.serial_driver import SerialDriver


class TestSerialDriverExtended:
    def test_build_move_packet(self):
        driver = SerialDriver("test", joint_dof=6)
        packet = driver._build_move_packet([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 2.0)
        assert isinstance(packet, bytes)
        # fmt = "<B6ff" -> 1 + 6*4 + 4 = 29 bytes
        assert len(packet) == struct.calcsize("<B6ff")
        parts = struct.unpack("<B6ff", packet)
        assert parts[0] == 0x01
        assert parts[1:7] == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        assert parts[7] == pytest.approx(2.0)

    def test_build_gripper_packet(self):
        driver = SerialDriver("test")
        packet = driver._build_gripper_packet(0.75, 0.3)
        parts = struct.unpack("<Bff", packet)
        assert parts[0] == 0x02
        assert parts[1] == pytest.approx(0.75)
        assert parts[2] == pytest.approx(0.3)

    def test_build_stop_packet(self):
        driver = SerialDriver("test")
        packet = driver._build_stop_packet()
        assert struct.unpack("<B", packet) == (0x03,)

    def test_build_state_request(self):
        driver = SerialDriver("test")
        packet = driver._build_state_request()
        assert struct.unpack("<B", packet) == (0x10,)

    def test_parse_state_response_valid(self):
        driver = SerialDriver("test", joint_dof=3)
        fmt = "<B3f3f3f"
        data = struct.pack(fmt, 0x10, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 10.0, 20.0, 30.0)
        state = driver._parse_state_response(data)
        assert state["positions"] == pytest.approx([1.0, 2.0, 3.0])
        assert state["velocities"] == pytest.approx([0.1, 0.2, 0.3])
        assert state["torques"] == pytest.approx([10.0, 20.0, 30.0])

    def test_parse_state_response_too_short(self):
        driver = SerialDriver("test", joint_dof=6)
        state = driver._parse_state_response(b"\x10")
        assert state == {}

    def test_get_joint_velocities_mock(self):
        driver = SerialDriver("test")
        driver.initialize()
        driver.start()
        assert driver.get_joint_velocities() == [0.0] * 6
        driver.stop()

    def test_get_joint_torques_mock(self):
        driver = SerialDriver("test")
        driver.initialize()
        driver.start()
        assert driver.get_joint_torques() == [0.0] * 6
        driver.stop()

    def test_move_joints_mock(self):
        driver = SerialDriver("test")
        driver.initialize()
        driver.start()
        assert driver.move_joints([0.1] * 6, duration=1.0) is True
        assert driver._driver_state.joint_positions == pytest.approx([0.1] * 6)
        driver.stop()

    def test_execute_trajectory(self):
        driver = SerialDriver("test")
        driver.initialize()
        driver.start()
        traj = TrajectoryCommand(
            waypoints=[[0.0] * 6, [0.1] * 6, [0.2] * 6],
            times=[1.0, 1.0, 1.0],
        )
        assert driver.execute_trajectory(traj) is True
        driver.stop()

    def test_execute_trajectory_not_connected(self):
        driver = SerialDriver("test")
        traj = TrajectoryCommand(waypoints=[[0.0] * 6], times=[1.0])
        # Not initialized/started: connected flag is False
        assert driver.execute_trajectory(traj) is False

    def test_set_gripper_mock(self):
        driver = SerialDriver("test")
        driver.initialize()
        driver.start()
        assert driver.set_gripper(0.8, force=0.4) is True
        assert driver._driver_state.gripper_state == pytest.approx(0.8)
        driver.stop()

    def test_emergency_stop_mock(self):
        driver = SerialDriver("test")
        driver.initialize()
        driver.start()
        driver.emergency_stop()
        assert driver._driver_state.error_code == 99
        assert "Emergency stop" in driver._driver_state.error_message
        driver.stop()

    def test_get_state_updates(self):
        driver = SerialDriver("test", joint_dof=3)
        driver.initialize()
        driver.start()
        state = driver.get_state()
        assert state is not None
        driver.stop()

    def test_read_state_no_serial(self):
        driver = SerialDriver("test")
        driver.initialize()
        driver.start()
        # _serial is None in mock mode
        assert driver._read_state() == {}
        driver.stop()
