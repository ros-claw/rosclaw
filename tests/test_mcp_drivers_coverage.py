"""Coverage tests for MuJoCoSimDriver and SerialDriver."""

import pytest
from unittest.mock import MagicMock, patch

from rosclaw.mcp_drivers.mujoco_sim_driver import MuJoCoSimDriver
from rosclaw.mcp_drivers.serial_driver import SerialDriver
from rosclaw.mcp_drivers.base import TrajectoryCommand


class TestMuJoCoSimDriverLifecycle:
    def test_initialize_import_error(self, caplog):
        import logging
        driver = MuJoCoSimDriver("test_bot")
        with patch.dict("sys.modules", {"mujoco": None}):
            with caplog.at_level(logging.WARNING, logger="rosclaw.mcp_drivers.mujoco_sim"):
                driver.initialize()
        assert "mujoco not available" in caplog.text
        assert driver._driver_state.connected is True

    def test_initialize_no_model_path(self, caplog):
        import logging
        driver = MuJoCoSimDriver("test_bot", model_path="")
        with caplog.at_level(logging.WARNING, logger="rosclaw.mcp_drivers.mujoco_sim"):
            driver.initialize()
        assert "No model path" in caplog.text
        assert driver._driver_state.connected is True

    def test_initialize_model_not_found(self, caplog):
        import logging
        driver = MuJoCoSimDriver("test_bot", model_path="/nonexistent/model.xml")
        with caplog.at_level(logging.WARNING, logger="rosclaw.mcp_drivers.mujoco_sim"):
            driver.initialize()
        assert "Model not found" in caplog.text
        assert driver._driver_state.connected is True

    def test_stop_clears_state(self):
        driver = MuJoCoSimDriver("test_bot")
        driver.initialize()
        driver._model = MagicMock()
        driver._data = MagicMock()
        driver.stop()
        assert driver._driver_state.connected is False
        assert driver._model is None
        assert driver._data is None


class TestMuJoCoSimDriverMockMode:
    def test_get_joint_positions_mock(self):
        driver = MuJoCoSimDriver("test_bot", joint_dof=3)
        driver.initialize()
        assert driver.get_joint_positions() == [0.0, 0.0, 0.0]

    def test_get_joint_velocities_mock(self):
        driver = MuJoCoSimDriver("test_bot", joint_dof=3)
        driver.initialize()
        assert driver.get_joint_velocities() == [0.0, 0.0, 0.0]

    def test_get_joint_torques_mock(self):
        driver = MuJoCoSimDriver("test_bot", joint_dof=3)
        driver.initialize()
        assert driver.get_joint_torques() == [0.0, 0.0, 0.0]

    def test_move_joints_mock(self):
        driver = MuJoCoSimDriver("test_bot", joint_dof=3)
        driver.initialize()
        driver.start()
        result = driver.move_joints([0.1, 0.2, 0.3], duration=1.0)
        assert result is True
        assert driver._driver_state.joint_positions == [0.1, 0.2, 0.3]

    def test_move_joints_not_connected(self):
        driver = MuJoCoSimDriver("test_bot", joint_dof=6)
        driver.initialize()
        driver.start()
        driver._driver_state.connected = False
        result = driver.move_joints([0.1] * 6)
        assert result is False

    def test_execute_trajectory_mock(self):
        driver = MuJoCoSimDriver("test_bot", joint_dof=3)
        driver.initialize()
        driver.start()
        traj = TrajectoryCommand(
            waypoints=[[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
            times=[1.0, 2.0],
        )
        assert driver.execute_trajectory(traj) is True

    def test_execute_trajectory_not_connected(self):
        driver = MuJoCoSimDriver("test_bot")
        traj = TrajectoryCommand(waypoints=[[0.0] * 6], times=[1.0])
        assert driver.execute_trajectory(traj) is False

    def test_set_gripper(self):
        driver = MuJoCoSimDriver("test_bot")
        driver.initialize()
        driver.start()
        assert driver.set_gripper(0.75, force=0.3) is True
        assert driver._driver_state.gripper_state == 0.75

    def test_emergency_stop_mock(self):
        driver = MuJoCoSimDriver("test_bot")
        driver.initialize()
        driver.start()
        driver.emergency_stop()
        assert driver._driver_state.error_code == 99
        assert "Emergency stop" in driver._driver_state.error_message

    def test_get_state_mock(self):
        driver = MuJoCoSimDriver("test_bot", joint_dof=3)
        driver.initialize()
        driver.start()
        state = driver.get_state()
        assert state.joint_positions == [0.0, 0.0, 0.0]
        assert state.joint_velocities == [0.0, 0.0, 0.0]
        assert state.joint_torques == [0.0, 0.0, 0.0]

    def test_get_mujoco_data_none(self):
        driver = MuJoCoSimDriver("test_bot")
        driver.initialize()
        assert driver.get_mujoco_data() is None


class TestSerialDriverLifecycle:
    def test_initialize_import_error(self, caplog):
        import logging
        driver = SerialDriver("test_bot", port="/dev/ttyUSB0")
        with patch.dict("sys.modules", {"serial": None}):
            with caplog.at_level(logging.WARNING, logger="rosclaw.mcp_drivers.serial"):
                driver.initialize()
        assert "pyserial not available" in caplog.text
        assert driver._driver_state.connected is True

    def test_initialize_port_error(self, caplog):
        import logging
        fake_serial = MagicMock()
        fake_serial.Serial.side_effect = Exception("port fail")
        driver = SerialDriver("test_bot", port="/dev/fake")
        with patch.dict("sys.modules", {"serial": fake_serial}):
            with caplog.at_level(logging.WARNING, logger="rosclaw.mcp_drivers.serial"):
                driver.initialize()
        assert "Could not open" in caplog.text
        assert driver._driver_state.connected is True

    def test_stop_no_serial(self):
        driver = SerialDriver("test_bot")
        driver._serial = None
        driver.stop()  # Should not raise

    def test_stop_with_closed_serial(self):
        driver = SerialDriver("test_bot")
        driver.initialize()
        driver.start()
        mock_serial = MagicMock()
        mock_serial.is_open = False
        driver._serial = mock_serial
        driver.stop()
        mock_serial.close.assert_not_called()

    def test_stop_with_open_serial(self):
        driver = SerialDriver("test_bot")
        driver.initialize()
        driver.start()
        mock_serial = MagicMock()
        mock_serial.is_open = True
        driver._serial = mock_serial
        driver.stop()
        mock_serial.close.assert_called_once()


class TestSerialDriverMockMode:
    def test_get_joint_positions_no_serial(self):
        driver = SerialDriver("test_bot", joint_dof=3)
        driver.initialize()
        assert driver.get_joint_positions() == [0.0, 0.0, 0.0]

    def test_get_joint_velocities_no_serial(self):
        driver = SerialDriver("test_bot", joint_dof=3)
        driver.initialize()
        assert driver.get_joint_velocities() == [0.0, 0.0, 0.0]

    def test_get_joint_torques_no_serial(self):
        driver = SerialDriver("test_bot", joint_dof=3)
        driver.initialize()
        assert driver.get_joint_torques() == [0.0, 0.0, 0.0]

    def test_move_joints_mock(self):
        driver = SerialDriver("test_bot", joint_dof=3)
        driver.initialize()
        driver.start()
        result = driver.move_joints([0.1, 0.2, 0.3], duration=1.0)
        assert result is True
        assert driver._driver_state.joint_positions == [0.1, 0.2, 0.3]

    def test_move_joints_not_connected(self):
        driver = SerialDriver("test_bot", joint_dof=6)
        driver.initialize()
        driver.start()
        driver._driver_state.connected = False
        result = driver.move_joints([0.1] * 6)
        assert result is False

    def test_execute_trajectory_mock(self):
        driver = SerialDriver("test_bot", joint_dof=3)
        driver.initialize()
        driver.start()
        traj = TrajectoryCommand(
            waypoints=[[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
            times=[1.0, 2.0],
        )
        assert driver.execute_trajectory(traj) is True

    def test_execute_trajectory_not_connected(self):
        driver = SerialDriver("test_bot", joint_dof=6)
        driver.initialize()
        driver.start()
        driver._driver_state.connected = False
        traj = TrajectoryCommand(waypoints=[[0.0] * 6], times=[1.0])
        assert driver.execute_trajectory(traj) is False

    def test_set_gripper_no_serial(self):
        driver = SerialDriver("test_bot")
        driver.initialize()
        driver.start()
        assert driver.set_gripper(0.75) is True
        assert driver._driver_state.gripper_state == 0.75

    def test_set_gripper_with_serial(self):
        driver = SerialDriver("test_bot")
        mock_serial = MagicMock()
        driver._serial = mock_serial
        driver.initialize()
        driver.start()
        assert driver.set_gripper(0.75, force=0.3) is True
        mock_serial.write.assert_called()

    def test_emergency_stop_no_serial(self):
        driver = SerialDriver("test_bot")
        driver.initialize()
        driver.start()
        driver.emergency_stop()
        assert driver._driver_state.error_code == 99

    def test_emergency_stop_with_serial(self):
        driver = SerialDriver("test_bot")
        mock_serial = MagicMock()
        driver._serial = mock_serial
        driver.initialize()
        driver.start()
        driver.emergency_stop()
        mock_serial.write.assert_called()

    def test_get_state_no_serial(self):
        driver = SerialDriver("test_bot", joint_dof=3)
        driver.initialize()
        state = driver.get_state()
        assert state is not None

    def test_build_packets(self):
        driver = SerialDriver("test_bot", joint_dof=3)
        move_pkt = driver._build_move_packet([0.1, 0.2, 0.3], 1.0)
        assert len(move_pkt) > 0
        gripper_pkt = driver._build_gripper_packet(0.5, 0.3)
        assert len(gripper_pkt) > 0
        stop_pkt = driver._build_stop_packet()
        assert len(stop_pkt) > 0
        state_req = driver._build_state_request()
        assert len(state_req) > 0

    def test_parse_state_response_valid(self):
        driver = SerialDriver("test_bot", joint_dof=3)
        import struct
        fmt = f"<B{3}f{3}f{3}f"
        data = struct.pack(fmt, 0x10, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 1.0, 2.0, 3.0)
        result = driver._parse_state_response(data)
        assert result["positions"] == pytest.approx([0.1, 0.2, 0.3])
        assert result["velocities"] == pytest.approx([0.01, 0.02, 0.03])
        assert result["torques"] == pytest.approx([1.0, 2.0, 3.0])

    def test_parse_state_response_too_short(self):
        driver = SerialDriver("test_bot", joint_dof=3)
        result = driver._parse_state_response(b"\x10\x00")
        assert result == {}

    def test_read_state_with_mock_serial(self):
        driver = SerialDriver("test_bot", joint_dof=3)
        import struct
        fmt = f"<B{3}f{3}f{3}f"
        data = struct.pack(fmt, 0x10, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 1.0, 2.0, 3.0)
        mock_serial = MagicMock()
        mock_serial.in_waiting = len(data)
        mock_serial.read.return_value = data
        driver._serial = mock_serial
        state = driver._read_state()
        assert state["positions"] == pytest.approx([0.1, 0.2, 0.3])
