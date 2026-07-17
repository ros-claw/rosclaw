"""Tests for MuJoCoSimDriver."""

import pytest

from rosclaw.mcp_drivers import MuJoCoSimDriver
from rosclaw.mcp_drivers.base import DriverState, TrajectoryCommand


class TestMuJoCoSimDriverLifecycle:
    def test_init_defaults(self):
        d = MuJoCoSimDriver()
        assert d.robot_id == "default_robot"
        assert d.joint_dof == 6
        assert d._model_path == ""

    def test_init_custom(self):
        d = MuJoCoSimDriver(robot_id="r1", model_path="/tmp/m.xml", joint_dof=7)
        assert d.robot_id == "r1"
        assert d.joint_dof == 7

    def test_initialize_empty_path(self):
        d = MuJoCoSimDriver(model_path="")
        with pytest.raises(RuntimeError, match="model_path is required"):
            d.initialize()
        assert not d.is_connected()
        assert d._model is None

    def test_initialize_missing_file(self):
        d = MuJoCoSimDriver(model_path="/tmp/nonexistent_model_12345.xml")
        with pytest.raises(RuntimeError, match="model not found"):
            d.initialize()
        assert not d.is_connected()
        assert d._model is None

    def test_stop_clears_state(self):
        d = MuJoCoSimDriver(fixture_mode=True)
        d.initialize()
        d.stop()
        assert not d.is_connected()
        assert d._model is None


class TestMuJoCoSimDriverFixtureMode:
    def test_get_joint_positions_fixture(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        d.initialize()
        pos = d.get_joint_positions()
        assert len(pos) == 6
        assert all(p == 0.0 for p in pos)

    def test_get_joint_velocities_fixture(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        d.initialize()
        vel = d.get_joint_velocities()
        assert len(vel) == 6
        assert all(v == 0.0 for v in vel)

    def test_get_joint_torques_fixture(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        d.initialize()
        torque = d.get_joint_torques()
        assert len(torque) == 6
        assert all(t == 0.0 for t in torque)

    def test_get_state_fixture(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        d.initialize()
        state = d.get_state()
        assert isinstance(state, DriverState)
        assert len(state.joint_positions) == 6
        assert len(state.joint_velocities) == 6
        assert len(state.joint_torques) == 6

    def test_move_joints_fixture(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        d.initialize()
        d.start()
        result = d.move_joints([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], duration=0.1)
        assert result is True
        d.stop()

    def test_move_joints_wrong_dof(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        d.initialize()
        d.start()
        with pytest.raises(ValueError):
            d.move_joints([0.1, 0.2], duration=0.1)
        d.stop()

    def test_move_joints_non_finite(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        d.initialize()
        d.start()
        with pytest.raises(ValueError):
            d.move_joints([float("inf"), 0, 0, 0, 0, 0], duration=0.1)
        d.stop()

    def test_move_joints_too_large(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        d.initialize()
        d.start()
        with pytest.raises(ValueError):
            d.move_joints([1e6, 0, 0, 0, 0, 0], duration=0.1)
        d.stop()

    def test_move_joints_not_initialized(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        with pytest.raises(RuntimeError):
            d.move_joints([0.1] * 6, duration=0.1)

    def test_set_gripper(self):
        d = MuJoCoSimDriver(fixture_mode=True)
        d.initialize()
        assert d.set_gripper(0.8) is True
        assert d.state.gripper_state == 0.8

    def test_emergency_stop(self):
        d = MuJoCoSimDriver(fixture_mode=True)
        d.initialize()
        d.emergency_stop()
        assert d.state.error_code == 99
        assert "Emergency stop" in d.state.error_message


class TestMuJoCoSimDriverTrajectory:
    def test_execute_trajectory_success(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        d.initialize()
        d.start()
        traj = TrajectoryCommand(
            waypoints=[[0.1] * 6, [0.2] * 6],
            times=[0.1, 0.1],
        )
        result = d.execute_trajectory(traj)
        assert result is True
        d.stop()

    def test_execute_trajectory_wrong_dof(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        d.initialize()
        d.start()
        traj = TrajectoryCommand(waypoints=[[0.1] * 3], times=[0.1])
        with pytest.raises(ValueError):
            d.execute_trajectory(traj)
        d.stop()

    def test_execute_trajectory_not_connected(self):
        d = MuJoCoSimDriver(joint_dof=6, fixture_mode=True)
        traj = TrajectoryCommand(waypoints=[[0.1] * 6], times=[0.1])
        result = d.execute_trajectory(traj)
        assert result is False


class TestMuJoCoSimDriverGetMujocoData:
    def test_get_mujoco_data_fixture(self):
        d = MuJoCoSimDriver(fixture_mode=True)
        d.initialize()
        assert d.get_mujoco_data() is None
