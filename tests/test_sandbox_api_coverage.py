"""Additional coverage tests for sandbox_api.py."""

from unittest.mock import MagicMock, patch

import pytest

from rosclaw.sandbox.sandbox_api import Sandbox, SandboxSession


class TestSandboxSession:
    def test_session_id(self):
        s = SandboxSession("test-123")
        assert s.session_id == "test-123"


class TestSandboxLoadModelAliases:
    def test_alias_universal_robots_ur5e(self):
        sandbox = Sandbox("universal_robots_ur5e", "empty")
        if sandbox.has_physics:
            assert sandbox._model is not None

    def test_alias_unitree_g1(self):
        Sandbox("unitree_g1", "empty")
        # g1 may or may not have model, just check no crash
        pass

    def test_alias_unitree_go2(self):
        Sandbox("unitree_go2", "empty")
        pass

    def test_nonexistent_robot_no_physics(self):
        sandbox = Sandbox("nonexistent_robot_xyz", "empty")
        assert not sandbox.has_physics
        assert sandbox._model is None
        assert sandbox._data is None


class TestSandboxStepEdgeCases:
    def test_step_no_model(self):
        sandbox = Sandbox("nonexistent", "empty")
        result = sandbox.step([0.0] * 6)
        assert result is None

    def test_step_empty_positions(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo not available")
        result = sandbox.step([])
        assert result is not None
        assert "qpos" in result

    def test_step_more_positions_than_actuators(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo not available")
        # UR5e has 6 actuators; pass 10 positions
        result = sandbox.step([0.1] * 10)
        assert result is not None


class TestSandboxResetClose:
    def test_reset_no_model(self):
        sandbox = Sandbox("nonexistent", "empty")
        sandbox.reset()  # should not crash

    def test_close_no_model(self):
        sandbox = Sandbox("nonexistent", "empty")
        sandbox.close()  # should not crash
        assert sandbox._model is None
        assert sandbox._data is None

    def test_reset_with_model(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo not available")
        sandbox.step([0.1] * 6)
        sandbox.reset()
        assert sandbox._data is not None

    def test_close_reopens(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo not available")
        sandbox.close()
        assert sandbox._model is None


class TestSandboxGetState:
    def test_get_state_no_model(self):
        sandbox = Sandbox("nonexistent", "empty")
        assert sandbox.get_state() is None

    def test_get_state_with_model(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo not available")
        state = sandbox.get_state()
        assert state is not None
        assert "qpos" in state
        assert "qvel" in state
        assert "time" in state


class TestSandboxGetObservationEdgeCases:
    def test_get_observation_no_model(self):
        sandbox = Sandbox("nonexistent", "empty")
        assert sandbox.get_observation() is None

    def test_get_observation_contacts_when_no_collision(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo not available")
        obs = sandbox.get_observation(normalize=True)
        assert isinstance(obs["contacts"], list)

    def test_get_observation_body_positions(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo not available")
        obs = sandbox.get_observation(normalize=True)
        bodies = obs["body_positions"]
        assert len(bodies) > 0
        # Check that body positions are 3D
        for _name, pos in bodies.items():
            assert len(pos) == 3


class TestSandboxCreateFactory:
    def test_create_factory_method(self):
        sandbox = Sandbox.create("ur5e", "empty")
        if sandbox.has_physics:
            assert sandbox._model is not None


class TestSandboxLoadModelImportError:
    def test_mujoco_import_error(self, caplog):
        import logging
        with patch.dict("sys.modules", {"mujoco": None}), caplog.at_level(logging.WARNING, logger="rosclaw.sandbox.sandbox_api"):
            sandbox = Sandbox("ur5e", "empty")
        assert not sandbox.has_physics
        assert "MuJoCo not installed" in caplog.text


class TestSandboxLoadModelCandidateFailures:
    def test_all_candidates_fail(self, tmp_path, caplog):
        import logging
        # Create a fake robot directory with unreadable XML
        zoo = tmp_path / "e-urdf-zoo"
        robot_dir = zoo / "fake_robot"
        robot_dir.mkdir(parents=True)
        (robot_dir / "scene.xml").write_text("<invalid>")
        (robot_dir / "robot.mjcf.xml").write_text("<invalid>")

        with patch("rosclaw.sandbox.sandbox_api.Path") as mock_path:
            mock_root = MagicMock()
            mock_root.parent.parent.parent.parent = zoo
            mock_path.return_value = mock_root
            mock_path.__truediv__ = lambda self, other: tmp_path / str(other)

            with caplog.at_level(logging.WARNING, logger="rosclaw.sandbox.sandbox_api"):
                sandbox = Sandbox("fake_robot", "empty")

        assert not sandbox.has_physics
