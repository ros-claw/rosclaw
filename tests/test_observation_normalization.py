"""Tests for enhanced observation normalization from scene state."""

import numpy as np
import pytest

from rosclaw.core.event_bus import EventBus
from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter
from rosclaw.sandbox.sandbox_api import Sandbox


class TestSandboxGetObservation:
    def test_get_observation_basic(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo model not available")

        obs = sandbox.get_observation(normalize=True)
        assert obs is not None
        assert "joint_positions" in obs
        assert "joint_positions_normalized" in obs
        assert "joint_velocities" in obs
        assert "joint_velocities_normalized" in obs
        assert "body_positions" in obs
        assert "contacts" in obs
        assert "time" in obs

    def test_get_observation_no_normalize(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo model not available")

        obs = sandbox.get_observation(normalize=False)
        assert obs is not None
        # Raw and normalized should be identical when normalize=False
        np.testing.assert_allclose(obs["joint_positions"], obs["joint_positions_normalized"])

    def test_normalized_positions_in_range(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo model not available")

        obs = sandbox.get_observation(normalize=True)
        norm_pos = np.array(obs["joint_positions_normalized"])
        # All normalized positions should be in [-1, 1]
        assert np.all(norm_pos >= -1.0)
        assert np.all(norm_pos <= 1.0)

    def test_normalized_positions_at_zero(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo model not available")

        # At home position (all zeros), normalized should be near center of range
        obs = sandbox.get_observation(normalize=True)
        norm_pos = np.array(obs["joint_positions_normalized"])
        # For UR5e, zero is roughly in the middle of most joint ranges
        assert np.all(np.abs(norm_pos) < 0.5)

    def test_velocities_clipped(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo model not available")

        obs = sandbox.get_observation(normalize=True)
        norm_vel = np.array(obs["joint_velocities_normalized"])
        # tanh-clipped velocities should be in [-1, 1]
        assert np.all(norm_vel >= -1.0)
        assert np.all(norm_vel <= 1.0)

    def test_body_positions_present(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo model not available")

        obs = sandbox.get_observation(normalize=True)
        bodies = obs["body_positions"]
        assert len(bodies) > 0
        # Should have base link and joint links
        assert any("base" in k.lower() or "world" in k.lower() for k in bodies)

    def test_contacts_list(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo model not available")

        obs = sandbox.get_observation(normalize=True)
        # Contacts may be empty at home position (no collision)
        assert isinstance(obs["contacts"], list)

    def test_get_observation_after_step(self):
        sandbox = Sandbox("ur5e", "empty")
        if not sandbox.has_physics:
            pytest.skip("MuJoCo model not available")

        # Step with some joint positions (MuJoCo ctrl drives toward target)
        target = [0.1, -0.2, 0.3, -0.1, 0.0, 0.0]
        for _ in range(50):  # multiple steps for position control to converge
            sandbox.step(target)
        obs = sandbox.get_observation(normalize=True)
        # Positions should have moved toward target
        assert obs["joint_positions"][0] > 0.0  # moved positive
        assert obs["joint_positions"][1] < 0.0  # moved negative

    def test_get_observation_no_model(self):
        sandbox = Sandbox("nonexistent_robot_12345", "empty")
        assert not sandbox.has_physics
        obs = sandbox.get_observation(normalize=True)
        assert obs is None


class TestRuntimeAdapterGetObservation:
    def test_get_observation_delegates(self):
        bus = EventBus()
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "ur5e"},
            event_bus=bus,
        )
        adapter.initialize()
        if not adapter.has_physics:
            pytest.skip("MuJoCo model not available")

        obs = adapter.get_observation(normalize=True)
        assert "joint_positions" in obs
        assert "joint_positions_normalized" in obs
        adapter.stop()

    def test_get_observation_no_physics(self):
        bus = EventBus()
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "nonexistent"},
            event_bus=bus,
        )
        adapter.initialize()
        obs = adapter.get_observation(normalize=True)
        assert obs == {}
        adapter.stop()

    def test_get_observation_without_normalize(self):
        bus = EventBus()
        adapter = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "empty", "robot_id": "ur5e"},
            event_bus=bus,
        )
        adapter.initialize()
        if not adapter.has_physics:
            pytest.skip("MuJoCo model not available")

        obs = adapter.get_observation(normalize=False)
        assert "joint_positions" in obs
        np.testing.assert_allclose(obs["joint_positions"], obs["joint_positions_normalized"])
        adapter.stop()
