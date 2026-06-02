"""Tests for G1 free-floating walking demo.

Validates:
  - Free-floating model loads correctly
  - Robot maintains balance with static pose + pulsed propulsion
  - Forward motion is achieved
  - Fall detection works correctly
  - GPU detection reports correctly
"""

import numpy as np

from rosclaw.examples.g1_free_floating_walk import (
    WalkState,
    check_fall,
    create_g1_free_floating_model,
    quat_to_rpy,
    run_walking_demo,
)


class TestG1FreeFloatingModel:
    """Unit tests for G1 free-floating model and utilities."""

    def test_model_xml_loads(self):
        import mujoco
        xml = create_g1_free_floating_model()
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.nq == 19  # 6 freejoint + 12 hinge joints + 1 for ???
        assert model.nu == 12  # 12 actuators

    def test_quat_to_rpy_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        rpy = quat_to_rpy(q)
        assert np.allclose(rpy, [0.0, 0.0, 0.0], atol=1e-6)

    def test_fall_detection_height(self):
        state = WalkState()
        assert check_fall(state, 0.2, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.3])) is True
        assert state.fall_detected is True
        assert "pelvis_height_low" in state.fall_reason

    def test_fall_detection_tilt(self):
        state = WalkState()
        # 50-degree pitch
        q = np.array([np.cos(np.radians(25)), 0.0, np.sin(np.radians(25)), 0.0])
        assert check_fall(state, 0.5, q, np.array([0.0, 0.0, 0.4])) is True
        assert state.fall_detected is True
        assert "excessive_tilt" in state.fall_reason

    def test_fall_detection_no_fall(self):
        state = WalkState()
        assert check_fall(state, 0.6, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5])) is False
        assert state.fall_detected is False


class TestG1WalkingDemo:
    """Integration tests for the walking demo."""

    def test_demo_runs_without_error(self):
        result = run_walking_demo(target_distance=0.05, duration_limit=20.0, verbose=False)
        assert "success" in result
        assert "distance_traveled" in result
        assert "fall_detected" in result
        assert "gpu_used" in result

    def test_demo_moves_forward(self):
        result = run_walking_demo(target_distance=0.05, duration_limit=30.0, verbose=False)
        assert result["distance_traveled"] > 0.02  # At least 2cm forward

    def test_demo_does_not_fall_immediately(self):
        result = run_walking_demo(target_distance=0.05, duration_limit=15.0, verbose=False)
        # Simulation may exit early when target is reached; check no fall occurred
        assert result["fall_detected"] is False

    def test_fall_detection_triggered_on_bad_pose(self):
        # This test verifies fall detection by checking a manually bad state
        state = WalkState()
        state.min_pelvis_height = 0.5
        assert check_fall(state, 0.4, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.4])) is True

    def test_gpu_detection_runs(self):
        result = run_walking_demo(target_distance=0.01, duration_limit=5.0, verbose=False)
        # gpu_used should be a boolean (True if JAX GPU available, False otherwise)
        assert isinstance(result["gpu_used"], bool)
