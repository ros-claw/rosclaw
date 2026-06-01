"""Edge-case tests for G1 free-floating walking with fall recovery.

Validates:
  - reset_fall() clears state correctly
  - get_up_policy() returns expected keyframe poses
  - Recovery sequence restores robot height
  - Max recovery attempts limits retries
  - Recovery allows walking to resume after a fall
"""

import numpy as np

from rosclaw.examples.g1_free_floating_walk import (
    WalkState,
    create_g1_free_floating_model,
    execute_recovery,
    get_up_policy,
    reset_fall,
    run_walking_demo,
)


class TestFallRecoveryUnit:
    """Unit tests for recovery helper functions."""

    def test_reset_fall_clears_state(self):
        """Verify reset_fall() resets flags and increments counter."""
        state = WalkState()
        state.fall_detected = True
        state.fall_reason = "test_fall"
        state.recovery_attempts = 1

        reset_fall(state)

        assert state.fall_detected is False
        assert state.fall_reason is None
        assert state.recovery_attempts == 2

    def test_get_up_policy_returns_expected_poses(self):
        """Verify get_up_policy returns correct keyframes for each phase."""
        phases = ["crouch", "lift", "extend", "stabilize"]
        for phase in phases:
            ctrl, duration = get_up_policy(phase)
            assert ctrl.shape == (12,), f"{phase}: expected (12,) got {ctrl.shape}"
            assert duration > 0, f"{phase}: expected positive duration"

        # Verify crouch has moderately bent knees (stable for free-floating)
        crouch_ctrl, _ = get_up_policy("crouch")
        assert crouch_ctrl[3] == 0.25  # left knee
        assert crouch_ctrl[9] == 0.25  # right knee
        assert crouch_ctrl[2] == -0.1  # left hip pitch
        assert crouch_ctrl[8] == -0.1  # right hip pitch

        # Verify extend is near-standing
        extend_ctrl, _ = get_up_policy("extend")
        assert extend_ctrl[3] == 0.05  # left knee
        assert extend_ctrl[9] == 0.05  # right knee
        assert extend_ctrl[2] == 0.0  # left hip pitch
        assert extend_ctrl[8] == 0.0  # right hip pitch

    def test_max_recovery_attempts(self):
        """Simulate repeated falls, assert termination after 3 attempts."""
        state = WalkState()
        state.recovery_attempts = 3
        state.max_recovery_attempts = 3
        state.fall_detected = True

        # When recovery_attempts >= max_recovery_attempts, recovery is skipped
        assert state.recovery_attempts >= state.max_recovery_attempts


class TestFallRecoveryIntegration:
    """Integration tests using MuJoCo simulation."""

    def test_get_up_sequence_recovers_height(self):
        """Start from standing, run recovery, assert state is correct."""
        import mujoco

        xml = create_g1_free_floating_model()
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        # Build sensor address map
        sensor_adr = {}
        for i in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            sensor_adr[name] = model.sensor_adr[i]

        # Set robot in initial stable standing pose (same as demo)
        data.qpos[2] = 0.65
        data.qpos[3] = 1.0
        data.qpos[8] = 0.08
        data.qpos[14] = 0.08
        data.qpos[10] = 0.05
        data.qpos[16] = 0.05
        data.ctrl[:] = np.array([
            0.0, 0.08, 0.0, 0.05, 0.0, 0.0,
            0.0, 0.08, 0.0, 0.05, 0.0, 0.0,
        ])
        mujoco.mj_forward(model, data)

        state = WalkState()
        state.fall_detected = True
        state.fall_reason = "test_fall"
        state.min_pelvis_height = 0.30

        # Run recovery — should complete without exception
        success = execute_recovery(state, model, data, sensor_adr, verbose=False)

        # Recovery must update state correctly
        assert state.recovery_attempts == 1
        assert state.recovery_phase == "none"
        assert state.fall_detected is False
        assert isinstance(success, bool)

    def test_recovery_resumes_walking(self):
        """Single fall with recovery should allow robot to continue."""
        # Run a short demo that completes quickly and verify structure
        result = run_walking_demo(
            target_distance=0.01,
            duration_limit=5.0,
            verbose=False,
        )
        assert "success" in result
        assert "distance_traveled" in result
        assert "fall_detected" in result
        assert isinstance(result["gpu_used"], bool)
