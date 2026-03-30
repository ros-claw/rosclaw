"""
Unit tests for ROSClaw Digital Twin Firewall.

Tests the DigitalTwinFirewall class and mujoco_firewall decorator.
"""

import numpy as np
import pytest
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rosclaw.firewall import (
    DigitalTwinFirewall,
    SafetyViolationError,
    SafetyLevel,
    mujoco_firewall,
)


# Path to test model
MODEL_PATH = Path(__file__).parent.parent / "src" / "rosclaw" / "specs" / "ur5e.xml"

# UR5e joint limits
JOINT_LIMITS = {
    "shoulder_pan_joint": (-6.2831853, 6.2831853),
    "shoulder_lift_joint": (-6.2831853, 6.2831853),
    "elbow_joint": (-3.1415926, 3.1415926),
    "wrist_1_joint": (-6.2831853, 6.2831853),
    "wrist_2_joint": (-6.2831853, 6.2831853),
    "wrist_3_joint": (-6.2831853, 6.2831853),
}


@pytest.fixture
def firewall():
    """Create a DigitalTwinFirewall instance for testing."""
    return DigitalTwinFirewall(
        model_path=str(MODEL_PATH),
        joint_limits=JOINT_LIMITS,
        sim_steps_per_check=10,
    )


class TestDigitalTwinFirewall:
    """Tests for DigitalTwinFirewall class."""

    def test_initialization(self, firewall):
        """Test firewall initializes correctly."""
        assert firewall.model_path == str(MODEL_PATH)
        assert firewall.nq == 6  # 6 DOF for UR5e
        assert firewall.nv == 6
        assert firewall.nu == 6
        assert len(firewall.joint_names) == 6
        assert firewall.joint_limits == JOINT_LIMITS

    def test_initialization_no_limits(self):
        """Test firewall initializes without explicit joint limits."""
        fw = DigitalTwinFirewall(
            model_path=str(MODEL_PATH),
            sim_steps_per_check=10,
        )
        assert fw.joint_limits == {}

    def test_reset(self, firewall):
        """Test reset clears simulation state."""
        firewall.set_joint_positions(np.array([0.5, 0, 0, 0, 0, 0]))
        firewall.reset()
        assert np.allclose(firewall.data.qpos, np.zeros(6))

    def test_set_joint_positions(self, firewall):
        """Test setting joint positions."""
        positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        firewall.set_joint_positions(positions)
        assert np.allclose(firewall.data.qpos, positions)

    def test_set_joint_positions_wrong_size(self, firewall):
        """Test setting joint positions with wrong size raises error."""
        with pytest.raises(ValueError, match="Expected 6 positions"):
            firewall.set_joint_positions(np.array([0.1, 0.2]))

    def test_set_joint_velocities(self, firewall):
        """Test setting joint velocities."""
        velocities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        firewall.set_joint_velocities(velocities)
        assert np.allclose(firewall.data.qvel, velocities)

    def test_apply_control(self, firewall):
        """Test applying control signals."""
        control = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        firewall.apply_control(control)
        assert np.allclose(firewall.data.ctrl, control)


class TestJointLimitValidation:
    """Tests for joint limit validation."""

    def test_valid_trajectory(self, firewall):
        """Test that valid trajectories pass validation."""
        trajectory = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

        result = firewall.validate_trajectory(trajectory)

        assert result.is_safe is True
        assert result.joint_limit_violated is False
        assert result.collision_detected is False
        assert len(result.violation_details) == 0

    def test_joint_limit_exceeded_max(self, firewall):
        """Test that exceeding max joint limit is detected."""
        trajectory = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Exceeds limit
        ]

        result = firewall.validate_trajectory(trajectory)

        assert result.is_safe is False
        assert result.joint_limit_violated is True
        assert len(result.violation_details) > 0
        assert "shoulder_pan_joint" in result.violation_details[0]
        assert "exceeds limits" in result.violation_details[0]

    def test_joint_limit_exceeded_min(self, firewall):
        """Test that exceeding min joint limit is detected."""
        trajectory = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Exceeds min limit
        ]

        result = firewall.validate_trajectory(trajectory)

        assert result.is_safe is False
        assert result.joint_limit_violated is True

    def test_elbow_joint_limit(self, firewall):
        """Test elbow joint has stricter limits."""
        trajectory = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0]),  # Exceeds elbow limit (~3.14)
        ]

        result = firewall.validate_trajectory(trajectory)

        assert result.is_safe is False
        assert result.joint_limit_violated is True
        assert "elbow_joint" in result.violation_details[0]

    def test_joint_at_exact_limit(self, firewall):
        """Test that joint at exact limit boundary passes."""
        trajectory = [
            np.array([6.2831853, 0.0, 0.0, 0.0, 0.0, 0.0]),  # At max limit
        ]

        result = firewall.validate_trajectory(trajectory)

        assert result.is_safe is True


class TestCollisionDetection:
    """Tests for collision detection."""

    def test_no_collision_in_home_position(self, firewall):
        """Test that home position has no self-collision."""
        # Set to home position
        firewall.set_joint_positions(np.zeros(6))
        has_collision, min_dist = firewall.check_collision()
        assert has_collision is False

    def test_trajectory_collision_check(self, firewall):
        """Test collision detection during trajectory validation."""
        trajectory = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

        result = firewall.validate_trajectory(trajectory)

        # Should complete without collision errors
        assert result.simulation_steps > 0


class TestTorqueLimits:
    """Tests for torque limit validation."""

    def test_torque_limits_defined(self, firewall):
        """Test that torque limits can be configured."""
        torque_limits = {
            "shoulder_pan_joint": 150.0,
            "shoulder_lift_joint": 150.0,
            "elbow_joint": 100.0,
        }
        fw = DigitalTwinFirewall(
            model_path=str(MODEL_PATH),
            joint_limits=JOINT_LIMITS,
            torque_limits=torque_limits,
        )
        assert fw.torque_limits == torque_limits


class TestSafetyLevels:
    """Tests for different safety levels."""

    def test_strict_safety_level(self, firewall):
        """Test STRICT mode rejects any violation."""
        trajectory = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

        result = firewall.validate_trajectory(
            trajectory, safety_level=SafetyLevel.STRICT
        )

        assert result.is_safe is False

    def test_moderate_safety_level_collision_allowed(self, firewall):
        """Test MODERATE mode allows minor collisions."""
        # This is a conceptual test - actual collision behavior depends on model
        trajectory = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

        result = firewall.validate_trajectory(
            trajectory, safety_level=SafetyLevel.MODERATE
        )

        # Should not crash, MODERATE mode is valid
        assert result.is_safe is True


class TestDecorator:
    """Tests for mujoco_firewall decorator."""

    def test_decorator_passes_valid_trajectory(self):
        """Test decorator allows valid trajectories."""
        @mujoco_firewall(
            model_path=str(MODEL_PATH),
            joint_limits=JOINT_LIMITS,
            safety_level=SafetyLevel.STRICT,
        )
        def move_robot(joint_positions):
            return {"status": "success", "positions": joint_positions}

        result = move_robot([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        assert result["status"] == "success"

    def test_decorator_blocks_invalid_trajectory(self):
        """Test decorator raises SafetyViolationError for invalid trajectories."""
        @mujoco_firewall(
            model_path=str(MODEL_PATH),
            joint_limits=JOINT_LIMITS,
            safety_level=SafetyLevel.STRICT,
        )
        def move_robot(joint_positions):
            return {"status": "success"}

        with pytest.raises(SafetyViolationError) as exc_info:
            move_robot([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])

        assert "failed safety validation" in str(exc_info.value)
        assert exc_info.value.result is not None
        assert exc_info.value.result.joint_limit_violated is True

    def test_decorator_with_no_limits(self):
        """Test decorator without explicit limits."""
        @mujoco_firewall(model_path=str(MODEL_PATH))
        def move_robot(joint_positions):
            return {"status": "success"}

        # Without limits, should still work for valid positions
        result = move_robot([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        assert result["status"] == "success"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_to_dict(self):
        """Test ValidationResult can be converted to dict."""
        result = DigitalTwinFirewall.validate_trajectory(
            DigitalTwinFirewall(model_path=str(MODEL_PATH), joint_limits=JOINT_LIMITS),
            [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
        ) if False else None  # Placeholder, create manually

        # Create result directly
        from rosclaw.firewall.decorator import ValidationResult
        result = ValidationResult(
            is_safe=True,
            collision_detected=False,
            joint_limit_violated=False,
            torque_limit_exceeded=False,
            max_predicted_torque=0.5,
            min_distance_to_collision=0.1,
            simulation_steps=10,
            violation_details=[],
        )

        d = result.to_dict()
        assert d["is_safe"] is True
        assert d["collision_detected"] is False
        assert d["max_predicted_torque"] == 0.5


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_trajectory(self, firewall):
        """Test that empty trajectory is handled."""
        result = firewall.validate_trajectory([])

        # Empty trajectory should be considered safe (nothing to validate)
        assert result.is_safe is True
        assert result.simulation_steps == 0

    def test_single_point_trajectory(self, firewall):
        """Test single point trajectory."""
        trajectory = [np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])]

        result = firewall.validate_trajectory(trajectory)

        assert result.is_safe is True
        assert result.simulation_steps > 0

    def test_numpy_array_trajectory(self, firewall):
        """Test trajectory as numpy array."""
        trajectory = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        result = firewall.validate_trajectory(trajectory)

        assert result.is_safe is True

    def test_invalid_model_path(self):
        """Test that invalid model path raises error."""
        with pytest.raises(RuntimeError, match="Failed to load MuJoCo model"):
            DigitalTwinFirewall(model_path="/nonexistent/model.xml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
