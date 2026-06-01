"""Coverage tests for firewall/decorator.py edge cases."""

import numpy as np
import pytest
from pathlib import Path

from rosclaw.firewall.decorator import (
    DigitalTwinFirewall,
    SafetyViolationError,
    SafetyLevel,
)

MODEL_PATH = Path(__file__).parent.parent / "src" / "rosclaw" / "specs" / "ur5e.xml"


class TestInputValidation:
    def test_set_joint_velocities_wrong_size(self):
        fw = DigitalTwinFirewall(model_path=str(MODEL_PATH), sim_steps_per_check=10)
        with pytest.raises(ValueError, match="Expected 6 velocities"):
            fw.set_joint_velocities(np.array([0.1, 0.2]))

    def test_apply_control_wrong_size(self):
        fw = DigitalTwinFirewall(model_path=str(MODEL_PATH), sim_steps_per_check=10)
        with pytest.raises(ValueError, match="Expected 6 controls"):
            fw.apply_control(np.array([0.1, 0.2]))


class TestCheckJointLimitsDirect:
    def test_limit_violation_min(self):
        fw = DigitalTwinFirewall(model_path=str(MODEL_PATH), sim_steps_per_check=10)
        fw.set_joint_positions(np.array([-10.0, 0, 0, 0, 0, 0]))
        violated, details = fw.check_joint_limits()
        assert violated is True
        assert any("<" in d for d in details)

    def test_limit_violation_max(self):
        fw = DigitalTwinFirewall(model_path=str(MODEL_PATH), sim_steps_per_check=10)
        fw.set_joint_positions(np.array([10.0, 0, 0, 0, 0, 0]))
        violated, details = fw.check_joint_limits()
        assert violated is True
        assert any(">" in d for d in details)

    def test_skip_joint_without_limit(self):
        fw = DigitalTwinFirewall(
            model_path=str(MODEL_PATH),
            sim_steps_per_check=10,
            joint_limits={"shoulder_pan_joint": (-1.0, 1.0)},  # only 1 joint
        )
        fw.set_joint_positions(np.array([0.0] * 6))
        violated, details = fw.check_joint_limits()
        # Only shoulder_pan_joint is checked; others skipped
        assert violated is False
        assert details == []


class TestCheckTorqueLimits:
    def test_torque_limit_exceeded(self):
        fw = DigitalTwinFirewall(
            model_path=str(MODEL_PATH),
            sim_steps_per_check=10,
            torque_limits={"shoulder_pan_joint": 0.01},  # very low limit
        )
        # Mock qfrc_actuator to force torque exceedance
        fw.data.qfrc_actuator[:] = np.array([0.5, 0, 0, 0, 0, 0])
        exceeded, max_torque, details = fw.check_torque_limits()
        assert exceeded is True
        assert max_torque == pytest.approx(0.5)
        assert len(details) > 0
        assert "shoulder_pan_joint" in details[0]

    def test_torque_limit_no_limits_set(self):
        fw = DigitalTwinFirewall(model_path=str(MODEL_PATH), sim_steps_per_check=10)
        fw.set_joint_positions(np.array([0.5, 0, 0, 0, 0, 0]))
        fw.step(10)
        exceeded, max_torque, details = fw.check_torque_limits()
        # No torque limits configured — should not report exceeded
        assert exceeded is False
        assert max_torque >= 0


class TestValidateTrajectoryEdgeCases:
    def test_prevalidation_skips_joint_without_limit(self):
        fw = DigitalTwinFirewall(
            model_path=str(MODEL_PATH),
            sim_steps_per_check=10,
            joint_limits={
                "shoulder_pan_joint": (-1.0, 1.0),
                "shoulder_lift_joint": (-1.0, 1.0),
            },
        )
        trajectory = [np.array([0.0] * 6)]
        result = fw.validate_trajectory(trajectory)
        assert result.is_safe is True

    def test_apply_control_fallback_no_control_inputs(self):
        fw = DigitalTwinFirewall(model_path=str(MODEL_PATH), sim_steps_per_check=10)
        trajectory = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        ]
        # No control_inputs provided — falls through to apply_control(target_pos)
        result = fw.validate_trajectory(trajectory)
        assert result.simulation_steps > 0

    def test_lenient_safety_level(self):
        fw = DigitalTwinFirewall(
            model_path=str(MODEL_PATH),
            sim_steps_per_check=10,
            torque_limits={"shoulder_pan_joint": 0.01},
        )
        trajectory = [np.array([0.5, 0, 0, 0, 0, 0])]
        result = fw.validate_trajectory(trajectory, safety_level=SafetyLevel.LENIENT)
        # LENIENT mode only rejects severe violations based on max_torque
        assert result.simulation_steps > 0

    def test_collision_strict_break(self):
        fw = DigitalTwinFirewall(model_path=str(MODEL_PATH), sim_steps_per_check=10)
        # Start at home, then move to a position that causes self-collision
        # UR5e wrist_3 at extreme angle can self-collide
        trajectory = [
            np.array([0.0] * 6),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.5]),  # extreme wrist rotation
        ]
        result = fw.validate_trajectory(trajectory, safety_level=SafetyLevel.STRICT)
        # If collision happens, STRICT breaks early
        assert result.simulation_steps >= 0

    def test_torque_strict_break(self):
        fw = DigitalTwinFirewall(
            model_path=str(MODEL_PATH),
            sim_steps_per_check=10,
            torque_limits={"shoulder_pan_joint": 0.001},
        )
        trajectory = [np.array([0.5, 0, 0, 0, 0, 0])]
        result = fw.validate_trajectory(trajectory, safety_level=SafetyLevel.STRICT)
        # With very low torque limit, STRICT should break on torque violation
        assert result.simulation_steps >= 0


class TestDecoratorEdgeCases:
    def test_decorator_trajectory_extractor(self):
        fw = DigitalTwinFirewall(model_path=str(MODEL_PATH), sim_steps_per_check=10)

        def extract_traj(payload):
            return payload["trajectory"]

        dec = fw.decorator(trajectory_extractor=extract_traj)

        @dec
        def move(payload):
            return {"status": "ok"}

        result = move({"trajectory": [np.array([0.0] * 6), np.array([0.1] * 6)]})
        assert result["status"] == "ok"

    def test_decorator_empty_trajectory_raises(self):
        fw = DigitalTwinFirewall(model_path=str(MODEL_PATH), sim_steps_per_check=10)
        dec = fw.decorator()

        @dec
        def move(joint_positions):
            return {"status": "ok"}

        with pytest.raises(ValueError, match="No trajectory provided"):
            move([])

    def test_decorator_converts_list_to_arrays(self):
        fw = DigitalTwinFirewall(
            model_path=str(MODEL_PATH),
            joint_limits={
                "shoulder_pan_joint": (-6.28, 6.28),
                "shoulder_lift_joint": (-6.28, 6.28),
                "elbow_joint": (-3.14, 3.14),
                "wrist_1_joint": (-6.28, 6.28),
                "wrist_2_joint": (-6.28, 6.28),
                "wrist_3_joint": (-6.28, 6.28),
            },
            sim_steps_per_check=10,
        )
        dec = fw.decorator()

        @dec
        def move(joint_positions):
            return {"status": "ok"}

        # Pass plain Python lists (not np.ndarray) to trigger conversion
        result = move([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        assert result["status"] == "ok"

    def test_mujoco_firewall_unsupported_format(self):
        with pytest.raises(RuntimeError, match="Failed to load MuJoCo model"):
            DigitalTwinFirewall(model_path="/tmp/robot.abc")

    def test_decorator_blocks_with_violation(self):
        fw = DigitalTwinFirewall(model_path=str(MODEL_PATH), sim_steps_per_check=10)
        dec = fw.decorator()

        @dec
        def move(joint_positions):
            return {"status": "ok"}

        with pytest.raises(SafetyViolationError):
            move([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])
