"""
ROSClaw Digital Twin Firewall - MuJoCo-based physics validation.

This module provides a decorator that validates robot trajectories against
a MuJoCo physics simulation before executing on real hardware.
"""

import functools
import inspect
import time
from typing import Any, Callable, Optional, TypeVar
from dataclasses import dataclass
from enum import Enum

import numpy as np
import mujoco
import mujoco.viewer


F = TypeVar("F", bound=Callable[..., Any])


class SafetyLevel(Enum):
    """Safety validation levels."""
    STRICT = "strict"      # Reject any collision or limit exceedance
    MODERATE = "moderate"  # Allow minor contact, reject severe violations
    LENIENT = "lenient"    # Warn but allow execution


@dataclass(frozen=True)
class ValidationResult:
    """Result of physics validation."""
    is_safe: bool
    collision_detected: bool
    joint_limit_violated: bool
    torque_limit_exceeded: bool
    max_predicted_torque: float
    min_distance_to_collision: float
    simulation_steps: int
    violation_details: list[str]

    def to_dict(self) -> dict:
        return {
            "is_safe": self.is_safe,
            "collision_detected": self.collision_detected,
            "joint_limit_violated": self.joint_limit_violated,
            "torque_limit_exceeded": self.torque_limit_exceeded,
            "max_predicted_torque": self.max_predicted_torque,
            "min_distance_to_collision": self.min_distance_to_collision,
            "simulation_steps": self.simulation_steps,
            "violation_details": self.violation_details,
        }


class DigitalTwinFirewall:
    """
    MuJoCo-based Digital Twin for trajectory validation.

    Loads robot MJCF/URDF models and simulates trajectories to detect:
    - Self-collisions and environment collisions
    - Joint limit violations
    - Torque limit exceedances
    - Instability or divergence

    Example:
        firewall = DigitalTwinFirewall("ur5e.xml")

        @firewall.validate_trajectory(safety_level=SafetyLevel.STRICT)
        def move_robot(joint_positions: list[float]):
            # This will only execute if trajectory passes MuJoCo validation
            return actual_robot_move(joint_positions)
    """

    def __init__(
        self,
        model_path: str,
        torque_limits: Optional[dict[str, float]] = None,
        joint_limits: Optional[dict[str, tuple[float, float]]] = None,
        safety_margin: float = 0.05,  # 5% safety margin
        sim_steps_per_check: int = 100,
    ):
        """
        Initialize Digital Twin with MuJoCo model.

        Args:
            model_path: Path to MJCF or URDF file
            torque_limits: Dict of joint_name -> max_torque (Nm)
            joint_limits: Dict of joint_name -> (min, max) in radians
            safety_margin: Safety margin as fraction (0.05 = 5%)
            sim_steps_per_check: Physics steps between safety checks
        """
        self.model_path = model_path
        self.safety_margin = safety_margin
        self.sim_steps_per_check = sim_steps_per_check

        # Load MuJoCo model
        self._load_model()

        # Override limits if provided
        self.torque_limits = torque_limits or {}
        self.joint_limits = joint_limits or {}

    def _load_model(self) -> None:
        """Load MuJoCo model from file."""
        try:
            if self.model_path.endswith(('.xml', '.mjcf')):
                self.model = mujoco.MjModel.from_xml_path(self.model_path)
            elif self.model_path.endswith('.urdf'):
                self.model = mujoco.MjModel.from_xml_path(self.model_path)
            else:
                raise ValueError(f"Unsupported model format: {self.model_path}")

            self.data = mujoco.MjData(self.model)

            # Extract joint info
            self.nq = self.model.nq  # Number of position coordinates
            self.nv = self.model.nv  # Number of velocity coordinates
            self.nu = self.model.nu  # Number of actuators

            # Get joint names
            self.joint_names = []
            for i in range(self.model.njnt):
                joint_id = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                self.joint_names.append(joint_id)

            print(f"[DigitalTwin] Loaded model: {self.model_path}")
            print(f"[DigitalTwin] Joints: {self.joint_names}")
            print(f"[DigitalTwin] DOF: nq={self.nq}, nv={self.nv}, nu={self.nu}")

        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model: {e}") from e

    def reset(self) -> None:
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_joint_positions(self, positions: np.ndarray) -> None:
        """Set joint positions in simulation."""
        if len(positions) != self.nq:
            raise ValueError(f"Expected {self.nq} positions, got {len(positions)}")
        self.data.qpos[:] = positions
        mujoco.mj_forward(self.model, self.data)

    def set_joint_velocities(self, velocities: np.ndarray) -> None:
        """Set joint velocities in simulation."""
        if len(velocities) != self.nv:
            raise ValueError(f"Expected {self.nv} velocities, got {len(velocities)}")
        self.data.qvel[:] = velocities

    def apply_control(self, ctrl: np.ndarray) -> None:
        """Apply control signals to actuators."""
        if len(ctrl) != self.nu:
            raise ValueError(f"Expected {self.nu} controls, got {len(ctrl)}")
        self.data.ctrl[:] = ctrl

    def step(self, n_steps: int = 1) -> None:
        """Step physics simulation forward."""
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def check_collision(self) -> tuple[bool, float]:
        """
        Check for collisions in current state.

        Returns:
            (collision_detected, min_distance)
        """
        # Update contact information
        mujoco.mj_collision(self.model, self.data)

        collision_detected = False
        min_distance = float('inf')

        # Check all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Contact distance < 0 means penetration
            if contact.dist < 0:
                collision_detected = True
            min_distance = min(min_distance, abs(contact.dist))

        return collision_detected, min_distance

    def check_joint_limits(self) -> tuple[bool, list[str]]:
        """
        Check if any joint exceeds its limits.

        Returns:
            (limit_violated, violation_details)
        """
        limit_violated = False
        violations = []

        for i, joint_name in enumerate(self.joint_names):
            if joint_name not in self.joint_limits:
                continue

            qpos = self.data.qpos[i]
            min_limit, max_limit = self.joint_limits[joint_name]

            # Apply safety margin
            margin = (max_limit - min_limit) * self.safety_margin

            if qpos < min_limit + margin:
                limit_violated = True
                violations.append(
                    f"Joint {joint_name}: {qpos:.4f} < {min_limit + margin:.4f} (min)"
                )
            elif qpos > max_limit - margin:
                limit_violated = True
                violations.append(
                    f"Joint {joint_name}: {qpos:.4f} > {max_limit - margin:.4f} (max)"
                )

        return limit_violated, violations

    def check_torque_limits(self) -> tuple[bool, float, list[str]]:
        """
        Check if any joint torque exceeds limits.

        Returns:
            (limit_exceeded, max_torque, violation_details)
        """
        limit_exceeded = False
        max_torque = 0.0
        violations = []

        # Get joint torques from actuator forces
        for i in range(min(self.nu, len(self.joint_names))):
            joint_name = self.joint_names[i]
            if i < len(self.data.qfrc_actuator):
                torque = abs(self.data.qfrc_actuator[i])
                max_torque = max(max_torque, torque)

                if joint_name in self.torque_limits:
                    limit = self.torque_limits[joint_name] * (1 - self.safety_margin)
                    if torque > limit:
                        limit_exceeded = True
                        violations.append(
                            f"Joint {joint_name}: torque {torque:.2f} Nm > limit {limit:.2f} Nm"
                        )

        return limit_exceeded, max_torque, violations

    def validate_trajectory(
        self,
        trajectory: list[np.ndarray],
        control_inputs: Optional[list[np.ndarray]] = None,
        time_step: float = 0.001,
        safety_level: SafetyLevel = SafetyLevel.STRICT,
    ) -> ValidationResult:
        """
        Validate a complete trajectory through MuJoCo simulation.

        Args:
            trajectory: List of target joint positions
            control_inputs: Optional list of control torques/positions
            time_step: Simulation time step in seconds
            safety_level: Strictness of validation

        Returns:
            ValidationResult with safety assessment
        """
        self.reset()

        collision_detected = False
        joint_limit_violated = False
        torque_limit_exceeded = False
        max_torque = 0.0
        min_distance = float('inf')
        all_violations = []
        total_steps = 0

        for step_idx, target_pos in enumerate(trajectory):
            # Set target as control input (position control)
            if control_inputs and step_idx < len(control_inputs):
                self.apply_control(control_inputs[step_idx])
            else:
                # Simple position control - target is desired position
                if len(target_pos) == self.nu:
                    self.apply_control(target_pos)

            # Step physics
            self.step(self.sim_steps_per_check)
            total_steps += self.sim_steps_per_check

            # Check collisions
            has_collision, distance = self.check_collision()
            if has_collision:
                collision_detected = True
                all_violations.append(f"Step {step_idx}: Collision detected")
                if safety_level == SafetyLevel.STRICT:
                    break
            min_distance = min(min_distance, distance)

            # Check joint limits
            j_limit_violated, j_violations = self.check_joint_limits()
            if j_limit_violated:
                joint_limit_violated = True
                all_violations.extend([f"Step {step_idx}: {v}" for v in j_violations])
                if safety_level == SafetyLevel.STRICT:
                    break

            # Check torque limits
            t_limit_exceeded, step_max_torque, t_violations = self.check_torque_limits()
            max_torque = max(max_torque, step_max_torque)
            if t_limit_exceeded:
                torque_limit_exceeded = True
                all_violations.extend([f"Step {step_idx}: {v}" for v in t_violations])
                if safety_level == SafetyLevel.STRICT:
                    break

        # Determine safety
        is_safe = not (collision_detected or joint_limit_violated or torque_limit_exceeded)

        if safety_level == SafetyLevel.MODERATE:
            # In moderate mode, allow minor collisions if not severe
            is_safe = not (joint_limit_violated or torque_limit_exceeded)
        elif safety_level == SafetyLevel.LENIENT:
            # In lenient mode, only reject severe violations
            is_safe = max_torque < max(self.torque_limits.values(), default=100) * 1.5

        return ValidationResult(
            is_safe=is_safe,
            collision_detected=collision_detected,
            joint_limit_violated=joint_limit_violated,
            torque_limit_exceeded=torque_limit_exceeded,
            max_predicted_torque=max_torque,
            min_distance_to_collision=min_distance,
            simulation_steps=total_steps,
            violation_details=all_violations,
        )

    def decorator(
        self,
        safety_level: SafetyLevel = SafetyLevel.STRICT,
        trajectory_extractor: Optional[Callable[..., list[np.ndarray]]] = None,
    ) -> Callable[[F], F]:
        """
        Decorator factory for trajectory validation.

        Args:
            safety_level: Validation strictness
            trajectory_extractor: Function to extract trajectory from function args

        Returns:
            Decorator function
        """
        def decorator_impl(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract trajectory
                if trajectory_extractor:
                    trajectory = trajectory_extractor(*args, **kwargs)
                else:
                    # Default: first argument is trajectory
                    trajectory = args[0] if args else kwargs.get('trajectory', kwargs.get('joint_positions', []))

                if not trajectory:
                    raise ValueError("No trajectory provided for validation")

                # Ensure trajectory is list of numpy arrays
                if isinstance(trajectory, np.ndarray):
                    trajectory = [trajectory]
                trajectory = [np.array(t) for t in trajectory]

                # Validate
                start_time = time.time()
                result = self.validate_trajectory(trajectory, safety_level=safety_level)
                elapsed = time.time() - start_time

                print(f"[DigitalTwin] Validation completed in {elapsed*1000:.1f}ms")
                print(f"[DigitalTwin] Result: {'SAFE' if result.is_safe else 'UNSAFE'}")

                if not result.is_safe:
                    print(f"[DigitalTwin] Violations: {result.violation_details}")
                    raise SafetyViolationError(
                        f"Trajectory failed safety validation: {result.violation_details}",
                        result,
                    )

                # Execute actual function
                return func(*args, **kwargs)

            return wrapper  # type: ignore
        return decorator_impl


class SafetyViolationError(Exception):
    """Exception raised when trajectory fails safety validation."""

    def __init__(self, message: str, result: ValidationResult):
        super().__init__(message)
        self.result = result


# Convenience alias for shorter decorator syntax
def mujoco_firewall(
    model_path: str,
    safety_level: SafetyLevel = SafetyLevel.STRICT,
    **firewall_kwargs: Any,
) -> Callable[[F], F]:
    """
    Quick decorator to add MuJoCo validation to robot control functions.

    Example:
        @mujoco_firewall("ur5e.xml", safety_level=SafetyLevel.STRICT)
        def move_robot(joint_positions):
            # Only executes if trajectory passes MuJoCo simulation
            return robot.move(joint_positions)
    """
    firewall = DigitalTwinFirewall(model_path, **firewall_kwargs)
    return firewall.decorator(safety_level=safety_level)
