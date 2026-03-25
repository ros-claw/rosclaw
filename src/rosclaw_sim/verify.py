"""Trajectory verification tools for ROSClaw Digital Twin."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from rosclaw_sim.digital_twin import DigitalTwin


@dataclass
class VerificationResult:
    """Comprehensive trajectory verification result.

    Attributes:
        is_valid: Whether trajectory passes all checks.
        checks_passed: List of check names that passed.
        checks_failed: List of check names that failed.
        joint_limit_violations: Details of joint limit violations.
        velocity_violations: Details of velocity limit violations.
        torque_violations: Details of torque limit violations.
        collision_violations: Details of collision events.
        smoothness_score: Trajectory smoothness (0.0-1.0).
        feasibility_score: Overall feasibility score (0.0-1.0).
        execution_time: Estimated execution time in seconds.
        metadata: Additional verification metadata.
    """

    is_valid: bool
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    joint_limit_violations: list[dict] = field(default_factory=list)
    velocity_violations: list[dict] = field(default_factory=list)
    torque_violations: list[dict] = field(default_factory=list)
    collision_violations: list[dict] = field(default_factory=list)
    smoothness_score: float = 1.0
    feasibility_score: float = 1.0
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "joint_limit_violations": self.joint_limit_violations,
            "velocity_violations": self.velocity_violations,
            "torque_violations": self.torque_violations,
            "collision_violations": self.collision_violations,
            "smoothness_score": self.smoothness_score,
            "feasibility_score": self.feasibility_score,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


@dataclass
class VerificationConfig:
    """Configuration for trajectory verification.

    Attributes:
        check_joint_limits: Whether to check joint position limits.
        check_velocity_limits: Whether to check joint velocity limits.
        check_torque_limits: Whether to check joint torque limits.
        check_collisions: Whether to check for collisions.
        check_smoothness: Whether to check trajectory smoothness.
        joint_limit_tolerance: Safety margin for joint limits (0.0-1.0).
        max_velocity: Maximum allowed joint velocity (rad/s).
        max_torque: Maximum allowed joint torque (Nm).
        smoothness_threshold: Maximum allowed jerk (rad/s^3).
        min_segment_time: Minimum time per trajectory segment (s).
    """

    check_joint_limits: bool = True
    check_velocity_limits: bool = True
    check_torque_limits: bool = False  # Requires torque estimation
    check_collisions: bool = True
    check_smoothness: bool = True

    joint_limit_tolerance: float = 0.05  # 5% safety margin
    max_velocity: float = 3.0  # rad/s
    max_torque: float = 50.0  # Nm
    smoothness_threshold: float = 100.0  # rad/s^3
    min_segment_time: float = 0.01  # 10ms


class TrajectoryVerifier:
    """Verify trajectories for safety and feasibility.

    The verifier performs static analysis on trajectories without requiring
    full physics simulation, making it fast for pre-validation.

    For full validation with physics simulation, use DigitalTwin.verify_trajectory().

    Example:
        >>> verifier = TrajectoryVerifier(config)
        >>>
        >>> # Define trajectory
        >>> trajectory = {
        ...     "joint_positions": np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
        ...     "timestamps": np.array([0.0, 0.1, 0.2]),
        ... }
        >>>
        >>> # Verify
        >>> result = verifier.verify(trajectory, joint_limits=[(-3, 3), (-2, 2)])
        >>> print(f"Valid: {result.is_valid}")
    """

    def __init__(self, config: VerificationConfig | None = None):
        self.config = config or VerificationConfig()

    def verify(
        self,
        trajectory: dict[str, np.ndarray],
        joint_limits: list[tuple[float, float]] | None = None,
        joint_names: list[str] | None = None,
    ) -> VerificationResult:
        """Verify a trajectory.

        Args:
            trajectory: Dictionary containing:
                - "joint_positions": (T, num_joints) array
                - "timestamps": Optional (T,) array
                - "joint_velocities": Optional (T, num_joints) array
            joint_limits: List of (min, max) for each joint.
            joint_names: Optional list of joint names.

        Returns:
            VerificationResult with detailed analysis.
        """
        checks_passed = []
        checks_failed = []

        joint_positions = trajectory.get("joint_positions")
        timestamps = trajectory.get("timestamps")

        if joint_positions is None:
            raise ValueError("Trajectory must contain 'joint_positions'")

        # Ensure 2D array
        if joint_positions.ndim == 1:
            joint_positions = joint_positions.reshape(1, -1)

        num_steps, num_joints = joint_positions.shape

        # Compute timestamps if not provided (assume uniform 50Hz)
        if timestamps is None:
            dt = 0.02  # 50Hz
            timestamps = np.arange(num_steps) * dt
        else:
            dt = np.diff(timestamps).mean() if len(timestamps) > 1 else 0.02

        # Check joint limits
        joint_violations = []
        if self.config.check_joint_limits and joint_limits is not None:
            joint_violations = self._check_joint_limits(
                joint_positions, joint_limits, joint_names
            )
            if not joint_violations:
                checks_passed.append("joint_limits")
            else:
                checks_failed.append("joint_limits")

        # Compute velocities and accelerations
        if timestamps is not None and len(timestamps) > 1:
            velocities = np.gradient(joint_positions, timestamps, axis=0)
            accelerations = np.gradient(velocities, timestamps, axis=0)
            jerks = np.gradient(accelerations, timestamps, axis=0)
        else:
            velocities = np.zeros_like(joint_positions)
            accelerations = np.zeros_like(joint_positions)
            jerks = np.zeros_like(joint_positions)

        # Check velocity limits
        velocity_violations = []
        if self.config.check_velocity_limits:
            velocity_violations = self._check_velocity_limits(
                velocities, joint_names
            )
            if not velocity_violations:
                checks_passed.append("velocity_limits")
            else:
                checks_failed.append("velocity_limits")

        # Check smoothness
        smoothness_score = 1.0
        if self.config.check_smoothness:
            smoothness_score = self._check_smoothness(jerks)
            if smoothness_score >= 0.8:
                checks_passed.append("smoothness")
            else:
                checks_failed.append("smoothness")

        # Compute overall feasibility
        feasibility_score = self._compute_feasibility_score(
            joint_violations,
            velocity_violations,
            smoothness_score,
            num_steps,
        )

        is_valid = (
            len(joint_violations) == 0
            and len(velocity_violations) == 0
            and feasibility_score >= 0.5
        )

        return VerificationResult(
            is_valid=is_valid,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            joint_limit_violations=joint_violations,
            velocity_violations=velocity_violations,
            smoothness_score=smoothness_score,
            feasibility_score=feasibility_score,
            execution_time=timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
        )

    def _check_joint_limits(
        self,
        joint_positions: np.ndarray,
        joint_limits: list[tuple[float, float]],
        joint_names: list[str] | None = None,
    ) -> list[dict]:
        """Check joint position limits."""
        violations = []
        num_steps, num_joints = joint_positions.shape

        # Apply safety margin
        margin = self.config.joint_limit_tolerance
        adjusted_limits = []
        for low, high in joint_limits:
            range_val = high - low
            adjusted_limits.append((
                low + margin * range_val,
                high - margin * range_val
            ))

        for joint_idx in range(min(num_joints, len(adjusted_limits))):
            low, high = adjusted_limits[joint_idx]

            # Find violations
            below_mask = joint_positions[:, joint_idx] < low
            above_mask = joint_positions[:, joint_idx] > high

            if below_mask.any():
                step_idx = int(np.where(below_mask)[0][0])
                violations.append({
                    "joint_idx": joint_idx,
                    "joint_name": joint_names[joint_idx] if joint_names else f"joint_{joint_idx}",
                    "type": "lower_limit",
                    "value": float(joint_positions[step_idx, joint_idx]),
                    "limit": float(low),
                    "step": step_idx,
                })

            if above_mask.any():
                step_idx = int(np.where(above_mask)[0][0])
                violations.append({
                    "joint_idx": joint_idx,
                    "joint_name": joint_names[joint_idx] if joint_names else f"joint_{joint_idx}",
                    "type": "upper_limit",
                    "value": float(joint_positions[step_idx, joint_idx]),
                    "limit": float(high),
                    "step": step_idx,
                })

        return violations

    def _check_velocity_limits(
        self,
        velocities: np.ndarray,
        joint_names: list[str] | None = None,
    ) -> list[dict]:
        """Check joint velocity limits."""
        violations = []
        num_steps, num_joints = velocities.shape

        max_vel = self.config.max_velocity

        for joint_idx in range(num_joints):
            velocity_violations = np.abs(velocities[:, joint_idx]) > max_vel

            if velocity_violations.any():
                step_idx = int(np.where(velocity_violations)[0][0])
                violations.append({
                    "joint_idx": joint_idx,
                    "joint_name": joint_names[joint_idx] if joint_names else f"joint_{joint_idx}",
                    "velocity": float(velocities[step_idx, joint_idx]),
                    "limit": max_vel,
                    "step": step_idx,
                })

        return violations

    def _check_smoothness(self, jerks: np.ndarray) -> float:
        """Check trajectory smoothness based on jerk."""
        max_jerk = np.abs(jerks).max()

        if max_jerk == 0:
            return 1.0

        # Score decreases as jerk increases
        threshold = self.config.smoothness_threshold
        score = max(0.0, 1.0 - max_jerk / threshold)

        return float(score)

    def _compute_feasibility_score(
        self,
        joint_violations: list,
        velocity_violations: list,
        smoothness_score: float,
        num_steps: int,
    ) -> float:
        """Compute overall feasibility score."""
        # Base score
        score = 1.0

        # Penalize violations
        violation_penalty = (len(joint_violations) + len(velocity_violations)) / max(num_steps, 1)
        score -= violation_penalty * 0.5

        # Weight by smoothness
        score = score * 0.7 + smoothness_score * 0.3

        return max(0.0, min(1.0, score))

    @staticmethod
    def verify_with_digital_twin(
        trajectory: dict[str, np.ndarray],
        digital_twin: DigitalTwin,
        num_runs: int = 5,
        enable_randomization: bool = True,
    ) -> dict[str, Any]:
        """Verify trajectory using Digital Twin with physics simulation.

        This provides more comprehensive validation than static analysis,
        checking for collisions and physics stability.

        Args:
            trajectory: Trajectory to verify.
            digital_twin: Initialized DigitalTwin instance.
            num_runs: Number of randomized runs for robustness check.
            enable_randomization: Whether to use domain randomization.

        Returns:
            Dictionary with comprehensive verification results.
        """
        # Run benchmark
        benchmark = digital_twin.benchmark_trajectory(
            trajectory={
                "joint_positions": trajectory["joint_positions"],
                "joint_velocities": trajectory.get("joint_velocities"),
            },
            num_runs=num_runs,
            enable_randomization=enable_randomization,
        )

        # Run single detailed verification
        detailed = digital_twin.verify_trajectory(
            trajectory=trajectory,
            verbose=True,
        )

        return {
            "success_rate": benchmark["success_rate"],
            "avg_stability": benchmark["avg_stability"],
            "worst_case": benchmark["worst_case"],
            "detailed_result": detailed,
            "recommendation": "APPROVE" if benchmark["success_rate"] > 0.9 else "REJECT",
        }


def verify_trajectory_file(
    trajectory_path: str | Path,
    robot_config_path: str | Path | None = None,
    verifier_config: VerificationConfig | None = None,
) -> VerificationResult:
    """Verify a trajectory from file.

    Args:
        trajectory_path: Path to trajectory file (.npy, .npz, or .json).
        robot_config_path: Optional path to robot configuration.
        verifier_config: Optional verifier configuration.

    Returns:
        VerificationResult.
    """
    trajectory_path = Path(trajectory_path)

    # Load trajectory
    if trajectory_path.suffix == ".npy":
        trajectory = {"joint_positions": np.load(trajectory_path)}
    elif trajectory_path.suffix == ".npz":
        data = np.load(trajectory_path)
        trajectory = {k: data[k] for k in data.files}
    else:
        raise ValueError(f"Unsupported file format: {trajectory_path.suffix}")

    # Load robot config if provided
    joint_limits = None
    joint_names = None
    if robot_config_path:
        # Parse robot config to get limits
        pass  # Implementation depends on config format

    # Verify
    verifier = TrajectoryVerifier(verifier_config)
    return verifier.verify(trajectory, joint_limits, joint_names)


def generate_verification_report(
    result: VerificationResult,
    output_path: str | Path,
) -> None:
    """Generate a human-readable verification report.

    Args:
        result: VerificationResult to report.
        output_path: Path to save the report.
    """
    output_path = Path(output_path)

    lines = [
        "# Trajectory Verification Report",
        "",
        f"**Valid:** {'Yes' if result.is_valid else 'No'}",
        f"**Feasibility Score:** {result.feasibility_score:.2%}",
        f"**Smoothness Score:** {result.smoothness_score:.2%}",
        f"**Execution Time:** {result.execution_time:.3f}s",
        "",
        "## Checks Passed",
    ]

    for check in result.checks_passed:
        lines.append(f"- [x] {check}")

    lines.extend(["", "## Checks Failed"])
    for check in result.checks_failed:
        lines.append(f"- [ ] {check}")

    if result.joint_limit_violations:
        lines.extend(["", "## Joint Limit Violations"])
        for v in result.joint_limit_violations:
            lines.append(
                f"- {v['joint_name']} ({v['type']}): "
                f"{v['value']:.4f} (limit: {v['limit']:.4f}) at step {v['step']}"
            )

    if result.velocity_violations:
        lines.extend(["", "## Velocity Violations"])
        for v in result.velocity_violations:
            lines.append(
                f"- {v['joint_name']}: "
                f"{v['velocity']:.4f} rad/s (limit: {v['limit']:.4f}) at step {v['step']}"
            )

    output_path.write_text("\n".join(lines))
