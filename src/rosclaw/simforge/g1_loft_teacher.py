"""SIM-only operational-space teacher for discovering lofted G1 strikes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class G1LoftTeacherConfig:
    """Bounded vertical foot-velocity teacher; zero target disables it."""

    target_vertical_speed_mps: float = 0.0
    velocity_gain_n_per_mps: float = 24.0
    maximum_vertical_force_n: float = 60.0
    target_forward_speed_mps: float = 0.0
    forward_velocity_gain_n_per_mps: float = 20.0
    maximum_forward_force_n: float = 80.0
    start_policy_frame: int = 230
    end_policy_frame: int = 335
    foot_strike_point_offset_m: tuple[float, float, float] = (0.13, 0.0, -0.025)
    maximum_foot_ball_distance_m: float = 0.0
    schema_version: str = "rosclaw.simforge.g1_loft_teacher_config.v4"

    def __post_init__(self) -> None:
        values = (
            self.target_vertical_speed_mps,
            self.velocity_gain_n_per_mps,
            self.maximum_vertical_force_n,
            self.target_forward_speed_mps,
            self.forward_velocity_gain_n_per_mps,
            self.maximum_forward_force_n,
            self.maximum_foot_ball_distance_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("G1 loft teacher config must be finite")
        if self.target_vertical_speed_mps != 0.0 and not (
            3.0 <= self.target_vertical_speed_mps <= 7.0
        ):
            raise ValueError("G1 loft teacher target speed must be zero or in [3, 7] m/s")
        if not 5.0 <= self.velocity_gain_n_per_mps <= 50.0:
            raise ValueError("G1 loft teacher velocity gain must be in [5, 50] N/(m/s)")
        if not 10.0 <= self.maximum_vertical_force_n <= 250.0:
            raise ValueError("G1 loft teacher force limit must be in [10, 250] N")
        if self.target_forward_speed_mps != 0.0 and not (
            5.0 <= self.target_forward_speed_mps <= 20.0
        ):
            raise ValueError("G1 loft teacher forward speed must be zero or in [5, 20] m/s")
        if not 5.0 <= self.forward_velocity_gain_n_per_mps <= 50.0:
            raise ValueError("G1 loft teacher forward gain must be in [5, 50] N/(m/s)")
        if not 10.0 <= self.maximum_forward_force_n <= 250.0:
            raise ValueError("G1 loft teacher forward force limit must be in [10, 250] N")
        if not 150 <= self.start_policy_frame < self.end_policy_frame <= 430:
            raise ValueError("G1 loft teacher policy window is invalid")
        if len(self.foot_strike_point_offset_m) != 3 or not all(
            math.isfinite(value) for value in self.foot_strike_point_offset_m
        ):
            raise ValueError("G1 loft teacher foot strike point must contain three finite values")
        if not (
            0.05 <= self.foot_strike_point_offset_m[0] <= 0.16
            and -0.04 <= self.foot_strike_point_offset_m[1] <= 0.04
            and -0.05 <= self.foot_strike_point_offset_m[2] <= 0.02
        ):
            raise ValueError("G1 loft teacher foot strike point lies outside the foot envelope")
        if self.maximum_foot_ball_distance_m != 0.0 and not (
            0.15 <= self.maximum_foot_ball_distance_m <= 1.0
        ):
            raise ValueError(
                "G1 loft teacher foot-ball distance must be zero or in [0.15, 1.0] m"
            )

    @property
    def enabled(self) -> bool:
        return self.target_vertical_speed_mps > 0.0 or self.target_forward_speed_mps > 0.0

    @property
    def config_hash(self) -> str:
        return hash_json(asdict(self))


@dataclass(frozen=True)
class G1LoftTeacherEffect:
    torque: np.ndarray
    vertical_force_n: float
    forward_force_n: float
    foot_vertical_speed_mps: float
    foot_forward_speed_mps: float
    active: bool


def project_g1_vertical_foot_force(
    *,
    jacobian_position: np.ndarray,
    generalized_velocity: np.ndarray,
    config: G1LoftTeacherConfig,
) -> G1LoftTeacherEffect:
    """Project the teacher force through a supplied G1 body Jacobian."""

    jacobian = np.asarray(jacobian_position, dtype=np.float64)
    velocity = np.asarray(generalized_velocity, dtype=np.float64)
    if (
        jacobian.ndim != 2
        or jacobian.shape[0] != 3
        or jacobian.shape[1] < 35
        or velocity.shape != (jacobian.shape[1],)
    ):
        raise ValueError(
            "G1 loft teacher expects matching Jacobian/velocity dimensions of at least 35"
        )
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(velocity)):
        raise FloatingPointError("G1 loft teacher inputs must be finite")
    foot_forward_speed = float(jacobian[0] @ velocity)
    foot_vertical_speed = float(jacobian[2] @ velocity)
    vertical_force = (
        0.0
        if config.target_vertical_speed_mps == 0.0
        else float(
            np.clip(
                config.velocity_gain_n_per_mps
                * (config.target_vertical_speed_mps - foot_vertical_speed),
                0.0,
                config.maximum_vertical_force_n,
            )
        )
    )
    forward_force = (
        0.0
        if config.target_forward_speed_mps == 0.0
        else float(
            np.clip(
                config.forward_velocity_gain_n_per_mps
                * (config.target_forward_speed_mps - foot_forward_speed),
                0.0,
                config.maximum_forward_force_n,
            )
        )
    )
    torque = (
        jacobian[2, 6:35] * vertical_force
        + jacobian[0, 6:35] * forward_force
    )
    if torque.shape != (29,) or not np.all(np.isfinite(torque)):
        raise FloatingPointError("G1 loft teacher emitted an invalid joint torque")
    return G1LoftTeacherEffect(
        torque=torque,
        vertical_force_n=vertical_force,
        forward_force_n=forward_force,
        foot_vertical_speed_mps=foot_vertical_speed,
        foot_forward_speed_mps=foot_forward_speed,
        active=vertical_force > 0.0 or forward_force > 0.0,
    )


def g1_loft_teacher_effect(
    *,
    model: Any,
    data: Any,
    right_ankle_body_id: int,
    config: G1LoftTeacherConfig,
    policy_frame: int,
    contact_observed: bool,
    ball_position: np.ndarray | None = None,
) -> G1LoftTeacherEffect:
    """Project a bounded upward task-space force into the 29 joint torques."""

    import mujoco

    zero = np.zeros(29, dtype=np.float64)
    if (
        not config.enabled
        or contact_observed
        or not config.start_policy_frame <= policy_frame <= config.end_policy_frame
    ):
        return G1LoftTeacherEffect(zero, 0.0, 0.0, 0.0, 0.0, False)
    jacobian_position = np.zeros((3, int(model.nv)), dtype=np.float64)
    jacobian_rotation = np.zeros((3, int(model.nv)), dtype=np.float64)
    foot_rotation = np.asarray(data.xmat[right_ankle_body_id], dtype=np.float64).reshape(3, 3)
    foot_point = np.asarray(
        data.xpos[right_ankle_body_id], dtype=np.float64
    ) + foot_rotation @ np.asarray(
        config.foot_strike_point_offset_m,
        dtype=np.float64,
    )
    if config.maximum_foot_ball_distance_m > 0.0:
        ball = np.asarray(ball_position, dtype=np.float64)
        if ball.shape != (3,) or not np.all(np.isfinite(ball)):
            raise ValueError("G1 loft teacher proximity gate requires a finite ball position")
        if float(np.linalg.norm(foot_point - ball)) > config.maximum_foot_ball_distance_m:
            return G1LoftTeacherEffect(zero, 0.0, 0.0, 0.0, 0.0, False)
    mujoco.mj_jac(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        foot_point,
        right_ankle_body_id,
    )
    return project_g1_vertical_foot_force(
        jacobian_position=jacobian_position,
        generalized_velocity=data.qvel,
        config=config,
    )


__all__ = [
    "G1LoftTeacherConfig",
    "G1LoftTeacherEffect",
    "g1_loft_teacher_effect",
    "project_g1_vertical_foot_force",
]
