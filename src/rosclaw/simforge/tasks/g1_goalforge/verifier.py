"""Independent trajectory verifier for GoalForge episodes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import GoalForgeEpisode
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class GoalForgeVerification:
    valid: bool
    physics_complete: bool
    joint_channels_complete: bool
    finite_state: bool
    result_consistent: bool
    errors: tuple[str, ...]
    independently_verified: bool = True
    schema_version: str = "rosclaw.g1_goalforge.verification.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["errors"] = list(self.errors)
        value["verification_hash"] = hash_json(value)
        return value


class GoalForgeVerifier:
    def verify(self, episode: GoalForgeEpisode) -> GoalForgeVerification:
        trace = episode.trajectory
        errors: list[str] = []
        required = {
            "time",
            "joint_position",
            "joint_velocity",
            "joint_torque",
            "pelvis_pose",
            "torso_quaternion",
            "com",
            "ball_pose",
            "ball_velocity",
            "left_foot_contact",
            "right_foot_contact",
        }
        missing = required - set(trace)
        if missing:
            errors.append("missing_channels=" + ",".join(sorted(missing)))
        lengths = {len(value) for value in trace.values()} if trace else set()
        if len(lengths) > 1:
            errors.append("channel_length_mismatch")
        joint_complete = all(
            name in trace
            and np.asarray(trace[name]).ndim == 2
            and np.asarray(trace[name]).shape[1] == 29
            for name in ("joint_position", "joint_velocity", "joint_torque")
        )
        if not joint_complete:
            errors.append("29_joint_channels_incomplete")
        finite = bool(trace) and all(
            np.all(np.isfinite(np.asarray(value)))
            for value in trace.values()
            if np.asarray(value).dtype.kind in "fiu"
        )
        if not finite:
            errors.append("non_finite_trajectory")
        physics_complete = bool(trace) and episode.result.physics_steps >= len(
            np.asarray(trace.get("time", ()))
        )
        if not physics_complete:
            errors.append("physics_steps_incomplete")
        consistent = _result_consistent(episode)
        if not consistent:
            errors.append("result_not_derived_from_trajectory")
        return GoalForgeVerification(
            valid=not errors,
            physics_complete=physics_complete,
            joint_channels_complete=joint_complete,
            finite_state=finite,
            result_consistent=consistent,
            errors=tuple(errors),
        )


def _result_consistent(episode: GoalForgeEpisode) -> bool:
    trace = episode.trajectory
    if not trace:
        return not episode.result.physics_executed
    pelvis = np.asarray(trace["pelvis_pose"], dtype=np.float64)
    ball = np.asarray(trace["ball_pose"], dtype=np.float64)
    velocity = np.asarray(trace["ball_velocity"], dtype=np.float64)
    if pelvis.shape[0] == 0 or ball.shape[0] == 0:
        return False
    maximum_speed = float(np.max(np.linalg.norm(velocity[:, :3], axis=1)))
    speed_consistent = episode.result.ball_speed_mps + 1e-6 >= maximum_speed
    height_consistent = math.isclose(
        episode.result.final_pelvis_height_m,
        float(pelvis[-1, 2]),
        abs_tol=0.03,
    )
    crossing_observed = (
        bool(np.any(np.asarray(trace["goal_crossing"], dtype=bool)))
        if "goal_crossing" in trace
        else bool(np.any(ball[:, 0] >= 5.0))
    )
    crossing_consistent = not episode.result.goal_crossed or crossing_observed
    contact_consistent = (
        not episode.result.kick_foot_contacted or episode.result.ball_contact_time_sec is not None
    )
    return speed_consistent and height_consistent and crossing_consistent and contact_consistent


__all__ = ["GoalForgeVerification", "GoalForgeVerifier"]
