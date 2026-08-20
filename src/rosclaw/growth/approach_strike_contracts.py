"""Task-neutral feature contracts for G1 approach-to-strike learning."""

from __future__ import annotations

EVENT_PHASE_NAMES = (
    "approach",
    "align_brake",
    "plant_bridge",
    "load",
    "swing",
    "contact",
    "follow_through",
    "recovery",
    "ready",
)
REWARD_NAMES = (
    "phase_progress",
    "ball_distance_progress",
    "upright",
    "action_smoothness",
    "contact_speed",
    "terminal_precision",
)
COST_NAMES = (
    "torque_projection",
    "torque_overdrive",
    "low_pelvis",
    "tilt_excess",
    "episode_safety_violation",
)
STATE_FEATURES = (
    *(f"joint_position.{index}" for index in range(29)),
    *(f"joint_velocity.{index}" for index in range(29)),
    "pelvis_height",
    "pelvis_velocity_x",
    "pelvis_velocity_y",
    "pelvis_velocity_z",
    "torso_quaternion_w",
    "torso_quaternion_x",
    "torso_quaternion_y",
    "torso_quaternion_z",
    "ball_relative_x",
    "ball_relative_y",
    "ball_relative_z",
    "ball_velocity_x",
    "ball_velocity_y",
    "ball_velocity_z",
    *(f"event_phase.{name}" for name in EVENT_PHASE_NAMES),
    *(f"base_joint_target.{index}" for index in range(29)),
)

__all__ = ["COST_NAMES", "EVENT_PHASE_NAMES", "REWARD_NAMES", "STATE_FEATURES"]
