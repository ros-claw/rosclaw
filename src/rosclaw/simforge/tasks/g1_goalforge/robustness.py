"""Safety-first robustness ordering for GoalForge candidates."""

from __future__ import annotations

from dataclasses import dataclass

from rosclaw.simforge.tasks.g1_goalforge.concepts import GoalForgeResult


@dataclass(frozen=True)
class GoalForgeScore:
    safe: bool
    goal_success: bool
    target_zone_success: bool
    score: float
    rejection_reason: str | None


def score_goalforge_result(result: GoalForgeResult) -> GoalForgeScore:
    if result.post_kick_fall:
        return GoalForgeScore(
            False, result.goal_crossed, result.target_zone_hit, -1000.0, "post_kick_fall"
        )
    if result.torque_limit_violation:
        return GoalForgeScore(
            False, result.goal_crossed, result.target_zone_hit, -1000.0, "torque_limit"
        )
    if result.joint_limit_violation:
        return GoalForgeScore(
            False, result.goal_crossed, result.target_zone_hit, -1000.0, "joint_limit"
        )
    if result.support_foot_slip_m > 0.08:
        return GoalForgeScore(
            False, result.goal_crossed, result.target_zone_hit, -500.0, "support_foot_slip"
        )
    if result.com_margin_min_m < -0.04:
        return GoalForgeScore(
            False, result.goal_crossed, result.target_zone_hit, -500.0, "com_outside_support"
        )
    score = (
        100.0 * float(result.target_zone_hit)
        + 25.0 * float(result.goal_crossed)
        - 20.0 * result.target_error_m
        + 2.0 * min(result.ball_speed_mps, 12.0)
        + 10.0 * result.robustness
    )
    return GoalForgeScore(True, result.goal_crossed, result.target_zone_hit, score, None)


__all__ = ["GoalForgeScore", "score_goalforge_result"]
