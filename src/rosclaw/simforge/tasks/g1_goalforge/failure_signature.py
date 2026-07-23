"""FailureSignatureV3 and deterministic routing for GoalForge."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    GOALFORGE_TASK_ID,
    GoalForgeResult,
    GoalForgeStatus,
    hash_json,
)


class FailureClassV3(StrEnum):
    TARGET = "TARGET_ERROR"
    CONTACT = "CONTACT_TIMING_ERROR"
    BALANCE = "BALANCE_ERROR"
    BODY_CONTROL = "BODY_CONTROL_ERROR"
    PERCEPTION = "PERCEPTION_ERROR"
    RUNTIME = "RUNTIME_FAULT"
    IMPOSSIBLE = "IMPOSSIBLE_TASK"


class RecoverabilityV3(StrEnum):
    RECOVERABLE = "RECOVERABLE"
    REQUIRES_RESET = "REQUIRES_RESET"
    UNRECOVERABLE = "UNRECOVERABLE"


@dataclass(frozen=True)
class RootCauseHypothesis:
    cause: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("root-cause confidence must be in [0, 1]")


@dataclass(frozen=True)
class FailureSignatureV3:
    failure_id: str
    stage: str
    body_hash: str
    scene_hash: str
    action_id: str
    policy_hash: str
    primary_class: FailureClassV3
    failure_type: GoalForgeStatus
    observations: tuple[tuple[str, Any], ...]
    root_cause_hypotheses: tuple[RootCauseHypothesis, ...]
    recoverability: RecoverabilityV3
    retry_budget: int
    safe_reset_state: str
    recommended_route: tuple[str, ...]
    schema_version: str = "rosclaw.failure_signature.v3"

    def __post_init__(self) -> None:
        for value in (self.body_hash, self.scene_hash, self.policy_hash):
            if not value.startswith("sha256:"):
                raise ValueError("FailureSignatureV3 hashes must be sha256 digests")
        if not 0 <= self.retry_budget <= 3:
            raise ValueError("FailureSignatureV3 retry budget must be in [0, 3]")
        if self.recoverability is RecoverabilityV3.UNRECOVERABLE and (
            self.retry_budget != 0 or self.recommended_route != ("STOP", "HUMAN")
        ):
            raise ValueError("unrecoverable GoalForge failures must stop")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["primary_class"] = self.primary_class.value
        value["failure_type"] = self.failure_type.value
        value["recoverability"] = self.recoverability.value
        value["observations"] = dict(self.observations)
        value["root_cause_hypotheses"] = [asdict(item) for item in self.root_cause_hypotheses]
        value["recommended_route"] = list(self.recommended_route)
        return value


class GoalForgeFailureRouter:
    """Fail-closed routing with bounded retry budgets."""

    def route(
        self,
        *,
        result: GoalForgeResult,
        body_hash: str,
        scene_hash: str,
        action_id: str,
        policy_hash: str,
        stage: str = "SWING_LEG",
    ) -> FailureSignatureV3:
        failure_class, recoverability, causes, routes = _route(result)
        retry_budget = (
            0
            if recoverability is RecoverabilityV3.UNRECOVERABLE
            else 1
            if failure_class is FailureClassV3.RUNTIME
            else 2
        )
        observation = {
            "ball_contact_time": result.ball_contact_time_sec,
            "ball_target_error_m": result.target_error_m,
            "ball_speed_mps": result.ball_speed_mps,
            "support_foot_slip_m": result.support_foot_slip_m,
            "torso_roll_peak": result.torso_roll_peak_rad,
            "torso_pitch_peak": result.torso_pitch_peak_rad,
            "com_margin_min": result.com_margin_min_m,
            "post_kick_stability_time": result.post_kick_stability_time_sec,
            "goal_crossed": result.goal_crossed,
            "post_kick_fall": result.post_kick_fall,
        }
        observation = {
            key: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in observation.items()
        }
        identity = {
            "task_id": GOALFORGE_TASK_ID,
            "stage": stage,
            "body_hash": body_hash,
            "scene_hash": scene_hash,
            "action_id": action_id,
            "policy_hash": policy_hash,
            "failure_type": result.status.value,
            "observations": observation,
        }
        return FailureSignatureV3(
            failure_id="failure_" + hash_json(identity).removeprefix("sha256:")[:24],
            stage=stage,
            body_hash=body_hash,
            scene_hash=scene_hash,
            action_id=action_id,
            policy_hash=policy_hash,
            primary_class=failure_class,
            failure_type=result.status,
            observations=tuple(sorted(observation.items())),
            root_cause_hypotheses=causes,
            recoverability=recoverability,
            retry_budget=retry_budget,
            safe_reset_state="STANDING_READY",
            recommended_route=routes,
        )


def _route(
    result: GoalForgeResult,
) -> tuple[
    FailureClassV3,
    RecoverabilityV3,
    tuple[RootCauseHypothesis, ...],
    tuple[str, ...],
]:
    status = result.status
    if status in {
        GoalForgeStatus.AGENT_LOST,
        GoalForgeStatus.POLICY_WORKER_CRASH,
        GoalForgeStatus.DDS_LOST,
        GoalForgeStatus.STATE_FEEDBACK_STALE,
        GoalForgeStatus.EXECUTOR_UNAVAILABLE,
    }:
        return (
            FailureClassV3.RUNTIME,
            RecoverabilityV3.REQUIRES_RESET,
            (RootCauseHypothesis("runtime_session_or_feedback_lost", 1.0),),
            ("RUNTIME_STOP", "RESET_SESSION"),
        )
    if status in {
        GoalForgeStatus.BALL_OUT_OF_REACH,
        GoalForgeStatus.ROBOT_NOT_STABLE,
        GoalForgeStatus.TASK_PHYSICALLY_IMPOSSIBLE,
        GoalForgeStatus.POLICY_BODY_INCOMPATIBLE,
    }:
        return (
            FailureClassV3.IMPOSSIBLE,
            RecoverabilityV3.UNRECOVERABLE,
            (RootCauseHypothesis("task_or_body_precondition_unsatisfied", 1.0),),
            ("STOP", "HUMAN"),
        )
    if status in {
        GoalForgeStatus.JOINT_LIMIT_EXCEEDED,
        GoalForgeStatus.TORQUE_LIMIT_EXCEEDED,
        GoalForgeStatus.ACTUATOR_SATURATION,
        GoalForgeStatus.BODY_CALIBRATION_MISMATCH,
        GoalForgeStatus.NON_FINITE_STATE,
    }:
        return (
            FailureClassV3.BODY_CONTROL,
            RecoverabilityV3.REQUIRES_RESET,
            (
                RootCauseHypothesis("bounded_controller_or_body_mismatch", 0.92),
                RootCauseHypothesis("candidate_too_aggressive", 0.78),
            ),
            ("KNOW", "SANDBOX", "STOP_REJECT"),
        )
    if status in {
        GoalForgeStatus.SUPPORT_FOOT_SLIP,
        GoalForgeStatus.COM_OUTSIDE_SUPPORT,
        GoalForgeStatus.POST_KICK_FALL,
        GoalForgeStatus.RECOVERY_STEP_FAILED,
        GoalForgeStatus.TORSO_OVERSHOOT,
    }:
        return (
            FailureClassV3.BALANCE,
            RecoverabilityV3.RECOVERABLE,
            (
                RootCauseHypothesis("support_friction_or_com_shift_mismatch", 0.86),
                RootCauseHypothesis("swing_or_recovery_too_aggressive", 0.74),
            ),
            ("MEMORY", "KNOW", "HOW", "AUTO", "STANCE_REPLAN"),
        )
    if status in {
        GoalForgeStatus.BALL_NOT_CONTACTED,
        GoalForgeStatus.EARLY_BALL_CONTACT,
        GoalForgeStatus.LATE_BALL_CONTACT,
        GoalForgeStatus.WRONG_FOOT_CONTACT,
    }:
        return (
            FailureClassV3.CONTACT,
            RecoverabilityV3.RECOVERABLE,
            (
                RootCauseHypothesis("contact_phase_or_trigger_delay_mismatch", 0.86),
                RootCauseHypothesis("stance_offset_mismatch", 0.68),
            ),
            ("MEMORY", "KNOW", "HOW", "AUTO"),
        )
    if status is GoalForgeStatus.BALL_OBSERVATION_STALE:
        return (
            FailureClassV3.PERCEPTION,
            RecoverabilityV3.REQUIRES_RESET,
            (RootCauseHypothesis("ball_observation_not_fresh", 1.0),),
            ("RUNTIME_STOP", "REOBSERVE"),
        )
    return (
        FailureClassV3.TARGET,
        RecoverabilityV3.RECOVERABLE,
        (
            RootCauseHypothesis("target_alignment_or_foot_yaw_error", 0.82),
            RootCauseHypothesis("ball_physics_belief_error", 0.66),
        ),
        ("MEMORY", "KNOW", "HOW", "AUTO"),
    )


__all__ = [
    "FailureClassV3",
    "FailureSignatureV3",
    "GoalForgeFailureRouter",
    "RecoverabilityV3",
    "RootCauseHypothesis",
]
