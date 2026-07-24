"""Failure-specific bounded interventions for GoalForge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    GoalForgeStatus,
    ShotParameters,
    hash_json,
)
from rosclaw.simforge.tasks.g1_goalforge.failure_signature import (
    FailureSignatureV3,
    RecoverabilityV3,
)
from rosclaw.twin.g1_kick.belief import KickTwinBelief


@dataclass(frozen=True)
class GoalForgeHowIntervention:
    failure_id: str
    parent_policy_hash: str
    patch: ShotParameters
    rationale: tuple[str, ...]
    bounded_retry_index: int
    schema_version: str = "rosclaw.g1_goalforge.how_intervention.v1"

    def __post_init__(self) -> None:
        if not self.parent_policy_hash.startswith("sha256:"):
            raise ValueError("How intervention requires parent policy hash")
        if not 1 <= self.bounded_retry_index <= 2:
            raise ValueError("How intervention retry index must be one or two")

    @property
    def intervention_hash(self) -> str:
        value = asdict(self)
        value["patch"] = self.patch.to_dict()
        return hash_json(value)


class GoalForgeHow:
    def advise(
        self,
        *,
        signature: FailureSignatureV3,
        current: ShotParameters,
        twin: KickTwinBelief,
        retry_index: int,
    ) -> GoalForgeHowIntervention:
        if signature.recoverability is RecoverabilityV3.UNRECOVERABLE:
            raise ValueError("How must not retry an unrecoverable GoalForge failure")
        status = signature.failure_type
        values: dict[str, float] = {}
        reasons: list[str] = []
        if status in {GoalForgeStatus.TARGET_MISS_LEFT, GoalForgeStatus.TARGET_MISS_RIGHT}:
            direction = -1.0 if status is GoalForgeStatus.TARGET_MISS_LEFT else 1.0
            values["foot_yaw_offset"] = _clip(
                current.foot_yaw_offset + direction * 0.045,
                -0.12,
                0.12,
            )
            values["stance_offset_y"] = _clip(
                current.stance_offset_y + direction * 0.025,
                -0.12,
                0.12,
            )
            values["pelvis_yaw_offset"] = _clip(
                current.pelvis_yaw_offset + direction * 0.10,
                -0.20,
                0.20,
            )
            reasons.append("correct_lateral_target_error")
        elif status in {
            GoalForgeStatus.BALL_NOT_CONTACTED,
            GoalForgeStatus.EARLY_BALL_CONTACT,
            GoalForgeStatus.LATE_BALL_CONTACT,
        }:
            timing = (
                -0.035
                if status is GoalForgeStatus.LATE_BALL_CONTACT
                else 0.035
                if status is GoalForgeStatus.EARLY_BALL_CONTACT
                else -twin.foot_contact_offset_mean[0]
            )
            values["contact_phase_offset"] = _clip(
                current.contact_phase_offset + timing,
                -0.10,
                0.10,
            )
            reasons.append("correct_contact_phase")
        elif status in {
            GoalForgeStatus.SUPPORT_FOOT_SLIP,
            GoalForgeStatus.COM_OUTSIDE_SUPPORT,
            GoalForgeStatus.POST_KICK_FALL,
            GoalForgeStatus.TORSO_OVERSHOOT,
        }:
            values["swing_speed_scale"] = _clip(
                current.swing_speed_scale - 0.08,
                0.80,
                1.15,
            )
            values["swing_amplitude"] = _clip(
                current.swing_amplitude - 0.06,
                0.75,
                1.15,
            )
            values["com_shift_y"] = _clip(
                current.com_shift_y + 0.025,
                -0.08,
                0.08,
            )
            values["recovery_step_length"] = _clip(
                current.recovery_step_length + 0.035,
                0.0,
                0.15,
            )
            reasons.append("reduce_slip_and_expand_recovery")
        elif status is GoalForgeStatus.SHOT_TOO_WEAK:
            values["swing_amplitude"] = _clip(
                current.swing_amplitude + 0.08,
                0.75,
                1.15,
            )
            reasons.append("increase_safe_ball_impulse")
        else:
            values["kick_trigger_delay"] = _clip(
                current.kick_trigger_delay + 0.02,
                0.0,
                0.20,
            )
            reasons.append("bounded_reobserve_before_trigger")
        patch = replace(
            current,
            stance_offset_x=values.get("stance_offset_x", current.stance_offset_x),
            stance_offset_y=values.get("stance_offset_y", current.stance_offset_y),
            pelvis_yaw_offset=values.get("pelvis_yaw_offset", current.pelvis_yaw_offset),
            com_shift_y=values.get("com_shift_y", current.com_shift_y),
            swing_amplitude=values.get("swing_amplitude", current.swing_amplitude),
            swing_speed_scale=values.get("swing_speed_scale", current.swing_speed_scale),
            foot_yaw_offset=values.get("foot_yaw_offset", current.foot_yaw_offset),
            contact_phase_offset=values.get("contact_phase_offset", current.contact_phase_offset),
            kick_trigger_delay=values.get("kick_trigger_delay", current.kick_trigger_delay),
            recovery_step_length=values.get("recovery_step_length", current.recovery_step_length),
            recovery_step_yaw=values.get("recovery_step_yaw", current.recovery_step_yaw),
            policy_type="parameter",
        )
        return GoalForgeHowIntervention(
            failure_id=signature.failure_id,
            parent_policy_hash=current.policy_hash,
            patch=patch,
            rationale=tuple(reasons),
            bounded_retry_index=retry_index,
        )


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


__all__ = ["GoalForgeHow", "GoalForgeHowIntervention"]
