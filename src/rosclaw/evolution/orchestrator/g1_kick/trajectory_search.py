"""Bounded keyframe candidates for GoalForge kick/recovery trajectories."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters, hash_json


@dataclass(frozen=True)
class TrajectoryCandidate:
    contact_pre_slowdown: float
    swing_keyframe_scale: float
    retract_keyframe_scale: float
    recovery_step_length: float
    parent_policy_hash: str
    schema_version: str = "rosclaw.g1_goalforge.trajectory_candidate.v1"

    def __post_init__(self) -> None:
        bounds = (
            (self.contact_pre_slowdown, 0.0, 0.20),
            (self.swing_keyframe_scale, 0.80, 1.12),
            (self.retract_keyframe_scale, 0.80, 1.15),
            (self.recovery_step_length, 0.0, 0.15),
        )
        if any(
            not math.isfinite(value) or not minimum <= value <= maximum
            for value, minimum, maximum in bounds
        ):
            raise ValueError("trajectory candidate is outside bounded patch space")
        if not self.parent_policy_hash.startswith("sha256:"):
            raise ValueError("trajectory candidate requires a parent policy hash")

    @property
    def candidate_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply(self, parent: ShotParameters) -> ShotParameters:
        if parent.policy_hash != self.parent_policy_hash:
            raise ValueError("trajectory candidate parent does not match")
        return replace(
            parent,
            swing_amplitude=self.swing_keyframe_scale,
            swing_speed_scale=max(0.80, 1.0 - self.contact_pre_slowdown),
            contact_phase_offset=max(
                -0.10,
                min(0.10, parent.contact_phase_offset - self.contact_pre_slowdown * 0.25),
            ),
            recovery_step_length=self.recovery_step_length,
            recovery_step_yaw=max(
                -0.15,
                min(0.15, (self.retract_keyframe_scale - 1.0) * 0.5),
            ),
            policy_type="trajectory",
        )


__all__ = ["TrajectoryCandidate"]
