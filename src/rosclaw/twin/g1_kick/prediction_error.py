"""Observed-vs-predicted residuals used to update the kick twin."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class KickPredictionError:
    predicted_ball_speed_mps: float
    observed_ball_speed_mps: float
    predicted_target_error_m: float
    observed_target_error_m: float
    predicted_contact_time_sec: float
    observed_contact_time_sec: float
    support_foot_slip_m: float
    torso_response_rad: float
    joint_tracking_rmse_rad: float
    source_episode_hash: str
    schema_version: str = "rosclaw.g1_kick.prediction_error.v1"

    def __post_init__(self) -> None:
        numeric = (
            self.predicted_ball_speed_mps,
            self.observed_ball_speed_mps,
            self.predicted_target_error_m,
            self.observed_target_error_m,
            self.predicted_contact_time_sec,
            self.observed_contact_time_sec,
            self.support_foot_slip_m,
            self.torso_response_rad,
            self.joint_tracking_rmse_rad,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
            raise ValueError("prediction errors must be finite and non-negative")
        if not self.source_episode_hash.startswith("sha256:"):
            raise ValueError("prediction error requires a source episode hash")

    @property
    def error_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["KickPredictionError"]
