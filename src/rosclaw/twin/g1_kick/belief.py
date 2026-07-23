"""Immutable and hash-addressed GoalForge twin belief."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class ScalarBelief:
    mean: float
    std: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) for value in (self.mean, self.std, self.minimum, self.maximum)
        ):
            raise ValueError("twin belief values must be finite")
        if self.std <= 0.0 or self.minimum >= self.maximum:
            raise ValueError("twin belief uncertainty/bounds are invalid")
        if not self.minimum <= self.mean <= self.maximum:
            raise ValueError("twin belief mean must be inside its bounds")


@dataclass(frozen=True)
class KickTwinBelief:
    ball_mass: ScalarBelief
    ball_ground_friction: ScalarBelief
    support_ground_friction: ScalarBelief
    restitution: ScalarBelief
    control_latency_ms: ScalarBelief
    foot_contact_offset_mean: tuple[float, float]
    foot_contact_offset_covariance: tuple[tuple[float, float], tuple[float, float]]
    update_count: int = 0
    parent_hash: str | None = None
    evidence_hashes: tuple[str, ...] = ()
    schema_version: str = "rosclaw.g1_kick.twin_belief.v1"

    def __post_init__(self) -> None:
        if self.update_count < 0:
            raise ValueError("twin update_count must be non-negative")
        if self.parent_hash is not None and not self.parent_hash.startswith("sha256:"):
            raise ValueError("twin parent hash must be a sha256 digest")
        if any(not value.startswith("sha256:") for value in self.evidence_hashes):
            raise ValueError("twin evidence hashes must be sha256 digests")
        covariance = self.foot_contact_offset_covariance
        if len(covariance) != 2 or any(len(row) != 2 for row in covariance):
            raise ValueError("foot contact covariance must be 2x2")
        if (
            covariance[0][0] <= 0.0
            or covariance[1][1] <= 0.0
            or abs(covariance[0][1] - covariance[1][0]) > 1e-12
        ):
            raise ValueError("foot contact covariance must be symmetric positive-diagonal")

    @classmethod
    def initial(cls) -> KickTwinBelief:
        return cls(
            ball_mass=ScalarBelief(0.42, 0.035, 0.36, 0.48),
            ball_ground_friction=ScalarBelief(0.12, 0.075, 0.03, 0.35),
            support_ground_friction=ScalarBelief(0.85, 0.20, 0.45, 1.25),
            restitution=ScalarBelief(0.55, 0.18, 0.20, 0.90),
            control_latency_ms=ScalarBelief(20.0, 18.0, 0.0, 80.0),
            foot_contact_offset_mean=(0.0, 0.0),
            foot_contact_offset_covariance=((0.0025, 0.0), (0.0, 0.0025)),
        )

    @property
    def belief_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def public_context(self) -> dict[str, float]:
        """Candidate-visible projection; never includes scenario hidden truth."""

        return {
            "ball_mass_belief": self.ball_mass.mean,
            "ball_mass_uncertainty": self.ball_mass.std,
            "ball_ground_friction_belief": self.ball_ground_friction.mean,
            "support_friction_belief": self.support_ground_friction.mean,
            "support_friction_uncertainty": self.support_ground_friction.std,
            "restitution_belief": self.restitution.mean,
            "control_latency_belief_ms": self.control_latency_ms.mean,
            "control_latency_uncertainty_ms": self.control_latency_ms.std,
            "foot_contact_offset_x": self.foot_contact_offset_mean[0],
            "foot_contact_offset_y": self.foot_contact_offset_mean[1],
        }


__all__ = ["KickTwinBelief", "ScalarBelief"]
