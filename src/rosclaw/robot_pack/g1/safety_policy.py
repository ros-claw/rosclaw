"""Permit/lease and immutable safety policy for simulation-only G1 kicks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_HARD_TORQUE_LIMITS,
    ShotParameters,
    hash_json,
)


@dataclass(frozen=True)
class G1KickPermit:
    session_id: str
    permit_id: str
    action_id: str
    body_hash: str
    policy_hash: str
    issued_at_monotonic: float
    expires_at_monotonic: float
    lease_deadline_monotonic: float
    hardware_authorized: bool = False
    evidence_domain: str = "SHADOW"
    schema_version: str = "rosclaw.g1.kick_permit.v1"

    def __post_init__(self) -> None:
        if not all((self.session_id, self.permit_id, self.action_id)):
            raise ValueError("G1 kick permit identity is required")
        if not self.body_hash.startswith("sha256:") or not self.policy_hash.startswith("sha256:"):
            raise ValueError("G1 kick permit must bind body and policy hashes")
        if not all(
            math.isfinite(value)
            for value in (
                self.issued_at_monotonic,
                self.expires_at_monotonic,
                self.lease_deadline_monotonic,
            )
        ):
            raise ValueError("G1 kick permit timestamps must be finite")
        if not (
            self.issued_at_monotonic < self.lease_deadline_monotonic <= self.expires_at_monotonic
        ):
            raise ValueError("G1 kick permit lease/expiry order is invalid")
        if self.hardware_authorized or self.evidence_domain != "SHADOW":
            raise ValueError("Phase 4 G1 permits are simulation-only SHADOW permits")

    @property
    def permit_hash(self) -> str:
        return hash_json(asdict(self))


class G1KickSafetyPolicy:
    def validate(
        self,
        *,
        permit: G1KickPermit,
        parameters: ShotParameters,
        expected_body_hash: str,
        now_monotonic: float,
    ) -> tuple[bool, tuple[str, ...]]:
        errors: list[str] = []
        if permit.body_hash != expected_body_hash:
            errors.append("body_hash_mismatch")
        if permit.policy_hash != parameters.policy_hash:
            errors.append("policy_hash_mismatch")
        if now_monotonic >= permit.expires_at_monotonic:
            errors.append("permit_expired")
        if now_monotonic >= permit.lease_deadline_monotonic:
            errors.append("lease_expired")
        if permit.hardware_authorized or permit.evidence_domain != "SHADOW":
            errors.append("non_shadow_authorization")
        if parameters.kick_foot != "right":
            errors.append("unsupported_kick_foot")
        return not errors, tuple(errors)

    @property
    def immutable_contract_hash(self) -> str:
        return hash_json(
            {
                "hard_torque_limits": G1_HARD_TORQUE_LIMITS,
                "joint_hard_limits": "qualified_body_mjcf",
                "e_stop": "immutable",
                "permit": "required",
                "lease": "required",
                "hardware_authorization": False,
                "evidence_domain": "SHADOW",
            }
        )


__all__ = ["G1KickPermit", "G1KickSafetyPolicy"]
