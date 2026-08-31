"""Separate simulation exploration permission from promotion evidence.

Exploration may admit declared contact-rich behavior to discover a skill, but
it never grants activation authority.  Promotion remains fail-closed and may
use a strictly tighter envelope.  Non-finite state, prohibited contacts and
unrealistic actuator commands are hard failures in both modes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")


class GrowthSafetyUse(StrEnum):
    EXPLORATION_SIM = "EXPLORATION_SIM"
    PROMOTION_SIM = "PROMOTION_SIM"


@dataclass(frozen=True)
class GrowthSafetyProfile:
    profile_id: str
    use: GrowthSafetyUse
    maximum_joint_limit_excess_rad: float
    maximum_normalized_actuator_command: float
    maximum_head_impact_speed_mps: float
    maximum_root_angular_speed_rad_s: float
    maximum_self_penetration_m: float
    always_allowed_contacts: tuple[str, ...]
    hard_fail_contacts: tuple[str, ...]
    phase_contact_permissions: Mapping[str, tuple[str, ...]]
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.continual.growth_safety_profile.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.profile_id):
            raise ValueError("growth safety profile id is invalid")
        thresholds = (
            self.maximum_joint_limit_excess_rad,
            self.maximum_normalized_actuator_command,
            self.maximum_head_impact_speed_mps,
            self.maximum_root_angular_speed_rad_s,
            self.maximum_self_penetration_m,
        )
        if (
            any(not math.isfinite(value) or value < 0.0 for value in thresholds)
            or self.maximum_normalized_actuator_command <= 0.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("growth safety thresholds or authority are invalid")
        allowed = _identifiers(self.always_allowed_contacts, label="always allowed contacts")
        hard = _identifiers(self.hard_fail_contacts, label="hard-fail contacts")
        if set(allowed) & set(hard):
            raise ValueError("a contact cannot be both allowed and hard-fail")
        permissions: dict[str, tuple[str, ...]] = {}
        for raw_phase, raw_contacts in self.phase_contact_permissions.items():
            phase = str(raw_phase)
            if not _IDENTIFIER.fullmatch(phase):
                raise ValueError("growth safety phase is invalid")
            contacts = _identifiers(raw_contacts, label="phase contacts")
            if set(contacts) & set(hard):
                raise ValueError("phase permissions cannot override hard-fail contacts")
            permissions[phase] = contacts
        if not permissions:
            raise ValueError("growth safety profile requires phase permissions")
        object.__setattr__(self, "always_allowed_contacts", allowed)
        object.__setattr__(self, "hard_fail_contacts", hard)
        object.__setattr__(
            self,
            "phase_contact_permissions",
            MappingProxyType(dict(sorted(permissions.items()))),
        )

    @property
    def profile_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "use": self.use.value,
            "maximum_joint_limit_excess_rad": self.maximum_joint_limit_excess_rad,
            "maximum_normalized_actuator_command": self.maximum_normalized_actuator_command,
            "maximum_head_impact_speed_mps": self.maximum_head_impact_speed_mps,
            "maximum_root_angular_speed_rad_s": self.maximum_root_angular_speed_rad_s,
            "maximum_self_penetration_m": self.maximum_self_penetration_m,
            "always_allowed_contacts": list(self.always_allowed_contacts),
            "hard_fail_contacts": list(self.hard_fail_contacts),
            "phase_contact_permissions": {
                phase: list(contacts) for phase, contacts in self.phase_contact_permissions.items()
            },
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class GrowthSafetyObservation:
    phase: str
    finite_state: bool
    joint_limit_excess_rad: float
    normalized_actuator_command: float
    head_impact_speed_mps: float
    root_angular_speed_rad_s: float
    self_penetration_m: float
    contacts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.phase) or not isinstance(self.finite_state, bool):
            raise ValueError("growth safety observation identity is invalid")
        values = (
            self.joint_limit_excess_rad,
            self.normalized_actuator_command,
            self.head_impact_speed_mps,
            self.root_angular_speed_rad_s,
            self.self_penetration_m,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("growth safety observation must contain finite non-negative values")
        contacts = _identifiers(self.contacts, label="observed contacts")
        object.__setattr__(self, "contacts", contacts)


@dataclass(frozen=True)
class GrowthSafetyDecision:
    profile_hash: str
    passed: bool
    promotion_eligible: bool
    hard_failure: bool
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw.continual.growth_safety_decision.v1"


def evaluate_growth_safety(
    profile: GrowthSafetyProfile,
    observation: GrowthSafetyObservation,
) -> GrowthSafetyDecision:
    reasons: list[str] = []
    if not observation.finite_state:
        reasons.append("NON_FINITE_STATE")
    thresholds = (
        (
            observation.joint_limit_excess_rad,
            profile.maximum_joint_limit_excess_rad,
            "JOINT_LIMIT_EXCESS",
        ),
        (
            observation.normalized_actuator_command,
            profile.maximum_normalized_actuator_command,
            "ACTUATOR_COMMAND_EXCESS",
        ),
        (
            observation.head_impact_speed_mps,
            profile.maximum_head_impact_speed_mps,
            "HEAD_IMPACT_EXCESS",
        ),
        (
            observation.root_angular_speed_rad_s,
            profile.maximum_root_angular_speed_rad_s,
            "ROOT_ANGULAR_SPEED_EXCESS",
        ),
        (
            observation.self_penetration_m,
            profile.maximum_self_penetration_m,
            "SELF_PENETRATION_EXCESS",
        ),
    )
    reasons.extend(label for value, maximum, label in thresholds if value > maximum)
    hard_contacts = set(observation.contacts) & set(profile.hard_fail_contacts)
    if hard_contacts:
        reasons.append("HARD_CONTACT:" + ",".join(sorted(hard_contacts)))
    permitted = set(profile.always_allowed_contacts)
    permitted.update(profile.phase_contact_permissions.get(observation.phase, ()))
    undeclared = set(observation.contacts) - permitted - set(profile.hard_fail_contacts)
    if undeclared:
        reasons.append("UNDECLARED_CONTACT:" + ",".join(sorted(undeclared)))
    passed = not reasons
    return GrowthSafetyDecision(
        profile_hash=profile.profile_hash,
        passed=passed,
        promotion_eligible=bool(passed and profile.use is GrowthSafetyUse.PROMOTION_SIM),
        hard_failure=not passed,
        reasons=tuple(reasons),
    )


def validate_profile_pair(
    exploration: GrowthSafetyProfile,
    promotion: GrowthSafetyProfile,
) -> None:
    """Require promotion to be at least as strict as exploration."""

    if (
        exploration.use is not GrowthSafetyUse.EXPLORATION_SIM
        or promotion.use is not GrowthSafetyUse.PROMOTION_SIM
    ):
        raise ValueError("growth safety pair has incorrect uses")
    exploration_thresholds = (
        exploration.maximum_joint_limit_excess_rad,
        exploration.maximum_normalized_actuator_command,
        exploration.maximum_head_impact_speed_mps,
        exploration.maximum_root_angular_speed_rad_s,
        exploration.maximum_self_penetration_m,
    )
    promotion_thresholds = (
        promotion.maximum_joint_limit_excess_rad,
        promotion.maximum_normalized_actuator_command,
        promotion.maximum_head_impact_speed_mps,
        promotion.maximum_root_angular_speed_rad_s,
        promotion.maximum_self_penetration_m,
    )
    if any(
        promotion_value > exploration_value
        for exploration_value, promotion_value in zip(
            exploration_thresholds,
            promotion_thresholds,
            strict=True,
        )
    ):
        raise ValueError("promotion safety thresholds cannot be looser than exploration")
    for phase, contacts in promotion.phase_contact_permissions.items():
        if not set(contacts).issubset(exploration.phase_contact_permissions.get(phase, ())):
            raise ValueError("promotion phase contacts must be a subset of exploration")
    if not set(promotion.always_allowed_contacts).issubset(exploration.always_allowed_contacts):
        raise ValueError("promotion always-allowed contacts must be a subset of exploration")
    if not set(exploration.hard_fail_contacts).issubset(promotion.hard_fail_contacts):
        raise ValueError("promotion cannot remove exploration hard-fail contacts")


def _identifiers(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if len(normalized) != len(set(normalized)) or any(
        not _IDENTIFIER.fullmatch(value) for value in normalized
    ):
        raise ValueError(f"{label} must be unique normalized identifiers")
    return normalized


__all__ = [
    "GrowthSafetyDecision",
    "GrowthSafetyObservation",
    "GrowthSafetyProfile",
    "GrowthSafetyUse",
    "evaluate_growth_safety",
    "validate_profile_pair",
]
