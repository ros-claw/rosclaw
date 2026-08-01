"""Truthful contracts for experience acquired outside ROSClaw Practice."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceDescriptor:
    provider: str
    dataset: str
    revision: str
    inventory_hash: str
    license_hash: str
    attribution: str
    revision_binding: str

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.provider, self.dataset, self.attribution)):
            raise ValueError("collective source identity and attribution must not be empty")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("collective source revision must be a full git commit")
        if not _SHA256.fullmatch(self.inventory_hash) or not _SHA256.fullmatch(
            self.license_hash
        ):
            raise ValueError("collective source commitments must be sha256 digests")
        if self.revision_binding not in {"VERIFIED", "UNVERIFIED_LOCAL_SNAPSHOT"}:
            raise ValueError("unknown collective source revision binding")


@dataclass(frozen=True)
class ExperienceCapsule:
    """A non-executable external experience record.

    Capsules carry semantics and provenance.  They cannot authorize a policy,
    hardware execution, or promotion evidence.
    """

    source: SourceDescriptor
    source_body: str
    target_body: str
    target_body_mapping: str
    task_semantics: tuple[str, ...]
    observation_semantics: tuple[str, ...]
    action_semantics: tuple[str, ...]
    modalities: tuple[str, ...]
    quality: str
    truth_level: str
    applicability: str
    training_eligible: bool
    promotion_evidence_eligible: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.experience_capsule.v1"

    def __post_init__(self) -> None:
        if self.truth_level not in {"T0", "T1", "T2", "T3", "T4", "T5"}:
            raise ValueError("unknown Dream truth level")
        if self.promotion_evidence_eligible:
            raise ValueError("collective capsules cannot be promotion evidence")
        if self.hardware_authorized:
            raise ValueError("collective capsules cannot authorize hardware")
        if not self.observation_semantics or not self.modalities:
            raise ValueError("collective capsule must describe its observations and modalities")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
