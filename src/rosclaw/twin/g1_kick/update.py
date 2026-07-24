"""Auditable record for one bounded twin update."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


@dataclass(frozen=True)
class KickTwinUpdate:
    parent_belief_hash: str
    child_belief_hash: str
    prediction_error_hash: str
    changed_parameters: tuple[str, ...]
    reasons: tuple[str, ...]
    hidden_truth_accessed: bool = False
    schema_version: str = "rosclaw.g1_kick.twin_update.v1"

    def __post_init__(self) -> None:
        hashes = (
            self.parent_belief_hash,
            self.child_belief_hash,
            self.prediction_error_hash,
        )
        if any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("twin update hashes must be sha256 digests")
        if self.hidden_truth_accessed:
            raise ValueError("KickTwin update must not access hidden truth")
        if not self.changed_parameters:
            raise ValueError("KickTwin update must change at least one belief")

    @property
    def update_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["KickTwinUpdate"]
