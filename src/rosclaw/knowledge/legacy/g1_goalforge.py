"""Immutable GoalForge task knowledge and candidate constraints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_HARD_TORQUE_LIMITS,
    GOALFORGE_TASK_ID,
    ShotParameters,
    hash_json,
)


@dataclass(frozen=True)
class GoalForgeKnowledge:
    body_hash: str
    kick_prior_hash: str
    task_id: str = GOALFORGE_TASK_ID
    maximum_retries: int = 2
    target_radius_m: float = 0.48
    maximum_support_slip_m: float = 0.08
    minimum_pelvis_height_m: float = 0.55
    immutable_fields: tuple[str, ...] = (
        "e_stop",
        "torque_hard_limit",
        "joint_hard_limit",
        "permit",
        "lease",
        "evidence_semantics",
        "hardware_authorization",
    )
    schema_version: str = "rosclaw.g1_goalforge.knowledge.v1"

    def __post_init__(self) -> None:
        if not self.body_hash.startswith("sha256:") or not self.kick_prior_hash.startswith(
            "sha256:"
        ):
            raise ValueError("GoalForge knowledge must bind body and prior hashes")
        if self.maximum_retries != 2:
            raise ValueError("GoalForge V4 bounds retries to two")

    @property
    def knowledge_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["hard_torque_limits"] = list(G1_HARD_TORQUE_LIMITS)
        return value

    def validate_candidate(
        self,
        *,
        candidate: ShotParameters,
        attempted_mutations: set[str] | frozenset[str] = frozenset(),
    ) -> tuple[bool, tuple[str, ...]]:
        errors: list[str] = []
        forbidden = set(self.immutable_fields) & set(attempted_mutations)
        if forbidden:
            errors.append("immutable_mutation=" + ",".join(sorted(forbidden)))
        if candidate.kick_foot != "right":
            errors.append("fixed_prior_only_supports_right_foot")
        if candidate.policy_type == "learned_adapter" and (candidate.dataset_snapshot_hash is None):
            errors.append("learned_candidate_missing_dataset")
        return not errors, tuple(errors)


__all__ = ["GoalForgeKnowledge"]
