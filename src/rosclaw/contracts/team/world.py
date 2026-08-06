"""SharedWorldSnapshotV1 / SharedWorldDeltaV1 (总纲 §10.5).

Shared state always carries time, frame, source, confidence, revision and
freshness, with explicit measurement/inference classification, merge policy,
tombstones and QoS. High-rate data never enters the LLM context raw.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class QoSSpec(ContractModel):
    SCHEMA = "rosclaw.world_qos.v1"

    schema_version: Literal["rosclaw.world_qos.v1"] = "rosclaw.world_qos.v1"
    reliability: Literal["reliable", "best_effort"] = "best_effort"
    history_depth: int = 1
    deadline_ms: int | None = None
    lifespan_ms: int | None = None


class ObjectState(ContractModel):
    SCHEMA = "rosclaw.world_object.v1"

    schema_version: Literal["rosclaw.world_object.v1"] = "rosclaw.world_object.v1"
    object_id: str
    pose: dict[str, Any] | None = None
    twist: dict[str, Any] | None = None
    covariance: list[float] | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_class: Literal["measurement", "inference"] = "measurement"
    evidence_ref: str | None = None
    tombstone: bool = False


class _WorldBase(ContractModel):
    team_id: str
    team_epoch: int
    world_revision: int
    source_member: str
    source_sensor: str | None = None
    observed_at: str
    published_at: str
    max_age_ms: int = 1000
    frame_id: str = "world"
    transform_revision: int = 0
    merge_policy: Literal["latest_valid", "covariance_fusion", "authoritative_source"] = (
        "latest_valid"
    )
    objects: list[ObjectState] = Field(default_factory=list)
    qos: QoSSpec = Field(default_factory=QoSSpec)


class SharedWorldSnapshotV1(_WorldBase):
    SCHEMA = "rosclaw.shared_world_snapshot.v1"
    HASH_PREFIX = "worlds"

    schema_version: Literal["rosclaw.shared_world_snapshot.v1"] = "rosclaw.shared_world_snapshot.v1"


class SharedWorldDeltaV1(_WorldBase):
    SCHEMA = "rosclaw.shared_world_delta.v1"
    HASH_PREFIX = "worldd"

    schema_version: Literal["rosclaw.shared_world_delta.v1"] = "rosclaw.shared_world_delta.v1"
    base_revision: int
