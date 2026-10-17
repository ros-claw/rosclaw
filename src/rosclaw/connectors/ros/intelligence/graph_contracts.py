"""Bounded read-only operational graph contracts; no ROS transport or admission."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rosclaw.contracts.common import content_hash

Status = Literal["SATISFIED", "VIOLATED", "UNKNOWN"]


class GraphRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ProvenanceEdgeV1(GraphRecord):
    id: str
    document_ref: str
    document_hash: str
    pointer: str
    value_hash: str
    observed_at: datetime
    freshness: Literal["FRESH", "STALE", "FUTURE", "UNKNOWN"]

    @model_validator(mode="after")
    def aware(self):
        if self.observed_at.tzinfo is None:
            raise ValueError("aware provenance time required")
        return self


class OperationalEntityV1(GraphRecord):
    id: str
    kind: Literal[
        "node", "topic", "service", "action", "frame", "urdf", "source", "run", "body", "world"
    ]
    name: str
    interface_type: str | None = None
    evidence_refs: list[str]


class OperationalEndpointV1(GraphRecord):
    id: str
    topic_id: str
    node_id: str
    kind: Literal["publisher", "subscriber"]
    gid: str | None = None
    identity_status: Literal["OBSERVED_GID", "NODE_ONLY", "AMBIGUOUS"]
    reliability: str | None = None
    durability: str | None = None
    liveliness: str | None = None
    deadline_ns: int | None = Field(None, ge=0)
    evidence_refs: list[str]


class OperationalDataflowV1(GraphRecord):
    id: str
    publisher_id: str
    subscriber_id: str
    observed_policy_status: Status
    # Four observed policies are not the entire RMW compatibility contract.
    full_compatibility: Literal["UNKNOWN"] = "UNKNOWN"
    delivery_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    evidence_refs: list[str]


class OperationalTransformV1(GraphRecord):
    id: str
    parent_id: str
    child_id: str
    static: bool
    authority: str | None
    age_ms: float | None
    clock_domain: Literal["ROS_SIM", "ROS_SYSTEM", "UNKNOWN"]
    evidence_refs: list[str]


class OperationalTimeV1(GraphRecord):
    id: str
    subject: str
    domain: Literal["ROS_SIM", "ROS_SYSTEM", "UNKNOWN"]
    receive_clock: Literal["STEADY", "UNKNOWN"] = "UNKNOWN"
    sample_rate_hz: float | None = Field(None, ge=0)
    receive_age_ms: float | None = Field(None, ge=0)
    message_age_ms: float | None = None
    evidence_refs: list[str]


class RosConstraintFindingV1(GraphRecord):
    id: str
    invariant: str
    status: Status
    scope: str
    evidence_refs: list[str]
    targets: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class RosOperationalGraphV1(GraphRecord):
    schema_version: Literal["rosclaw.ros_operational_graph.v1"] = "rosclaw.ros_operational_graph.v1"
    graph_hash: str
    source_ref: str
    snapshot_hash: str
    case_hash: str
    source_urdf_sha256: str
    requirements_hash: str
    reviewed_at: datetime
    max_age_ms: float = Field(gt=0, le=500)
    evidence_domain: Literal["SEALED_REPLAY", "OBSERVATION"]
    snapshot_freshness: Literal["FRESH", "STALE", "FUTURE"]
    source_status: Literal["MATCHED", "MISMATCH", "UNKNOWN"]
    current_authority: Literal["NONE"] = "NONE"
    current_l1: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"
    entities: list[OperationalEntityV1] = Field(max_length=2048)
    endpoints: list[OperationalEndpointV1] = Field(max_length=1024)
    dataflows: list[OperationalDataflowV1] = Field(max_length=4096)
    transforms: list[OperationalTransformV1] = Field(max_length=1024)
    time_facts: list[OperationalTimeV1] = Field(max_length=2048)
    findings: list[RosConstraintFindingV1] = Field(max_length=4096)
    provenance: dict[str, ProvenanceEdgeV1] = Field(max_length=8192)
    completeness: dict[str, bool]
    limitations: tuple[str, ...] = (
        "historical_or_observed_diagnosis_not_L1_or_motion_authority",
        "missing_endpoint_GIDs_and_TF_authorities_cannot_be_inferred",
        "four_policy_QoS_check_is_not_complete_RMW_compatibility_or_delivery",
        "graph_Body_reference_is_not_a_new_Body_or_source_verification",
    )

    @model_validator(mode="after")
    def linked(self):
        if self.reviewed_at.tzinfo is None:
            raise ValueError("aware review clock required")
        ids = [
            x.id
            for x in self.entities
            + self.endpoints
            + self.dataflows
            + self.transforms
            + self.time_facts
            + self.findings
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("unique graph record identities required")
        entities = {x.id: x for x in self.entities}
        endpoints = {x.id: x for x in self.endpoints}
        for ep in self.endpoints:
            if (
                ep.topic_id not in entities
                or entities[ep.topic_id].kind != "topic"
                or ep.node_id not in entities
                or entities[ep.node_id].kind != "node"
            ):
                raise ValueError("endpoint requires actual topic and node references")
        for flow in self.dataflows:
            if flow.publisher_id not in endpoints or flow.subscriber_id not in endpoints:
                raise ValueError("dataflow endpoint reference missing")
            pub, sub = endpoints[flow.publisher_id], endpoints[flow.subscriber_id]
            if pub.kind != "publisher" or sub.kind != "subscriber" or pub.topic_id != sub.topic_id:
                raise ValueError("dataflow requires same-topic publisher to subscriber")
        for tf in self.transforms:
            if any(
                k not in entities or entities[k].kind != "frame"
                for k in (tf.parent_id, tf.child_id)
            ):
                raise ValueError("transform frame reference missing")
        for record in (
            self.entities
            + self.endpoints
            + self.dataflows
            + self.transforms
            + self.time_facts
            + self.findings
        ):
            if any(ref not in self.provenance for ref in record.evidence_refs):
                raise ValueError("unresolved provenance reference")
        for key, value in self.provenance.items():
            if key != value.id:
                raise ValueError("provenance key/id mismatch")
        return self

    @classmethod
    def from_dict(cls, data):
        graph = cls.model_validate(data)
        payload = graph.model_dump(mode="json")
        digest = payload.pop("graph_hash")
        if not digest or digest != content_hash("rosopgraph", payload):
            raise ValueError("operational graph hash mismatch")
        return graph


class ReadOnlyProbeRequestV1(GraphRecord):
    id: str
    operation: Literal[
        "SOURCE_CAPTURE",
        "ENDPOINT_QOS",
        "TF_AUTHORITY_WINDOW",
        "CLOCK_TF_WINDOW",
        "NATIVE_DIAGNOSTICS",
    ]
    targets: list[str]
    finding_ids: list[str]
    required_evidence: list[str]
    provider: Literal[
        "EXISTING_NATIVE_READ_ONLY_PROBE", "PROVIDER_EXTENSION_REQUIRED", "OPERATOR_SOURCE_REVIEW"
    ]
    max_duration_sec: int = Field(ge=1, le=5)
    side_effects: Literal["NONE"] = "NONE"
    execution_authority: Literal["NONE"] = "NONE"
    execution_status: Literal["PROPOSED_NOT_EXECUTED"] = "PROPOSED_NOT_EXECUTED"


class MissingEvidencePlanV1(GraphRecord):
    schema_version: Literal["rosclaw.ros_missing_evidence_plan.v1"] = (
        "rosclaw.ros_missing_evidence_plan.v1"
    )
    plan_hash: str
    graph_hash: str
    snapshot_hash: str
    source_ref: str
    status: Literal["NEEDS_READ_ONLY_PROBE", "NO_MISSING_EVIDENCE", "BLOCKED_SOURCE_MISMATCH"]
    requests: list[ReadOnlyProbeRequestV1] = Field(max_length=8)
    deferred_finding_ids: list[str]
    total_budget_sec: int = Field(ge=0, le=30)
    execution_authority: Literal["NONE"] = "NONE"
    current_l1: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"
