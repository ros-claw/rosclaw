"""Sealed read-only diagnostic proposals. Observation is never causal verification."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from rosclaw.contracts.common import content_hash

from ..intelligence.graph_contracts import GraphRecord, ReadOnlyProbeRequestV1


class RosHypothesisV1(GraphRecord):
    id: str
    issue_id: str
    description: str
    predicate: Literal[
        "RECEIVE_GAP",
        "QOS_CONFLICT",
        "CLOCK_CONFLICT",
        "TF_TIME_CONFLICT",
        "TF_AUTHORITY_CONFLICT",
        "SOURCE_CONSUMER_CONFLICT",
        "UNRESOLVED_PATH",
    ]
    targets: list[str] = Field(max_length=64)
    status: Literal["SUPPORTED_OBSERVATION", "CONTRADICTED_OBSERVATION", "UNKNOWN"]
    supporting_evidence: list[str] = Field(max_length=512)
    contradicting_evidence: list[str] = Field(max_length=512)
    confounders: tuple[str, ...]
    causal_verification: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"


class DiagnosticPredictionV1(GraphRecord):
    hypothesis_id: str
    if_supported: str
    if_contradicted: str
    on_missing_or_conflicting_evidence: Literal["UNKNOWN"] = "UNKNOWN"


class DiagnosticProbeV1(GraphRecord):
    request: ReadOnlyProbeRequestV1
    predictions: list[DiagnosticPredictionV1] = Field(min_length=1, max_length=64)
    discriminated_hypotheses: int = Field(ge=1, le=64)
    selection_basis: Literal["AVAILABLE_PROVIDER_THEN_SUPPORTED_AND_UNKNOWN_CANDIDATE_COVERAGE"] = (
        "AVAILABLE_PROVIDER_THEN_SUPPORTED_AND_UNKNOWN_CANDIDATE_COVERAGE"
    )


class RosDiagnosticExperimentV1(GraphRecord):
    schema_version: Literal["rosclaw.ros_diagnostic_experiment.v1"] = (
        "rosclaw.ros_diagnostic_experiment.v1"
    )
    experiment_hash: str
    graph_hash: str
    snapshot_hash: str
    source_ref: str
    proposed_at: datetime
    evidence_domain: Literal["SEALED_REPLAY", "OBSERVATION"]
    status: Literal["NEEDS_READ_ONLY_PROBE", "NO_TARGET_ISSUE", "BLOCKED_SOURCE_OR_TIME"]
    hypotheses: list[RosHypothesisV1] = Field(max_length=64)
    probes: list[DiagnosticProbeV1] = Field(max_length=8)
    deferred_hypothesis_ids: list[str] = Field(max_length=64)
    budget_sec: int = Field(ge=1, le=30)
    max_probes: int = Field(ge=1, le=8)
    total_duration_sec: int = Field(ge=0, le=30)
    intervention_class: Literal["READ_ONLY_PROBE"] = "READ_ONLY_PROBE"
    side_effects: Literal["NONE"] = "NONE"
    on_missing_evidence: Literal["UNKNOWN"] = "UNKNOWN"
    execution_status: Literal["PROPOSED_NOT_EXECUTED"] = "PROPOSED_NOT_EXECUTED"
    execution_authority: Literal["NONE"] = "NONE"
    current_l1: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"

    @model_validator(mode="after")
    def linked(self):
        if self.proposed_at.tzinfo is None:
            raise ValueError("aware proposal clock required")
        ids = [h.id for h in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate hypothesis identity")
        planned = set()
        request_ids = []
        for probe in self.probes:
            request_ids.append(probe.request.id)
            predictions = [p.hypothesis_id for p in probe.predictions]
            if len(predictions) != len(set(predictions)) or not set(predictions) <= set(ids):
                raise ValueError("unresolved or repeated prediction")
            if probe.discriminated_hypotheses != len(predictions):
                raise ValueError("candidate coverage mismatch")
            if not 1 <= len(probe.request.targets) <= 64:
                raise ValueError("bounded nonempty probe targets required")
            planned.update(predictions)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("duplicate probe identity")
        deferred = self.deferred_hypothesis_ids
        if len(deferred) != len(set(deferred)) or set(deferred) != set(ids) - planned:
            raise ValueError("explicit complete deferred coverage required")
        if self.total_duration_sec != sum(p.request.max_duration_sec for p in self.probes):
            raise ValueError("probe duration sum mismatch")
        if self.total_duration_sec > self.budget_sec:
            raise ValueError("diagnostic budget exceeded")
        return self

    @classmethod
    def from_dict(cls, data):
        obj = cls.model_validate(data)
        payload = obj.model_dump(mode="json")
        digest = payload.pop("experiment_hash")
        if not digest or digest != content_hash("rosdiagnostic", payload):
            raise ValueError("diagnostic experiment hash mismatch")
        return obj


class DiagnosticAssessmentV1(GraphRecord):
    schema_version: Literal["rosclaw.ros_diagnostic_assessment.v1"] = (
        "rosclaw.ros_diagnostic_assessment.v1"
    )
    assessment_hash: str
    experiment_hash: str
    before_graph_hash: str
    after_graph_hash: str
    status: Literal["OBSERVATIONAL_UPDATE", "ABSTAIN"]
    reasons: list[str]
    hypotheses: list[RosHypothesisV1] = Field(max_length=64)
    withdrawn_hypothesis_ids: list[str] = Field(max_length=64)
    causal_verification: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"
    probe_execution: Literal["NOT_ATTESTED_BY_GRAPH"] = "NOT_ATTESTED_BY_GRAPH"
    execution_authority: Literal["NONE"] = "NONE"

    @classmethod
    def from_dict(cls, data):
        obj = cls.model_validate(data)
        payload = obj.model_dump(mode="json")
        digest = payload.pop("assessment_hash")
        if not digest or digest != content_hash("rosdiagnosticassessment", payload):
            raise ValueError("diagnostic assessment hash mismatch")
        return obj
