"""Diagnostic-only engineering IR. None of these contracts grants authority."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rosclaw.contracts.common import content_hash


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class EvidenceRefV1(StrictModel):
    source_ref: str
    snapshot_hash: str
    pointer: str
    observed_at: str
    value_hash: str
    classification: Literal["MEASURED", "INFERRED", "UNKNOWN"]


class ConstraintDeltaV1(StrictModel):
    id: str
    invariant: str
    status: Literal["SATISFIED", "VIOLATED", "UNKNOWN"]
    expected: str
    observed: Any = None
    evidence_refs: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)


class RepairIntentV1(StrictModel):
    id: str
    semantic_field: Literal[
        "localization.tf_broadcast",
        "sensor.scan_bridge_state",
        "navigation.managed_nodes",
        "costmap.radius",
        "control.route_review",
    ]
    proposed_value: Any
    classification: Literal["INFERRED_CANDIDATE"] = "INFERRED_CANDIDATE"
    stage_only: Literal[True] = True
    applicable_now: Literal[False] = False
    evidence_refs: list[str]
    preconditions: list[str]
    proof_obligations: tuple[str, ...] = (
        "new_World_fresh_source_and_Body_binding",
        "independent_full_L1_READY",
        "two_canonical_navigation_receipts",
        "independent_stop_zero_collision_and_source_continuity",
    )

    @model_validator(mode="after")
    def typed_semantic_value(self):
        value = self.proposed_value
        valid = {
            "localization.tf_broadcast": type(value) is bool,
            "sensor.scan_bridge_state": value == "running",
            "navigation.managed_nodes": type(value) is list
            and bool(value)
            and all(type(v) is str and v for v in value),
            "costmap.radius": type(value) in (int, float) and math.isfinite(value) and value > 0,
            "control.route_review": value is None,
        }
        if not valid[self.semantic_field]:
            raise ValueError("typed semantic candidate value required")
        return self


class RosEngineeringCaseV1(StrictModel):
    schema_version: Literal["rosclaw.ros_engineering_case.v1"] = "rosclaw.ros_engineering_case.v1"
    mission: Literal["navigation.navigate_to_pose"] = "navigation.navigate_to_pose"
    case_hash: str
    snapshot_hash: str
    reviewed_at: str
    evidence_domain: Literal["SEALED_REPLAY", "OBSERVATION"]
    run_id: str | None
    body_hash: str | None
    source_urdf_sha256: str
    current_l1: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"
    current_authority: Literal["NONE"] = "NONE"
    admission: Literal["NOT_ADMITTED"] = "NOT_ADMITTED"
    source_status: Literal["MATCHED", "MISMATCH", "UNKNOWN"]
    constraints: list[ConstraintDeltaV1]
    repair_candidates: list[RepairIntentV1]
    evidence_index: dict[str, EvidenceRefV1]
    derived_documents: dict[str, Any]
    navigation_observation_status: str
    diagnosis_issue_codes: list[str]
    unassessed_checks: tuple[str, ...] = (
        "full_L1_geometry_and_effective_footprint",
        "daemon_admission",
        "physical_navigation",
        "current_source_continuity",
    )

    @classmethod
    def from_dict(cls, data):
        case = cls.model_validate(data)
        payload = case.model_dump(mode="json")
        digest = payload.pop("case_hash")
        if not digest or digest != content_hash("roscase", payload):
            raise ValueError("engineering case hash mismatch")
        return case
