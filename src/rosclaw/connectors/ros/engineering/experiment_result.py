"""Bound original experiment records, never physical attestation from JSON claims."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from rosclaw.contracts.common import content_hash

from ..intelligence.graph_contracts import GraphRecord
from .intervention_plan import RosInterventionPlanV1, _encoded


class ExperimentArtifactV1(GraphRecord):
    kind: Literal[
        "native",
        "physics",
        "owner",
        "L1",
        "source_binding",
        "canonical_receipt",
        "independent_stop",
        "strict_age",
        "consumer_output",
        "native_error",
        "independent_review",
    ]
    locator: str = Field(min_length=1, max_length=2048)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExperimentPhaseV1(GraphRecord):
    phase: Literal["BEFORE", "AFTER", "ROLLBACK"]
    run_id: str = Field(min_length=1, max_length=256)
    world_instance_id: str = Field(min_length=1, max_length=256)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    started_at: datetime
    finished_at: datetime
    staging_hash: str
    observed_node: str = Field(min_length=1, max_length=512)
    observed_parameter: str = Field(min_length=1, max_length=256)
    observed_value: Any
    parameter_captured_at: datetime
    native_captured_at: datetime
    source_urdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_scene_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    controllers_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirements_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    declared_body_limits_hash: str
    body_snapshot_hash: str | None = None
    reported_l1: Literal["NOT_EVALUATED", "UNKNOWN", "BLOCKED", "READY_FOR_SIM_TEST"]
    reported_completed_action_ids: list[str] = Field(default_factory=list, max_length=2)
    artifacts: list[ExperimentArtifactV1] = Field(max_length=256)
    evidence_domain: Literal["SIMULATION"] = "SIMULATION"
    source_authentication: Literal["NOT_ESTABLISHED_BY_HASH"] = "NOT_ESTABLISHED_BY_HASH"

    @model_validator(mode="after")
    def valid_window(self):
        if (
            self.started_at.tzinfo is None
            or self.finished_at.tzinfo is None
            or self.finished_at < self.started_at
        ):
            raise ValueError("aware ordered original phase clocks required")
        for at in (self.parameter_captured_at, self.native_captured_at):
            if at.tzinfo is None or not self.started_at <= at <= self.finished_at:
                raise ValueError("original readback and native capture must be within their phase")
        if not 0 <= (self.native_captured_at - self.parameter_captured_at).total_seconds() <= 0.5:
            raise ValueError("original per-node readback cannot be renewed by a phase envelope")
        ids = self.reported_completed_action_ids
        if len(ids) != len(set(ids)) or any(not value or len(value) > 256 for value in ids):
            raise ValueError("bounded unique original action identities required")
        locators = [pin.locator for pin in self.artifacts]
        if len(locators) != len(set(locators)):
            raise ValueError("one original artifact cannot stand in for multiple proof roles")
        return self


class RosExperimentResultV1(GraphRecord):
    schema_version: Literal["rosclaw.ros_experiment_result.v1"] = "rosclaw.ros_experiment_result.v1"
    result_hash: str
    intervention_plan_hash: str
    status: Literal["INCOMPLETE_ORIGINAL_EVIDENCE", "NEEDS_INDEPENDENT_ORIGINAL_VERIFICATION"]
    phases: list[ExperimentPhaseV1] = Field(min_length=3, max_length=3)
    wall_sec: float = Field(ge=0, le=3600)
    missing_evidence: list[str] = Field(max_length=64)
    reference_checks: Literal["STRUCTURAL_BINDINGS_ONLY"] = "STRUCTURAL_BINDINGS_ONLY"
    phase_reports: Literal["SOURCE_ASSERTED_NOT_AUTHENTICATED_BY_PINS"] = (
        "SOURCE_ASSERTED_NOT_AUTHENTICATED_BY_PINS"
    )
    original_files_read: Literal[False] = False
    independent_physical_verification: Literal["NOT_ATTESTED_BY_THIS_CONTRACT"] = (
        "NOT_ATTESTED_BY_THIS_CONTRACT"
    )
    causal_verification: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"
    memory_promotion: Literal["NOT_GRANTED"] = "NOT_GRANTED"
    execution_authority: Literal["NONE"] = "NONE"
    current_l1: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"
    next_step: Literal["REPLAY_PINNED_ORIGINALS_WITH_EXISTING_INDEPENDENT_SDK_VERIFIERS"] = (
        "REPLAY_PINNED_ORIGINALS_WITH_EXISTING_INDEPENDENT_SDK_VERIFIERS"
    )

    @classmethod
    def from_dict(cls, data):
        obj = cls.model_validate(data)
        document = obj.model_dump(mode="json")
        digest = document.pop("result_hash")
        if not digest or digest != content_hash("rosexperimentresult", document):
            raise ValueError("experiment result hash mismatch")
        return obj


def bind_experiment_result(plan, phases):
    """Validate plan/phase identities and budgets without trusting success booleans.

    Pins are declarations. This pure function does not read any original, verify
    a daemon signature, inspect physics, execute a probe or promote a mechanism.
    The existing independent SDK replay remains a separate mandatory operation.
    """
    plan = RosInterventionPlanV1.from_dict(plan.model_dump(mode="json"))
    if plan.status != "CONDITIONAL_SIM_STAGING_PROPOSAL":
        raise ValueError("a concrete reviewed single-variable proposal is required")
    records = [ExperimentPhaseV1.model_validate(value) for value in phases]
    if [p.phase for p in records] != ["BEFORE", "AFTER", "ROLLBACK"]:
        raise ValueError("exactly three ordered original phase records required")
    if len({p.run_id for p in records}) != 3 or len({p.world_instance_id for p in records}) != 3:
        raise ValueError("independent new run and World instance identities required")
    for previous, current in zip(records, records[1:], strict=False):
        if previous.finished_at > current.started_at:
            raise ValueError(
                "phases overlap; independent closed-before-next-World records required"
            )
    wall = (records[-1].finished_at - records[0].started_at).total_seconds()
    if wall > plan.request.max_wall_sec:
        raise ValueError("preregistered total experiment wall budget exceeded")
    for key in [
        "source_revision",
        "image_id",
        "source_urdf_sha256",
        "physical_scene_signature",
        "controllers_sha256",
        "requirements_sha256",
        "declared_body_limits_hash",
    ]:
        if len({getattr(p, key) for p in records}) != 1:
            raise ValueError(
                "changed protected source/geometry/controller/requirements/policy binding: " + key
            )
    if records[0].source_revision != plan.request.source_revision:
        raise ValueError("phase source differs from the declared frozen experiment revision")
    if [p.staging_hash for p in records] != [
        plan.before_staging_hash,
        plan.after_staging_hash,
        plan.rollback_staging_hash,
    ]:
        raise ValueError("original phase staging differs from sealed proposal or rollback")
    original = plan.rollback_documents
    for key in plan.changed_path:
        original = original[key]
    for phase in records:
        expected = plan.request.expected_after_value if phase.phase == "AFTER" else original
        if (
            phase.observed_node != plan.request.node
            or phase.observed_parameter != plan.request.parameter
            or _encoded(phase.observed_value) != _encoded(expected)
        ):
            raise ValueError("actual original phase readback contradicts the sealed hypothesis")
        if (
            phase.native_captured_at - phase.parameter_captured_at
        ).total_seconds() * 1000 > plan.max_age_ms:
            raise ValueError("original parameter exceeds the graph-bound freshness window")
    actions = [action for p in records for action in p.reported_completed_action_ids]
    if len(actions) != len(set(actions)):
        raise ValueError("action identities reused across independent Worlds")
    receipt_pins = [pin for p in records for pin in p.artifacts if pin.kind == "canonical_receipt"]
    if (
        len(receipt_pins) > plan.request.max_total_goals
        or len(actions) > plan.request.max_total_goals
    ):
        raise ValueError("preregistered total action budget exceeded")
    if len({pin.sha256 for pin in receipt_pins}) != len(receipt_pins):
        raise ValueError("the same original receipt bytes cannot be counted twice")
    missing = []
    for phase in records:
        kinds = [pin.kind for pin in phase.artifacts]
        required = {"native", "physics", "owner"}
        if phase.phase == "AFTER":
            required |= {
                "L1",
                "source_binding",
                "independent_stop",
                "strict_age",
                "independent_review",
            }
            if phase.reported_l1 != "READY_FOR_SIM_TEST" or not phase.body_snapshot_hash:
                missing.append(
                    "AFTER: fresh current Body binding and reported full L1 READY required"
                )
            if (
                len(phase.reported_completed_action_ids) != plan.request.max_after_goals
                or kinds.count("canonical_receipt") != plan.request.max_after_goals
            ):
                missing.append(
                    "AFTER: two distinct original canonical receipts/action identities required"
                )
        if plan.request.parameter.endswith(".sensor_frame"):
            required.add("consumer_output")
        for kind in sorted(required - set(kinds)):
            missing.append(phase.phase + ":" + kind)
    result = RosExperimentResultV1(
        result_hash="",
        intervention_plan_hash=plan.plan_hash,
        status="INCOMPLETE_ORIGINAL_EVIDENCE"
        if missing
        else "NEEDS_INDEPENDENT_ORIGINAL_VERIFICATION",
        phases=records,
        wall_sec=wall,
        missing_evidence=missing,
    )
    payload = result.model_dump(mode="json")
    payload.pop("result_hash")
    return RosExperimentResultV1.from_dict(
        {"result_hash": content_hash("rosexperimentresult", payload), **payload}
    )
