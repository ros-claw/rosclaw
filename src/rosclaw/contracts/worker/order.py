"""WorkOrderV1 / WorkResultV1 (总纲 §9.4, §9.5).

A WorkOrder is the minimal bounded contract given to a worker: explicit
inputs, outputs, tools, data scope, deadline, budget and verification. It
never carries permits, API keys, passwords or private ledger data —
credentials are injected by the adapter at runtime by reference.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from rosclaw.contracts.common import ContractModel

#: WorkOrder run-track states (ADR-0003 dual-track lifecycle).
WORK_ORDER_STATES = (
    "DRAFT",
    "OFFERED",
    "CLAIMED",
    "RUNNING",
    "SUBMITTED",
    "VERIFYING",
    "ACCEPTED",
    "BLOCKED",
    "FAILED",
    "EXPIRED",
    "CANCELLED",
)


class DataScope(ContractModel):
    SCHEMA = "rosclaw.work_data_scope.v1"

    schema_version: Literal["rosclaw.work_data_scope.v1"] = "rosclaw.work_data_scope.v1"
    readable_paths: list[str] = Field(default_factory=list)
    writable_paths: list[str] = Field(default_factory=list)
    memory_namespaces: list[str] = Field(default_factory=list)


class BudgetEnvelope(ContractModel):
    SCHEMA = "rosclaw.work_budgets.v1"

    schema_version: Literal["rosclaw.work_budgets.v1"] = "rosclaw.work_budgets.v1"
    wall_time_sec: int = 1800
    model_tokens: int = 150_000
    monetary_microunits: int = 3_000_000
    max_tool_calls: int = 300
    max_children: int = 0


class WorkOrderLease(ContractModel):
    SCHEMA = "rosclaw.work_lease.v1"

    schema_version: Literal["rosclaw.work_lease.v1"] = "rosclaw.work_lease.v1"
    lease_id: str
    issued_at: str
    expires_at: str


class ExpectedOutput(ContractModel):
    SCHEMA = "rosclaw.expected_output.v1"

    schema_version: Literal["rosclaw.expected_output.v1"] = "rosclaw.expected_output.v1"
    schema_ref: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class WorkVerification(ContractModel):
    SCHEMA = "rosclaw.work_verification.v1"

    schema_version: Literal["rosclaw.work_verification.v1"] = "rosclaw.work_verification.v1"
    deterministic: list[str] = Field(default_factory=list)
    independent_review: bool = False


class SideEffectPolicy(ContractModel):
    SCHEMA = "rosclaw.side_effect_policy.v1"

    schema_version: Literal["rosclaw.side_effect_policy.v1"] = "rosclaw.side_effect_policy.v1"
    class_: str = Field(
        default="none",
        alias="class",
        description="none | sandbox_process | workspace_write | network_write | physical",
    )
    idempotency_key: str | None = None
    physical_side_effects: Literal["forbidden", "team_task_only"] = "forbidden"

    @model_validator(mode="after")
    def _side_effects_require_idempotency(self) -> SideEffectPolicy:
        if self.class_ not in ("none", "sandbox_process") and not self.idempotency_key:
            raise ValueError(f"side-effect class {self.class_!r} requires an idempotency_key")
        if self.class_ == "physical" and self.physical_side_effects == "forbidden":
            raise ValueError("physical side-effect class is forbidden in P0/P1")
        return self


class WorkOrderV1(ContractModel):
    SCHEMA = "rosclaw.work_order.v1"
    HASH_PREFIX = "wo"

    schema_version: Literal["rosclaw.work_order.v1"] = "rosclaw.work_order.v1"
    work_order_id: str
    mission_id: str
    task_id: str | None = None
    issued_by: str
    assigned_to: str | None = None
    capability: str
    goal: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    data_scope: DataScope = Field(default_factory=DataScope)
    budgets: BudgetEnvelope = Field(default_factory=BudgetEnvelope)
    lease: WorkOrderLease | None = None
    expected_output: ExpectedOutput = Field(default_factory=ExpectedOutput)
    verification: WorkVerification = Field(default_factory=WorkVerification)
    side_effect_policy: SideEffectPolicy = Field(default_factory=SideEffectPolicy)
    delegation_depth: int = 0
    # 规格 §19.5：防递归爆炸——默认只允许 primary→worker 一层。
    max_delegation_depth: int = 1
    parent_work_order_id: str | None = None
    root_work_order_id: str | None = None
    status: str = "DRAFT"

    @model_validator(mode="after")
    def _status_legal(self) -> WorkOrderV1:
        if self.status not in WORK_ORDER_STATES:
            raise ValueError(f"illegal work order status {self.status!r}")
        return self


class ResultArtifact(ContractModel):
    SCHEMA = "rosclaw.result_artifact.v1"

    schema_version: Literal["rosclaw.result_artifact.v1"] = "rosclaw.result_artifact.v1"
    ref: str
    media_type: str = "application/octet-stream"
    digest: str | None = None


class ResultClaim(ContractModel):
    SCHEMA = "rosclaw.result_claim.v1"

    schema_version: Literal["rosclaw.result_claim.v1"] = "rosclaw.result_claim.v1"
    claim: str
    evidence_refs: list[str] = Field(default_factory=list)


class WorkUsage(ContractModel):
    SCHEMA = "rosclaw.work_usage.v1"

    schema_version: Literal["rosclaw.work_usage.v1"] = "rosclaw.work_usage.v1"
    wall_time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_microunits: int = 0


class WorkResultV1(ContractModel):
    """A submission from a worker. ``COMPLETED`` ≠ accepted by ROSClaw."""

    SCHEMA = "rosclaw.work_result.v1"
    HASH_PREFIX = "wres"

    schema_version: Literal["rosclaw.work_result.v1"] = "rosclaw.work_result.v1"
    work_order_id: str
    worker_id: str
    lease_id: str
    status: Literal["COMPLETED", "FAILED", "BLOCKED"] = "COMPLETED"
    started_at: str | None = None
    finished_at: str | None = None
    summary: str = ""
    artifacts: list[ResultArtifact] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[ResultClaim] = Field(default_factory=list)
    usage: WorkUsage = Field(default_factory=WorkUsage)
    children: list[str] = Field(default_factory=list)
    worker_trace_ref: str | None = None
    warnings: list[str] = Field(default_factory=list)
