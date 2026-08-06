"""TaskGraphV1 / TaskNodeV1 / TaskGraphPatchV1 (总纲 §4.3).

The model may only *propose* a TaskGraphPatchV1; the MissionStore performs
DAG validation, field validation, permission checks, budget checks and a
revision CAS before committing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel, ValidationError


class TaskKind(StrEnum):
    PERCEIVE = "perceive"
    REASON = "reason"
    CREATE_ARTIFACT = "create_artifact"
    VALIDATE = "validate"
    REQUEST_ACTION = "request_action"
    COORDINATE = "coordinate"
    WAIT = "wait"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    NEEDS_REBINDING = "NEEDS_REBINDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Assignee(ContractModel):
    SCHEMA = "rosclaw.task_assignee.v1"

    schema_version: Literal["rosclaw.task_assignee.v1"] = "rosclaw.task_assignee.v1"
    type: Literal["native", "worker", "robot"] = "native"
    id: str | None = None


class Lease(ContractModel):
    SCHEMA = "rosclaw.task_lease.v1"

    schema_version: Literal["rosclaw.task_lease.v1"] = "rosclaw.task_lease.v1"
    lease_id: str | None = None
    expires_at: str | None = None


class TaskConstraints(ContractModel):
    SCHEMA = "rosclaw.task_constraints.v1"

    schema_version: Literal["rosclaw.task_constraints.v1"] = "rosclaw.task_constraints.v1"
    body_id: str | None = None
    freshness_ms: int | None = None
    risk_tier: str = "LOW"
    deadline: str | None = None


class TaskVerification(ContractModel):
    SCHEMA = "rosclaw.task_verification.v1"

    schema_version: Literal["rosclaw.task_verification.v1"] = "rosclaw.task_verification.v1"
    schema_ref: str | None = None
    verifier: str | None = Field(None, description="e.g. deterministic:pose_bounds")


class TaskNodeV1(ContractModel):
    SCHEMA = "rosclaw.task_node.v1"
    HASH_PREFIX = "task"

    schema_version: Literal["rosclaw.task_node.v1"] = "rosclaw.task_node.v1"
    task_id: str
    mission_id: str
    kind: TaskKind
    goal: str
    dependencies: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    required_capabilities: list[str] = Field(default_factory=list)
    assignee: Assignee = Field(default_factory=Assignee)
    lease: Lease = Field(default_factory=Lease)
    verification: TaskVerification = Field(default_factory=TaskVerification)
    status: TaskStatus = TaskStatus.PENDING
    attempt: int = 0
    max_attempts: int = 3
    artifacts: list[str] = Field(default_factory=list)
    trace_id: str | None = None


class TaskGraphV1(ContractModel):
    SCHEMA = "rosclaw.task_graph.v1"
    HASH_PREFIX = "tgraph"

    schema_version: Literal["rosclaw.task_graph.v1"] = "rosclaw.task_graph.v1"
    mission_id: str
    revision: int = 0
    nodes: list[TaskNodeV1] = Field(default_factory=list)

    def node_ids(self) -> set[str]:
        return {n.task_id for n in self.nodes}

    def validate_dag(self) -> None:
        """Raise ValidationError on dangling deps, duplicates, or cycles."""
        ids: set[str] = set()
        for node in self.nodes:
            if node.task_id in ids:
                raise ValidationError(f"duplicate task_id {node.task_id!r}")
            ids.add(node.task_id)
        for node in self.nodes:
            for dep in node.dependencies:
                if dep not in ids:
                    raise ValidationError(f"task {node.task_id!r} depends on unknown task {dep!r}")
        # Kahn topological check.
        indegree = {n.task_id: 0 for n in self.nodes}
        for node in self.nodes:
            indegree[node.task_id] += len(node.dependencies)
        queue = [tid for tid, deg in indegree.items() if deg == 0]
        seen = 0
        while queue:
            current = queue.pop()
            seen += 1
            for node in self.nodes:
                if current in node.dependencies:
                    indegree[node.task_id] -= 1
                    if indegree[node.task_id] == 0:
                        queue.append(node.task_id)
        if seen != len(self.nodes):
            raise ValidationError("task graph contains a dependency cycle")


class PatchOperation(ContractModel):
    SCHEMA = "rosclaw.task_patch_op.v1"

    schema_version: Literal["rosclaw.task_patch_op.v1"] = "rosclaw.task_patch_op.v1"
    op: Literal["add_node", "remove_node", "update_node", "set_status"]
    node: TaskNodeV1 | None = None
    task_id: str | None = None
    status: TaskStatus | None = None


class TaskGraphPatchV1(ContractModel):
    """A *proposal* from the model. Never treated as committed state."""

    SCHEMA = "rosclaw.task_graph_patch.v1"
    HASH_PREFIX = "tgpatch"

    schema_version: Literal["rosclaw.task_graph_patch.v1"] = "rosclaw.task_graph_patch.v1"
    patch_id: str
    mission_id: str
    base_revision: int
    operations: list[PatchOperation] = Field(default_factory=list)
    rationale: str = ""
    proposed_by: str = Field(..., description="actor id, e.g. agent:rosclaw-native:body_01")
    context_revision: int = 0
