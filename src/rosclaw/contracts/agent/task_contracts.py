"""通用任务合约（九审 §6/§25.11，NINE-4）。

机器可执行的任务合约体系——不写死任何特定任务（无 star 字段）：
UserTurn → GoalEnvelope → OutcomeContract → TaskSpecV3 → TaskGraphV1
→ TaskResultV3 + ArtifactRef + EvidenceClaim + WorkOrderV2。
"""

from __future__ import annotations

from typing import Any, Literal

from rosclaw.contracts.agent.task_graph import TaskGraphV1, TaskNodeV1  # noqa: F401
from rosclaw.contracts.common import ContractModel


class GoalEnvelopeV1(ContractModel):
    """模型/解析器对一次用户 turn 的目标理解（§6.2）。"""

    SCHEMA = "rosclaw.goal_envelope.v1"
    schema_version: Literal["rosclaw.goal_envelope.v1"] = "rosclaw.goal_envelope.v1"
    turn_id: str
    kind: str  # CONVERSATION/RESEARCH/INSPECTION/DIAGNOSIS/CREATION/SIMULATION/...
    goal: str
    target: dict[str, Any] = {}
    requested_mode: str = "SIMULATION"
    deliverables: list[str] = []
    constraints: list[str] = []
    ambiguities: list[str] = []
    confidence: float = 0.0
    route_hint: str = ""


class OutcomeContractV1(ContractModel):
    """任务何时算完成（§6.3）。"""

    SCHEMA = "rosclaw.outcome_contract.v1"
    schema_version: Literal["rosclaw.outcome_contract.v1"] = "rosclaw.outcome_contract.v1"
    goal: str
    required_outcomes: list[str] = []
    deliverables: list[str] = []
    required_evidence: str = "COMMAND_REPLAY"
    success_criteria: dict[str, Any] = {}
    partial_policy: str = "report_partial_with_recovery"


class TaskSpecV3(ContractModel):
    """机器可执行任务（§6.4）。"""

    SCHEMA = "rosclaw.task_spec.v3"
    schema_version: Literal["rosclaw.task_spec.v3"] = "rosclaw.task_spec.v3"
    task_type: str
    goal: str = ""
    robot_id: str = ""
    mode: str = "SIMULATION"
    parameters: dict[str, Any] = {}
    constraints: dict[str, Any] = {}
    outcome: OutcomeContractV1 | None = None
    budget: dict[str, Any] = {}
    safety_policy: str = ""
    required_capabilities: list[str] = []
    caused_by_turn_id: str = ""


# TaskNodeV1/TaskGraphV1 复用既有合约（contracts/agent/task_graph.py）
# ——不重复定义（schema 冲突），顶部 re-export。


class ArtifactRefV1(ContractModel):
    """产物引用（§18.1）——大结果不进模型上下文。"""

    SCHEMA = "rosclaw.artifact_ref.v1"
    schema_version: Literal["rosclaw.artifact_ref.v1"] = "rosclaw.artifact_ref.v1"
    artifact_id: str
    task_id: str = ""
    kind: str = ""
    mime_type: str = ""
    path: str = ""
    sha256: str = ""
    bytes: int = 0
    producer: str = ""
    evidence_level: str = "PLANNED"
    preview: bool = False
    created_at: str = ""


class EvidenceClaimV1(ContractModel):
    """证据声明（§18.2）。"""

    SCHEMA = "rosclaw.evidence_claim.v1"
    schema_version: Literal["rosclaw.evidence_claim.v1"] = "rosclaw.evidence_claim.v1"
    claim: str
    subject: str = ""
    source: str = ""
    observation: str = ""
    evidence_level: str = "PLANNED"
    verifier: str = ""
    confidence: float = 0.0
    limitations: list[str] = []
    artifact_refs: list[str] = []
    receipt_refs: list[str] = []


class WorkOrderV2(ContractModel):
    """Worker 工作单（§15.3）——无物理权限声明内置。"""

    SCHEMA = "rosclaw.work_order.v2"
    schema_version: Literal["rosclaw.work_order.v2"] = "rosclaw.work_order.v2"
    work_order_id: str
    parent_task_id: str = ""
    role: str = ""
    goal: str = ""
    inputs: list[str] = []
    deliverables: list[str] = []
    constraints: list[str] = []
    budget: dict[str, Any] = {}
    verification: str = "kernel_review_required"
    physical_authority: str = "NONE"
