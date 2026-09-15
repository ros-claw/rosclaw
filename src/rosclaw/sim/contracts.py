"""仿真域契约骨架（PR-MH0，ADR-0014）。

Maturity: experimental（ADR-0000 §4）。MH0 阶段无跨进程消费者；
首个跨进程消费者落地时晋升至 ``rosclaw.contracts`` 包。

全部契约带公共信封 ``backend`` / ``backend_version`` / ``created_at`` /
``digest``；``digest`` 经 :meth:`SimContract.with_digest` 填充，自身不
参与内容哈希。canonical JSON + sha256、未知字段前向兼容、未知主版本
fail-closed 均由 ``rosclaw.contracts.common.ContractModel`` 提供。
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Self

from rosclaw.contracts.common import ContractModel

__all__ = [
    "SimContract",
    "SimulationBackendCapabilities",
    "ModelReference",
    "ModelInspection",
    "StateSnapshot",
    "ObservationRequest",
    "ObservationResult",
    "RolloutRequest",
    "SimulationTrace",
    "ModelPatch",
    "ModelPatchResult",
    "AuditRequest",
    "AuditResult",
    "ExperimentBranch",
    "ExperimentResult",
    "ComparisonResult",
    "SimulationEvidenceBundle",
    "SimulationReceipt",
]


class SimContract(ContractModel):
    """仿真契约公共信封。

    ``created_at`` 默认空串以保证骨架实例 digest 确定性；生产者填真实
    时间戳时，digest 是对"内容（含 created_at）"的承诺——同一逻辑对象
    两次创建 digest 不同属设计使然，去重靠 model_ref / states_digest
    等业务摘要字段。
    """

    HASH_PREFIX: ClassVar[str] = "sim"
    HASH_EXCLUDE_FIELD: ClassVar[str] = "digest"

    backend: str = ""
    backend_version: str = ""
    created_at: str = ""  # ISO-8601 UTC
    digest: str = ""

    def with_digest(self) -> Self:
        """返回 digest 已盖章的副本（幂等）。"""
        return self.model_copy(update={"digest": self.canonical_hash()})


class SimulationBackendCapabilities(SimContract):
    """后端运行时能力探测结果（真探测，非版本字符串比较）。"""

    SCHEMA: ClassVar[str] = "rosclaw.sim.backend_capabilities.v1"
    HASH_PREFIX: ClassVar[str] = "simcap"
    schema_version: Literal["rosclaw.sim.backend_capabilities.v1"] = (
        "rosclaw.sim.backend_capabilities.v1"
    )

    capabilities: dict[str, bool] = {}
    details: dict[str, str] = {}
    platform: str = ""
    gl_backend: str = ""
    missing_required: list[str] = []


class ModelReference(SimContract):
    """模型不可变引用：修改模型 = 新 patch → 新 ModelReference。"""

    SCHEMA: ClassVar[str] = "rosclaw.sim.model_reference.v1"
    HASH_PREFIX: ClassVar[str] = "simref"
    schema_version: Literal["rosclaw.sim.model_reference.v1"] = "rosclaw.sim.model_reference.v1"

    model_ref: str = ""
    model_digest: str = ""
    parent_model_ref: str | None = None
    source: dict[str, Any] = {}  # {"kind": "task"|"eurdf"|"menagerie", "ref": ...}
    compiled: bool = False


class ModelInspection(SimContract):
    """编译态模型检查：summary 仅供 Agent 理解，结构化字段为权威。"""

    SCHEMA: ClassVar[str] = "rosclaw.sim.model_inspection.v1"
    HASH_PREFIX: ClassVar[str] = "simins"
    schema_version: Literal["rosclaw.sim.model_inspection.v1"] = "rosclaw.sim.model_inspection.v1"

    model_ref: str = ""
    nq: int = 0
    nv: int = 0
    nu: int = 0
    joints: list[dict[str, Any]] = []
    actuators: list[dict[str, Any]] = []
    sensors: list[str] = []
    cameras: list[str] = []
    sites: list[str] = []
    summary: str = ""


class StateSnapshot(SimContract):
    """可继续仿真的完整状态；绑定 model_digest，跨模型 fail closed。"""

    SCHEMA: ClassVar[str] = "rosclaw.sim.state_snapshot.v1"
    HASH_PREFIX: ClassVar[str] = "simsta"
    schema_version: Literal["rosclaw.sim.state_snapshot.v1"] = "rosclaw.sim.state_snapshot.v1"

    model_ref: str = ""
    model_digest: str = ""
    time: float = 0.0
    qpos: list[float] = []
    qvel: list[float] = []
    act: list[float] = []
    ctrl: list[float] = []


class ObservationRequest(SimContract):
    """观测请求：channels 语义化命名（如 joint_positions / site_pose:x）。"""

    SCHEMA: ClassVar[str] = "rosclaw.sim.observation_request.v1"
    HASH_PREFIX: ClassVar[str] = "simobq"
    schema_version: Literal["rosclaw.sim.observation_request.v1"] = (
        "rosclaw.sim.observation_request.v1"
    )

    model_ref: str = ""
    state_ref: str = ""
    channels: list[str] = []
    at_time: float | None = None


class ObservationResult(SimContract):
    SCHEMA: ClassVar[str] = "rosclaw.sim.observation_result.v1"
    HASH_PREFIX: ClassVar[str] = "simobs"
    schema_version: Literal["rosclaw.sim.observation_result.v1"] = (
        "rosclaw.sim.observation_result.v1"
    )

    request_digest: str = ""
    model_ref: str = ""
    time: float = 0.0
    values: dict[str, Any] = {}


class RolloutRequest(SimContract):
    """有界 rollout 请求（预算由执行层强制：max_steps/max_duration/…）。"""

    SCHEMA: ClassVar[str] = "rosclaw.sim.rollout_request.v1"
    HASH_PREFIX: ClassVar[str] = "simrol"
    schema_version: Literal["rosclaw.sim.rollout_request.v1"] = "rosclaw.sim.rollout_request.v1"

    model_ref: str = ""
    initial_state_ref: str | None = None
    controller: dict[str, Any] = {}
    duration_s: float = 0.0
    seed: int = 0


class SimulationTrace(SimContract):
    SCHEMA: ClassVar[str] = "rosclaw.sim.simulation_trace.v1"
    HASH_PREFIX: ClassVar[str] = "simtrc"
    schema_version: Literal["rosclaw.sim.simulation_trace.v1"] = "rosclaw.sim.simulation_trace.v1"

    request_digest: str = ""
    model_ref: str = ""
    model_digest: str = ""
    steps: int = 0
    timestep_s: float = 0.0
    states_digest: str = ""


class ModelPatch(SimContract):
    """结构化模型补丁（MjSpec 路径；不鼓励 XML 字符串改写）。"""

    SCHEMA: ClassVar[str] = "rosclaw.sim.model_patch.v1"
    HASH_PREFIX: ClassVar[str] = "simpat"
    schema_version: Literal["rosclaw.sim.model_patch.v1"] = "rosclaw.sim.model_patch.v1"

    model_ref: str = ""
    patch_ops: list[dict[str, Any]] = []
    description: str = ""


class ModelPatchResult(SimContract):
    SCHEMA: ClassVar[str] = "rosclaw.sim.model_patch_result.v1"
    HASH_PREFIX: ClassVar[str] = "simpar"
    schema_version: Literal["rosclaw.sim.model_patch_result.v1"] = (
        "rosclaw.sim.model_patch_result.v1"
    )

    patch_digest: str = ""
    ok: bool = False
    new_model_ref: str | None = None
    error_code: str = ""


class AuditRequest(SimContract):
    SCHEMA: ClassVar[str] = "rosclaw.sim.audit_request.v1"
    HASH_PREFIX: ClassVar[str] = "simauq"
    schema_version: Literal["rosclaw.sim.audit_request.v1"] = "rosclaw.sim.audit_request.v1"

    subject_ref: str = ""
    checks: list[str] = []
    profile: str = "strict"


class AuditResult(SimContract):
    """机器可读审计结果：PASS/FAIL + 逐项检查 + 证据 ref 列表。"""

    SCHEMA: ClassVar[str] = "rosclaw.sim.audit_result.v1"
    HASH_PREFIX: ClassVar[str] = "simaur"
    schema_version: Literal["rosclaw.sim.audit_result.v1"] = "rosclaw.sim.audit_result.v1"

    request_digest: str = ""
    status: str = ""  # PASS | FAIL | WARN
    checks: dict[str, Any] = {}
    warnings: list[str] = []
    violations: list[str] = []
    evidence: list[str] = []


class ExperimentBranch(SimContract):
    """从同一状态/模型分叉的实验分支；分支初始 digest 必须一致。"""

    SCHEMA: ClassVar[str] = "rosclaw.sim.experiment_branch.v1"
    HASH_PREFIX: ClassVar[str] = "simbr"
    schema_version: Literal["rosclaw.sim.experiment_branch.v1"] = "rosclaw.sim.experiment_branch.v1"

    base_model_ref: str = ""
    base_state_ref: str = ""
    branch_id: str = ""
    patch_refs: list[str] = []


class ExperimentResult(SimContract):
    SCHEMA: ClassVar[str] = "rosclaw.sim.experiment_result.v1"
    HASH_PREFIX: ClassVar[str] = "simexp"
    schema_version: Literal["rosclaw.sim.experiment_result.v1"] = "rosclaw.sim.experiment_result.v1"

    branch_ref: str = ""
    trace_refs: list[str] = []
    metrics: dict[str, float] = {}
    success: bool = False


class ComparisonResult(SimContract):
    """实验对比：指标表 + best + Pareto 候选，不让 LLM 肉眼比 JSON。"""

    SCHEMA: ClassVar[str] = "rosclaw.sim.comparison_result.v1"
    HASH_PREFIX: ClassVar[str] = "simcmp"
    schema_version: Literal["rosclaw.sim.comparison_result.v1"] = "rosclaw.sim.comparison_result.v1"

    subject_refs: list[str] = []
    metric_table: list[dict[str, Any]] = []
    best_ref: str = ""
    pareto_refs: list[str] = []


class SimulationReceipt(SimContract):
    """仿真实验回执（规格 §31）：永远 SIMULATED，永不生成 REAL permit。

    双层 digest（规格 §56）：``states_digest`` = raw（逐状态），
    ``semantic_digest`` = 语义（指标容差比较层）。
    """

    SCHEMA: ClassVar[str] = "rosclaw.sim.receipt.v1"
    HASH_PREFIX: ClassVar[str] = "simrcp"
    schema_version: Literal["rosclaw.sim.receipt.v1"] = "rosclaw.sim.receipt.v1"

    model_ref: str = ""
    model_digest: str = ""
    initial_state_ref: str = ""
    action_digest: str = ""
    trace_ref: str = ""
    final_state_ref: str = ""
    seed: int = 0
    steps: int = 0
    simulation_time_s: float = 0.0
    success: bool | None = None
    metrics: dict[str, Any] = {}
    audit_ref: str = ""
    artifacts: list[str] = []
    states_digest: str = ""
    semantic_digest: str = ""
    receipt_ref: str = ""
    trust_level: str = "SIMULATED"
    usable_for_real_execution: bool = False


class SimulationEvidenceBundle(SimContract):
    """仿真证据包：永远 SIMULATED，永不生成 REAL permit。"""

    SCHEMA: ClassVar[str] = "rosclaw.sim.evidence_bundle.v1"
    HASH_PREFIX: ClassVar[str] = "simevb"
    schema_version: Literal["rosclaw.sim.evidence_bundle.v1"] = "rosclaw.sim.evidence_bundle.v1"

    model_ref: str = ""
    model_digest: str = ""
    trace_refs: list[str] = []
    audit_refs: list[str] = []
    capabilities_digest: str = ""
    trust_level: str = "SIMULATED"
    usable_for_real_execution: bool = False
