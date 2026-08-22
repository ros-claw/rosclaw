"""AgentEventV2 — the only channel of UI state (大纲 §9.2).

All UI state comes from journaled events with a per-mission monotonic
sequence. TUI reconnects via ``after_sequence``. Visibility tiers:
USER (always render), DEBUG (collapsed by default), AUDIT (compliance).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class AgentEventType(StrEnum):
    # Agent 生命周期（批次 B：settled 是 TUI 停 spinner 的唯一可靠信号）。
    AGENT_STARTED = "agent.started"
    AGENT_SETTLED = "agent.settled"
    AGENT_FAILED = "agent.failed"
    # Turn
    TURN_ACCEPTED = "turn.accepted"
    TURN_ENDED = "turn.ended"
    TURN_CANCEL_REQUESTED = "turn.cancel.requested"
    # 文本
    MESSAGE_STARTED = "message.started"
    MESSAGE_ENDED = "message.ended"
    MISSION_STATE_CHANGED = "mission.state.changed"
    MISSION_RENAMED = "mission.renamed"
    MISSION_ARCHIVED = "mission.archived"
    CONTEXT_COMPILATION_STARTED = "context.compilation.started"
    CONTEXT_COMPILATION_COMPLETED = "context.compilation.completed"
    CONTEXT_USAGE = "context.usage"
    MODEL_SELECTED = "model.selected"
    MODEL_REQUEST_STARTED = "model.request.started"
    MODEL_TEXT_DELTA = "model.text.delta"
    MODEL_RETRY_SCHEDULED = "model.retry.scheduled"
    MODEL_FAILOVER = "model.failover"
    MODEL_REQUEST_ENDED = "model.request.ended"
    MODEL_TOOL_CALL_PROPOSED = "model.tool_call.proposed"
    TOOL_PROPOSED = "tool.proposed"
    # PR-N5C：effect 执行前冻结（单一 Effect Contract——审批/并发/
    # Verifier 读冻结结果）。
    TOOL_EFFECT_RESOLVED = "tool.effect_resolved"
    TOOL_STARTED = "tool.started"
    TOOL_PROGRESS = "tool.progress"
    TOOL_COMPLETED = "tool.completed"
    TASK_GRAPH_COMMITTED = "task_graph.committed"
    WORKER_OFFERED = "worker.offered"
    WORKER_CLAIMED = "worker.claimed"
    WORKER_STARTED = "worker.started"
    WORKER_SUBMITTED = "worker.submitted"
    WORKER_VERIFYING = "worker.verifying"
    WORKER_ACCEPTED = "worker.accepted"
    WORKER_FAILED = "worker.failed"
    WORKER_EXPIRED = "worker.expired"
    WORKER_COMPLETED = "worker.completed"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"
    GRANT_REVOKED = "grant.revoked"
    GRANT_CONSUMED = "grant.consumed"
    ACTION_PROPOSED = "action.proposed"
    ACTION_DISPATCHED = "action.dispatched"
    ACTION_PROGRESS = "action.progress"
    ACTION_RECEIPT = "action.receipt"
    RECEIPT_RECEIVED = "receipt.received"
    VERIFICATION_STARTED = "verification.started"
    VERIFICATION_COMPLETED = "verification.completed"
    COMPACTION_STARTED = "compaction.started"
    COMPACTION_COMPLETED = "compaction.completed"
    MISSION_COMPLETED = "mission.completed"
    MISSION_FAILED = "mission.failed"
    # PR-HP1：NativeEventV2 硬不变量（调整方案 §四）——输入门/操作/
    # 会话活性。
    INPUT_PERSISTED = "input.persisted"
    INPUT_DISPATCHED = "input.dispatched"
    OPERATION_STARTED = "operation.started"
    OPERATION_OUTPUT = "operation.output"
    OPERATION_COMPLETED = "operation.completed"
    OPERATION_FAILED = "operation.failed"
    SESSION_IDLE = "session.idle"
    SESSION_DEGRADED = "session.degraded"
    TURN_FAILED = "turn.failed"
    # 系统
    WARNING = "warning"
    CAPABILITIES_CHANGED = "capabilities.changed"
    CONFIG_RELOADED = "config.reloaded"
    ERROR = "error"


class Visibility(StrEnum):
    USER = "USER"
    DEBUG = "DEBUG"
    AUDIT = "AUDIT"


class AgentEventV2(ContractModel):
    SCHEMA = "rosclaw.agent_event.v2"
    HASH_PREFIX = "aevt"

    schema_version: Literal["rosclaw.agent_event.v2"] = "rosclaw.agent_event.v2"
    event_id: str
    sequence: int = 0  # assigned by the store, per-mission monotonic
    mission_id: str
    turn_id: str | None = None
    task_id: str | None = None
    trace_id: str | None = None
    timestamp: str
    type: AgentEventType
    visibility: Visibility = Visibility.USER
    payload: dict[str, Any] = Field(default_factory=dict)
    # PR-HP1：一等链路段（落库列，不埋 payload——resume/trace/投影
    # 直接读列）。
    session_id: str | None = None
    #: 任务 revision（输入绑定的 task revision）。
    revision: int | None = None
    #: Harness 原生 item id（Pi item 等——只在 binding 语义内使用）。
    item_id: str | None = None
    #: 工具调用 id（tool.* 事件的调用关联）。
    call_id: str | None = None
    #: 长操作 id（operation.* 事件关联）。
    operation_id: str | None = None
    #: 内容是否进入了模型上下文（可重放日志的模型可见性标注）。
    model_visible: bool | None = None
