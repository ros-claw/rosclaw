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
    TURN_ACCEPTED = "turn.accepted"
    MISSION_STATE_CHANGED = "mission.state.changed"
    CONTEXT_COMPILATION_STARTED = "context.compilation.started"
    CONTEXT_COMPILATION_COMPLETED = "context.compilation.completed"
    MODEL_REQUEST_STARTED = "model.request.started"
    MODEL_TEXT_DELTA = "model.text.delta"
    MODEL_TOOL_CALL_PROPOSED = "model.tool_call.proposed"
    TOOL_STARTED = "tool.started"
    TOOL_PROGRESS = "tool.progress"
    TOOL_COMPLETED = "tool.completed"
    TASK_GRAPH_COMMITTED = "task_graph.committed"
    WORKER_OFFERED = "worker.offered"
    WORKER_STARTED = "worker.started"
    WORKER_COMPLETED = "worker.completed"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"
    ACTION_PROPOSED = "action.proposed"
    ACTION_DISPATCHED = "action.dispatched"
    ACTION_PROGRESS = "action.progress"
    RECEIPT_RECEIVED = "receipt.received"
    VERIFICATION_STARTED = "verification.started"
    VERIFICATION_COMPLETED = "verification.completed"
    COMPACTION_STARTED = "compaction.started"
    COMPACTION_COMPLETED = "compaction.completed"
    MISSION_COMPLETED = "mission.completed"
    MISSION_FAILED = "mission.failed"
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
