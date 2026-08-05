"""EmbodiedContextEnvelopeV1（重构规格 §14，PR-PNA-2）。

每个认知 turn 由 rosclaw-agent 的 extension 经 pi-bridge 拉取并注入：
具身事实不靠 session 历史记忆——每轮重新注入最新 Body/Self/Mission/
pending approval/active action/safety。envelope 带 TTL 与内容 hash；
过期或 hash 不匹配 → context stale → 禁止物理动作。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class EmbodiedContextEnvelopeV1(ContractModel):
    SCHEMA = "rosclaw.embodied_context.v1"

    schema_version: Literal["rosclaw.embodied_context.v1"] = "rosclaw.embodied_context.v1"
    mission_id: str
    mission_revision: int = 0
    context_revision: int = 0
    generated_at: str
    expires_at: str
    body: dict[str, Any] = Field(default_factory=dict)
    self_state: dict[str, Any] = Field(default_factory=dict)
    task_graph: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    pending_approvals: list[dict[str, Any]] = Field(default_factory=list)
    active_actions: list[dict[str, Any]] = Field(default_factory=list)
    receipts: list[dict[str, Any]] = Field(default_factory=list)
    workers: list[dict[str, Any]] = Field(default_factory=list)
    memory_summary: dict[str, Any] = Field(default_factory=dict)
    safety: dict[str, Any] = Field(default_factory=dict)
    tool_policy: dict[str, Any] = Field(default_factory=dict)
    freshness: dict[str, Any] = Field(default_factory=dict)
    hash: str = ""
