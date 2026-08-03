"""CompactionEntryV1 (大纲 §8.2).

压缩只是 Context View 的变化：canonical journal 永不删除；summary 永远
是不可信对话上下文，不是物理事实。`task_graph_revision/context_revision`
只是压缩发生时的记录——下次运行仍从 MissionStore 读取最新版本。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class CompactionSummary(ContractModel):
    SCHEMA = "rosclaw.compaction_summary.v1"

    schema_version: Literal["rosclaw.compaction_summary.v1"] = "rosclaw.compaction_summary.v1"
    goal: str = ""
    user_constraints: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    progress: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class CompactionEntryV1(ContractModel):
    SCHEMA = "rosclaw.compaction_entry.v1"
    HASH_PREFIX = "cmp"

    schema_version: Literal["rosclaw.compaction_entry.v1"] = "rosclaw.compaction_entry.v1"
    compaction_id: str
    mission_id: str
    created_at: str
    reason: Literal["manual", "threshold", "overflow"] = "manual"
    summary: CompactionSummary
    first_kept_event_id: str = Field(
        ..., description="第一个被保留的 conversation 事件/消息标识（切点之后全部保留）"
    )
    tokens_before: int = 0
    tokens_after: int = 0
    evidence_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    task_graph_revision: int = 0
    context_revision: int = 0
    summary_model: str = "deterministic-fallback"
    usage: dict[str, Any] = Field(default_factory=dict)
    # 批次 A（补充实施文档 §3.3/§8.10）：可审计的覆盖范围与来源。
    #: 被压缩区间的 entry_id 列表（journal append 时稳定赋号）。
    covered_entry_ids: list[str] = Field(default_factory=list)
    #: 被压缩区间 canonical JSON 的内容 hash（审计对账用）。
    covered_span_hash: str = ""
    #: 被本条取代的上一条 compaction id。
    supersedes: str | None = None
    #: 生成摘要时的 prompt/provider/model（deterministic fallback 留空）。
    prompt_version: str = ""
    provider: str = ""
    model: str = ""
    #: 被压缩区间内受保护、不可拆的原子组标识（观测/授权/回执对等）。
    protected_groups: list[str] = Field(default_factory=list)
