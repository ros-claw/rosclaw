"""Persistent compaction engine (PR-07, 大纲 §8；借鉴 Pi compaction 算法，不 import）。

铁律（§8.1）：
- canonical journal 永不删除；summary 永远是 untrusted conversation
  context，不是物理事实。
- 物理事实（宪法/Body/Snapshot/TaskGraph/Grant/Permit/Action/Receipt/
  lease/epoch/预算/已验证证据）每次从权威存储重新编译，绝不进 summary。
- 切分永不拆对：tool call/result、WorkOrder/Result、approval 请求/决定、
  action proposal/Permit/Receipt、问答对——优先在完整 user turn 边界切。
- `_persisted_count` 在 view 缩短后重置为新 view 长度（§8 风险点）。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from rosclaw.agentd.context.tokens import estimate_tokens
from rosclaw.contracts.agent.compaction import CompactionEntryV1, CompactionSummary

COMPACTION_MARKER_ROLE = "compaction"

#: 配对边界：assistant(tool_calls) 与其 tool result 必须同侧。
_PAIR_ROLES = ("tool",)


def compute_history_budget(
    *,
    context_window: int,
    protected_tokens: int,
    tool_schema_tokens: int,
    max_output_tokens: int = 16_384,
    safety_margin: int = 4_096,
) -> int:
    """§8.5：history_budget = window - protected - tools - output - margin。"""
    budget = (
        context_window - protected_tokens - tool_schema_tokens - max_output_tokens - safety_margin
    )
    return max(budget, 2_000)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            total += estimate_tokens(json.dumps(content, ensure_ascii=False))
        if message.get("tool_calls"):
            total += estimate_tokens(json.dumps(message["tool_calls"], ensure_ascii=False))
    return total


def _turn_boundary_ok(messages: list[dict[str, Any]], cut: int) -> bool:
    """切点不得切开 tool_call/tool_result 配对（§8.4）。"""
    if cut <= 0 or cut >= len(messages):
        return True
    left = messages[cut - 1]
    # 左侧是带 tool_calls 的 assistant，而右侧还有它的 tool result → 切断。
    if left.get("tool_calls"):
        return False
    # 右侧是 tool result，而其 assistant 在左侧 → 切断。
    return messages[cut].get("role") not in _PAIR_ROLES


def find_cut_point(messages: list[dict[str, Any]], *, keep_recent_tokens: int = 20_000) -> int:
    """倒序累计 keepRecentTokens 找切点（Pi 算法），对齐到合法边界。"""
    accumulated = 0
    cut = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        accumulated += _message_tokens(messages[index])
        if accumulated > keep_recent_tokens:
            cut = min(len(messages), index + 1)
            break
    else:
        return 0  # 全部塞得下
    # 向前对齐到合法边界（不拆对）。
    while cut < len(messages) and not _turn_boundary_ok(messages, cut):
        cut += 1
    # 尽量落在 user turn 开头。
    while 0 < cut < len(messages) and messages[cut].get("role") != "user":
        if not _turn_boundary_ok(messages, cut + 1):
            break
        cut += 1
    return cut


def _message_tokens(message: dict[str, Any]) -> int:
    total = 0
    content = message.get("content")
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif isinstance(content, list):
        total += estimate_tokens(json.dumps(content, ensure_ascii=False))
    if message.get("tool_calls"):
        total += estimate_tokens(json.dumps(message["tool_calls"], ensure_ascii=False))
    return total


def deterministic_summary(
    span: list[dict[str, Any]], *, goal: str, focus: str | None = None
) -> CompactionSummary:
    """确定性 fallback 摘要（summary 模型失败时也用它）。"""
    user_texts = [
        str(m.get("content") or "")
        for m in span
        if m.get("role") == "user" and not str(m.get("content") or "").startswith("[")
    ]
    decisions: list[str] = []
    for m in span:
        content = str(m.get("content") or "")
        if "rosclaw.decision.v1" in content:
            decisions.append(content[:200])
    constraints = [t[:200] for t in user_texts[-5:] if t]
    if focus:
        constraints.insert(0, f"[focus] {focus}")
    return CompactionSummary(
        goal=goal,
        user_constraints=constraints,
        decisions=decisions[-10:],
        progress=[f"压缩了 {len(span)} 条历史消息"],
        open_questions=[],
        blocked=[],
        next_steps=["继续当前 Mission"],
    )


class CompactionStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, entry: CompactionEntryV1) -> None:
        self._conn.execute(
            "INSERT INTO compaction_entries (compaction_id, mission_id, created_at, "
            "reason, entry_json, tokens_before, tokens_after) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entry.compaction_id,
                entry.mission_id,
                entry.created_at,
                entry.reason,
                entry.model_dump_json(),
                entry.tokens_before,
                entry.tokens_after,
            ),
        )

    def latest(self, mission_id: str) -> CompactionEntryV1 | None:
        row = self._conn.execute(
            "SELECT entry_json FROM compaction_entries WHERE mission_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (mission_id,),
        ).fetchone()
        return CompactionEntryV1(**json.loads(row["entry_json"])) if row else None

    def list(self, mission_id: str) -> list[CompactionEntryV1]:
        rows = self._conn.execute(
            "SELECT entry_json FROM compaction_entries WHERE mission_id = ? ORDER BY created_at",
            (mission_id,),
        ).fetchall()
        return [CompactionEntryV1(**json.loads(r["entry_json"])) for r in rows]

    def count(self, mission_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM compaction_entries WHERE mission_id = ?",
            (mission_id,),
        ).fetchone()
        return int(row["n"])


def build_compaction_view(
    entry: CompactionEntryV1, kept: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """view = summary 标记消息（untrusted）+ 保留消息。"""
    summary_text = (
        "[compaction summary — UNTRUSTED conversation context, NOT physical facts]\n"
        f"goal: {entry.summary.goal}\n"
        f"user_constraints: {entry.summary.user_constraints}\n"
        f"decisions: {len(entry.summary.decisions)} recorded\n"
        f"progress: {entry.summary.progress}\n"
        f"next_steps: {entry.summary.next_steps}\n"
        f"(compact_id={entry.compaction_id}, tokens {entry.tokens_before}→{entry.tokens_after})"
    )
    return [
        {
            "role": COMPACTION_MARKER_ROLE,
            "content": summary_text,
            "compaction_id": entry.compaction_id,
        },
        *kept,
    ]


def restore_view_from_journal(journal: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """重启恢复：从最新 compaction 标记（含）开始（§8.3 canonical 仍在）。"""
    last_marker = -1
    for index, message in enumerate(journal):
        if message.get("role") == COMPACTION_MARKER_ROLE:
            last_marker = index
    if last_marker < 0:
        return list(journal)
    return journal[last_marker:]
