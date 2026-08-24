"""P0-A 红测试（0824 总纲 §19.P0-A）：Session replay 与 UI 去重。

红测试先行——幂等 append / unique index / dedup 不存在时必须红。

验收（文档原文）：
- 注入 429、断流、重连和重复 event：整段 transcript 不得重复；
- resume 前后 transcript digest 一致；
- provider retry 记录 attempt，不重复追加既有事件。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path


def _store(tmp_path: Path):
    from rosclaw.agentd.events import AgentEventStore
    from rosclaw.storage.migrations import MigrationRunner

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, "sqlite")
    return AgentEventStore(conn), conn


def _digest(events) -> str:
    canonical = json.dumps(
        [
            {
                "seq": e.sequence, "type": e.type.value,
                "payload": e.payload, "session_id": e.session_id,
                "call_id": e.call_id, "item_id": e.item_id,
            }
            for e in events
        ],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class TestIdempotentAppend:
    def test_same_event_id_appended_once(self, tmp_path: Path) -> None:
        """重复 event（provider retry/重连重放）落账一次——
        (session_id, event_id) unique。"""
        from rosclaw.contracts.agent.agent_event import AgentEventType

        store, conn = _store(tmp_path)
        first = asyncio.run(store.append(
            "mis_1", AgentEventType.TOOL_EFFECT_RESOLVED,
            {"effect": "sim"}, session_id="s1", call_id="c1",
            event_id="s1:turn1:c1:effect",
        ))
        second = asyncio.run(store.append(
            "mis_1", AgentEventType.TOOL_EFFECT_RESOLVED,
            {"effect": "sim"}, session_id="s1", call_id="c1",
            event_id="s1:turn1:c1:effect",
        ))
        assert first.sequence == second.sequence, "重复事件获得了新序号"
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM agent_events WHERE session_id='s1'"
        ).fetchone()
        assert int(rows["n"]) == 1, "重复 event 重复落账"

    def test_duplicate_not_republished(self, tmp_path: Path) -> None:
        """重复 event 不再 publish——live 消费者不得见第二张卡。"""
        from rosclaw.contracts.agent.agent_event import AgentEventType

        store, _conn = _store(tmp_path)
        queue = store.bus.subscribe("mis_1")
        asyncio.run(store.append(
            "mis_1", AgentEventType.OPERATION_COMPLETED,
            {"operation_id": "op_1"}, session_id="s1",
            event_id="s1:op_1:completed",
        ))
        asyncio.run(store.append(
            "mis_1", AgentEventType.OPERATION_COMPLETED,
            {"operation_id": "op_1"}, session_id="s1",
            event_id="s1:op_1:completed",
        ))
        assert queue.qsize() == 1, f"重复 event 被重复 publish（{queue.qsize()} 张卡）"

    def test_replay_digest_stable_across_duplicate_injection(
        self, tmp_path: Path
    ) -> None:
        """resume 前后 transcript digest 一致——注入重复不改变账本。"""
        from rosclaw.contracts.agent.agent_event import AgentEventType

        store, _conn = _store(tmp_path)
        asyncio.run(store.append(
            "mis_1", AgentEventType.OPERATION_STARTED,
            {"operation_id": "op_1"}, session_id="s1",
            event_id="s1:op_1:started",
        ))
        before = _digest(store.replay("mis_1"))
        # 断流重连 → 同一批事件重放注入。
        for _ in range(3):
            asyncio.run(store.append(
                "mis_1", AgentEventType.OPERATION_STARTED,
                {"operation_id": "op_1"}, session_id="s1",
                event_id="s1:op_1:started",
            ))
        after = _digest(store.replay("mis_1"))
        assert before == after, "注入重复后 transcript digest 改变"

    def test_unique_index_exists(self, tmp_path: Path) -> None:
        """migration 提供 (session_id, event_id) unique 兜底。"""
        _store(tmp_path)
        conn = sqlite3.connect(":memory:")
        from rosclaw.storage.migrations import MigrationRunner

        MigrationRunner().apply(conn, "sqlite")
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND sql LIKE '%session_id%event_id%'"
        ).fetchall()
        assert rows, "缺 (session_id, event_id) unique index migration"

    def test_caller_event_id_roundtrip(self, tmp_path: Path) -> None:
        """幂等返回的是同一事件（调用方可安全重试）。"""
        from rosclaw.contracts.agent.agent_event import AgentEventType

        store, _conn = _store(tmp_path)
        first = asyncio.run(store.append(
            "mis_1", AgentEventType.TOOL_EFFECT_RESOLVED,
            {"a": 1}, session_id="s1", event_id="s1:x",
        ))
        second = asyncio.run(store.append(
            "mis_1", AgentEventType.TOOL_EFFECT_RESOLVED,
            {"a": 1}, session_id="s1", event_id="s1:x",
        ))
        assert first.event_id == second.event_id == "s1:x"
