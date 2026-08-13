"""Agent event journal + live bus (PR-02, 大纲 §9.3).

Order is mandatory:

```text
domain operation
→ transaction writes event (per-mission sequence MAX+1)
→ projection updated
→ event bus publishes the *committed* event
→ TUI receives
```

Reconnection replays the journal via ``after_sequence`` and then attaches
to the live bus.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from rosclaw.contracts.agent.agent_event import (
    AgentEventType,
    AgentEventV2,
    Visibility,
)
from rosclaw.contracts.common import new_id


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class MissionEventBus:
    """In-process fan-out of committed events to live SSE subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, mission_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._subscribers[mission_id].append(queue)
        return queue

    def unsubscribe(self, mission_id: str, queue: asyncio.Queue) -> None:
        if queue in self._subscribers.get(mission_id, []):
            self._subscribers[mission_id].remove(queue)

    def publish(self, event: AgentEventV2) -> None:
        for queue in self._subscribers.get(event.mission_id, []):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer: drop DEBUG first, never block the domain op.
                if event.visibility is not Visibility.DEBUG:
                    try:
                        queue.get_nowait()
                        queue.put_nowait(event)
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        pass


class AgentEventStore:
    """Journal write + replay on the MissionStore's connection."""

    def __init__(self, conn: sqlite3.Connection, bus: MissionEventBus | None = None) -> None:
        self._conn = conn
        self._lock = asyncio.Lock()
        self.bus = bus or MissionEventBus()

    async def append(
        self,
        mission_id: str,
        type: AgentEventType,
        payload: dict | None = None,
        *,
        visibility: Visibility = Visibility.USER,
        turn_id: str | None = None,
        task_id: str | None = None,
        trace_id: str | None = None,
    ) -> AgentEventV2:
        async with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_seq FROM agent_events "
                "WHERE mission_id = ?",
                (mission_id,),
            ).fetchone()
            sequence = int(row["next_seq"])
            event = AgentEventV2(
                event_id=new_id("evt"),
                sequence=sequence,
                mission_id=mission_id,
                turn_id=turn_id,
                task_id=task_id,
                trace_id=trace_id,
                timestamp=_utcnow(),
                type=type,
                visibility=visibility,
                payload=payload or {},
            )
            self._conn.execute(
                "INSERT INTO agent_events (event_id, mission_id, sequence, turn_id, "
                "task_id, trace_id, type, visibility, payload_json, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    mission_id,
                    sequence,
                    turn_id,
                    task_id,
                    trace_id,
                    type.value,
                    visibility.value,
                    json.dumps(payload or {}, sort_keys=True, ensure_ascii=False),
                    event.timestamp,
                ),
            )
        # Publish only after the journal row is committed.
        self.bus.publish(event)
        return event

    def replay(
        self,
        mission_id: str,
        *,
        after_sequence: int = 0,
        before_sequence: int | None = None,
        visibility: Visibility | None = None,
        limit: int = 1000,
    ) -> list[AgentEventV2]:
        query = (
            "SELECT * FROM agent_events WHERE mission_id = ? AND sequence > ?"
            + (" AND sequence < ?" if before_sequence is not None else "")
            + (" AND visibility = ?" if visibility else "")
            + " ORDER BY sequence LIMIT ?"
        )
        params: list = [mission_id, after_sequence]
        if before_sequence is not None:
            params.append(before_sequence)
        if visibility:
            params.append(visibility.value)
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def latest_sequence(self, mission_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(sequence), 0) AS s FROM agent_events WHERE mission_id = ?",
            (mission_id,),
        ).fetchone()
        return int(row["s"])

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> AgentEventV2:
        return AgentEventV2(
            event_id=row["event_id"],
            sequence=row["sequence"],
            mission_id=row["mission_id"],
            turn_id=row["turn_id"],
            task_id=row["task_id"],
            trace_id=row["trace_id"],
            timestamp=row["timestamp"],
            type=AgentEventType(row["type"]),
            visibility=Visibility(row["visibility"]),
            payload=json.loads(row["payload_json"]),
        )


async def stream_events(
    store: AgentEventStore,
    mission_id: str,
    *,
    after_sequence: int = 0,
) -> AsyncIterator[AgentEventV2]:
    """一个 authoritative event stream：journal replay → live bus（Channel 设计 §21）。

    供 ACP / TUI SSE / 未来 AG-UI 共用，替代各消费方自己的轮询循环。

    保证：
    - 无丢失：先 attach live bus 再 replay journal，起始空窗由 queue 覆盖；
      live queue 溢出（bus 满时丢最旧事件）产生的 sequence 空洞从 journal
      补齐——sequence 是 per-mission 严格单调无空洞的，空洞只可能来自丢事件。
    - 无重复：live 事件 sequence <= 已发送水位时跳过。
    - sequence 单调递增。
    - 重连：调用方记录最后收到的 sequence，以 ``after_sequence`` 重新订阅。
    """
    queue = store.bus.subscribe(mission_id)  # 先挂 live，关闭 replay 竞态窗口
    sent = after_sequence
    try:
        # 1) journal replay（分页直到追平当前 journal 末尾）
        while True:
            batch = store.replay(mission_id, after_sequence=sent, limit=500)
            if not batch:
                break
            for event in batch:
                if event.sequence > sent:
                    sent = event.sequence
                    yield event
            if len(batch) < 500:
                break
        # 2) live bus
        while True:
            event = await queue.get()
            if event.sequence <= sent:
                continue  # 与 replay 重叠的部分去重
            if event.sequence > sent + 1:
                # live queue 溢出丢过事件——从 journal 补齐空洞。
                for missed in store.replay(
                    mission_id, after_sequence=sent, before_sequence=event.sequence
                ):
                    if missed.sequence > sent:
                        sent = missed.sequence
                        yield missed
            sent = event.sequence
            yield event
    finally:
        store.bus.unsubscribe(mission_id, queue)
