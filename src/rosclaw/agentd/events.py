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
                if event.visibility is Visibility.DEBUG:
                    continue
                # PR-HP1：饱和必须显式 OVERLOADED——USER/AUDIT 不得静默
                # 丢：驱逐最旧 + 放入当前事件 + 保证队列里有
                # session.degraded 标记（消费者据此知道发生了丢弃并可从
                # journal 按 cursor 补放）。不阻塞执行。
                import contextlib as _cl

                with _cl.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()  # evict oldest
                marker = AgentEventV2(
                    event_id=new_id("evt"),
                    sequence=0,
                    mission_id=event.mission_id,
                    timestamp=_utcnow(),
                    type=AgentEventType.SESSION_DEGRADED,
                    visibility=Visibility.USER,
                    payload={
                        "reason": "OVERLOADED",
                        "note": "live 队列饱和——部分事件已从流中驱逐；"
                        "journal 完整，按 cursor 重放可补",
                    },
                )
                # 标记去重：队尾已是 degraded 则不重复插。
                with _cl.suppress(asyncio.QueueFull):
                    queue.put_nowait(marker)
                with _cl.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)


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
        session_id: str | None = None,
        revision: int | None = None,
        item_id: str | None = None,
        call_id: str | None = None,
        operation_id: str | None = None,
        model_visible: bool | None = None,
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
                session_id=session_id,
                revision=revision,
                item_id=item_id,
                call_id=call_id,
                operation_id=operation_id,
                model_visible=model_visible,
            )
            # PR-HP1：一等链路段落库（029 迁移列）。
            self._conn.execute(
                "INSERT INTO agent_events (event_id, mission_id, sequence, turn_id, "
                "task_id, trace_id, type, visibility, payload_json, timestamp, "
                "session_id, revision, item_id, call_id, operation_id, "
                "model_visible) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    session_id,
                    revision,
                    item_id,
                    call_id,
                    operation_id,
                    None if model_visible is None else (1 if model_visible else 0),
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
        keys = set(row.keys())  # sqlite3.Row 不支持 __contains__——取一次键集
        model_visible_raw = row["model_visible"] if "model_visible" in keys else None
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
            session_id=row["session_id"] if "session_id" in keys else None,
            revision=row["revision"] if "revision" in keys else None,
            item_id=row["item_id"] if "item_id" in keys else None,
            call_id=row["call_id"] if "call_id" in keys else None,
            operation_id=row["operation_id"] if "operation_id" in keys else None,
            model_visible=(
                None if model_visible_raw is None else bool(model_visible_raw)
            ),
        )
