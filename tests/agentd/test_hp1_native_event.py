"""PR-HP1 红测试（调整方案 §四.HP1）：NativeEventV2 与事件日志。

红测试先行——合约字段/事件类型/input 事件/背压标记不存在时必须红。

硬不变量（方案原文）：
1. 用户输入必须先持久化、绑定 task/revision，再交 Harness——
   input.persisted 没成功，不得开始模型请求；
2. 每个事件单调 sequence；resume 从 cursor 重放；
3. TUI 消费落后不能阻塞执行；队列饱和必须显式 OVERLOADED，
   不能丢输入（也不静默丢 USER/AUDIT 事件——留 session.degraded
   标记）；
4. 事件字段：sessionId/taskId/revision/turnId/callId/operationId/
   modelVisible 一等列（不是埋在 payload 里）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class TestContractFields:
    def test_new_event_types_exist(self) -> None:
        from rosclaw.contracts.agent.agent_event import AgentEventType

        for value in (
            "input.persisted",
            "input.dispatched",
            "operation.started",
            "operation.output",
            "operation.completed",
            "operation.failed",
            "session.idle",
            "session.degraded",
            "turn.failed",
        ):
            assert AgentEventType(value).value == value, f"缺事件类型 {value}"

    def test_event_first_class_linkage_fields(self) -> None:
        """session_id/revision/item_id/call_id/operation_id/
        model_visible 是合约一等字段。"""
        from rosclaw.contracts.agent.agent_event import AgentEventV2

        event = AgentEventV2.model_validate_contract({
            "schema_version": "rosclaw.agent_event.v2",
            "event_id": "evt_1",
            "sequence": 1,
            "mission_id": "mis_1",
            "session_id": "sess_1",
            "task_id": "task_1",
            "revision": 2,
            "turn_id": "turn_1",
            "item_id": "item_1",
            "call_id": "call_1",
            "operation_id": "op_1",
            "model_visible": True,
            "timestamp": "2026-08-21T00:00:00+00:00",
            "type": "input.persisted",
            "payload": {},
        })
        assert event.session_id == "sess_1"
        assert event.revision == 2
        assert event.call_id == "call_1"
        assert event.operation_id == "op_1"
        assert event.model_visible is True

    def test_golden_schema_updated(self) -> None:
        from rosclaw.contracts.agent.agent_event import AgentEventV2

        golden = (
            REPO / "tests" / "contracts" / "golden" / "rosclaw.agent_event.v2.json"
        )
        current = AgentEventV2.model_json_schema()
        current["$id"] = "rosclaw://schemas/rosclaw.agent_event.v2"
        current["title"] = "rosclaw.agent_event.v2"
        assert json.loads(golden.read_text(encoding="utf-8")) == current, (
            "agent_event.v2 schema 漂移——本 PR 加性扩展需重新导出 golden"
        )


class TestJournalColumns:
    def test_linkage_fields_roundtrip_through_journal(self, tmp_path: Path) -> None:
        """一等列真实落库（不是只在 payload）并能 replay 回来。"""
        import sqlite3

        from rosclaw.agentd.events import AgentEventStore
        from rosclaw.contracts.agent.agent_event import AgentEventType
        from rosclaw.storage.migrations import MigrationRunner

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        # 需要一个 mission 行（外键）
        conn.execute(
            "INSERT INTO missions (mission_id, owner_principal, goal_json, "
            "body_id, effective_body_hash, mode, state, budgets_json, "
            "authorization_json, created_at, updated_at) "
            "VALUES ('mis_1', 'u', '{}', 'sim/ur5e', 'h', 'SIMULATION', "
            "'ACTIVE', '{}', '{}', '2026-08-21T00:00:00Z', "
            "'2026-08-21T00:00:00Z')"
        )
        store = AgentEventStore(conn)

        async def run() -> None:
            await store.append(
                "mis_1", AgentEventType.TOOL_STARTED,
                {"k": "v"},
                session_id="sess_1", task_id="task_1", revision=3,
                call_id="call_9", operation_id="op_2", model_visible=True,
            )

        asyncio.run(run())
        events = store.replay("mis_1")
        assert len(events) == 1
        ev = events[0]
        assert ev.session_id == "sess_1"
        assert ev.revision == 3
        assert ev.call_id == "call_9"
        assert ev.operation_id == "op_2"
        assert ev.model_visible is True


class TestInputPersistedGate:
    async def test_bind_emits_input_persisted_with_linkage(
        self, tmp_path: Path
    ) -> None:
        """pi.task.bind 成功 → input.persisted 落 journal（session/task/
        revision/文本 hash 一等携带）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from rosclaw.contracts.agent.agent_event import AgentEventType
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.task.bind",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
                "backend_native_id": "pi_1",
                "message_id": "msg_hp1",
                "text": "画五角星",
                "cwd": str(tmp_path),
                "body_id": mission.body_binding.body_id,
            },
        )
        assert result.get("ok"), result
        events = service.events_replay(mission.mission_id, limit=100)
        persisted = [
            e for e in events if e.type is AgentEventType.INPUT_PERSISTED
        ]
        assert persisted, "bind 成功但 journal 无 input.persisted"
        ev = persisted[0]
        assert ev.session_id == "pi_1"
        assert ev.task_id
        assert ev.revision >= 1
        assert ev.model_visible is True
        # 文本不裸写全文的限制不适用用户输入本身，但必须可审计：
        # payload 带 hash + 字节数。
        assert ev.payload.get("text_sha256")
        assert ev.payload.get("bytes", 0) > 0
        await service.close()

    async def test_no_dispatched_without_persisted(self, tmp_path: Path) -> None:
        """input.dispatched 前必须有对应 message 的 input.persisted
        （顺序不变量）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        # 直接 dispatched（无 bind）→ 拒绝。
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.input.dispatched",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "message_id": "msg_never_persisted",
            },
        )
        assert not result.get("ok"), result
        assert result.get("code") in ("INPUT_NOT_PERSISTED", "INVALID_ARGUMENT")
        await service.close()

    async def test_dispatched_after_bind_ok(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from rosclaw.contracts.agent.agent_event import AgentEventType
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        bind = await bridge._dispatch(
            "user:local:1000", 1, "pi.task.bind",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
                "backend_native_id": "pi_1",
                "message_id": "msg_hp1b",
                "text": "继续",
                "cwd": str(tmp_path),
                "body_id": mission.body_binding.body_id,
            },
        )
        assert bind.get("ok"), bind
        dispatched = await bridge._dispatch(
            "user:local:1000", 1, "pi.input.dispatched",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "message_id": "msg_hp1b",
            },
        )
        assert dispatched.get("ok"), dispatched
        events = service.events_replay(mission.mission_id, limit=100)
        types = [e.type for e in events]
        p_idx = types.index(AgentEventType.INPUT_PERSISTED)
        d_idx = types.index(AgentEventType.INPUT_DISPATCHED)
        assert p_idx < d_idx, "persisted 必须先于 dispatched"
        await service.close()


class TestBackpressure:
    def test_saturated_queue_gets_explicit_degraded_marker(self) -> None:
        """队列饱和：DEBUG 可丢；USER/AUDIT 不得静默丢——消费者必须
        收到显式 session.degraded 标记（OVERLOADED 语义）。"""
        from rosclaw.agentd.events import MissionEventBus
        from rosclaw.contracts.agent.agent_event import (
            AgentEventType,
            AgentEventV2,
            Visibility,
        )

        bus = MissionEventBus()
        queue = bus.subscribe("mis_1")
        base = {
            "schema_version": "rosclaw.agent_event.v2",
            "mission_id": "mis_1",
            "timestamp": "2026-08-21T00:00:00+00:00",
            "payload": {},
        }
        # 塞满（maxsize=1024），消费者不读。
        for i in range(1024):
            bus.publish(AgentEventV2(
                event_id=f"evt_{i}", sequence=i + 1, type=AgentEventType.TOOL_PROGRESS,
                visibility=Visibility.USER, **base,
            ))
        assert queue.full()
        # 再发一条 USER——不得静默丢：队列里必须出现 degraded 标记。
        bus.publish(AgentEventV2(
            event_id="evt_over", sequence=1025,
            type=AgentEventType.ACTION_RECEIPT, visibility=Visibility.USER, **base,
        ))
        seen_types: list[str] = []
        while not queue.empty():
            seen_types.append(queue.get_nowait().type.value)
        assert "session.degraded" in seen_types, (
            "饱和时无显式 OVERLOADED 标记——事件被静默丢弃"
        )

    def test_debug_events_droppable_without_marker(self) -> None:
        from rosclaw.agentd.events import MissionEventBus
        from rosclaw.contracts.agent.agent_event import (
            AgentEventType,
            AgentEventV2,
            Visibility,
        )

        bus = MissionEventBus()
        queue = bus.subscribe("mis_1")
        base = {
            "schema_version": "rosclaw.agent_event.v2",
            "mission_id": "mis_1",
            "timestamp": "2026-08-21T00:00:00+00:00",
            "payload": {},
        }
        for i in range(1024):
            bus.publish(AgentEventV2(
                event_id=f"evt_{i}", sequence=i + 1, type=AgentEventType.TOOL_PROGRESS,
                visibility=Visibility.USER, **base,
            ))
        bus.publish(AgentEventV2(
            event_id="evt_dbg", sequence=1025,
            type=AgentEventType.CONTEXT_USAGE, visibility=Visibility.DEBUG, **base,
        ))
        # DEBUG 可丢（设计允许）：不驱逐 USER 内容、不加标记、不抛错。
        head = queue.get_nowait()
        assert head.event_id == "evt_0", "DEBUG 丢弃不应驱逐既有 USER 事件"
        assert head.type.value != "session.degraded"


class TestReplayCursor:
    def test_resume_replay_from_cursor_no_loss_no_dup(self, tmp_path: Path) -> None:
        """resume 从 cursor 重放：无丢无重（边界 sequence 精确）。"""
        import sqlite3

        from rosclaw.agentd.events import AgentEventStore
        from rosclaw.contracts.agent.agent_event import AgentEventType
        from rosclaw.storage.migrations import MigrationRunner

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        conn.execute(
            "INSERT INTO missions (mission_id, owner_principal, goal_json, "
            "body_id, effective_body_hash, mode, state, budgets_json, "
            "authorization_json, created_at, updated_at) "
            "VALUES ('mis_1', 'u', '{}', 'sim/ur5e', 'h', 'SIMULATION', "
            "'ACTIVE', '{}', '{}', '2026-08-21T00:00:00Z', "
            "'2026-08-21T00:00:00Z')"
        )
        store = AgentEventStore(conn)

        async def run() -> None:
            for _ in range(10):
                await store.append("mis_1", AgentEventType.TOOL_PROGRESS)

        asyncio.run(run())
        # 消费到 7 断了——从 cursor=7 重放必须恰好拿到 8,9,10。
        tail = store.replay("mis_1", after_sequence=7)
        assert [e.sequence for e in tail] == [8, 9, 10], tail
        # 全程单调无洞。
        all_events = store.replay("mis_1")
        assert [e.sequence for e in all_events] == list(range(1, 11))
