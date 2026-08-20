"""NINE-2 红测试（九审 §6.1/§7/§25.4-7）：UserTurnV2 + 唯一任务因果链。

红测试先行——当前用户输入不落任何 ROSClaw 账本（Pi JSONL 之外的
因果断裂），任务没有 caused_by_turn_id：

1. pi.turn.record：NL 输入 → UserTurnV2 持久化（turn_id/delivery_seq/
   session/source/text_hash）；同 session delivery 去重；
2. rosclaw_task 创建的 task 带 caused_by_turn_id（= 该 session 最近
   的用户 turn）——每个副作用可追溯到 turn；
3. TurnStore 只记 interactive 用户输入——extension 注入/系统事件
   不能伪装 user turn。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _setup


async def _record_turn(service, tmp_path: Path, text: str, *, session: str = "pi_1"):
    from rosclaw.agentd.pi_bridge.server import PiBridgeServer

    bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
    return await bridge._dispatch(
        "user:local:1000", 1, "pi.turn.record",
        {"token": service.control_token, "pi_session_id": session, "text": text,
         "source": "interactive"},
    )


class TestUserTurnStore:
    async def test_record_persists_turn(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        result = await _record_turn(service, tmp_path, "画一个五角星")
        assert result.get("ok"), result
        turn = result.get("turn") or {}
        assert turn.get("turn_id", "").startswith("turn_")
        assert turn.get("delivery_seq") == 1
        assert turn.get("text_hash", "").startswith("sha256:")
        assert turn.get("source") == "interactive"
        assert turn.get("mission_id") == mission.mission_id
        # 第二个输入 seq=2。
        second = await _record_turn(service, tmp_path, "再画一次")
        assert second["turn"]["delivery_seq"] == 2
        await service.close()

    async def test_non_interactive_source_refused(self, tmp_path: Path) -> None:
        """extension 注入/系统事件不得伪装 interactive user turn。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.turn.record",
            {"token": service.control_token, "pi_session_id": "pi_1",
             "text": "系统注入", "source": "extension_injection"},
        )
        assert not result.get("ok"), "非 interactive 来源竟落账为用户 turn"
        await service.close()


class TestTaskCausedByTurn:
    async def test_task_has_caused_by_turn_id(self, tmp_path: Path) -> None:
        """PR-H9：因果链归 TaskKernel——task_revisions.user_message_id
        锚定触发消息（输入事务先持久化再投递），task_records/
        caused_by_turn_id 已删。"""
        service, mission = await _setup(tmp_path)
        turn = await _record_turn(service, tmp_path, "画一个五角星")
        # 输入事务：同一消息绑定 kernel 任务（与 InputController 同入口）。
        bound = service._task_kernel.bind_message(
            mission_id=mission.mission_id, session_ref="pi_1",
            backend_native_id="pi_1",
            message_id=turn["turn"]["turn_id"],
            text="画一个五角星", cwd=str(service._home),
        )
        row = service._store.connection.execute(
            "SELECT user_message_id FROM task_revisions "
            "WHERE task_id = ? ORDER BY revision DESC LIMIT 1",
            (bound["task_id"],),
        ).fetchone()
        assert row and row[0] == turn["turn"]["turn_id"], (
            f"task revision 缺 user_message_id 锚点: {row}"
        )
        await service.close()
