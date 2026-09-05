"""Input Arbiter TurnDisposition 契约测试（大道至简 R0-1 语义）。

0827 原版：已知 recipe → owner=TASK_ROUTER + suppress（确定性链
抢答）。2026-09-05 大道至简方案判定该双脑架构为假成功结构性
根因（hello→五角星），聊天主路径自动路由整体退役——
**所有自然语言的 Owner 恒为 PI_CONVERSATION，suppress 恒 false**。

闭环断言：
1. 画路径指令/普通问答/重放——disposition 恒 PI_CONVERSATION +
   suppress=false + 无 auto_task + 零任务；
2. 协议字段保留（版本倾斜兼容）：input_id/owner/task_id/
   suppress_model_turn 四键齐全。
"""

from __future__ import annotations

from pathlib import Path

import pytest


async def _persist(bridge, service, mission_id: str, message_id: str, text: str):
    return await bridge._dispatch(
        "user:local:1000", 1, "pi.input.persist",
        {
            "token": service.control_token,
            "mission_id": mission_id,
            "session_ref": "pi_1",
            "message_id": message_id,
            "text": text,
        },
    )


class TestTurnDisposition:
    async def test_directive_owned_by_pi(self, tmp_path: Path) -> None:
        """画路径指令也无条件归 Pi——确定性链不再认领任何输入。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await _persist(
            bridge, service, mission.mission_id, "msg_td_1",
            "画一个五角星，给我 GIF 和 MP4",
        )
        assert result.get("ok"), result
        disposition = result.get("turn_disposition")
        assert disposition, f"缺 TurnDisposition：{result}"
        assert disposition.get("owner") == "PI_CONVERSATION", disposition
        assert disposition.get("suppress_model_turn") is False, disposition
        assert disposition.get("task_id") == "", disposition
        assert "auto_task" not in result, result
        assert disposition.get("input_id"), disposition
        await service.close()

    async def test_question_owned_by_conversation(
        self, tmp_path: Path
    ) -> None:
        """疑问句 → owner=PI_CONVERSATION，不 suppress（走模型）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await _persist(
            bridge, service, mission.mission_id, "msg_td_q",
            "怎么画五角星？",
        )
        assert result.get("ok"), result
        disposition = result.get("turn_disposition")
        assert disposition, f"缺 TurnDisposition：{result}"
        assert disposition.get("owner") == "PI_CONVERSATION", disposition
        assert disposition.get("suppress_model_turn") is False, disposition
        assert not result.get("auto_task"), result
        await service.close()

    async def test_replay_no_task_no_suppress(self, tmp_path: Path) -> None:
        """同一 message_id 重投递：不建任务、不 suppress——路由
        去重随路由一起退役，重放幂等由 input 落账保证。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        for _ in range(2):
            result = await _persist(
                bridge, service, mission.mission_id, "msg_td_replay",
                "画一个五角星",
            )
            assert result.get("ok"), result
            disposition = result.get("turn_disposition") or {}
            assert disposition.get("suppress_model_turn") is False
            assert disposition.get("owner") == "PI_CONVERSATION"
            assert "auto_task" not in result
        rows = service._task_kernel._conn.execute(
            "SELECT COUNT(*) AS n FROM tasks"
        ).fetchone()
        assert int(rows["n"]) == 0, f"聊天输入竟建了任务：{rows['n']}"
        await service.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
