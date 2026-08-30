"""0827 体验审计 P0-1/P0-2 红测试：Input Arbiter——一条输入只有一个
Owner（双控制者根治）。

0827 实证事故：pi.input.persist 自动启动任务后，同一条用户指令仍
被送给模型——确定性链与 Native Agent 同时接管同一输入，产生两个
相反的终态（模型宣布"✅ 任务完成"，watcher 随后报"交付 MISSING"）。

闭环断言：
1. pi.input.persist 响应携带权威 TurnDisposition：
   已知 recipe → owner=TASK_ROUTER + suppress_model_turn=true +
   task_id；普通问答 → owner=PI_CONVERSATION + suppress=false；
2. 重放幂等：同一 message_id 重投递（PTY 重发/网络重试）仍然
   suppress_model_turn=true 且不创建第二个任务——重放不得掉进
   模型路径（否则重发即双执行）；
3. 疑问句永不 suppress（讨论形式走模型）。
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
    async def test_directive_claimed_by_task_router(
        self, tmp_path: Path
    ) -> None:
        """已知 recipe 指令 → disposition owner=TASK_ROUTER，suppress
        model turn，task_id 与 auto_task 一致。"""
        from rosclaw.agentd.auto_route import reset_routed_for_tests
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        reset_routed_for_tests()
        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await _persist(
            bridge, service, mission.mission_id, "msg_td_1",
            "画一个五角星，给我 GIF 和 MP4",
        )
        assert result.get("ok"), result
        disposition = result.get("turn_disposition")
        assert disposition, f"缺 TurnDisposition：{result}"
        assert disposition.get("owner") == "TASK_ROUTER", disposition
        assert disposition.get("suppress_model_turn") is True, disposition
        auto = result.get("auto_task") or {}
        assert disposition.get("task_id") == str(auto.get("task_id")), (
            disposition, auto,
        )
        assert disposition.get("input_id"), disposition
        await service.close()

    async def test_question_owned_by_conversation(
        self, tmp_path: Path
    ) -> None:
        """疑问句 → owner=PI_CONVERSATION，不 suppress（讨论走模型）。"""
        from rosclaw.agentd.auto_route import reset_routed_for_tests
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        reset_routed_for_tests()
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

    async def test_replay_still_suppressed_no_second_task(
        self, tmp_path: Path
    ) -> None:
        """同一 message_id 重投递：仍 suppress（不能掉进模型路径）
        且不创建第二个 task。"""
        from rosclaw.agentd.auto_route import reset_routed_for_tests
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        reset_routed_for_tests()
        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        first = await _persist(
            bridge, service, mission.mission_id, "msg_td_replay",
            "画一个五角星",
        )
        assert first.get("ok"), first
        first_task = str((first.get("auto_task") or {}).get("task_id"))
        assert first_task, first
        # 重放（PTY 重发/投递重试）——同 message_id。
        second = await _persist(
            bridge, service, mission.mission_id, "msg_td_replay",
            "画一个五角星",
        )
        assert second.get("ok"), second
        disposition = second.get("turn_disposition") or {}
        assert disposition.get("suppress_model_turn") is True, (
            f"重放掉进模型路径=双执行：{second}"
        )
        assert disposition.get("owner") == "TASK_ROUTER", disposition
        assert str(
            (second.get("auto_task") or {}).get("task_id")
        ) == first_task, (first_task, second)
        kernel = service._task_kernel
        rows = kernel._conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()
        assert int(rows["n"]) == 1, f"重放创建了第二个任务：{rows['n']}"
        await service.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
