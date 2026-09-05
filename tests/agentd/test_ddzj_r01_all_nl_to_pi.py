"""大道至简 R0-1（ROSClaw_Native_Agent大道至简深度重构方案_2026-09-05）：
聊天主路径删除确定性自动路由——**所有自然语言无条件进入 Pi**。

0905 体验实证（rosclaw体验0905.txt）：双脑架构——本地确定性大脑
（auto_route/SHAPE_MARKERS/RECIPE_COVERAGE）抢答自然语言，Pi 沦为
补救程序——是「hello→五角星」假成功的结构性根因。方案 R0：
- 删除聊天主路径 auto_route / explain_route / suppress_model_turn；
- 固定 recipe 只保留为显式 demo（R0-2），不再拦截用户对话；
- 用户目标是否完成由看过全部过程的 Pi 判断，Kernel 只报客观
  执行事实。

闭环断言（钉死新语义，防回流）：
1. 「画一个五角星」→ 无 auto_task、零幽灵任务、
   owner=PI_CONVERSATION、suppress_model_turn=false；
2. 投诉文本、解释性追问同样无条件进 Pi（EXPLAIN_HANDLER 退役）；
3. 同一 message_id 重投递 → 仍零任务（重放不再依赖路由去重）；
4. 机制声明：pi_bridge persist 路径无 maybe_auto_route /
   is_explain_followup 引用，auto_route/explain_route 模块删除。
"""

from __future__ import annotations

import importlib.util

import pytest


class TestAllNaturalLanguageGoesToPi:
    async def _persist(self, tmp_path, text: str, msg: str):
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.input.persist",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
                "message_id": msg,
                "text": text,
            },
        )
        tasks = service._task_kernel._conn.execute(
            "SELECT COUNT(*) AS n FROM tasks"
        ).fetchone()["n"]
        await service.close()
        return result, int(tasks)

    async def test_star_directive_goes_to_pi(self, tmp_path) -> None:
        """已知形状的画路径指令也无条件进 Pi——确定性链不再抢答
        （0905 前：零模型回合自动画星=双脑架构的招牌假成功）。"""
        result, tasks = await self._persist(tmp_path, "画一个五角星", "m1")
        assert "auto_task" not in result, result
        assert tasks == 0, f"聊天输入竟建了任务: {tasks}"
        disposition = result.get("turn_disposition") or {}
        assert disposition.get("owner") == "PI_CONVERSATION", disposition
        assert disposition.get("suppress_model_turn") is False, disposition

    async def test_complaint_goes_to_pi(self, tmp_path) -> None:
        result, tasks = await self._persist(
            tmp_path, "你画的居然是个五角星！我要的是立方体！", "m1"
        )
        assert "auto_task" not in result, result
        assert tasks == 0
        assert (result.get("turn_disposition") or {}).get(
            "owner"
        ) == "PI_CONVERSATION"

    async def test_explain_followup_goes_to_pi(self, tmp_path) -> None:
        """解释性追问也进 Pi——Pi 有只读工具（artifact list 等）自己
        回答；EXPLAIN_HANDLER 确定性回答退役（方案：所有自然语言
        无条件进 Pi，唯一 Owner）。"""
        result, tasks = await self._persist(
            tmp_path, "你这是啥？给我看看证据", "m1"
        )
        assert "explain" not in result, result
        assert "auto_task" not in result, result
        assert tasks == 0
        disposition = result.get("turn_disposition") or {}
        assert disposition.get("owner") == "PI_CONVERSATION", disposition
        assert disposition.get("suppress_model_turn") is False, disposition

    async def test_replay_creates_no_task(self, tmp_path) -> None:
        """同一 message_id 重投递：路由去重随路由一起退役——重放
        天然零任务（persist 幂等由 input 落账保证）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        for _ in range(2):
            result = await bridge._dispatch(
                "user:local:1000", 1, "pi.input.persist",
                {
                    "token": service.control_token,
                    "mission_id": mission.mission_id,
                    "session_ref": "pi_1",
                    "message_id": "m1",
                    "text": "画一个五角星",
                },
            )
            assert "auto_task" not in result, result
        tasks = service._task_kernel._conn.execute(
            "SELECT COUNT(*) AS n FROM tasks"
        ).fetchone()["n"]
        await service.close()
        assert int(tasks) == 0

    def test_persist_path_has_no_router_refs(self) -> None:
        import inspect

        from rosclaw.agentd.pi_bridge import server

        src = inspect.getsource(server)
        assert "maybe_auto_route" not in src
        assert "is_explain_followup" not in src
        assert "EXPLAIN_HANDLER" not in src

    def test_router_modules_deleted(self) -> None:
        assert importlib.util.find_spec("rosclaw.agentd.auto_route") is None
        assert importlib.util.find_spec("rosclaw.agentd.explain_route") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
