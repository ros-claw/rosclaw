"""0905 体验复核机制修正（用户决定：不搞情绪词护栏——投诉由模型
自己处理，机制上经语义覆盖自然落到模型路径）。

0903 日志实证："你画的居然是个五角星！我要的是立方体！"——
若按首个可识别形状执行 = 再画一遍五角星并 PASS（投诉变二次假
成功）。正确机制（非情绪检测）：
- cube 注册为"已知未覆盖"形状——立方体诉求（含投诉里的）经
  覆盖门禁落到模型路径；
- 多形状条款：文中提到的每个形状都是独立条款——五角星引用被
  满足不算数，立方体未覆盖照样拦截。

闭环断言：
1. 投诉文本（含 star5+cube）→ 不自动路由 + 零幽灵任务（经覆盖
   门禁，无任何情绪词检测）；
2. "画个立方体" → 不自动路由（cube 已知未覆盖）；
3. "画五角星"/"画五角星改成画圆形" → 照常自动路由（回归护栏）；
4. requirements 多形状编译：star5+cube 两条条款都在。
"""

from __future__ import annotations

import pytest


class TestComplaintViaCoverageSemantics:
    async def _persist(self, tmp_path, text: str, msg: str):
        from rosclaw.agentd.auto_route import reset_routed_for_tests
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        reset_routed_for_tests()
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

    async def test_angry_complaint_falls_to_model_via_cube(
        self, tmp_path
    ) -> None:
        """投诉（star5 引用 + cube 诉求）→ 覆盖门禁落模型——不是
        情绪词拦截。"""
        result, tasks = await self._persist(
            tmp_path,
            "你画的居然是个五角星！也没在3D仿真里展示！"
            "我要的是立方体！你个混蛋！",
            "m1",
        )
        assert not result.get("auto_task"), result
        assert tasks == 0, f"投诉竟建了任务: {tasks}"

    async def test_cube_request_falls_to_model(self, tmp_path) -> None:
        """画立方体 → cube 已知未覆盖 → 模型路径（模型自己说明能力
        边界或尽力而为——不画星冒充）。"""
        result, tasks = await self._persist(
            tmp_path, "那请用ur5机械臂画个立方体，我想看到仿真视频", "m1"
        )
        assert not result.get("auto_task"), result
        assert tasks == 0

    async def test_star_still_auto_routes(self, tmp_path) -> None:
        result, tasks = await self._persist(tmp_path, "画一个五角星", "m1")
        assert result.get("auto_task"), result
        assert tasks == 1

    async def test_revision_to_circle_still_auto_routes(self, tmp_path) -> None:
        """多形状但全覆盖 → 照常执行（改成语义不受影响）。"""
        result, tasks = await self._persist(
            tmp_path, "画五角星改成画圆形", "m1"
        )
        assert result.get("auto_task"), result
        assert tasks == 1


class TestMultiShapeCompilation:
    def test_all_mentioned_shapes_become_clauses(self) -> None:
        from rosclaw.task_kernel.requirements import compile_requirements

        reqs = compile_requirements("画五角星改成画圆形")
        verifiers = {r.verifier for r in reqs}
        assert "shape.star5" in verifiers
        assert "shape.circle" in verifiers

    def test_complaint_compiles_cube_clause(self) -> None:
        from rosclaw.task_kernel.requirements import compile_requirements
        from rosclaw.task_kernel.task_router import RECIPE_COVERAGE

        reqs = compile_requirements(
            "你画的居然是个五角星！我要的是立方体！"
        )
        verifiers = {r.verifier for r in reqs}
        assert "shape.star5" in verifiers
        assert "shape.cube" in verifiers
        coverage = RECIPE_COVERAGE["recipe:sim.draw_path"]
        assert "shape.cube" not in coverage, (
            "cube 不得在覆盖表——登记≠支持执行"
        )

    def test_no_emotion_markers_in_router(self) -> None:
        """机制声明：路由层不做情绪词检测（用户决定）。"""
        import inspect

        from rosclaw.task_kernel import task_router

        src = inspect.getsource(task_router)
        assert "_COMPLAINT_MARKERS" not in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
