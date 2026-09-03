"""0902 审计 R0-2 红测试：RequirementCompiler + 语义覆盖率门禁。

0902 实证（P0 事故）：用户新增"红色圆柱笔 + 3D 实际轨迹 + 不要 2D"，
自动路由只识别"机械臂+轨迹+画"关键词命中五角星 recipe——新材料性
条件被吞，旧视频复用，宣布 PASS/DELIVERED 假成功。

硬规则（审计 §3.1）：
- 快速配方只有在语义覆盖率 100% 时才能执行；
- 任一 must/forbidden 条款未被 recipe 覆盖 → 不自动路由（交
  Native Agent 编译 TaskSpec——不猜）；
- 未覆盖时不得创建幽灵任务（任务创建之前就拦截）。

闭环断言：
1. 编译器把工具/颜色/轨迹可见/平面/禁止/形状编译为可核验条款；
2. 纯五角星（全覆盖）→ 自动路由不受影响（回归护栏）；
3. 持笔/轨迹可见/不要 2D/未知形状 → 不自动路由 + 零幽灵任务；
4. 条款随 TaskSpec 冻结（R0-3 逐条验收的输入面）。
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestRequirementCompiler:
    def test_plain_star_only_shape_requirement(self) -> None:
        from rosclaw.task_kernel.requirements import compile_requirements

        reqs = compile_requirements("画一个五角星")
        verifiers = {r.verifier for r in reqs}
        assert verifiers == {"shape.star5"}, verifiers

    def test_tool_and_color_and_trace_and_forbid(self) -> None:
        from rosclaw.task_kernel.requirements import compile_requirements

        reqs = compile_requirements(
            "末端持红色圆柱笔，在 3D 画面里显示本次实际轨迹，不要 2D"
        )
        verifiers = {r.verifier for r in reqs}
        assert "receipt.tool_ref" in verifiers
        assert "render.tool_color" in verifiers
        assert "receipt.overlays.actual_eef_trace" in verifiers
        forbidden = [r for r in reqs if r.level == "forbidden"]
        assert any(r.verifier == "delivery.not_2d_only" for r in forbidden)

    def test_vertical_plane_requirement(self) -> None:
        from rosclaw.task_kernel.requirements import compile_requirements

        reqs = compile_requirements("垂直桌子画五角星")
        assert any(r.verifier == "plan.plane.vertical" for r in reqs)

    def test_unknown_shape_is_known_requirement(self) -> None:
        """画正方形：已知形状词但 recipe 不认——条款出现（未覆盖由
        门禁拦），不允许画错形状冒充。"""
        from rosclaw.task_kernel.requirements import compile_requirements

        reqs = compile_requirements("画一个正方形")
        assert any(r.verifier == "shape.square" for r in reqs)


class TestCoverageGate:
    """server 级：auto-route 门禁（PiBridgeServer._dispatch 直达）。"""

    async def _persist(self, tmp_path: Path, text: str, msg: str):
        from rosclaw.agentd.auto_route import reset_routed_for_tests
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        # 进程内路由去重表是全局的——每个用例独立（不复用
        # message_id，否则跨用例误判"已路由"）。
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
        kernel = service._task_kernel
        task_count = kernel._conn.execute(
            "SELECT COUNT(*) AS n FROM tasks"
        ).fetchone()["n"]
        return result, int(task_count)

    async def test_plain_star_still_auto_routes(self, tmp_path: Path) -> None:
        """回归护栏：全覆盖输入不受影响（recipe 仍自动执行）。"""
        result, tasks = await self._persist(
            tmp_path, "画一个五角星，给我 GIF 和 MP4", "m1"
        )
        assert result.get("auto_task"), result
        assert tasks == 1

    async def test_tool_requirement_blocks_autoroute(self, tmp_path: Path) -> None:
        """持笔 → recipe 未覆盖 → 不自动路由 + 零幽灵任务。"""
        result, tasks = await self._persist(
            tmp_path, "画一个五角星，机械臂末端持笔", "m1"
        )
        assert not result.get("auto_task"), result
        assert tasks == 0, f"未覆盖竟建了任务（幽灵任务）: {tasks}"

    async def test_trace_overlay_now_auto_routes(self, tmp_path: Path) -> None:
        """R2-3：渲染器已支持 actual_eef_trace overlay（render_from_spec
        证据绑定）——该要求被 recipe 真实覆盖，自动路由。"""
        result, tasks = await self._persist(
            tmp_path, "画五角星视频，在 3D 画面里显示本次实际轨迹", "m1"
        )
        assert result.get("auto_task"), result
        assert tasks == 1

    async def test_forbid_2d_now_auto_routes(self, tmp_path: Path) -> None:
        """R2-3：3D 场景视频在账即满足"不要只有 2D"禁止项——自动路由。"""
        result, tasks = await self._persist(
            tmp_path, "画五角星，不要 2D", "m1"
        )
        assert result.get("auto_task"), result
        assert tasks == 1

    async def test_unknown_shape_blocks_autoroute(self, tmp_path: Path) -> None:
        result, tasks = await self._persist(
            tmp_path, "画一个正方形", "m1"
        )
        assert not result.get("auto_task"), result
        assert tasks == 0


class TestRequirementsFrozenInSpec:
    def test_requirements_frozen_in_task_spec(self, tmp_path: Path) -> None:
        """条款随 TaskSpec 冻结——R0-3 逐条验收的输入面。"""
        import sqlite3

        from rosclaw.storage.migrations import MigrationRunner
        from rosclaw.task_kernel.service import TaskKernel

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        kernel = TaskKernel(conn, tmp_path)
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_1", text="画五角星，末端持红色圆柱笔",
            cwd=str(tmp_path), body_id="sim/ur5e",
        )
        spec = kernel.get_task_spec(str(bound["task_id"]))
        assert spec is not None
        reqs = spec.get("requirements") or []
        verifiers = {r["verifier"] for r in reqs}
        assert "receipt.tool_ref" in verifiers
        assert "render.tool_color" in verifiers


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
