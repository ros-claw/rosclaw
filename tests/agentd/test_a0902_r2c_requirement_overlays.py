"""0902 审计 R2-3 红测试：需求驱动的能力组合——overlay 要求让
draw_path recipe 走 RenderSpec 渲染（§4.3/R2.3），不再是"recipe
覆盖表与实现两张皮"。

0902 事故的完整闭环：用户说"在 3D 画面里显示本次实际轨迹，不要
2D"——R0-2 门禁此前把它挡在 recipe 外（交模型），因为渲染器根本
不支持 overlay；R2-2 渲染器支持后，recipe 必须真实交付 overlay
（不是改表宣称覆盖）。

闭环断言：
1. 端到端：任务 spec 带 receipt.overlays.actual_eef_trace 要求 →
   recipe 场景渲染走 render_from_spec → receipt.overlays_applied
   含 actual_eef_trace → finish_task 逐条 verdict SATISFIED →
   任务 SUCCEEDED（真 PASS，不是宣称）；
2. requirement_check 读 receipt 的 overlays_applied（R2-2 的诚实
   字段名）——旧 overlays 键兼容；
3. 覆盖表诚实扩展：RECIPE_COVERAGE[recipe:sim.draw_path] 现在含
   receipt.overlays.actual_eef_trace 与 delivery.not_2d_only——
   有实现才覆盖；
4. 门禁行为更新（R0-2 测试随能力演进）：轨迹叠加/不要 2D 现在
   自动路由（能力真实存在）；持笔/颜色/接触仍不路由（无资产/
   无证据通道——诚实边界不动）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


class TestOverlayRequirementEndToEnd:
    async def test_overlay_requirement_delivered_and_verified(
        self, tmp_path: Path
    ) -> None:
        """轨迹叠加要求 → recipe 真实交付（receipt 证据）→ 逐条
        verdict SATISFIED → SUCCEEDED。"""
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
                "message_id": "msg_overlay_1",
                # 圆形（非五角星——通用性证据）+ 轨迹叠加 + 不要 2D。
                "text": "画一个圆形轨迹，在 3D 画面里显示本次实际轨迹，不要 2D",
            },
        )
        assert result.get("ok"), result
        auto = result.get("auto_task")
        assert auto, f"overlay 要求已可覆盖——仍被挡在 recipe 外: {result}"
        kernel = service._task_kernel
        task_id = str(auto["task_id"])
        deadline = asyncio.get_event_loop().time() + 300
        while asyncio.get_event_loop().time() < deadline:
            task = kernel.get_task(task_id)
            if task and task["state"] in ("SUCCEEDED", "FAILED", "REPAIR_REQUIRED"):
                break
            await asyncio.sleep(2)
        task = kernel.get_task(task_id)
        assert task["state"] == "SUCCEEDED", (
            f"终态 {task['state']}：{task.get('terminal_reason')}"
        )
        # receipt 证据：场景渲染的 overlays_applied 含实际轨迹。
        receipts = list(
            (tmp_path / "sim" / "traces").glob("*/render_receipt.json")
        )
        assert receipts, "无 render receipt"
        applied = [
            o for r in receipts
            for o in (json.loads(r.read_text()).get("overlays_applied") or [])
        ]
        assert "actual_eef_trace" in applied, f"overlay 未真实绘制: {applied}"
        # 逐条 verdict（验收行 checks_json 的 requirement_coverage——
        # 在 verifications 表，不在任务行）。
        row = kernel._conn.execute(
            "SELECT checks_json FROM verifications WHERE task_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        checks = json.loads(row["checks_json"] if row else "{}")
        coverage = checks.get("requirement_coverage") or []
        overlay_req = next(
            (c for c in coverage
             if c.get("verifier") == "receipt.overlays.actual_eef_trace"),
            None,
        )
        assert overlay_req, f"验收缺 overlay 条款: {coverage}"
        assert overlay_req["status"] == "SATISFIED", overlay_req
        await service.close()


class TestOverlayVerifierField:
    def test_overlays_applied_field_read(self) -> None:
        from rosclaw.task_kernel.requirement_check import check_requirements

        receipt = {"overlays_applied": ["actual_eef_trace"]}
        verdicts = check_requirements(
            home=Path("/nonexistent-home"),
            requirements=[{"req_id": "r1", "level": "must", "claim": "x",
              "verifier": "receipt.overlays.actual_eef_trace"}],
            artifacts=[],
            receipts=[receipt],
        )
        assert verdicts[0]["status"] == "SATISFIED", verdicts

    def test_overlays_absent_is_violated(self) -> None:
        from rosclaw.task_kernel.requirement_check import check_requirements

        verdicts = check_requirements(
            home=Path("/nonexistent-home"),
            requirements=[{"req_id": "r1", "level": "must", "claim": "x",
              "verifier": "receipt.overlays.actual_eef_trace"}],
            artifacts=[],
            receipts=[{"overlays_applied": []}],
        )
        assert verdicts[0]["status"] == "VIOLATED", verdicts


class TestCoverageTableHonesty:
    def test_draw_path_covers_overlay_and_not_2d(self) -> None:
        from rosclaw.task_kernel.task_router import RECIPE_COVERAGE

        coverage = RECIPE_COVERAGE["recipe:sim.draw_path"]
        assert "receipt.overlays.actual_eef_trace" in coverage
        assert "delivery.not_2d_only" in coverage

    def test_tool_color_contact_still_uncovered(self) -> None:
        """无资产/无证据通道的条款仍不覆盖（诚实边界不动）。"""
        from rosclaw.task_kernel.task_router import RECIPE_COVERAGE

        coverage = RECIPE_COVERAGE["recipe:sim.draw_path"]
        assert "receipt.tool_ref" not in coverage
        assert "render.tool_color" not in coverage
        assert "verification.contact" not in coverage


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
