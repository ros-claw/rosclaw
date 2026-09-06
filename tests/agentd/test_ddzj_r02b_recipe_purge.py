"""大道至简 R0-2b 机制测试：生产 recipe 链与语义 Gate 物理删除。

方案 R0：「生产环境只保留一套 Pi Agent Loop」——
- TaskExecutionService / plan_templates（draw_path recipe）删除；
  固定流程只在显式 demo（product/demo.py ur5e-star，R0-2a）；
- task_router（route_recipe/RECIPE_COVERAGE/compile_recipe_inputs/
  is_task_directive）、requirements（SHAPE_MARKERS/
  compile_requirements）、requirement_check 全部删除——Kernel
  不再从目标文本编译/裁决语义条款，只报客观执行事实；
- rosclaw_task（goal→recipe 模型工具）从注册面/分发面/wire 面
  全部删除（残留 adapter 是双脑架构的活口）。

闭环断言：
1. 五个模块 import 不到；
2. finish_task 验收面无 requirement_coverage（客观事实：
   deliverables/verdict/receipt）；
3. rosclaw_task 在 TS 注册面与 PY 分发面均不存在；
4. demo 链仍是唯一保留形态（ur5e-star 可跑——R0-2a 已证）。
"""

from __future__ import annotations

import importlib.util

import pytest


class TestProductionRecipeChainDeleted:
    @pytest.mark.parametrize(
        "module",
        [
            "rosclaw.agentd.task_execution",
            "rosclaw.agentd.plan_templates",
            "rosclaw.task_kernel.task_router",
            "rosclaw.task_kernel.requirements",
            "rosclaw.task_kernel.requirement_check",
        ],
    )
    def test_module_gone(self, module: str) -> None:
        assert importlib.util.find_spec(module) is None, (
            f"{module} 仍在——语义 Gate/生产 recipe 链未删净"
        )

    def test_no_router_symbols_anywhere_in_src(self) -> None:
        """route_recipe/RECIPE_COVERAGE/compile_requirements/
        SHAPE_MARKERS 在 src/ 无任何引用（注释外的活引用）。"""
        import subprocess
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src"
        for symbol in (
            "route_recipe", "RECIPE_COVERAGE", "compile_requirements",
            "SHAPE_MARKERS", "compile_recipe_inputs", "is_task_directive",
        ):
            hits = subprocess.run(
                ["grep", "-rn", symbol, str(src)],
                capture_output=True, text=True,
            ).stdout.strip()
            assert not hits, f"{symbol} 仍有活引用:\n{hits[:400]}"

    def test_finish_task_no_semantic_clause_gate(self, tmp_path) -> None:
        """finish_task 不再做目标文本条款验收——客观事实面
        （无 deliverable 要求+无产物 → 诚实失败；有证据 → PASS）。"""
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
            cwd=str(tmp_path), body_id="",
        )
        spec = kernel.get_task_spec(str(bound["task_id"]))
        assert spec is not None
        # 语义条款面已空——不再从目标文本编译"红色圆柱笔"等条款。
        assert not (spec.get("requirements") or []), spec.get("requirements")

    def test_rosclaw_task_tool_gone(self) -> None:
        import inspect

        from rosclaw.agentd.pi_bridge import tool_dispatch

        src = inspect.getsource(tool_dispatch)
        assert '"rosclaw_task"' not in src
        assert "async def _task(" not in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


class TestCapabilityReceiptIsTrustedEvidence:
    """recipe 链删除后，Pi 驱动的 capability 动力学 rollout 必须有
    合法的终态路径（否则所有 Pi 具身任务永远 RUNNING——比假成功
    更糟）。受信证据第四态：kernel:capability:* 生产的
    simulation_receipt.json 在账（receipt 由 SimTrajectoryService
    内核代码路径写入，非模型可拼）。"""

    def _kernel(self, tmp_path):
        import sqlite3

        from rosclaw.storage.migrations import MigrationRunner
        from rosclaw.task_kernel.service import TaskKernel

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        return TaskKernel(conn, tmp_path)

    def _task_with_receipt(self, kernel, tmp_path, producer: str):
        bound = kernel.bind_message(
            mission_id="m1", session_ref="s1", backend_native_id="s1",
            message_id="msg_1", text="画一个五角星", cwd=str(tmp_path),
            body_id="sim/ur5e",
        )
        task_id = str(bound["task_id"])
        kernel.note_tool_use(task_id, "rosclaw_execute")
        trace_dir = tmp_path / "sim" / "traces" / "trace_test"
        trace_dir.mkdir(parents=True)
        receipt = trace_dir / "simulation_receipt.json"
        receipt.write_text('{"ok": true}', encoding="utf-8")
        artifact = kernel.register_artifact(
            task_id=task_id, path=str(receipt),
            media_type="application/json", producer=producer,
        )
        return task_id, str(artifact["artifact_id"])

    def test_capability_receipt_allows_terminal(self, tmp_path) -> None:
        kernel = self._kernel(tmp_path)
        task_id, art = self._task_with_receipt(
            kernel, tmp_path, "kernel:capability:ur5e_simulate_cartesian_trajectory",
        )
        result = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art],
        )
        assert not any(
            "PLAN_AUTHORITY_MISSING" in f for f in result.get("failures", [])
        ), result.get("failures")

    def test_model_produced_receipt_still_rejected(self, tmp_path) -> None:
        """模型手拼的同路径文件不算数（producer 非 kernel）。"""
        kernel = self._kernel(tmp_path)
        task_id, art = self._task_with_receipt(
            kernel, tmp_path, "model:rosclaw_artifact_register",
        )
        result = kernel.finish_task(
            task_id=task_id, summary="done", artifact_ids=[art],
        )
        assert any(
            "PLAN_AUTHORITY_MISSING" in f for f in result.get("failures", [])
        ), result.get("failures")
