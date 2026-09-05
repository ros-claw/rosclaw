"""0902 审计 R0-2：RequirementCompiler 编译器单测。

大道至简 R0-1 后：聊天主路径自动路由已删除（见
test_ddzj_r01_all_nl_to_pi.py），本文件只保留编译器纯函数与
TaskSpec 冻结面；覆盖率门禁将随 R0-2（Kernel 只报客观事实）
整体退役。
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
