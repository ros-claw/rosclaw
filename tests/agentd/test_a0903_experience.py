"""0903/0905 体验复核机制断言（大道至简 R0-1 后保留面）。

聊天路由语义已由 test_ddzj_r01_all_nl_to_pi.py 钉死（所有自然
语言无条件进 Pi——投诉/立方体/五角星都不再有任何自动执行）。
本文件只保留与路由无关的机制声明：
1. 路由层不做情绪词检测（0905 用户决定——投诉由模型自己处理）；
2. compile_requirements 多形状条款（R0-2 将随语义 Gate 整体退役，
   在此之前保持已冻结行为）。
"""

from __future__ import annotations

import pytest


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
