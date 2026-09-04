"""0903 体验日志复核红测试（rosclaw体验0903.txt 实证）：

投诉/批评不是确定性任务指令——"你画的居然是个五角星！我要的
是立方体！" 含可识别形状词，若不拦截会**再画一遍五角星并
PASS**（投诉变二次假成功）。情绪/批评标记 → 交模型（道歉+诚实
解释），不进确定性链。
"""

from __future__ import annotations

import pytest


class TestComplaintNotDirective:
    """投诉含形状词不得触发确定性链（0903 实证：批评"你画的居然
    是个五角星"被当成新指令又画了一遍五角星）。"""

    def test_angry_complaint_not_directive(self) -> None:
        from rosclaw.task_kernel.task_router import is_task_directive

        assert not is_task_directive(
            "你画的居然是个五角星！也没在3D仿真里展示！你压根就没思考吧！"
            "我要的是立方体！你个混蛋！"
        )
        assert not is_task_directive("你画的什么鬼东西？根本不是我要的")
        assert not is_task_directive("居然画错了，这不是我要的")

    def test_legit_revision_still_directive(self) -> None:
        """回归护栏：正常修订仍是指令（0901 的"不对——加红色圆柱笔"
        是合法修订，不含情绪词）。"""
        from rosclaw.task_kernel.task_router import is_task_directive

        assert is_task_directive(
            "不对——加红色圆柱笔，在 3D 画面里显示本次实际轨迹，不要 2D"
        )
        assert is_task_directive("画一个五角星")

    def test_angry_complaint_not_auto_routed(self) -> None:
        """端到端（编译层）：投诉文本即使进了路由前置也不得路由
        ——零任务。"""
        from rosclaw.task_kernel.task_router import is_task_directive

        complaint = ("你画的居然是个五角星！我要的是立方体！"
                     "你压根就没思考吧！")
        assert not is_task_directive(complaint)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
