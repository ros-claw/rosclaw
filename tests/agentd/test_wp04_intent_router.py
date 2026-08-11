"""WP-P0-5 红测试（总纲 §7.1）：两级路由——确定性 Intent Router。

红测试先行——已知任务（画五角星）仍要走一次模型来回才敢动手：
模型请求预算 SLO 是 ≤2（目标 1 次理解）。确定性 Intent Router 对
高价值已知任务直接产出 TaskSpec（零模型回合）；未知目标返回
None 交模型（不瞎猜）。
"""

from __future__ import annotations


class TestIntentRouter:
    def test_star_goal_routes_to_draw_shape(self) -> None:
        from rosclaw.agentd.intent_router import route_intent

        for text in (
            "我想用机械臂画个五角星",
            "让机械臂画五角星",
            "draw a five-pointed star",
            "用 UR5e 画一个五角星",
        ):
            spec = route_intent(text)
            assert spec is not None, f"未路由: {text}"
            assert spec["goal"] == "draw_shape"
            assert spec["parameters"]["shape"] == "star5"

    def test_unknown_goal_returns_none(self) -> None:
        """未知目标不交模型瞎猜——返回 None 走模型路径。"""
        from rosclaw.agentd.intent_router import route_intent

        assert route_intent("帮我分析这段控制器日志") is None
        assert route_intent("今天天气怎么样") is None
        # 近似但非任务（只提"星"不提画/机械臂）不误路由。
        assert route_intent("给我讲个关于星星的故事") is None

    def test_router_extracts_radius(self) -> None:
        from rosclaw.agentd.intent_router import route_intent

        spec = route_intent("画一个半径 0.2 米的五角星")
        assert spec is not None
        assert abs(spec["parameters"]["radius_m"] - 0.2) < 1e-9
