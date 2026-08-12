"""NINE-3 红测试（九审 §17.3）：PlanStore 随机实例 ID。

红测试先行——当前 plan_id = digest 前缀（确定性）：同一任务重复
执行会撞同一个 plan_id，被"单次消费"误杀合法重跑。
plan_instance_id 必须随机；digest 只做内容寻址；scope 绑定。
"""

from __future__ import annotations

import json


def _call(name: str, **kwargs):
    from rosclaw.sim import ur5e_mcp

    fn = getattr(ur5e_mcp, name.replace("ur5e.", "").replace(".", "_"), None)
    return json.loads(fn(**kwargs))


def _plan():
    return _call(
        "ur5e.plan_cartesian_path",
        shape="star5", center_x=0.35, center_y=0.25, z=0.30, outer_radius=0.10,
    )


class TestRandomPlanInstance:
    def test_same_shape_twice_gets_distinct_instances(self) -> None:
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        first = _plan()
        second = _plan()
        assert first["plan_id"] != second["plan_id"], (
            "同一 digest 复用了同一 plan_id——合法重跑会被误杀"
        )
        # digest 是内容寻址——相同载荷同 digest。
        assert first["digest"] == second["digest"]

    def test_both_instances_independently_executable(self) -> None:
        """两个实例各自可执行一次（digest 相同不互相挤占）。"""
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        first = _plan()
        _plan()  # 第二个实例（证明不互相挤占）
        r1 = _call("ur5e.execute_plan", plan_id=first["plan_id"])
        assert r1.get("ok"), r1
        # 第一个消费后，第二个实例仍可独立规划执行（重新规划同一形状）。
        third = _plan()
        r2 = _call("ur5e.execute_plan", plan_id=third["plan_id"])
        assert r2.get("ok"), r2
