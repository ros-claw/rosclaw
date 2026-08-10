"""PR-EIGHT-3 红测试（八审 §1.4/P0-3）：PlanStore + 不透明 plan_id。

红测试先行——当前 plan_cartesian_path 把完整轨迹（waypoints +
插值 points + hash）返回给模型，execute_cartesian_path 又要求模型
原样传回并猜 hash：LLM 搬运权威载荷，token 浪费且必出 schema/
hash 错（八审真实会话实测）。

正确设计：模型只见 plan_id + digest + 摘要；execute_plan(plan_id)
由执行器从 PlanStore 取回 canonical payload 复验执行；单次消费、
TTL、容量限制、篡改 fail closed。
"""

from __future__ import annotations

import json


def _call(name: str, **kwargs):
    from rosclaw.sim import ur5e_mcp

    fn = getattr(ur5e_mcp, name.replace("ur5e.", "").replace(".", "_"), None)
    assert fn is not None, f"ur5e sim 缺工具 {name}"
    return json.loads(fn(**kwargs))


def _plan():
    return _call(
        "ur5e.plan_cartesian_path",
        shape="star5", center_x=0.35, center_y=0.25, z=0.30, outer_radius=0.10,
    )


class TestPlanHandleView:
    def test_plan_result_has_no_payload_arrays(self) -> None:
        """模型可见的 plan 结果不得含完整插值点/航点数组。"""
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        result = _plan()
        assert result.get("ok"), result
        assert result.get("plan_id"), f"缺 plan_id: {result}"
        assert result.get("digest"), "缺 digest"
        assert result.get("summary"), "缺人类可读摘要"
        assert result.get("point_count", 0) > 10
        # 关键：无 points/waypoints 载荷泄漏。
        assert "points" not in result, "模型视图泄漏完整插值点"
        assert "waypoints" not in result, "模型视图泄漏航点数组"
        assert "trajectory" not in result, "模型视图泄漏 trajectory 对象"

    def test_execute_plan_by_id(self) -> None:
        """execute_plan(plan_id)：执行器取回载荷、复验、执行；receipt
        引用 plan_id/digest。"""
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        plan = _plan()
        result = _call("ur5e.execute_plan", plan_id=plan["plan_id"])
        assert result.get("ok"), result
        assert result.get("plan_id") == plan["plan_id"]
        assert result.get("digest") == plan["digest"]
        assert result.get("points_executed", 0) > 10

    def test_unknown_plan_id_refused(self) -> None:
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        try:
            result = _call("ur5e.execute_plan", plan_id="plan_nonexistent")
        except Exception as exc:
            assert "plan" in str(exc).lower()
            return
        assert not result.get("ok", True), f"未知 plan_id 竟执行: {result}"

    def test_plan_single_use(self) -> None:
        """同一 plan_id 重复执行必须 fail closed（单次消费）。"""
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        plan = _plan()
        first = _call("ur5e.execute_plan", plan_id=plan["plan_id"])
        assert first.get("ok"), first
        try:
            second = _call("ur5e.execute_plan", plan_id=plan["plan_id"])
        except Exception as exc:
            assert "consum" in str(exc).lower() or "used" in str(exc).lower() or "plan" in str(exc).lower()
            return
        assert not second.get("ok", True), "plan_id 被重复消费"

    def test_plan_store_capacity_evicts_oldest(self) -> None:
        """容量限制：超出后最旧 plan 被驱逐（fail closed 执行）。"""
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        first_id = _plan()["plan_id"]
        for i in range(40):
            _call(
                "ur5e.plan_cartesian_path",
                shape="star5", center_x=0.35, center_y=0.25, z=0.30,
                outer_radius=round(0.05 + i * 0.005, 3),
            )
        try:
            result = _call("ur5e.execute_plan", plan_id=first_id)
        except Exception:
            return  # 驱逐后拒绝——通过
        assert not result.get("ok", True), "被驱逐的 plan 竟仍可执行"

    def test_verify_drawing_defaults_to_last_plan(self) -> None:
        """verify_drawing 不传 hash——用最近执行 plan 的 digest。"""
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        plan = _plan()
        _call("ur5e.execute_plan", plan_id=plan["plan_id"])
        verdict = _call("ur5e.verify_drawing")
        assert verdict.get("ok"), verdict
        assert verdict["verification"]["verdict"] == "PASS"


class TestCatalogHidesLegacyExecutor:
    def test_manifest_uses_execute_plan(self) -> None:
        """kit manifest：execute_plan 在动作清单；旧
        execute_cartesian_path 不在（dev 兼容层，模型默认不可见）。"""
        from rosclaw.sim.robot_kit import load_first_party_kits

        kits = {k.kit_id: k for k in load_first_party_kits()}
        kit = kits["rosclaw/ur5e-sim"]
        assert "ur5e.execute_plan" in kit.action_tools
        assert "ur5e.execute_cartesian_path" not in kit.action_tools, (
            "旧整轨迹工具仍对模型可见——模型会继续搬运载荷"
        )
