"""PR-SEVEN-4 红测试（七审 §2.4/§6）：UR5e 五角星任务级闭环。

红测试先行——当前 UR5e SIM 只是内存状态机（move_to_pose 把 pose
从 A 改到 B），没有笛卡尔轨迹/插值/trace/几何验证：

1. plan_cartesian_path（COMPUTE）：五角星 10 顶点交替 + 闭合，按
   最大线段插值，canonical hash 稳定（同输入同 hash），全部点在
   安全工作空间内（越界即拒）；
2. execute_cartesian_path（SIM_ACTION）：整条轨迹一个 ExactAction
   （不是每个插值点一张卡）；trajectory hash 不符即拒；执行产出
   时间序列 trace（不是只覆盖最后 pose）；
3. get_cartesian_trace（OBSERVE）：返回完整 trace + trajectory hash；
4. verify_drawing（COMPUTE）：端点误差/路径 RMSE/最大误差/闭合误差
   全部过阈值才 PASS；trace 被篡改必须 FAIL；
5. 证据：SVG + trajectory JSON + 同一 hash 绑定。
"""

from __future__ import annotations

import json
import math


def _call(name: str, **kwargs):
    """直接调 sim 工具函数（单元层；MCP 路径由 journey 覆盖）。"""
    from rosclaw.sim import ur5e_mcp

    fn = getattr(ur5e_mcp, name.replace("ur5e.", "").replace(".", "_"), None)
    assert fn is not None, f"ur5e sim 缺工具 {name}"
    return json.loads(fn(**kwargs))


class TestPlanCartesianPath:
    def test_star_plan_ten_vertices_closed_stable_hash(self) -> None:
        result = _call(
            "ur5e.plan_cartesian_path",
            shape="star5", center_x=0.35, center_y=0.25, z=0.30,
            outer_radius=0.10, include_payload=True,
        )
        assert result.get("ok"), result
        points = result["trajectory"]["points"]
        # 10 个交替顶点 + 回到起点 = 11 个航点（未插值）。
        waypoints = result["trajectory"]["waypoints"]
        assert len(waypoints) == 11, f"五角星顶点数错: {len(waypoints)}"
        assert waypoints[0] == waypoints[-1], "轨迹未闭合"
        # 插值后点数更多且每段不超 max_segment。
        assert len(points) > len(waypoints)
        for a, b in zip(points, points[1:], strict=False):
            seg = math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))
            assert seg <= result["trajectory"]["max_segment_m"] + 1e-9
        # canonical hash 稳定。
        again = _call(
            "ur5e.plan_cartesian_path",
            shape="star5", center_x=0.35, center_y=0.25, z=0.30,
            outer_radius=0.10, include_payload=True,
        )
        assert again["trajectory"]["hash"] == result["trajectory"]["hash"]
        assert result["trajectory"]["hash"]

    def test_out_of_workspace_plan_rejected(self) -> None:
        """中心/半径越出安全空间 → 规划即拒（fail closed）。"""
        result = None
        try:
            result = _call(
                "ur5e.plan_cartesian_path",
                shape="star5", center_x=0.95, center_y=0.0, z=0.30,
                outer_radius=0.20,
            )
        except Exception as exc:
            assert "workspace" in str(exc).lower() or "safe" in str(exc).lower()
            return
        # 或以结构化失败返回。
        assert result is not None and not result.get("ok", True), (
            f"越界轨迹竟通过规划: {str(result)[:200]}"
        )


class TestExecuteAndVerify:
    def test_execute_trace_and_verify_loop(self) -> None:
        """plan → execute（单动作）→ trace → verify PASS；篡改 trace
        必须 FAIL。"""
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        plan = _call(
            "ur5e.plan_cartesian_path",
            shape="star5", center_x=0.35, center_y=0.25, z=0.30,
            outer_radius=0.10, include_payload=True,
        )
        trajectory = plan["trajectory"]
        executed = _call(
            "ur5e.execute_cartesian_path",
            trajectory=trajectory,
        )
        assert executed.get("ok"), executed
        assert executed.get("trajectory_hash") == trajectory["hash"]
        # trace 是时间序列——点数与插值点一致。
        trace = _call("ur5e.get_cartesian_trace", include_points=True)
        assert trace.get("ok")
        points = trace["trace"]["points"]
        assert len(points) == len(trajectory["points"])
        assert points[0].get("t") is not None, "trace 缺时间序列"
        # verify：端点/RMSE/最大/闭合误差全过 → PASS。
        verdict = _call(
            "ur5e.verify_drawing",
            expected_trajectory_hash=trajectory["hash"],
        )
        assert verdict.get("ok")
        assert verdict["verification"]["verdict"] == "PASS", verdict
        assert verdict["verification"]["closure_error_m"] < 1e-6
        assert verdict["verification"]["rmse_m"] < 1e-6
        # 篡改 trace → FAIL。
        ur5e_mcp._state["trace"][5]["x"] += 0.5
        verdict2 = _call(
            "ur5e.verify_drawing",
            expected_trajectory_hash=trajectory["hash"],
        )
        assert verdict2["verification"]["verdict"] == "FAIL", (
            "trace 被篡改竟仍 PASS"
        )

    def test_execute_rejects_hash_mismatch(self) -> None:
        """execute 的 trajectory 与 hash 不符（篡改一个点）→ 拒绝。"""
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        plan = _call(
            "ur5e.plan_cartesian_path",
            shape="star5", center_x=0.35, center_y=0.25, z=0.30,
            outer_radius=0.10, include_payload=True,
        )
        trajectory = dict(plan["trajectory"])
        trajectory["points"] = [dict(p) for p in trajectory["points"]]
        trajectory["points"][3]["x"] += 0.1  # 篡改但保留原 hash
        result = None
        try:
            result = _call("ur5e.execute_cartesian_path", trajectory=trajectory)
        except Exception as exc:
            assert "hash" in str(exc).lower()
            return
        assert result is not None and not result.get("ok", True), (
            "篡改轨迹竟被执行"
        )

    def test_evidence_bundle_svg_and_json(self) -> None:
        """执行证据：SVG + trajectory JSON 与同一 hash 绑定。"""
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp.reset_simulation()
        plan = _call(
            "ur5e.plan_cartesian_path",
            shape="star5", center_x=0.35, center_y=0.25, z=0.30,
            outer_radius=0.10, include_payload=True,
        )
        _call("ur5e.execute_cartesian_path", trajectory=plan["trajectory"])
        trace = _call("ur5e.get_cartesian_trace", include_points=True)
        assert trace["trace"]["trajectory_hash"] == plan["trajectory"]["hash"]
        svg = trace["trace"].get("svg", "")
        assert svg.startswith("<svg") and "<polyline" in svg or "<polygon" in svg, (
            "缺 SVG 证据"
        )
