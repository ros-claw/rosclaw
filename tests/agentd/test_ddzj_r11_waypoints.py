"""大道至简 R1-1 红测试：plan_cartesian 必须接受任意路径点
（ROSClaw_Native_Agent大道至简深度重构方案_2026-09-05 §R1）。

0905 体验实证（rosclaw体验0905.txt）：用户要「画 hello」，模型第一
次调 plan_cartesian_path 直接 ValidationError——工具只收 star5/
circle 枚举，模型无路可走只能拿五角星冒充。「画什么由 Pi 决定；
ROSClaw 负责把 Pi 给出的路径可靠地变成仿真、视频和验证结果」。

闭环断言：
1. 内置规划器接受任意 waypoints（多笔画/抬笔经 contact=False
   表达）——插值、工作空间校验、canonical hash、句柄摘要；
2. 越界点 fail-closed（规划即拒）；
3. 内置工具 trajectory_generate_planar_path：waypoints 直通；
   既无 shape 也无 waypoints → 报错（删除 shape 默认 star5 的
   静默兜底——「hello→五角星」的开关级根因）；
4. MCP 工具 ur5e.plan_cartesian_path：waypoints 直通；
   无 shape 且无 waypoints → 报错；
5. 自定义路径可真实仿真（simulate 接受该 plan_id 出 trace——
   可规划还必须可执行）；
6. 摘要诚实：自定义路径的 summary 不得出现形状名冒充。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

#: 两笔画 "HI" 形路径（h 一竖一横、i 一竖一点；contact=False 段
#: 为抬笔移动）。全部点在安全工作空间内（半径 0.3~0.4m，z=0.3m）。
_HI_WAYPOINTS = [
    # h 竖
    {"x": 0.35, "y": 0.20, "z": 0.30},
    {"x": 0.35, "y": 0.20, "z": 0.36},
    # 抬笔到横杠起点
    {"x": 0.35, "y": 0.26, "z": 0.33, "contact": False},
    # h 横
    {"x": 0.35, "y": 0.32, "z": 0.33},
    # 抬笔到 i
    {"x": 0.35, "y": 0.38, "z": 0.30, "contact": False},
    # i 竖
    {"x": 0.35, "y": 0.38, "z": 0.34},
]


class TestBuiltinWaypointPlanning:
    def test_arbitrary_waypoints_accepted(self, tmp_path: Path) -> None:
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        svc = SimTrajectoryService(tmp_path)
        plan = svc.generate_planar_path(
            waypoints=_HI_WAYPOINTS, max_segment_m=0.02,
        )
        assert plan["plan_id"]
        assert plan["point_count"] > len(_HI_WAYPOINTS), "未插值"
        summary = str(plan.get("summary", ""))
        assert "五角星" not in summary and "star5" not in summary, summary
        assert "自定义" in summary or "custom" in summary.lower(), summary

    def test_out_of_workspace_point_rejected(self, tmp_path: Path) -> None:
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        svc = SimTrajectoryService(tmp_path)
        bad = [dict(p) for p in _HI_WAYPOINTS]
        bad[2] = {"x": 0.95, "y": 0.95, "z": 0.30}  # 半径 1.34m 越界
        with pytest.raises(ValueError, match="workspace|outside"):
            svc.generate_waypoint_path(waypoints=bad)

    def test_contact_flags_recorded(self, tmp_path: Path) -> None:
        """抬笔段随 plan 持久化（渲染/验收层需要区分绘制与移动）。"""
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        svc = SimTrajectoryService(tmp_path)
        plan = svc.generate_planar_path(
            waypoints=_HI_WAYPOINTS, max_segment_m=0.02,
        )
        payload = svc.get_plan_payload(str(plan["plan_id"]))
        contacts = [bool(w.get("contact", True))
                    for w in payload.get("waypoints", [])]
        assert contacts.count(False) == 2, f"抬笔标记丢失: {contacts}"

    def test_custom_plan_is_executable(self, tmp_path: Path) -> None:
        """可规划必须可执行——自定义 plan_id 直接进动力学 rollout。"""
        from rosclaw.agentd.sim_trajectory import SimTrajectoryService

        svc = SimTrajectoryService(tmp_path)
        plan = svc.generate_planar_path(
            waypoints=_HI_WAYPOINTS, max_segment_m=0.02,
        )
        rollout = svc.simulate_cartesian_trajectory(str(plan["plan_id"]))
        assert rollout.get("ok") is True, rollout
        assert rollout.get("trace_id"), rollout


class TestBuiltinToolSurface:
    def _registry(self):
        from rosclaw.agentd.tools import BuiltinToolRegistry

        return BuiltinToolRegistry(body_id="sim/ur5e", body_summary="UR5e")

    def test_tool_accepts_waypoints(self, tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        registry = self._registry()
        tools = registry.strict_tools(["trajectory_generate_planar_path"])
        assert [t.name for t in tools] == ["trajectory_generate_planar_path"]
        schema = tools[0].parameters
        props = schema.get("properties", {})
        assert "waypoints" in props, f"工具契约无 waypoints: {props}"
        assert "shape" not in schema.get("required", []), (
            "shape 不应再是必填（waypoints 是通用入口）"
        )

    def test_no_silent_star5_default(self, tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        """既无 shape 也无 waypoints → 报错。tools.py 曾有
        arguments.get('shape', 'star5') 静默兜底——hello 画成五角星
        的开关级根因，钉死防回流。"""
        import inspect

        from rosclaw.agentd import tools

        src = inspect.getsource(tools)
        assert 'get("shape", "star5")' not in src
        assert "get('shape', 'star5')" not in src


class TestMcpWaypointPlanning:
    def test_mcp_tool_accepts_waypoints(self) -> None:
        from rosclaw.sim import ur5e_mcp

        result = json.loads(ur5e_mcp.plan_cartesian_path(
            waypoints=_HI_WAYPOINTS,
        ))
        assert result["ok"] is True, result
        assert result["plan_id"]
        assert result["point_count"] > len(_HI_WAYPOINTS)
        assert "五角星" not in result["summary"], result["summary"]

    def test_mcp_tool_requires_shape_or_waypoints(self) -> None:
        from rosclaw.sim import ur5e_mcp

        with pytest.raises((ValueError, TypeError)):
            ur5e_mcp.plan_cartesian_path(
                shape="", center_x=0.35, center_y=0.25, z=0.30,
                outer_radius=0.10,
            )

    def test_mcp_tool_out_of_workspace_rejected(self) -> None:
        from rosclaw.sim import ur5e_mcp

        with pytest.raises(ValueError, match="outside|radius"):
            ur5e_mcp.plan_cartesian_path(
                waypoints=[
                    {"x": 0.35, "y": 0.25, "z": 0.30},
                    {"x": 0.95, "y": 0.95, "z": 0.30},
                ],
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
