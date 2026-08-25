"""P1-C3 红测试（0824 总纲 §15.4/P1-C）：PathSpec3D + reach/draw 验收。

真实缺口：`plane=xy` 硬编码（generate_planar_path 对非 xy 诚实拒
绝）——竖直画板/斜面/任意法向的 draw 与 reach 无法表达；§15.4
的 plane_pose/tool_pose_constraint/sampling/timing 不在契约里。

断言：
1. PathSpec3D 契约字段（geometry/plane_pose/tool_pose_constraint/
   sampling/timing）存在且被校验；
2. 任意平面生成：normal=[0,1,0]（竖直面）→ 全部点满足平面方程
   n·p = offset（1e-6）；waypoints 朝向沿 -normal；xy 默认行为
   不变（回退兼容）；
3. reach 路径：waypoints = approach（target+normal*h）→ contact
   （target 精确命中）；
4. 旅程：竖直面 draw 端到端（rollout + tracking PASS）；reach 到
   目标点端到端（末端误差 < 5mm）。
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pytest

from rosclaw.agentd.sim_trajectory import SimTrajectoryService
from rosclaw.contracts.agent.pose_trajectory import PoseTrajectorySpecV1
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.service import TaskKernel


class TestPathSpec3DContract:
    def test_optional_3d_fields_present(self) -> None:
        spec = PoseTrajectorySpecV1(
            frame_id="world",
            tool_frame="attachment_site",
            contact_plane={"normal_xyz": [0.0, 1.0, 0.0], "offset_m": 0.3},
            waypoints=[{
                "position_m": [0.35, 0.3, 0.2],
                "orientation_xyzw": [1.0, 0.0, 0.0, 0.0],
                "kind": "contact",
            }],
            digest="sha256:x",
            geometry="shape",
            plane_pose={
                "position_m": [0.0, 0.3, 0.0],
                "orientation_xyzw": [1.0, 0.0, 0.0, 0.0],
            },
            tool_pose_constraint={"axis": "tool_z", "tolerance_deg": 5.0},
            sampling={"max_segment_m": 0.02},
            timing={"speed_mps": 0.05, "accel_mps2": 0.2},
        )
        assert spec.plane_pose is not None
        assert spec.tool_pose_constraint["tolerance_deg"] == 5.0
        assert spec.timing["speed_mps"] == 0.05

    def test_fields_default_absent(self) -> None:
        """3D 扩展字段是可选的——旧规格（无这些字段）仍然合法。"""
        spec = PoseTrajectorySpecV1(
            frame_id="world",
            tool_frame="attachment_site",
            contact_plane={"normal_xyz": [0.0, 0.0, 1.0], "offset_m": 0.1},
            waypoints=[{
                "position_m": [0.3, 0.0, 0.1],
                "orientation_xyzw": [1.0, 0.0, 0.0, 0.0],
                "kind": "contact",
            }],
            digest="sha256:x",
        )
        assert spec.plane_pose is None
        assert spec.timing is None


class TestArbitraryPlane:
    def test_vertical_plane_points_satisfy_equation(self, tmp_path: Path) -> None:
        service = SimTrajectoryService(tmp_path)
        plan = service.generate_planar_path(
            shape="star5", center_m=[0.35, 0.0, 0.25], scale_m=0.1,
            plane_normal_xyz=[0.0, 1.0, 0.0], plane_offset_m=0.0,
        )
        for point in plan["points"]:
            # normal=[0,1,0], offset=0 → y ≡ 0
            assert abs(point["y"] - 0.0) < 1e-9, point
        spec = plan["spec"]
        assert spec["contact_plane"]["normal_xyz"] == [0.0, 1.0, 0.0]
        # 竖直面：点在 xz 上有形状（z 有变化）。
        zs = [p["z"] for p in plan["points"]]
        assert max(zs) - min(zs) > 0.1, "竖直面形状没有 z 向展开"

    def test_xy_default_unchanged(self, tmp_path: Path) -> None:
        service = SimTrajectoryService(tmp_path)
        plan = service.generate_planar_path(
            shape="star5", center_m=[0.35, 0.0, 0.12], scale_m=0.12,
        )
        assert all(abs(p["z"] - 0.12) < 1e-9 for p in plan["points"])
        assert plan["spec"]["contact_plane"]["normal_xyz"] == [0.0, 0.0, 1.0]

    def test_non_unit_normal_rejected(self, tmp_path: Path) -> None:
        service = SimTrajectoryService(tmp_path)
        with pytest.raises(ValueError, match="unit|单位"):
            service.generate_planar_path(
                shape="star5", center_m=[0.35, 0.0, 0.2], scale_m=0.1,
                plane_normal_xyz=[0.0, 2.0, 0.0], plane_offset_m=0.0,
            )


class TestReachPath:
    def test_reach_waypoints_hit_target(self, tmp_path: Path) -> None:
        service = SimTrajectoryService(tmp_path)
        plan = service.generate_reach_path(
            target_m=[0.35, 0.1, 0.25], approach_m=0.05,
        )
        spec = plan["spec"]
        waypoints = spec["waypoints"]
        contact = [w for w in waypoints if w["kind"] == "contact"]
        assert contact, "缺 contact 航点"
        final = contact[-1]["position_m"]
        assert math.dist(final, [0.35, 0.1, 0.25]) < 1e-9, final
        approach = [w for w in waypoints if w["kind"] == "approach"]
        assert approach, "缺 approach 航点"
        # approach 在 target 沿法向（+z）上方。
        assert approach[0]["position_m"][2] > final[2]


class TestJourneys:
    def _kernel(self, tmp_path: Path) -> TaskKernel:
        conn = sqlite3.connect(tmp_path / "k.db")
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, "sqlite")
        return TaskKernel(conn, tmp_path)

    def test_vertical_draw_end_to_end(self, tmp_path: Path) -> None:
        """竖直面 draw：plan → rollout → tracking PASS（任意平面闭环）。"""
        service = SimTrajectoryService(tmp_path)
        plan = service.generate_planar_path(
            shape="star5", center_m=[0.35, 0.0, 0.25], scale_m=0.1,
            plane_normal_xyz=[0.0, 1.0, 0.0], plane_offset_m=0.0,
        )
        rollout = service.simulate_cartesian_trajectory(plan["plan_id"])
        assert rollout["ok"] is True
        tracking = rollout["tracking"]
        # 绝对阈值与既有 xy 基线一致（接触段 ~0.02 达成，0.03 上限）。
        assert tracking["max_error_m"] < 0.03, tracking

    def test_reach_end_to_end(self, tmp_path: Path) -> None:
        """arm reach：末端到达目标（最终位置误差 < 5mm）。"""
        service = SimTrajectoryService(tmp_path)
        plan = service.generate_reach_path(target_m=[0.35, 0.1, 0.25])
        rollout = service.simulate_cartesian_trajectory(plan["plan_id"])
        assert rollout["ok"] is True
        # reach 的验收是最终到达精度（不是过渡窗口的瞬时误差）。
        trace = json.loads(
            (tmp_path / "sim" / "traces" / rollout["trace_id"]
             / "trace.json").read_text(encoding="utf-8")
        )
        final = trace["actual"][-1]
        assert math.dist(
            (final["x"], final["y"], final["z"]), (0.35, 0.1, 0.25)
        ) < 0.005, final
