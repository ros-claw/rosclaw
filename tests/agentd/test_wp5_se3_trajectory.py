"""WP-5 红测试（0823 审计冻结序列 §五）：SE(3) PoseTrajectorySpecV1。

红测试先行——位姿轨迹规格不存在时必须红。

1. 规划产出 SE(3) 规格：每个航点 position_m + orientation_xyzw
   （单位四元数）+ kind（transit/approach/contact/lift）——不再是
   纯位置点列；
2. approach/lift 显式建模在规格里（contact 段前有 approach 降下、
   后有 lift 抬升）——不再是 simulate 里的临时拼接；
3. contact_plane + tool_frame 进规格（接触平面法向/工具坐标系是
   契约，不是隐含约定）；
4. 规格随 plan 持久化且 digest 可反查；
5. trace 记录实际朝向（quat_xyzw）——朝向跟踪可验收；
6. metrics 有朝向误差指标（工具轴偏离接触平面法向的角度）；
7. 旧格式 plan（纯 points）向后兼容——默认朝向=工具轴垂直于
   接触平面；
8. verify_tracking 支持朝向阈值（PASS/FAIL 诚实）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


def _make_sim(home: Path):
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService

    return SimTrajectoryService(home)


def _unit(q: list[float], tol: float = 1e-3) -> bool:
    return abs(math.sqrt(sum(v * v for v in q)) - 1.0) < tol


class TestPoseTrajectorySpec:
    def test_plan_carries_se3_spec(self, tmp_path: Path) -> None:
        sim = _make_sim(tmp_path)
        plan = sim.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
        )
        spec = plan.get("spec")
        assert spec is not None, "plan 缺 SE(3) 位姿规格（仍只是纯位置点列）"
        assert spec["schema_version"] == "rosclaw.pose_trajectory_spec.v1"
        assert spec["tool_frame"], "spec 缺 tool_frame"
        plane = spec["contact_plane"]
        assert plane["normal_xyz"] == [0.0, 0.0, 1.0]
        assert plane["offset_m"] == pytest.approx(0.30, abs=1e-6)
        waypoints = spec["waypoints"]
        assert len(waypoints) > 10
        for wp in waypoints:
            assert len(wp["position_m"]) == 3
            assert len(wp["orientation_xyzw"]) == 4
            assert _unit(wp["orientation_xyzw"]), f"非单位四元数: {wp}"
            assert wp["kind"] in ("transit", "approach", "contact", "lift")
        assert spec["digest"].startswith("sha256:")

    def test_spec_validates_against_contract(self, tmp_path: Path) -> None:
        from rosclaw.contracts.agent.pose_trajectory import PoseTrajectorySpecV1

        sim = _make_sim(tmp_path)
        plan = sim.generate_planar_path(
            shape="circle", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
        )
        parsed = PoseTrajectorySpecV1.model_validate(plan["spec"])
        assert parsed.waypoints, "契约解析后无航点"

    def test_approach_lift_explicit_in_spec(self, tmp_path: Path) -> None:
        """approach 降下 → contact 接触 → lift 抬升 是规格的一等
        航点段，不是 simulate 临时拼的。"""
        sim = _make_sim(tmp_path)
        plan = sim.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
        )
        kinds = [wp["kind"] for wp in plan["spec"]["waypoints"]]
        first_contact = kinds.index("contact")
        last_contact = len(kinds) - 1 - kinds[::-1].index("contact")
        assert "approach" in kinds[:first_contact], "contact 前无 approach 段"
        assert "lift" in kinds[last_contact + 1 :], "contact 后无 lift 段"
        # approach 段终点 == contact 起点（同一位置，降下到位）。
        wps = plan["spec"]["waypoints"]
        approach_end = wps[first_contact - 1]["position_m"]
        contact_start = wps[first_contact]["position_m"]
        assert approach_end == pytest.approx(contact_start, abs=1e-9)
        # approach 段起点高于接触平面（抬升高度）。
        approach_start = wps[first_contact - 2]["position_m"] if first_contact >= 2 else None
        if approach_start is not None and wps[first_contact - 2]["kind"] == "approach":
            assert approach_start[2] > contact_start[2] + 0.05

    def test_spec_persisted_in_plan_record(self, tmp_path: Path) -> None:
        sim = _make_sim(tmp_path)
        plan = sim.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
        )
        record = json.loads(
            (tmp_path / "sim" / "plans" / f"{plan['plan_id']}.json").read_text(
                encoding="utf-8"
            )
        )
        assert record.get("spec", {}).get("schema_version") == (
            "rosclaw.pose_trajectory_spec.v1"
        ), "plan 落盘记录缺 SE(3) 规格"
        # digest 可反查：规格内容寻址。
        import hashlib

        canonical = json.dumps(
            record["spec"]["waypoints"], sort_keys=True
        ).encode()
        assert record["spec"]["digest"] == "sha256:" + hashlib.sha256(
            canonical
        ).hexdigest()


class TestSe3Simulation:
    def test_trace_records_actual_orientation(self, tmp_path: Path) -> None:
        sim = _make_sim(tmp_path)
        plan = sim.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
        )
        run = sim.simulate_cartesian_trajectory(plan["plan_id"])
        trace = json.loads(
            Path(run["artifacts"]["trace_json"]).read_text(encoding="utf-8")
        )
        assert trace["actual"], "trace 无实际轨迹"
        for pt in trace["actual"]:
            assert "quat_xyzw" in pt, "trace 缺实际朝向——朝向不可验收"
            assert _unit(pt["quat_xyzw"], tol=1e-2)

    def test_orientation_tracking_metrics(self, tmp_path: Path) -> None:
        """工具轴应保持在接触平面法向附近（画图姿态）——朝向误差
        是一等验收指标。"""
        sim = _make_sim(tmp_path)
        plan = sim.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
        )
        run = sim.simulate_cartesian_trajectory(plan["plan_id"])
        metrics = json.loads(
            Path(run["artifacts"]["metrics_json"]).read_text(encoding="utf-8")
        )
        assert "max_orientation_error_deg" in metrics, (
            "metrics 缺朝向误差指标"
        )
        assert metrics["max_orientation_error_deg"] < 25.0, (
            f"工具轴偏离法向 {metrics['max_orientation_error_deg']}°——"
            "6-DOF IK 未跟踪朝向"
        )

    def test_backward_compat_legacy_plan(self, tmp_path: Path) -> None:
        """旧格式 plan（纯 points + hash）仍可仿真——朝向默认工具轴
        垂直接触平面。"""
        sim = _make_sim(tmp_path)
        plan = sim.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
        )
        # 手工退化记录：删掉 spec，只留旧字段。
        path = tmp_path / "sim" / "plans" / f"{plan['plan_id']}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record.pop("spec", None)
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        run = sim.simulate_cartesian_trajectory(plan["plan_id"])
        assert run["ok"] is True
        trace = json.loads(
            Path(run["artifacts"]["trace_json"]).read_text(encoding="utf-8")
        )
        assert "quat_xyzw" in trace["actual"][0]

    def test_verify_tracking_orientation_threshold(self, tmp_path: Path) -> None:
        sim = _make_sim(tmp_path)
        plan = sim.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.05,
        )
        run = sim.simulate_cartesian_trajectory(plan["plan_id"])
        ok = sim.verify_tracking(
            run["trace_id"],
            max_tracking_error_m=0.05,
            max_orientation_error_deg=30.0,
        )
        assert ok["verdict"] == "PASS", ok
        strict = sim.verify_tracking(
            run["trace_id"],
            max_tracking_error_m=0.05,
            max_orientation_error_deg=1e-6,
        )
        assert strict["verdict"] == "FAIL", "零容差下应诚实 FAIL"
