"""十四审 PR-14.6 红测试：SIM 动力学闭环正式能力（总纲 §5）。

红测试先行——修复前必须红：
1. trajectory.generate_planar_path：star5/circle 闭合采样 + 工作空间
   校验 + 非法即拒（形状任务只传参数，不再临时开发仿真器）；
2. ur5e.simulate_cartesian_trajectory：真实 MuJoCo 动力学 rollout
   （SIM_DYN_ROLLOUT 证据）——trace 非空 + 跟踪误差指标；
3. simulation.render_trace：实际轨迹 GIF ≥30 帧（可打开的产物）；
4. simulation.verify_tracking：阈值判定 PASS/FAIL 诚实；
5. Task Runner simulate_trajectory：TaskSpec acceptance 驱动验收，
   交付 trace/metrics/GIF——全程 COMPUTE，不需要人工审批；
6. 五角星用例黑盒：用户句子级请求只需参数，不收 WorkOrder。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.agentd.sim_trajectory import SimTrajectoryService
from tests.agentd.test_pi_tool_bridge import _setup


def _ctx(service, mission, idem: str):
    from rosclaw.agentd.pi_bridge.action_admission import ActionRequestContext

    snapshot = service.snapshot(mission.mission_id)
    return ActionRequestContext(
        pi_session_id="pi_test",
        mission_id=mission.mission_id,
        context_revision=snapshot.context_revision,
        body_hash=mission.body_binding.effective_body_hash,
        mode=mission.mode.value,
        idempotency_key=idem,
        context_lease_id="",
    )


@pytest.fixture()
def sim_svc(tmp_path: Path) -> SimTrajectoryService:
    return SimTrajectoryService(tmp_path)


class TestGeneratePlanarPath:
    def test_star5_closed_sampled(self, sim_svc: SimTrajectoryService) -> None:
        plan = sim_svc.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.10,
            plane="xy", max_segment_m=0.02,
        )
        assert plan["ok"] and plan["plan_id"]
        points = plan["points"]
        assert len(points) > 30
        first, last = points[0], points[-1]
        assert abs(first["x"] - last["x"]) < 1e-9
        assert abs(first["y"] - last["y"]) < 1e-9
        assert all(abs(p["z"] - 0.30) < 1e-9 for p in points)

    def test_circle_supported(self, sim_svc: SimTrajectoryService) -> None:
        """圆与五角星走同一组合——不为每个形状写新仿真器。"""
        plan = sim_svc.generate_planar_path(
            shape="circle", center_m=[0.35, 0.25, 0.30], scale_m=0.08,
            plane="xy", max_segment_m=0.02,
        )
        assert plan["ok"]
        import math

        for p in plan["points"][:-1]:
            r = math.hypot(p["x"] - 0.35, p["y"] - 0.25)
            assert abs(r - 0.08) < 1e-6

    def test_invalid_rejected(self, sim_svc: SimTrajectoryService) -> None:
        with pytest.raises(ValueError, match="shape"):
            sim_svc.generate_planar_path(
                shape="mobius", center_m=[0.35, 0.25, 0.30], scale_m=0.1,
                plane="xy", max_segment_m=0.02,
            )
        with pytest.raises(ValueError, match="workspace|radius|range"):
            sim_svc.generate_planar_path(
                shape="star5", center_m=[3.0, 3.0, 0.30], scale_m=0.1,
                plane="xy", max_segment_m=0.02,
            )


class TestSimulateRenderVerify:
    def test_full_dynamics_loop(self, sim_svc: SimTrajectoryService) -> None:
        plan = sim_svc.generate_planar_path(
            shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.10,
            plane="xy", max_segment_m=0.03,
        )
        # 动力学 rollout——真实 MuJoCo 物理（不是命令回放）。
        result = sim_svc.simulate_cartesian_trajectory(plan["plan_id"])
        assert result["ok"], result.get("error")
        assert result["evidence_level"] == "SIM_DYN_ROLLOUT"
        assert result["physics_executed"] is True
        assert result["point_count"] >= 20  # 每个 IK 航点一个物理采样
        assert result["trace_id"]
        metrics = result["tracking"]
        assert metrics["max_error_m"] >= 0.0
        assert metrics["mean_error_m"] >= 0.0
        # 渲染——实际 eef 轨迹 GIF（可打开的产物）。
        render = sim_svc.render_trace(result["trace_id"], format="gif")
        assert render["ok"], render.get("error")
        gif = Path(render["artifact"]["path"])
        assert gif.exists() and gif.stat().st_size > 0
        assert render["artifact"]["frames"] >= 30
        from PIL import Image

        with Image.open(gif) as img:
            assert img.format == "GIF"
        # 验证——宽阈值 PASS；不可能阈值诚实 FAIL。
        passed = sim_svc.verify_tracking(
            result["trace_id"], max_tracking_error_m=0.10
        )
        assert passed["ok"] and passed["verdict"] == "PASS"
        failed = sim_svc.verify_tracking(
            result["trace_id"], max_tracking_error_m=1e-9
        )
        assert failed["ok"] and failed["verdict"] == "FAIL"


class TestTaskRunnerSimulateTrajectory:
    async def test_task_end_to_end(self, tmp_path: Path) -> None:
        """TaskSpec（含 acceptance）→ VERIFIED + trace/metrics/GIF +
        SIM_DYN_ROLLOUT 证据——全 COMPUTE，无人工审批。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.task_runner import TaskRunner

        ctx = _ctx(service, mission, "idem_sim_task_1")
        runner = TaskRunner(service)
        result = await runner.run(
            request_ctx=ctx,
            goal="simulate_trajectory",
            parameters={
                "shape": "star5",
                "center_m": [0.35, 0.25, 0.30],
                "radius_m": 0.10,
                "acceptance": {
                    "trace_nonempty": True,
                    "animation_min_frames": 30,
                    "max_tracking_error_m": 0.10,
                },
            },
            caller_pid=None,
            caller_uid=None,
        )
        assert result["state"] == "VERIFIED", result.get("error")
        assert result["evidence_level"] == "SIM_DYN_ROLLOUT"
        assert result["policy"] == "AUTO_SIM", result
        artifacts = result.get("deliverables") or result.get("artifacts") or []
        names = " ".join(str(a) for a in artifacts)
        assert ".gif" in names, names
        assert "trace" in names
        verification = result.get("verification") or {}
        assert verification.get("verdict") == "PASS"
        await service.close()

    async def test_impossible_acceptance_honest_fail(self, tmp_path: Path) -> None:
        """acceptance 不可能达成 → 诚实 FAIL/INCONCLUSIVE，绝不报喜。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.task_runner import TaskRunner

        ctx = _ctx(service, mission, "idem_sim_task_2")
        runner = TaskRunner(service)
        result = await runner.run(
            request_ctx=ctx,
            goal="simulate_trajectory",
            parameters={
                "shape": "star5",
                "acceptance": {"max_tracking_error_m": 1e-9},
            },
            caller_pid=None,
            caller_uid=None,
        )
        assert result["state"] in ("FAILED", "INCONCLUSIVE"), result
        assert result["state"] != "VERIFIED"
        await service.close()

    async def test_compile_invalid_zero_proposal(self, tmp_path: Path) -> None:
        """非法参数编译期失败（零提案）——不启动任何执行。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.task_runner import TaskRunner

        ctx = _ctx(service, mission, "idem_sim_task_3")
        runner = TaskRunner(service)
        result = await runner.run(
            request_ctx=ctx,
            goal="simulate_trajectory",
            parameters={"shape": "hypercube"},
            caller_pid=None,
            caller_uid=None,
        )
        assert result["state"] == "FAILED"
        assert "shape" in (result.get("error") or "")
        await service.close()
