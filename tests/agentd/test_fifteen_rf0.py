"""十五审 PR-RF-0 红测试：术语/边界冻结 + 失败体验固化（Gate 3/4）。

红测试先行——当前必须红：
1. rosclaw_task_submit 治理工具存在（模型不再自由挑 Worker）；
2. 一个任务只有一个 owning execution——同一 TaskSpec 重复提交返回
   同一执行，绝不裂变出第二 Worker（用户原诉："为什么要开好几次
   worker，还都失败了"）；
3. delegate 的 worker_id 自由选择被废弃（router 决定执行者）。
"""

from __future__ import annotations

from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


class TestGovernanceToolSurface:
    async def test_task_submit_registered(self, tmp_path: Path) -> None:
        """task_submit 在工具表（无为而治：模型只交 TaskSpec 目标合同）。"""
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_task_submit",
                mission=mission.mission_id,
                idem="idem_rf0_submit",
                arguments={
                    "goal": "让 UR5e 在仿真中画五角星并交付动画和误差指标",
                    "required_capabilities": ["simulation.ur5e", "trajectory"],
                    "effects": "simulation_only",
                    "deliverables": [{"type": "image/gif"}],
                    "acceptance": {"max_tracking_error_m": 0.05},
                },
            )
        )
        assert result.ok, result.summary
        assert result.error_code != "TOOL_UNKNOWN"
        await service.close()

    async def test_delegate_worker_id_free_choice_deprecated(
        self, tmp_path: Path
    ) -> None:
        """delegate(worker_id=...) 废弃——ExecutionRouter 按注册表+策略
        决定执行者；模型指定的 worker_id 不再被直接采纳。"""
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_rf0_delegate",
                arguments={"goal": "x", "worker_id": "worker:native:basic"},
            )
        )
        # 新语义：带 worker_id 的 delegate 被废弃拒绝（或明确忽略并
        # 提示 router 决策），绝不静默按模型选择执行。
        assert (
            result.error_code == "DELEGATE_DEPRECATED"
            or "router" in result.summary.lower()
            or "路由" in result.summary
        ), result.summary
        await service.close()


class TestOneTaskOneExecution:
    async def test_duplicate_submit_returns_same_execution(
        self, tmp_path: Path
    ) -> None:
        """Gate 3：同一任务的重复提交返回同一 owning execution——
        不创建第二个 Worker。"""
        from rosclaw.agentd.control_plane import TaskControlPlane

        service, mission = await _setup(tmp_path)
        plane = TaskControlPlane(service)
        spec = {
            "goal": "调研项目结构并写 STRUCTURE.md",
            "required_capabilities": ["repo.analyze"],
            "effects": "workspace_only",
        }
        first = await plane.submit(mission.mission_id, spec, idem="rf0_a")
        second = await plane.submit(mission.mission_id, spec, idem="rf0_b")
        assert first["execution_id"] == second["execution_id"]
        assert second.get("attached") is True
        executions = plane.executions_for(mission.mission_id)
        assert len(executions) == 1, f"裂变出 {len(executions)} 个执行——Gate 3 回归"
        await service.close()

    async def test_verifier_failure_feeds_same_session(
        self, tmp_path: Path
    ) -> None:
        """验收失败的证据反馈回同一 session 修复，不自动新建 Worker。"""
        from rosclaw.agentd.control_plane import TaskControlPlane

        service, mission = await _setup(tmp_path)
        plane = TaskControlPlane(service)
        spec = {"goal": "x", "required_capabilities": [], "effects": "workspace_only"}
        first = await plane.submit(mission.mission_id, spec, idem="rf0_c")
        await plane.report_verifier_failure(
            first["execution_id"], evidence="deliverable missing"
        )
        executions = plane.executions_for(mission.mission_id)
        assert len(executions) == 1
        assert executions[0]["state"] in ("REPAIRING", "RUNNING")
        await service.close()
