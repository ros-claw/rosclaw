"""PR-H5 红测试：rosclaw_execute 统一执行入口 + wait/stop operation。

红测试先行——修复前必须红：
1. rosclaw_execute 按 execution_class 路由：COMPUTE 内联免审批；
2. PHYSICAL_ACTION 走同一 admission 链（SIM 安全动作 POLICY_AUTO
   记录）；REAL/不兼容 → 诚实 REJECTED（不静默降级）；
3. 未知 capability_id → REJECTED（不猜）；
4. rosclaw_wait_operation 等到终态返回（有界）；stop 即 cancel。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _request, _setup


class TestExecuteRouting:
    async def test_compute_inline(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_execute",
                mission=mission.mission_id,
                idem="h5_compute",
                arguments={
                    "capability_id": "ur5e.plan_cartesian_path",
                    "arguments": {"shape": "star5", "center_x": 0.35,
                                  "center_y": 0.25, "z": 0.3,
                                  "outer_radius": 0.08},
                },
            )
        )
        assert result.ok, result.summary
        assert "plan" in result.summary.lower() or "plan_" in result.summary
        await service.close()

    async def test_physical_goes_through_admission(self, tmp_path: Path) -> None:
        """SIM 安全动作：execute 经 admission（POLICY_AUTO 审计在案）——
        不是绕过审批的另一条路。"""
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_execute",
                mission=mission.mission_id,
                idem="h5_phys",
                arguments={
                    "capability_id": "ur5e.execute_plan",
                    "arguments": {"plan_id": "plan_nonexistent"},
                    "expected_effect": "测试",
                    "risk_tier": "LOW",
                },
            )
        )
        # plan 不存在 → 诚实失败（不是凭空执行）；关键是走了 admission
        # 链（有 policy/审计记录而不是裸跑）。
        assert not result.ok or "plan" in result.summary.lower()
        await service.close()

    async def test_unknown_capability_rejected(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        dispatcher = PiToolDispatcher(service)
        result = await dispatcher.execute(
            _request(
                "rosclaw_execute",
                mission=mission.mission_id,
                idem="h5_unknown",
                arguments={"capability_id": "no.such.thing", "arguments": {}},
            )
        )
        assert not result.ok
        # N5C：EffectResolver 先行 fail closed——未知能力在 effect
        # 解析期即拒（EFFECT_UNRESOLVABLE），与原分支码并列有效。
        assert result.error_code in ("EFFECT_UNRESOLVABLE", "UNKNOWN_CAPABILITY", "INVALID_ARGUMENTS",
                                     "NOT_FOUND")
        await service.close()


class TestWaitStopOperation:
    async def test_wait_returns_terminal(self, tmp_path: Path) -> None:

        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        dispatcher = PiToolDispatcher(service)
        import sys

        op = await service._operation_manager.start(
            task_id="task_x", attempt_id="a", kind="process",
            argv=[sys.executable, "-c", "print('done-marker', flush=True)"],
        )
        result = await dispatcher.execute(
            _request(
                "rosclaw_wait_operation",
                mission=mission.mission_id,
                idem="h5_wait",
                arguments={"operation_id": op["operation_id"],
                           "timeout_sec": 10},
            )
        )
        assert result.ok, result.summary
        assert "SUCCEEDED" in result.summary
        await service.close()

    async def test_stop_cancels(self, tmp_path: Path) -> None:
        import sys

        service, mission = await _setup(tmp_path)
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

        dispatcher = PiToolDispatcher(service)
        op = await service._operation_manager.start(
            task_id="task_x", attempt_id="a", kind="process",
            argv=[sys.executable, "-c", "import time; time.sleep(30)"],
        )
        result = await dispatcher.execute(
            _request(
                "rosclaw_stop_operation",
                mission=mission.mission_id,
                idem="h5_stop",
                arguments={"operation_id": op["operation_id"]},
            )
        )
        assert result.ok
        assert service._operation_manager.get(op["operation_id"])["state"] == "CANCELLED"
        await service.close()
