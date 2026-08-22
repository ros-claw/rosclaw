"""PR-N7 红测试（调整方案 §六）：按副作用管理安全。

红测试先行——330s 轮询/lease 全域门不存在替代前必须红。

1. `_request_action()` 不再在模型工具中轮询 330 秒：创建审批后立即
   返回 WAITING_APPROVAL + approval_id；审批可安全过期，但不卡住
   Harness 回合；
2. Context Lease 只约束依赖真实身体状态的动作：SIM 域
   （simulation_state）物理动作不再需要真实 context lease；REAL 域
   照旧硬要求；
3. 审批后携带 approval_id 再次调用即执行（审批事件恢复回合后，
   模型用 approval_id 续接，不重新建卡轮询）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path


async def _request_action(service, mission, *, idem: str, extra_args=None):
    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
    from tests.agentd.test_pi_tool_bridge import _request

    return await PiToolDispatcher(service).execute(
        caller_pid=1,
        caller_uid=1000,
        request=_request(
            "rosclaw_request_action",
            mission=mission.mission_id,
            idem=idem,
            arguments={
                "capability_id": "ur5e.move_joints",
                "arguments": {"joints": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]},
                "risk_tier": "LOW",
                **(extra_args or {}),
            },
        ),
    )


class TestNoPollingInModelTool:
    async def test_waiting_approval_returns_immediately(
        self, tmp_path: Path
    ) -> None:
        """SIM ask 策略（用户开了 ask-every-time）：创建审批卡后立即
        返回 WAITING_APPROVAL——不再轮询 330s 卡死回合。"""
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        # 开 ask-every-time（POLICY_AUTO 关闭 → 需要人工）。
        safety = service._home / "agent" / "safety.json"
        safety.parent.mkdir(parents=True, exist_ok=True)
        safety.write_text(json.dumps({"sim_policy": "ask"}), encoding="utf-8")
        started = time.monotonic()
        result = await _request_action(service, mission, idem="n7_1")
        elapsed = time.monotonic() - started
        assert elapsed < 10, f"审批等待卡了 {elapsed:.1f}s（330s 轮询未消除）"
        assert result.status == "WAITING_APPROVAL", result
        assert result.approval_id, result
        await service.close()


class TestLeaseOnlyForRealBodyState:
    async def test_sim_domain_action_needs_no_real_context_lease(
        self, tmp_path: Path
    ) -> None:
        """SIM 域（simulation_state）动作不要求真实 context lease——
        本地 MuJoCo 仿真不要求真实 body freshness。"""
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        # 无 context_lease_id——当前被 CONTEXT_LEASE_REQUIRED 硬拒；
        # N7：sim-only 效果域豁免（SIM 任务不被 REAL lease 阻挡）。
        result = await _request_action(service, mission, idem="n7_2")
        assert result.error_code != "CONTEXT_LEASE_REQUIRED", (
            f"SIM 域动作仍被真实 context lease 阻挡: {result.error_code}"
        )
        # ask 默认关 → POLICY_AUTO 自动执行（SIM 安全动作）。
        assert result.ok, result
        await service.close()

    async def test_real_body_domain_still_requires_lease(
        self, tmp_path: Path
    ) -> None:
        """REAL 域（physical_body）动作的 lease 要求不变——fail
        closed 不动摇。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
        from rosclaw.agentd.tooling.descriptor import physical_action_descriptor
        from tests.agentd.test_pi_tool_bridge import _request, _setup

        service, mission = await _setup(tmp_path)
        # 注册一个 REAL 域能力（无 SIMULATION_STATE_ONLY 标记）。
        service._tool_catalog.register(physical_action_descriptor(
            "realbody.move",
            source="mcp:realbody",
            supported_modes=["SIMULATION", "SHADOW", "REAL"],
            required_body_types=["sim/ur5e"],
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "additionalProperties": False,
            },
        ))
        result = await PiToolDispatcher(service).execute(
            caller_pid=1, caller_uid=1000,
            request=_request(
                "rosclaw_request_action",
                mission=mission.mission_id,
                idem="n7_3",
                arguments={
                    "capability_id": "realbody.move",
                    "arguments": {"x": 1.0},
                    "risk_tier": "LOW",
                },
            ),
        )
        assert not result.ok
        assert result.error_code in (
            "CONTEXT_LEASE_REQUIRED", "CONTEXT_NOT_FRESH",
        ), result.error_code
        await service.close()


class TestApprovalResume:
    async def test_reinvoke_with_approval_id_executes(
        self, tmp_path: Path
    ) -> None:
        """审批决定后：模型携带 approval_id 再次调用即执行
        （审批事件恢复回合的续接路径——不重建卡、不轮询）。"""
        from tests.agentd.test_pi_tool_bridge import _setup

        service, mission = await _setup(tmp_path)
        safety = service._home / "agent" / "safety.json"
        safety.parent.mkdir(parents=True, exist_ok=True)
        safety.write_text(json.dumps({"sim_policy": "ask"}), encoding="utf-8")
        first = await _request_action(service, mission, idem="n7_4a")
        assert first.status == "WAITING_APPROVAL"
        approval_id = first.approval_id
        # operator 批准（broker 决定——与 operatord 同路径）。
        grant = service._broker.decide(
            approval_id, principal=mission.owner_principal, approve=True,
            decided_by="user:local:1000",
        )
        assert grant is not None
        # 模型携带 approval_id 续接执行。
        second = await _request_action(
            service, mission, idem="n7_4b",
            extra_args={"approval_id": approval_id},
        )
        assert second.ok, second
        assert second.status in ("COMPLETED", "EXECUTED"), second
        await service.close()
