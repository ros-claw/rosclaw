"""PR-SIX-3 红测试（六审 §6）：Body—Capability—Executor 闭环。

红测试先行——以下缺陷修复前必须红：

1. UR5e body 上调 LIMO 动作（required_body_types 不含 sim/ur5e）必须
   在建卡前 BODY_CAPABILITY_MISMATCH——当前 admission 根本不看
   body compatibility（CI journey 就在 sim/ur5e 上执行了
   limo.speaker.play_tone）；
2. PHYSICAL_ACTION 未声明 body scope（required_body_types 为空）必须
   建卡前隔离（BODY_SCOPE_MISSING）；
3. SIM executor 必须按 (body, capability source) 路由——capability
   source 无对应 executor 时 fail closed，不能用全局 _sim_channel
   执行任意物理动作；
4. ExactAction 记录 executor_identity，execute 按它路由；
5. pi.capabilities 只返回当前 body 兼容的动作能力（附排除原因）——
   模型不再靠猜 capability ID。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.agentd.test_pi_approval import _setup_with_operatord
from tests.agentd.test_pi_tool_bridge import (
    _issue_lease,
)


def _register_action(service, tool_id: str, *, source: str, body_types: list[str]):
    from rosclaw.contracts.agent.tool import (
        ExecutionClass,
        ToolDescriptorV2,
        ToolEvidenceClass,
        ToolSideEffectClass,
    )

    service._tool_catalog.register(
        ToolDescriptorV2(
            tool_id=tool_id,
            source=source,
            execution_class=ExecutionClass.PHYSICAL_ACTION,
            description="six-3 测试动作",
            input_schema={"type": "object", "additionalProperties": False},
            supported_modes=["SIMULATION"],
            required_body_types=body_types,
            evidence_class=ToolEvidenceClass.SIMULATED,
            risk_tier="LOW",
            model_callable=False,
            requires_exact_action_grant=True,
            side_effect_class=ToolSideEffectClass.IRREVERSIBLE,
        )
    )


async def _propose(service, mission, capability_id: str, *, idem: str, session="pi_1"):
    from rosclaw.agentd.pi_bridge.action_admission import (
        ActionAdmissionService,
        ActionRequestContext,
    )

    snapshot = service.snapshot(mission.mission_id)
    lease = await _issue_lease(service, mission, session)
    ctx = ActionRequestContext(
        pi_session_id=session,
        mission_id=mission.mission_id,
        context_revision=snapshot.context_revision,
        body_hash=mission.body_binding.effective_body_hash,
        mode=mission.mode.value,
        idempotency_key=idem,
        context_lease_id=lease,
    )
    admission = ActionAdmissionService(service)
    return await admission.propose(
        caller_pid=1, caller_uid=1000,
        request=ctx,
        capability_id=capability_id,
        arguments={},
        expected_effect="six-3",
        risk_tier="LOW",
    )


class TestBodyCapabilityGate:
    async def test_limo_capability_on_ur5e_body_rejected_before_card(
        self, tmp_path: Path
    ) -> None:
        """UR5e body × LIMO 动作 → BODY_CAPABILITY_MISMATCH，零 approval。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        assert mission.body_binding.body_id == "sim/ur5e"
        _register_action(
            service, "limo.speaker.play_tone", source="mcp:limo-sim",
            body_types=["sim/limo"],
        )
        with pytest.raises(ToolBridgeError) as excinfo:
            await _propose(service, mission, "limo.speaker.play_tone", idem="idem_six3_x")
        assert excinfo.value.code == "BODY_CAPABILITY_MISMATCH", (
            f"LIMO 动作在 UR5e body 上竟进入建卡: {excinfo.value.code}"
        )
        assert service.pending_approvals(mission.mission_id) == []
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_physical_action_without_body_scope_quarantined(
        self, tmp_path: Path
    ) -> None:
        """PHYSICAL_ACTION 无 body scope 声明 → BODY_SCOPE_MISSING + 隔离。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        _register_action(
            service, "sim.scopeless.action", source="native:agentd", body_types=[]
        )
        with pytest.raises(ToolBridgeError) as excinfo:
            await _propose(service, mission, "sim.scopeless.action", idem="idem_six3_scope")
        assert excinfo.value.code in ("BODY_SCOPE_MISSING", "CAPABILITY_QUARANTINED")
        assert service._tool_catalog.quarantine_reason("sim.scopeless.action"), (
            "无 body scope 的物理动作未被隔离"
        )
        assert service.pending_approvals(mission.mission_id) == []
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_compatible_action_records_executor_identity(
        self, tmp_path: Path
    ) -> None:
        """body 兼容的动作建卡成功，ExactAction 记录 capability source
        与 executor_identity（执行路由凭据）。"""
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        _register_action(
            service, "test.arm.move_to_pose", source="mcp:test-arm-sim",
            body_types=["sim/ur5e"],
        )
        card = await _propose(service, mission, "test.arm.move_to_pose", idem="idem_six3_ok")
        row = service._store.connection.execute(
            "SELECT request_json FROM operator_requests WHERE request_id = ?",
            (card["approval_id"],),
        ).fetchone()
        exact = json.loads(json.loads(row["request_json"])["exact_action_json"])
        assert exact["capability_source"] == "mcp:test-arm-sim"
        assert exact.get("executor_identity"), (
            "ExactAction 缺 executor_identity——execute 无法按 (body,source) 路由"
        )
        await operatord.stop()
        await agent_server.stop()
        await service.close()


class TestExecutorRouting:
    async def test_capability_without_executor_fails_closed(self, tmp_path: Path) -> None:
        """capability source 无对应 SIM executor → 执行 fail closed
        （EXECUTOR_FOR_BODY_UNAVAILABLE），grant 零消费。"""
        from rosclaw.agentd.operator_socket import operator_call

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        _register_action(
            service, "test.arm.move_to_pose", source="mcp:test-arm-sim",
            body_types=["sim/ur5e"],
        )
        # 不注册任何 mcp:ur5e-sim executor——批准后执行必须 fail closed。
        card = await _propose(service, mission, "test.arm.move_to_pose", idem="idem_six3_route")
        listed = await operator_call(sock, "approvals.list", {"mission_id": mission.mission_id})
        entry = next(a for a in listed["approvals"] if a["request_id"] == card["approval_id"])
        decided = await operator_call(
            sock, "approvals.decide",
            {"request_id": entry["request_id"],
             "display_hash": entry["display_hash"], "approve": True},
        )
        assert decided.get("ok"), decided
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
            ActionRequestContext,
        )

        snapshot = service.snapshot(mission.mission_id)
        lease = await _issue_lease(service, mission, "pi_1")
        ctx = ActionRequestContext(
            pi_session_id="pi_1",
            mission_id=mission.mission_id,
            context_revision=snapshot.context_revision,
            body_hash=mission.body_binding.effective_body_hash,
            mode=mission.mode.value,
            idempotency_key="idem_six3_route",
            context_lease_id=lease,
        )
        outcome = await ActionAdmissionService(service).execute(
            card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000
        )
        assert outcome["executed"] is False, "无 executor 的动作竟执行成功"
        assert outcome.get("error_code") in (
            "EXECUTOR_FOR_BODY_UNAVAILABLE", "ACTION_FAILED"
        )
        row = service._store.connection.execute(
            "SELECT consumed FROM mission_grants WHERE request_id = ?",
            (card["approval_id"],),
        ).fetchone()
        assert row is not None and row["consumed"] == 0, "失败动作竟消费了 grant"
        await operatord.stop()
        await agent_server.stop()
        await service.close()


class TestCapabilitiesSurface:
    async def test_capabilities_lists_only_body_compatible(self, tmp_path: Path) -> None:
        """pi.capabilities：当前 body 的动作能力只含兼容项；不兼容项进
        excluded 并附原因（模型不再靠猜 capability ID）。"""
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        _register_action(
            service, "test.arm.move_to_pose", source="mcp:test-arm-sim",
            body_types=["sim/ur5e"],
        )
        # 七审 PR-SEVEN-2：executor_state 进入 listing——本用例测 body
        # 兼容面，给测试 source 注册一个 executor（否则正确落入
        # excluded[EXECUTOR_FOR_BODY_UNAVAILABLE]）。
        service._sim_executors["mcp:test-arm-sim"] = object()
        _register_action(
            service, "limo.speaker.play_tone", source="mcp:limo-sim",
            body_types=["sim/limo"],
        )
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.capabilities",
            {"token": service.control_token, "mission_id": mission.mission_id},
        )
        assert result.get("ok"), result
        action_ids = [c["capability_id"] for c in result.get("action_capabilities", [])]
        assert "test.arm.move_to_pose" in action_ids, (
            f"body 兼容动作未列出: {action_ids}"
        )
        assert "limo.speaker.play_tone" not in action_ids, (
            "body 不兼容动作竟列为可用——模型会照单调用"
        )
        excluded = {
            e["capability_id"]: e.get("reason") for e in result.get("excluded", [])
        }
        assert excluded.get("limo.speaker.play_tone") == "BODY_CAPABILITY_MISMATCH"
        await operatord.stop()
        await agent_server.stop()
        await service.close()
