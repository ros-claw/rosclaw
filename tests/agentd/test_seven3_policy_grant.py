"""PR-SEVEN-3 红测试（七审 §2.5）：SIM Policy Auto-Grant。

红测试先行——当前 SIM 的每个动作都要人工审批卡（本地内存里移动
一个虚拟点也要先登记独立授权身份）。策略：

- developer + SIMULATION + 第一方 kit + SIMULATION_STATE_ONLY 效果
  → POLICY_AUTO 自动 grant（全链绑定 + 单次消费 + SIM receipt）；
- host/network/persistent/cost 效果即卡（SIM 也不例外）；
- /safety sim ask-every-time 恢复逐卡；
- REAL 永远人工（policy grant 不得跨域）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.agentd.test_pi_tool_bridge import _issue_lease, _setup


async def _propose(service, mission, capability_id: str, *, idem: str, args=None):
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
        idempotency_key=idem,
        context_lease_id=lease,
    )
    card = await ActionAdmissionService(service).propose(
        request=ctx,
        capability_id=capability_id,
        arguments=args if args is not None else {},
        expected_effect="seven-3",
        risk_tier="LOW",
        caller_pid=1,
        caller_uid=1000,
    )
    return ctx, card


class TestPolicyAutoGrant:
    async def test_safe_sim_action_auto_grants_without_operator(
        self, tmp_path: Path
    ) -> None:
        """安全 SIM 动作（UR5e kit 第一方、sim-only 效果）：无 operatord、
        无人工卡——propose 后即 APPROVED，execute 直接产出 receipt。"""
        from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService

        service, mission = await _setup(tmp_path)  # 无 operatord
        ctx, card = await _propose(
            service, mission, "ur5e.move_to_pose",
            idem="idem_seven3_auto",
            args={"x": 0.35, "y": 0.25, "z": 0.45},
        )
        # 政策自动批准——不等待人工。
        status = ActionAdmissionService(service).decision_status(card["approval_id"])
        assert status["status"] == "APPROVED", (
            f"安全 SIM 动作竟在等人工: {status}"
        )
        outcome = await ActionAdmissionService(service).execute(
            card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000
        )
        assert outcome["executed"] is True, outcome
        assert outcome["receipt_id"], "缺 receipt"
        # 审计链：decision authority 记录为 POLICY_AUTO。
        row = service._store.connection.execute(
            "SELECT status, decided_by FROM operator_requests WHERE request_id = ?",
            (card["approval_id"],),
        ).fetchone()
        assert row["status"] == "APPROVED"
        assert "POLICY_AUTO" in str(row["decided_by"]), (
            f"decided_by 未记录政策授权: {row['decided_by']}"
        )
        await service.close()

    async def test_host_mutation_effect_still_needs_human(
        self, tmp_path: Path
    ) -> None:
        """host 副作用能力即使在 SIM 也不得自动放行——卡保持 PENDING。"""
        from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService
        from rosclaw.contracts.agent.tool import (
            ExecutionClass,
            ToolDescriptorV2,
            ToolEvidenceClass,
            ToolSideEffectClass,
        )

        service, mission = await _setup(tmp_path)
        service._tool_catalog.register(
            ToolDescriptorV2(
                tool_id="sim.host.write_file",
                source="native:agentd",
                execution_class=ExecutionClass.PHYSICAL_ACTION,
                description="写宿主文件的 SIM 动作（必须人工）。",
                input_schema={"type": "object", "additionalProperties": False},
                supported_modes=["SIMULATION"],
                required_body_types=["sim/ur5e"],
                evidence_class=ToolEvidenceClass.SIMULATED,
                risk_tier="LOW",
                model_callable=False,
                requires_exact_action_grant=True,
                side_effect_class=ToolSideEffectClass.IRREVERSIBLE,
                effect_domain="HOST_MUTATION",
            )
        )
        _ctx, card = await _propose(
            service, mission, "sim.host.write_file", idem="idem_seven3_host"
        )
        status = ActionAdmissionService(service).decision_status(card["approval_id"])
        assert status["status"] == "PENDING", (
            f"host 副作用竟自动批准: {status}"
        )
        await service.close()

    async def test_ask_every_time_requires_operator(self, tmp_path: Path) -> None:
        """用户开启 ask-every-time 后，安全 SIM 也要人工卡。"""
        from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService

        service, mission = await _setup(tmp_path)
        # /safety sim ask-every-time 的持久化文件。
        safety = tmp_path / "agent" / "safety.json"
        safety.parent.mkdir(parents=True, exist_ok=True)
        safety.write_text(json.dumps({"sim_policy": "ask"}), encoding="utf-8")
        _ctx, card = await _propose(
            service, mission, "ur5e.move_to_pose",
            idem="idem_seven3_ask",
            args={"x": 0.35, "y": 0.25, "z": 0.45},
        )
        status = ActionAdmissionService(service).decision_status(card["approval_id"])
        assert status["status"] == "PENDING", (
            f"ask-every-time 下竟自动批准: {status}"
        )
        await service.close()

    async def test_policy_grant_single_consumption(self, tmp_path: Path) -> None:
        """POLICY_AUTO 的 grant 同样单次消费——第二次 execute 拒绝。"""
        from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission = await _setup(tmp_path)
        ctx, card = await _propose(
            service, mission, "ur5e.move_to_pose",
            idem="idem_seven3_once",
            args={"x": 0.35, "y": 0.25, "z": 0.45},
        )
        admission = ActionAdmissionService(service)
        first = await admission.execute(
            card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000
        )
        assert first["executed"] is True
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.execute(
                card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000
            )
        assert excinfo.value.code in (
            "GRANT_CONSUMED", "TXN_STATE_INVALID", "TXN_TERMINAL"
        )
        await service.close()

    async def test_real_mode_never_policy_auto(self, tmp_path: Path) -> None:
        """REAL/SHADOW 不得政策自动授权——policy 判定函数结构性拒绝。"""
        from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService

        service, mission = await _setup(tmp_path)
        admission = ActionAdmissionService(service)
        # 直接验证判定函数：非 SIMULATION 一律 False。
        from types import SimpleNamespace

        real_mission = SimpleNamespace(mode=SimpleNamespace(value="REAL"))
        assert admission._policy_auto_applies(real_mission, None) is False
        shadow_mission = SimpleNamespace(mode=SimpleNamespace(value="SHADOW"))
        assert admission._policy_auto_applies(shadow_mission, None) is False
        await service.close()
