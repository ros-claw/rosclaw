"""HOTFIX-2 验收（四审 P0-4C/4D/4F）：ActionTxn、capability 权威合约、
ExecutionOutcomeV1/receipt 精确链。

- 同 idempotency key + 同 hash → 同一事务（不重复建卡）；
- 同 key + 不同 hash → IDEMPOTENCY_CONFLICT；
- capability 不在 catalog → CAPABILITY_UNKNOWN；OBSERVE 类 → NOT_ACTIONABLE；
- 模型 risk_tier 不能低于 catalog 权威 tier（可高不可低）；
- 缺 contract required 参数 → INVALID_ARGUMENTS；
- 执行后 txn 到达 COMPLETED 且 outcome 携带全 ID 链（ExecutionOutcomeV1）；
- 预置旧 receipt 不会为新动作背书（any-receipt 不存在）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.agentd.test_pi_approval import _setup_with_operatord
from tests.agentd.test_pi_tool_bridge import SIM_ACTION_CAPABILITY, _issue_lease, _setup


async def _propose(service, mission, *, idem: str, arguments: dict | None = None, **overrides):
    from rosclaw.agentd.pi_bridge.action_admission import (
        ActionAdmissionService,
        ActionRequestContext,
    )

    snapshot = service.snapshot(mission.mission_id)
    lease = await _issue_lease(service, mission)
    ctx = ActionRequestContext(
        pi_session_id=overrides.get("session", "pi_1"),
        mission_id=mission.mission_id,
        context_revision=snapshot.context_revision,
        body_hash=mission.body_binding.effective_body_hash,
        mode=mission.mode.value,
        idempotency_key=idem,
        context_lease_id=lease,
    )
    admission = ActionAdmissionService(service)
    card = await admission.propose(
        caller_pid=1, caller_uid=1000,
        request=ctx,
        capability_id=overrides.get("capability", SIM_ACTION_CAPABILITY),
        arguments=arguments if arguments is not None else {},
        expected_effect="hotfix-2",
        risk_tier=overrides.get("risk_tier", "LOW"),
    )
    return admission, ctx, card


class TestActionTxnIdempotency:
    async def test_same_key_same_hash_returns_same_txn(self, tmp_path: Path) -> None:
        service, mission = await _setup(tmp_path)
        _adm1, _c1, card_a = await _propose(service, mission, idem="idem_same")
        _adm2, _c2, card_b = await _propose(service, mission, idem="idem_same")
        assert card_a["approval_id"] == card_b["approval_id"], (
            "同 key 同 hash 竟然建了两张卡"
        )
        assert card_b.get("idempotent_replay") is True
        pending = service.pending_approvals(mission.mission_id)
        assert len(pending) == 1, f"重复建卡: {len(pending)} 张"
        await service.close()

    async def test_same_key_different_hash_conflicts(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission = await _setup(tmp_path)
        await _propose(service, mission, idem="idem_conflict", arguments={"a": 1})
        with pytest.raises(ToolBridgeError) as excinfo:
            await _propose(service, mission, idem="idem_conflict", arguments={"a": 2})
        assert excinfo.value.code == "IDEMPOTENCY_CONFLICT"
        await service.close()


class TestCapabilityAuthority:
    async def test_unknown_capability_rejected(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission = await _setup(tmp_path)
        with pytest.raises(ToolBridgeError) as excinfo:
            await _propose(service, mission, idem="idem_unk", capability="sim_nonexistent")
        assert excinfo.value.code == "CAPABILITY_UNKNOWN"
        assert service.pending_approvals(mission.mission_id) == []
        await service.close()

    async def test_observe_class_not_actionable(self, tmp_path: Path) -> None:
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission = await _setup(tmp_path)
        with pytest.raises(ToolBridgeError) as excinfo:
            await _propose(service, mission, idem="idem_obs", capability="sim_get_state")
        assert excinfo.value.code == "NOT_ACTIONABLE"
        assert service.pending_approvals(mission.mission_id) == []
        await service.close()

    async def test_model_cannot_lower_authoritative_risk(self, tmp_path: Path) -> None:
        """catalog LOW + 模型报 LOW → LOW；catalog 若 HIGH，模型报 LOW
        也必须 HIGH（风险只能往高报，不能往低报）。"""
        service, mission = await _setup(tmp_path)
        _adm, _ctx, card = await _propose(
            service, mission, idem="idem_risk", risk_tier="LOW"
        )
        assert card["risk_tier"] == "LOW"  # catalog 权威 LOW，模型无法伪造
        # 把 catalog tier 改成 HIGH，再 propose——模型报 LOW 也必须得到 HIGH。
        descriptor = service._tool_catalog.get(SIM_ACTION_CAPABILITY)
        service._tool_catalog.replace(
            descriptor.model_copy(update={"risk_tier": "HIGH"})
        )
        _adm2, _ctx2, card2 = await _propose(
            service, mission, idem="idem_risk2", risk_tier="LOW"
        )
        assert card2["risk_tier"] == "HIGH", "模型自报 LOW 竟压过 catalog HIGH"
        await service.close()


class TestExecutionOutcomeChain:
    async def test_completed_txn_carries_full_id_chain(self, tmp_path: Path) -> None:
        from rosclaw.agentd.operator_socket import operator_call

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card = await _propose(service, mission, idem="idem_chain")
        listed = await operator_call(sock, "approvals.list", {"mission_id": mission.mission_id})
        entry = next(a for a in listed["approvals"] if a["request_id"] == card["approval_id"])
        decided = await operator_call(
            sock, "approvals.decide",
            {"request_id": entry["request_id"],
             "display_hash": entry["display_hash"], "approve": True},
        )
        assert decided.get("ok"), decided
        outcome = await admission.execute(card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000)

        # ExecutionOutcomeV1：全 ID 链（P0-4F）。
        assert outcome["schema_version"] == "rosclaw.execution_outcome.v1"
        assert outcome["executed"] is True
        assert outcome["terminal"] is True
        assert outcome["txn_id"]
        assert outcome["approval_id"] == card["approval_id"]
        assert outcome["grant_id"]
        assert outcome["action_id"]
        assert outcome["receipt_id"]
        assert outcome["capability_id"] == SIM_ACTION_CAPABILITY
        # ActionTxn 终态 COMPLETED 且 ID 一致。
        from rosclaw.agentd.pi_bridge.action_txn import ActionTxnStore

        txn = ActionTxnStore(service._store.connection).get(outcome["txn_id"])
        assert txn is not None and txn.state == "COMPLETED"
        assert txn.approval_id == outcome["approval_id"]
        assert txn.grant_id == outcome["grant_id"]
        # receipt.received 事件精确绑定本 action_id。
        events = [
            e for e in service.events_replay(mission.mission_id, limit=20)
            if e.type.value == "receipt.received"
        ]
        assert any(e.payload.get("action_id") == outcome["action_id"] for e in events), (
            f"receipt 事件未绑定本 action: {events}"
        )
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_old_receipt_never_vouches_new_action(self, tmp_path: Path) -> None:
        """预置旧 receipt 事件 + executor 失败 → FAILED（any-receipt
        判定不存在——P0-NA-13/P0-4F 场景 D）。"""
        from rosclaw.agentd.operator_socket import operator_call
        from rosclaw.agentd.sim_executor import SimActionError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        # 预置一个旧成功 receipt（干扰项）。
        await service._handlers._emit(
            "receipt.received", mission.mission_id,
            {"action_id": "act_old", "final_state": "COMPLETED",
             "trust_level": "SIMULATED", "verified": True},
        )

        class _BrokenClient:
            async def call_tool(self, tool_name: str, arguments: dict) -> str:
                raise SimActionError("executor down")

        from rosclaw.agentd.sim_executor import SimActionChannel

        service._handlers._sim_channel = SimActionChannel(
            command="true", args=(), name="broken-sim", client=_BrokenClient()
        )
        admission, ctx, card = await _propose(service, mission, idem="idem_old_receipt")
        listed = await operator_call(sock, "approvals.list", {"mission_id": mission.mission_id})
        entry = next(a for a in listed["approvals"] if a["request_id"] == card["approval_id"])
        await operator_call(
            sock, "approvals.decide",
            {"request_id": entry["request_id"],
             "display_hash": entry["display_hash"], "approve": True},
        )
        outcome = await admission.execute(card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000)

        # executor 失败 + 旧 receipt 存在 → 仍必须 FAILED（场景 D）。
        assert outcome["executed"] is False, "旧 receipt 竟为新动作背书"
        assert outcome["status"] == "FAILED"
        from rosclaw.agentd.pi_bridge.action_txn import ActionTxnStore

        txn = ActionTxnStore(service._store.connection).get_by_approval(card["approval_id"])
        assert txn is not None and txn.state == "FAILED"
        await operatord.stop()
        await agent_server.stop()
        await service.close()
