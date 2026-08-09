"""PR-HF5-3 红测试（五审 P0-5D/5E）：ActionTxnV2 + ReceiptV2 + reconcile。

红测试先行：

1. execute 必须要求 ActionTxn 存在（legacy 卡拒绝执行）；
2. 跨 Mission 复用：B 的 fresh context 不得执行 A 的已批准卡
   （同 revision 也不行）；
3. 全链一致性：request/approval/grant/txn 的 session/mission/body/
   mode/capability/args_hash 任一不符即拒；
4. CAS transition：并发 execute 只能 dispatch 一次；
5. txn 过期不能执行；
6. terminal_receipt=True 但无 receipt 对象 → 不是 COMPLETED；
7. receipt 的 action_id/capability/args_hash/grant 与 txn 不符 →
   FAILED/RECONCILE_REQUIRED；
8. SIM executor 对 ok=false/driver=failed 的 domain 失败不报
   COMPLETED；
9. tool 结果透出完整脱敏 outcome 链（txn/action/receipt/capability）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.agentd.test_pi_approval import _setup_with_operatord
from tests.agentd.test_pi_tool_bridge import (
    SIM_ACTION_CAPABILITY,
    _issue_lease,
)


async def _propose(service, mission, *, idem: str, session="pi_1", arguments=None):
    from rosclaw.agentd.pi_bridge.action_admission import (
        ActionAdmissionService,
        ActionRequestContext,
    )

    snapshot = service.snapshot(mission.mission_id)
    lease = _issue_lease(service, mission, session)
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
    card = await admission.propose(
        request=ctx,
        capability_id=SIM_ACTION_CAPABILITY,
        arguments=arguments if arguments is not None else {},
        expected_effect="hf5-3",
        risk_tier="LOW",
    )
    return admission, ctx, card


async def _approve(service, mission, sock, approval_id):
    from rosclaw.agentd.operator_socket import operator_call

    listed = await operator_call(sock, "approvals.list", {"mission_id": mission.mission_id})
    entry = next(a for a in listed["approvals"] if a["request_id"] == approval_id)
    decided = await operator_call(
        sock, "approvals.decide",
        {"request_id": entry["request_id"],
         "display_hash": entry["display_hash"], "approve": True},
    )
    assert decided.get("ok"), decided


class TestChainConsistency:
    async def test_execute_requires_action_txn(self, tmp_path: Path) -> None:
        """legacy 卡（无 ActionTxn）明确拒绝执行——不返回空 txn_id。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card = await _propose(service, mission, idem="idem_legacy")
        await _approve(service, mission, sock, card["approval_id"])
        # 删掉 txn（模拟 legacy 卡）——execute 必须拒绝。
        txn_row = service._store.connection.execute(
            "SELECT txn_id FROM action_txns WHERE approval_id = ?",
            (card["approval_id"],),
        ).fetchone()
        assert txn_row is not None
        service._store.connection.execute(
            "DELETE FROM action_txns WHERE txn_id = ?", (txn_row["txn_id"],)
        )
        service._store.connection.commit()
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.execute(card["approval_id"], request=ctx)
        assert excinfo.value.code in ("TXN_REQUIRED", "LEGACY_TXN_UNEXECUTABLE")
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_cross_mission_approval_rejected_same_revision(
        self, tmp_path: Path
    ) -> None:
        """B 的 fresh context 不得执行 A 的已批准卡（同 revision 也不行）。"""
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionRequestContext,
        )
        from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, _ctx_a, card_a = await _propose(service, mission, idem="idem_a")
        await _approve(service, mission, sock, card_a["approval_id"])
        # Mission B：独立 session pi_2 合法 writer + fresh context + 同 revision。
        mission_b = service.create_mission("mission-b", mode="SIMULATION")
        bindings = SessionBindingStore(service._store.connection)
        bindings.bind(
            pi_session_id="pi_2", pi_session_path="", mission_id=mission_b.mission_id,
            body_id="sim/ur5e", execution_mode="SIMULATION",
            created_by="user:local:1000",
        )
        bindings.acquire_lease(
            mission_id=mission_b.mission_id, pi_session_id="pi_2",
            owner_pid=2, owner_uid=1000,
        )
        snapshot_b = service.snapshot(mission_b.mission_id)
        lease_b = _issue_lease(service, mission_b, "pi_2")
        ctx_b = ActionRequestContext(
            pi_session_id="pi_2",
            mission_id=mission_b.mission_id,
            context_revision=snapshot_b.context_revision,
            body_hash=mission_b.body_binding.effective_body_hash,
            mode=mission_b.mode.value,
            idempotency_key="idem_b",
            context_lease_id=lease_b,
        )
        # 用 B 的上下文执行 A 的卡——必须拒绝（grant 不消费）。
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.execute(card_a["approval_id"], request=ctx_b)
        assert excinfo.value.code in (
            "CHAIN_MISMATCH",
            "MISSION_MISMATCH",
            "CONTEXT_REVISION_MISMATCH",
        )
        row = service._store.connection.execute(
            "SELECT consumed FROM mission_grants WHERE request_id = ?",
            (card_a["approval_id"],),
        ).fetchone()
        assert row is not None and row["consumed"] == 0, "A 的 grant 竟被 B 消费"
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_expired_txn_cannot_execute(self, tmp_path: Path) -> None:
        """txn 过期（expires_at 已过）→ 拒绝执行。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card = await _propose(service, mission, idem="idem_exp")
        await _approve(service, mission, sock, card["approval_id"])
        # 把 txn 的 expires_at 改到过去。
        service._store.connection.execute(
            "UPDATE action_txns SET expires_at = ? WHERE approval_id = ?",
            ("2000-01-01T00:00:00+00:00", card["approval_id"]),
        )
        service._store.connection.commit()
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.execute(card["approval_id"], request=ctx)
        assert excinfo.value.code in ("TXN_EXPIRED", "GRANT_EXPIRED")
        await operatord.stop()
        await agent_server.stop()
        await service.close()


class TestReceiptContract:
    async def test_terminal_bool_without_receipt_is_not_completed(
        self, tmp_path: Path
    ) -> None:
        """terminal_receipt=True 但缺 receipt 对象/ID → 不得 COMPLETED。"""

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card = await _propose(service, mission, idem="idem_noreceipt")

        # 让 handler 返回 terminal_receipt=True 但 evidence_ref 为空
        # （模拟缺 receipt 的路径）。
        original = service._handlers.request_action

        async def _broken(decision):
            outcome = await original(decision)
            outcome.terminal_receipt = True
            outcome.evidence_ref = None  # 缺 receipt
            return outcome

        service._handlers.request_action = _broken
        await _approve(service, mission, sock, card["approval_id"])
        outcome = await admission.execute(card["approval_id"], request=ctx)
        assert outcome["status"] != "COMPLETED", (
            "terminal_receipt=True 但无 receipt 竟 COMPLETED"
        )
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_mcp_domain_failure_not_completed(self, tmp_path: Path) -> None:
        """SIM executor 对 ok=false/driver=failed 的 domain 失败不得报
        COMPLETED（transport 没错≠动作成功）。"""
        from rosclaw.agentd.sim_executor import SimActionChannel

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )

        class _DomainFailClient:
            async def call_tool(self, tool_name: str, arguments: dict) -> str:
                import json

                return json.dumps({"ok": False, "driver": "failed"})

        service._handlers._sim_channel = SimActionChannel(
            command="true", args=(), name="fail-sim", client=_DomainFailClient()
        )
        admission, ctx, card = await _propose(service, mission, idem="idem_domfail")
        await _approve(service, mission, sock, card["approval_id"])
        outcome = await admission.execute(card["approval_id"], request=ctx)
        assert outcome["status"] != "COMPLETED", (
            "domain ok=false 竟报 COMPLETED"
        )
        assert outcome["executed"] is False
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_outcome_carries_full_sanitized_chain(self, tmp_path: Path) -> None:
        """ExecutionOutcome 必须透出完整脱敏 ID 链（txn/action/receipt/
        capability/verified）——不是 bool+文本摘要。"""

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card = await _propose(service, mission, idem="idem_full")
        await _approve(service, mission, sock, card["approval_id"])
        outcome = await admission.execute(card["approval_id"], request=ctx)
        assert outcome["executed"] is True
        # 全链字段非空。
        for field in ("txn_id", "approval_id", "grant_id", "action_id",
                      "receipt_id", "capability_id"):
            assert outcome.get(field), f"outcome 缺 {field}"
        # action_id 与 receipt_id 是独立身份（不再 receipt_id = action_id）。
        assert outcome["receipt_id"], "缺独立 receipt_id"
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_receipt_id_independent_from_action_id(self, tmp_path: Path) -> None:
        """P0-5E 独立身份：receipt_id 不得等于 action_id——receipt 是
        独立证据对象（有自己的 ID），evidence_ref 绑定 receipt_id，
        receipt.received 事件同时携带两个 ID。"""

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card = await _propose(service, mission, idem="idem_rcpt_id")
        await _approve(service, mission, sock, card["approval_id"])
        outcome = await admission.execute(card["approval_id"], request=ctx)
        assert outcome["executed"] is True
        action_id = outcome["action_id"]
        receipt_id = outcome["receipt_id"]
        assert receipt_id != action_id, (
            f"receipt_id 竟等于 action_id（{receipt_id}）——receipt 不是独立证据对象"
        )
        # evidence_ref 绑定 receipt_id（不是 action_id）。
        assert outcome["evidence_ref"] == f"receipt://{receipt_id}"
        # 事件同时携带两个 ID 且与本动作精确一致。
        events = service.events_replay(mission.mission_id, limit=50)
        matched = [
            e for e in events
            if e.type.value == "receipt.received"
            and e.payload.get("receipt_id") == receipt_id
        ]
        assert matched, "receipt.received 事件缺独立 receipt_id"
        assert matched[0].payload.get("action_id") == action_id
        await operatord.stop()
        await agent_server.stop()
        await service.close()
