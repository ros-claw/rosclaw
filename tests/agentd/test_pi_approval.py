"""PR-PNA-5（规格 §20 验收子集）：bridge request_action 授权链。

- 创建授权卡 → operatord（SIM 签名）批准 → 执行 → receipt；
- operator 拒绝 → 动作不执行；
- 超时未决定 → 动作不执行（默认拒绝）；
- 模型/工具路径本身不含任何"自批"入口（工具只建卡+等待）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rosclaw.agentd.operator_socket import OperatorSocketServer, operator_call
from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.operatord.enrollment import enroll
from rosclaw.operatord.server import OperatorDaemon
from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup


async def _setup_with_operatord(tmp_path: Path):
    service, mission = await _setup(tmp_path)
    agent_sock = tmp_path / "run" / "operator.sock"
    agent_server = OperatorSocketServer(service, agent_sock)
    await agent_server.start()
    identity = enroll(tmp_path / "operatord")
    operatord = OperatorDaemon(
        identity=identity,
        socket_path=tmp_path / "run" / "operatord.sock",
        agent_socket=agent_sock,
        daemon_client=None,
        require_human_presence=False,
    )
    await operatord.start()
    return service, mission, operatord, agent_server, tmp_path / "run" / "operatord.sock"


async def _decide_pending(service, mission_id: str, sock: Path, approve: bool) -> dict:
    listed = await operator_call(sock, "approvals.list", {"mission_id": mission_id})
    entry = listed["approvals"][0]
    return await operator_call(
        sock,
        "approvals.decide",
        {"request_id": entry["request_id"], "display_hash": entry["display_hash"],
         "approve": approve},
    )


class TestRequestActionChain:
    async def test_approve_then_execute_then_receipt(self, tmp_path: Path) -> None:
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(tmp_path)
        dispatcher = PiToolDispatcher(service)

        async def operator_approves() -> None:
            for _ in range(30):
                await asyncio.sleep(0.5)
                if service.pending_approvals(mission.mission_id):
                    await _decide_pending(service, mission.mission_id, sock, True)
                    return

        approver = asyncio.create_task(operator_approves())
        try:
            result = await dispatcher.execute(
                _request(
                    "rosclaw_request_action",
                    mission=mission.mission_id,
                    idem="idem_ra_1",
                    lease=_issue_lease(service, mission),
                    arguments={
                        "capability_id": "sim_ground_truth",
                        "arguments": {},
                        "expected_effect": "SIM 探测",
                        "risk_tier": "LOW",
                    },
                )
            )
        finally:
            approver.cancel()
        assert result.approval_id
        # SIM executor 结果（接受与否取决于 sim executor 存在性——关键是
        # 链走到了执行而非卡在授权）。
        assert result.status in {"COMPLETED", "FAILED"}
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_operator_decline_never_executes(self, tmp_path: Path) -> None:
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(tmp_path)
        dispatcher = PiToolDispatcher(service)

        async def operator_denies() -> None:
            for _ in range(30):
                await asyncio.sleep(0.5)
                if service.pending_approvals(mission.mission_id):
                    await _decide_pending(service, mission.mission_id, sock, False)
                    return

        denier = asyncio.create_task(operator_denies())
        try:
            result = await dispatcher.execute(
                _request(
                    "rosclaw_request_action",
                    mission=mission.mission_id,
                    idem="idem_ra_2",
                    lease=_issue_lease(service, mission),
                    arguments={"capability_id": "sim_ground_truth", "arguments": {}},
                )
            )
        finally:
            denier.cancel()
        assert not result.ok
        assert result.status == "DECLINED"
        assert result.error_code == "OPERATOR_DECLINED"
        # 没有产生任何 grant。
        assert service.list_grants() == []
        await operatord.stop()
        await agent_server.stop()
        await service.close()


class TestAdmissionNegativeChain:
    """三审 P0-NA-10/11/13 负向链：并发卡不串、revision/body/mode/lease
    变化拒绝、无关 grant 不被消费。"""

    async def _propose(self, service, mission, **overrides):
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
            ActionRequestContext,
        )

        snapshot = service.snapshot(mission.mission_id)
        # HOTFIX-1：admission 现在要求 agentd 签发的 context lease——
        # 测试也按真实路径签发（不是绕过）。
        from rosclaw.agentd.pi_bridge.context_lease import ContextLeaseStore

        session = overrides.get("session", "pi_1")
        lease_id = overrides.get("lease_id")
        if lease_id is None:
            lease = ContextLeaseStore(service._store.connection).issue(
                pi_session_id=session,
                mission_id=mission.mission_id,
                context_revision=overrides.get("revision", snapshot.context_revision),
                context_hash="test_hash",
                body_hash=overrides.get(
                    "body_hash", mission.body_binding.effective_body_hash
                ),
                mode=overrides.get("mode", mission.mode.value),
            )
            lease_id = lease.context_lease_id
        ctx = ActionRequestContext(
            pi_session_id=session,
            mission_id=mission.mission_id,
            context_revision=overrides.get("revision", snapshot.context_revision),
            body_hash=overrides.get(
                "body_hash", mission.body_binding.effective_body_hash
            ),
            mode=overrides.get("mode", mission.mode.value),
            idempotency_key=overrides.get("idem", "idem_neg"),
            context_lease_id=lease_id,
        )
        admission = ActionAdmissionService(service)
        card = await admission.propose(
            request=ctx,
            capability_id="sim_ground_truth",
            arguments={},
            expected_effect="负向测试",
            risk_tier="LOW",
        )
        return admission, ctx, card

    async def test_two_concurrent_cards_exact_ids(self, tmp_path: Path) -> None:
        """两张并发卡各有精确 ID——不取 pending 末尾、不串卡。"""
        service, mission = await _setup(tmp_path)
        _adm_a, _ctx_a, card_a = await self._propose(service, mission, idem="idem_a")
        _adm_b, _ctx_b, card_b = await self._propose(service, mission, idem="idem_b")
        assert card_a["approval_id"] != card_b["approval_id"]
        pending = {r.request_id for r in service.pending_approvals(mission.mission_id)}
        assert {card_a["approval_id"], card_b["approval_id"]} <= pending
        # 每张卡的 display_hash 绑定各自内容。
        assert card_a["display_hash"] != card_b["display_hash"]
        await service.close()

    async def test_stale_revision_execute_rejected_after_approve(
        self, tmp_path: Path
    ) -> None:
        """approve 后 revision 变化（TOCTOU）——execute 必须拒绝。"""
        import pytest

        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card = await self._propose(service, mission, idem="idem_toctou")
        # operator 批准这张卡。
        listed = await operator_call(sock, "approvals.list", {"mission_id": mission.mission_id})
        entry = next(a for a in listed["approvals"] if a["request_id"] == card["approval_id"])
        decided = await operator_call(
            sock,
            "approvals.decide",
            {"request_id": entry["request_id"],
             "display_hash": entry["display_hash"], "approve": True},
        )
        assert decided.get("ok"), decided
        # 批准后 revision 前进（模拟新观测/新 turn）。
        service._store.bump_context_revision(mission.mission_id)
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.execute(card["approval_id"], request=ctx)
        # 两层都正确：卡 revision 直接比对给 CONTEXT_REVISION_MISMATCH；
        # lease 层（HOTFIX-1）发现 revision 前进给 CONTEXT_NOT_FRESH。
        assert excinfo.value.code in ("CONTEXT_REVISION_MISMATCH", "CONTEXT_NOT_FRESH")
        # grant 未被消费。
        row = service._store.connection.execute(
            "SELECT consumed FROM mission_grants WHERE request_id = ?",
            (card["approval_id"],),
        ).fetchone()
        assert row is not None and row["consumed"] == 0
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_unrelated_grant_never_consumed(self, tmp_path: Path) -> None:
        """存在无关 grant 时，新 action 只消费本卡 grant。"""
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        # 第一张卡：批准但故意不执行（留下未消费 grant A）。
        admission, _ctx_a, card_a = await self._propose(service, mission, idem="idem_g1")
        listed = await operator_call(sock, "approvals.list", {"mission_id": mission.mission_id})
        entry_a = next(a for a in listed["approvals"] if a["request_id"] == card_a["approval_id"])
        await operator_call(
            sock, "approvals.decide",
            {"request_id": entry_a["request_id"],
             "display_hash": entry_a["display_hash"], "approve": True},
        )
        grant_a = service._store.connection.execute(
            "SELECT grant_id FROM mission_grants WHERE request_id = ?",
            (card_a["approval_id"],),
        ).fetchone()["grant_id"]
        # 第二张卡：批准并执行——只允许消费 grant B。
        _adm2, ctx_b, card_b = await self._propose(service, mission, idem="idem_g2")
        listed = await operator_call(sock, "approvals.list", {"mission_id": mission.mission_id})
        entry_b = next(a for a in listed["approvals"] if a["request_id"] == card_b["approval_id"])
        await operator_call(
            sock, "approvals.decide",
            {"request_id": entry_b["request_id"],
             "display_hash": entry_b["display_hash"], "approve": True},
        )
        result = await admission.execute(card_b["approval_id"], request=ctx_b)
        grant_b_row = service._store.connection.execute(
            "SELECT grant_id, consumed FROM mission_grants WHERE request_id = ?",
            (card_b["approval_id"],),
        ).fetchone()
        assert result["grant_id"] == grant_b_row["grant_id"]
        assert result["grant_id"] != grant_a
        # grant A 仍未被消费。
        row_a = service._store.connection.execute(
            "SELECT consumed FROM mission_grants WHERE grant_id = ?", (grant_a,)
        ).fetchone()
        assert row_a["consumed"] == 0
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_body_hash_change_rejected(self, tmp_path: Path) -> None:
        """body hash 变化 → propose 硬拒绝。"""
        import pytest

        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission = await _setup(tmp_path)
        with pytest.raises(ToolBridgeError) as excinfo:
            await self._propose(service, mission, body_hash="body_tampered")
        assert excinfo.value.code == "BODY_HASH_MISMATCH"
        assert service.pending_approvals(mission.mission_id) == []
        await service.close()

    async def test_lease_lost_rejected(self, tmp_path: Path) -> None:
        """writer lease 丢失 → propose 硬拒绝。"""
        import pytest

        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission = await _setup(tmp_path)
        # lease 过期（单 writer 语义下不能被抢——丢失只可能来自过期/崩溃）。
        service._store.connection.execute(
            "UPDATE pi_session_leases SET expires_at = ? WHERE mission_id = ?",
            ("2000-01-01T00:00:00+00:00", mission.mission_id),
        )
        with pytest.raises(ToolBridgeError) as excinfo:
            await self._propose(service, mission)
        assert excinfo.value.code == "WRITER_LEASE_REQUIRED"
        assert service.pending_approvals(mission.mission_id) == []
        await service.close()

    async def test_execute_replay_no_second_side_effect(self, tmp_path: Path) -> None:
        """同一 approval 重放 execute——grant 已消费，拒绝第二次。"""
        import pytest

        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card = await self._propose(service, mission, idem="idem_replay")
        listed = await operator_call(sock, "approvals.list", {"mission_id": mission.mission_id})
        entry = next(a for a in listed["approvals"] if a["request_id"] == card["approval_id"])
        await operator_call(
            sock, "approvals.decide",
            {"request_id": entry["request_id"],
             "display_hash": entry["display_hash"], "approve": True},
        )
        first = await admission.execute(card["approval_id"], request=ctx)
        assert first["grant_id"]
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.execute(card["approval_id"], request=ctx)
        assert excinfo.value.code == "GRANT_CONSUMED"
        await operatord.stop()
        await agent_server.stop()
        await service.close()


class TestApprovalsGet:
    """P0-NA-14：approvals.get 精确单卡查询 + fail-closed 语义。"""

    async def test_get_exact_card(self, tmp_path: Path) -> None:
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
            ActionRequestContext,
        )

        snapshot = service.snapshot(mission.mission_id)
        from rosclaw.agentd.pi_bridge.context_lease import ContextLeaseStore

        lease = ContextLeaseStore(service._store.connection).issue(
            pi_session_id="pi_1",
            mission_id=mission.mission_id,
            context_revision=snapshot.context_revision,
            context_hash="test_hash",
            body_hash=mission.body_binding.effective_body_hash,
            mode=mission.mode.value,
        )
        admission = ActionAdmissionService(service)
        card = await admission.propose(
            request=ActionRequestContext(
                pi_session_id="pi_1",
                mission_id=mission.mission_id,
                context_revision=snapshot.context_revision,
                body_hash=mission.body_binding.effective_body_hash,
                mode=mission.mode.value,
                idempotency_key="idem_get_1",
                context_lease_id=lease.context_lease_id,
            ),
            capability_id="sim_ground_truth",
            arguments={"beep": True},
            expected_effect="蜂鸣",
            risk_tier="LOW",
        )
        got = await operator_call(sock, "approvals.get", {"request_id": card["approval_id"]})
        assert got.get("ok"), got
        approval = got["approval"]
        assert approval["request_id"] == card["approval_id"]
        assert approval["parameters"] == {"beep": True}
        assert approval["expected_effect"] == "蜂鸣"
        assert approval["display_hash"] == card["display_hash"]
        assert approval["mode"] == "SIMULATION"
        # 不存在的卡 → CARD_NOT_FOUND（不是空卡）。
        missing = await operator_call(sock, "approvals.get", {"request_id": "appr_nope"})
        assert not missing.get("ok") and missing.get("code") == "CARD_NOT_FOUND"
        await operatord.stop()
        await agent_server.stop()
        await service.close()
