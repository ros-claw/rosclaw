"""PR-SIX-2 红测试（六审 §4）：ExactActionV3 真正不可变。

红测试先行——以下缺陷修复前必须红：

1. 真实 propose 建卡的 exact.context_hash 必须等于验证过的 context
   lease 的 hash 且非空（当前明确写死 context_hash=""）；
2. ExactAction/Approval/ActionTxn 的 expires_at 按统一策略生成且一致
   （当前 ExactAction=now、Approval=now+600s 分裂），execute 尊重最短
   TTL；
3. execute 完整复验 canonical chain——逐字段篡改 exact_action_json
   （capability/version/source/mission/mode/body_hash/context_revision/
   context_hash/normalized_arguments/risk/expiry/intent_hash）全部
   拒绝且 grant 零消费（当前 execute 根本不解析 exact_action_json）；
4. executor 收到的参数必须来自 ExactAction.normalized_arguments——
   篡改 display.parameters 不得影响执行（当前 executor 吃 display
   对象）；
5. PHYSICAL_ACTION 的 input_schema 未声明 additionalProperties:false
   必须在建卡前被隔离（fail closed）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.agentd.test_pi_approval import _setup_with_operatord
from tests.agentd.test_pi_tool_bridge import (
    SIM_ACTION_CAPABILITY,
    _issue_lease,
)


async def _propose_and_approve(service, mission, sock, *, idem: str, session="pi_1"):
    from rosclaw.agentd.operator_socket import operator_call
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
    card = await admission.propose(
        caller_pid=1, caller_uid=1000,
        request=ctx,
        capability_id=SIM_ACTION_CAPABILITY,
        arguments={},
        expected_effect="six-2",
        risk_tier="LOW",
    )
    listed = await operator_call(sock, "approvals.list", {"mission_id": mission.mission_id})
    entry = next(a for a in listed["approvals"] if a["request_id"] == card["approval_id"])
    decided = await operator_call(
        sock, "approvals.decide",
        {"request_id": entry["request_id"],
         "display_hash": entry["display_hash"], "approve": True},
    )
    assert decided.get("ok"), decided
    return admission, ctx, card, lease


def _stored_request_json(service, approval_id: str) -> dict:
    row = service._store.connection.execute(
        "SELECT request_json FROM operator_requests WHERE request_id = ?",
        (approval_id,),
    ).fetchone()
    return json.loads(row["request_json"])


def _rewrite_exact_action(service, approval_id: str, mutate) -> None:
    """篡改存储卡里的 exact_action_json（execute 必须发现）。"""
    request = _stored_request_json(service, approval_id)
    exact = json.loads(request["exact_action_json"])
    mutate(exact)
    request["exact_action_json"] = json.dumps(exact)
    service._store.connection.execute(
        "UPDATE operator_requests SET request_json = ? WHERE request_id = ?",
        (json.dumps(request), approval_id),
    )
    service._store.connection.commit()


def _grant_consumed(service, approval_id: str) -> int:
    row = service._store.connection.execute(
        "SELECT consumed FROM mission_grants WHERE request_id = ?", (approval_id,)
    ).fetchone()
    return int(row["consumed"]) if row else -1


class TestExactActionBinding:
    async def test_exact_action_binds_real_context_hash(self, tmp_path: Path) -> None:
        """真实建卡的 context_hash 非空且等于 lease 验证过的 hash。"""
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, _ctx, card, lease_id = await _propose_and_approve(
            service, mission, sock, idem="idem_six2_hash"
        )
        exact = json.loads(_stored_request_json(service, card["approval_id"])["exact_action_json"])
        # lease 的真实 context_hash。
        row = service._store.connection.execute(
            "SELECT context_hash FROM pi_context_leases WHERE context_lease_id = ?",
            (lease_id,),
        ).fetchone()
        assert row and row["context_hash"], "lease 本身无 context_hash（前置条件）"
        assert exact["context_hash"] == row["context_hash"], (
            f"ExactAction context_hash 未绑定真实值: {exact['context_hash']!r} "
            f"vs lease {row['context_hash']!r}"
        )
        assert exact["context_hash"] != "", "context_hash 仍为空字符串"
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_unified_ttl_across_exact_approval_txn(self, tmp_path: Path) -> None:
        """ExactAction/Approval/ActionTxn 的 expires_at 统一（同一策略、
        同一时间边界）——不再 ExactAction=now / Approval=+600s 分裂。"""
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        _admission, _ctx, card, _lease = await _propose_and_approve(
            service, mission, sock, idem="idem_six2_ttl"
        )
        stored = _stored_request_json(service, card["approval_id"])
        exact = json.loads(stored["exact_action_json"])
        approval_expires = stored["expires_at"]
        txn_row = service._store.connection.execute(
            "SELECT expires_at FROM action_txns WHERE approval_id = ?",
            (card["approval_id"],),
        ).fetchone()
        assert exact["expires_at"] == approval_expires, (
            f"ExactAction 与 Approval 有效期分裂: {exact['expires_at']} vs {approval_expires}"
        )
        assert txn_row and txn_row["expires_at"] == approval_expires, (
            f"ActionTxn 与 Approval 有效期分裂: "
            f"{txn_row and txn_row['expires_at']} vs {approval_expires}"
        )
        await operatord.stop()
        await agent_server.stop()
        await service.close()


class TestExecuteCanonicalRevalidation:
    @pytest.mark.parametrize(
        "field,tamper",
        [
            ("capability_id", lambda v: "limo.evidence.doctored"),
            ("capability_version", lambda v: "9.9.9"),
            ("capability_source", lambda v: "mcp:evil"),
            ("mission_id", lambda v: "mis_other"),
            ("mode", lambda v: "REAL"),
            ("body_hash", lambda v: "body_deadbeef"),
            ("context_revision", lambda v: v + 100),
            ("context_hash", lambda v: "sha256:tampered"),
            ("authoritative_risk_tier", lambda v: "LOW" if v != "LOW" else "CRITICAL"),
            ("action_intent_hash", lambda v: "sha256:forged"),
        ],
    )
    async def test_tampered_exact_action_field_rejected(
        self, tmp_path: Path, field: str, tamper
    ) -> None:
        """逐字段篡改存储卡的 exact_action_json——execute 必须
        CHAIN_MISMATCH 且 grant 零消费。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card, _lease = await _propose_and_approve(
            service, mission, sock, idem=f"idem_six2_tamper_{field}"
        )
        _rewrite_exact_action(
            service, card["approval_id"], lambda e: e.__setitem__(field, tamper(e[field]))
        )
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.execute(card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000)

        assert excinfo.value.code in (
            "CHAIN_MISMATCH", "EXACT_ACTION_INVALID", "EXACT_ACTION_EXPIRED",
            "CONTEXT_HASH_MISMATCH", "BODY_HASH_MISMATCH", "MODE_MISMATCH",
            "CONTEXT_REVISION_MISMATCH", "MISSION_MISMATCH",
        ), f"篡改 {field} 的错误码不符: {excinfo.value.code}"
        assert _grant_consumed(service, card["approval_id"]) == 0, (
            f"篡改 {field} 后 grant 竟被消费"
        )
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_tampered_normalized_arguments_rejected(self, tmp_path: Path) -> None:
        """篡改 normalized_arguments——arguments_hash 复算必须发现。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card, _lease = await _propose_and_approve(
            service, mission, sock, idem="idem_six2_tamper_args"
        )

        def _mutate(exact: dict) -> None:
            exact["normalized_arguments"]["volume_percent"] = 99

        _rewrite_exact_action(service, card["approval_id"], _mutate)
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.execute(card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000)

        assert excinfo.value.code in ("CHAIN_MISMATCH", "EXACT_ACTION_INVALID")
        assert _grant_consumed(service, card["approval_id"]) == 0
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_expired_exact_action_cannot_execute(self, tmp_path: Path) -> None:
        """ExactAction 过期（在 approval/txn TTL 内）→ execute 拒绝。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card, _lease = await _propose_and_approve(
            service, mission, sock, idem="idem_six2_expiry"
        )
        _rewrite_exact_action(
            service,
            card["approval_id"],
            lambda e: e.__setitem__("expires_at", "2000-01-01T00:00:00+00:00"),
        )
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.execute(card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000)

        assert excinfo.value.code in (
            "EXACT_ACTION_EXPIRED", "TXN_EXPIRED", "CHAIN_MISMATCH", "GRANT_EXPIRED"
        )
        assert _grant_consumed(service, card["approval_id"]) == 0
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_executor_receives_exact_normalized_arguments(
        self, tmp_path: Path
    ) -> None:
        """executor 收到的必须是 ExactAction.normalized_arguments
        （canonical 一致）——不是另一份可变 display 对象。"""
        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        captured: list[dict] = []
        original_channel = service._sim_executors["native:agentd"]
        original_client = original_channel._client

        class _SpyClient:
            async def call_tool(self, tool_name: str, arguments: dict) -> str:
                captured.append(dict(arguments))
                return await original_client.call_tool(tool_name, arguments)

        original_channel._client = _SpyClient()
        admission, ctx, card, _lease = await _propose_and_approve(
            service, mission, sock, idem="idem_six2_exec_args"
        )
        exact = json.loads(_stored_request_json(service, card["approval_id"])["exact_action_json"])
        outcome = await admission.execute(card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000)

        assert outcome["executed"] is True
        assert captured, "executor 未被调用"
        assert captured[0] == exact["normalized_arguments"], (
            f"executor 收到 {captured[0]}，不是 ExactAction.normalized_arguments "
            f"{exact['normalized_arguments']}"
        )
        await operatord.stop()
        await agent_server.stop()
        await service.close()

    async def test_tampered_display_parameters_rejected(self, tmp_path: Path) -> None:
        """批准后篡改 display.parameters（operator 看到的展示对象）——
        display_hash 复算必须发现（CHAIN_MISMATCH），grant 零消费；
        展示对象绝不能成为执行参数来源。"""
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        admission, ctx, card, _lease = await _propose_and_approve(
            service, mission, sock, idem="idem_six2_display_tamper"
        )
        request = _stored_request_json(service, card["approval_id"])
        request["action_display"]["parameters"] = {"frequency_hz": 1}
        service._store.connection.execute(
            "UPDATE operator_requests SET request_json = ? WHERE request_id = ?",
            (json.dumps(request), card["approval_id"]),
        )
        service._store.connection.commit()
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.execute(card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000)

        assert excinfo.value.code == "CHAIN_MISMATCH"
        assert _grant_consumed(service, card["approval_id"]) == 0
        await operatord.stop()
        await agent_server.stop()
        await service.close()


class TestStrictPhysicalSchema:
    async def test_non_strict_physical_schema_quarantined(self, tmp_path: Path) -> None:
        """PHYSICAL_ACTION 的 input_schema 未声明 additionalProperties:false
        → 建卡前拒绝（fail closed，不放行未知参数）。"""
        from rosclaw.agentd.pi_bridge.action_admission import (
            ActionAdmissionService,
            ActionRequestContext,
        )
        from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError
        from rosclaw.contracts.agent.tool import (
            ExecutionClass,
            ToolDescriptorV2,
            ToolEvidenceClass,
            ToolSideEffectClass,
        )

        service, mission, operatord, agent_server, sock = await _setup_with_operatord(
            tmp_path
        )
        service._tool_catalog.register(
            ToolDescriptorV2(
                tool_id="sim.loose.action",
                source="native:agentd",
                execution_class=ExecutionClass.PHYSICAL_ACTION,
                description="松散 schema 的物理动作（应被隔离）。",
                input_schema={"type": "object", "additionalProperties": True},
                supported_modes=["SIMULATION"],
                # 有 body scope——本测试专测 schema 严格性（SIX-3 的
                # body scope 门禁另有测试）。
                required_body_types=["sim/ur5e"],
                evidence_class=ToolEvidenceClass.SIMULATED,
                risk_tier="LOW",
                model_callable=False,
                requires_exact_action_grant=True,
                side_effect_class=ToolSideEffectClass.IRREVERSIBLE,
            )
        )
        snapshot = service.snapshot(mission.mission_id)
        lease = await _issue_lease(service, mission, "pi_1")
        ctx = ActionRequestContext(
            pi_session_id="pi_1",
            mission_id=mission.mission_id,
            context_revision=snapshot.context_revision,
            body_hash=mission.body_binding.effective_body_hash,
            mode=mission.mode.value,
            idempotency_key="idem_six2_loose",
            context_lease_id=lease,
        )
        admission = ActionAdmissionService(service)
        with pytest.raises(ToolBridgeError) as excinfo:
            await admission.propose(
                caller_pid=1, caller_uid=1000,
                request=ctx,
                capability_id="sim.loose.action",
                arguments={"anything": "goes"},
                expected_effect="loose",
                risk_tier="LOW",
            )
        assert excinfo.value.code in (
            "CAPABILITY_QUARANTINED", "SCHEMA_NOT_STRICT", "INVALID_ARGUMENTS"
        ), f"松散 schema 物理动作竟进入建卡: {excinfo.value.code}"
        assert service.pending_approvals(mission.mission_id) == []
        await operatord.stop()
        await agent_server.stop()
        await service.close()
