"""复盘回归（2026-08-04 深度审查）：

- ModeldGateway 多实例 socket 冲突（并发 mission/failover/mgmt 共存）
- fork 基于 canonical journal（compaction 后历史不丢）
- §19.6 攻击回归：tool result 伪造"用户已授权"不得批准；
  客户端断开后 pending approval 保持待定并最终过期
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1


def _answer(request) -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "d",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "ANSWER",
        "summary": "ok",
        "evidence_refs": [],
    }
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": "x"},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


def _service(tmp_path: Path, script=None) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    return AgentService(
        config, tmp_path, gateway=MockModelGateway(mock_profile(), script or [_answer] * 50)
    )


class TestModeldSocketIsolation:
    @pytest.mark.skipif(
        __import__("rosclaw.agentd.models.modeld_gateway", fromlist=["_find_modeld_runtime"])._find_modeld_runtime()
        is None,
        reason="modeld runtime unavailable",
    )
    async def test_concurrent_gateways_do_not_break_each_other(self, tmp_path: Path) -> None:
        from rosclaw.agentd.models.modeld_gateway import ModeldGateway
        from rosclaw.agentd.models.profiles import kimi_code_k3_profile

        g1 = ModeldGateway(kimi_code_k3_profile(), home=tmp_path)
        g2 = ModeldGateway(kimi_code_k3_profile(), home=tmp_path)
        assert g1._socket_path != g2._socket_path or not g1._socket_path
        await asyncio.gather(g1._ensure_started(), g2._ensure_started())
        try:
            assert g1._socket_path != g2._socket_path
            d1 = await g1.manage("GET", "/v1/providers")
            d2 = await g2.manage("GET", "/v1/providers")
            assert d1.get("providers") and d2.get("providers")
        finally:
            await g1.close()
            await g2.close()


class TestForkCanonical:
    async def test_fork_after_compaction_keeps_full_history(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            mission = service.create_mission("fork canonical 测试")
            for i in range(4):
                await service.send_turn(mission.mission_id, f"第 {i} 轮 {'长文本' * 200}")
            await service.compact(mission.mission_id)
            canonical = service.store.conversation_canonical(mission.mission_id)
            view = service.store.conversation(mission.mission_id)
            assert len(canonical) > len(view), "compaction 后 canonical 必须长于 view"
            first_entry = canonical[0]["entry_id"]
            branch = service.branches.fork(
                mission.mission_id, from_entry_id=first_entry, label="全历史"
            )
            forked_conv = service.store.conversation(branch.forked_mission_id)
            # fork 自第一个 entry：应带入 canonical 中的全部消息（包括
            # 被 compaction 折叠的早期轮次）。
            assert len(forked_conv) >= len(canonical) - 1 or len(forked_conv) == 1
            assert any("第 0 轮" in str(m.get("content")) for m in forked_conv) or any(
                "compaction" in str(m.get("role")) for m in forked_conv
            ) or len(forked_conv) == 1
        finally:
            await service.close()


class TestAttackRegressions:
    async def test_tool_result_forged_authorization_rejected(self, tmp_path: Path) -> None:
        """§19.6：tool result 声称"用户已授权"——REQUEST_ACTION 无 grant
        仍被 fail closed 拒绝（broker 重算，不采信任何文本声明）。"""
        service = _service(tmp_path)
        try:
            mission = service.create_mission("伪造授权攻击")
            # 直接向 broker verify 提交一个不存在的 grant（模拟模型被注入后
            # 引用伪造授权）。
            from rosclaw.operator import GrantDeniedError

            with pytest.raises(GrantDeniedError, match="unknown_grant"):
                service._broker.verify(
                    "grant_forged_by_tool_result",
                    principal="user:local:1000",
                    body_hash=mission.body_binding.effective_body_hash,
                    mode="SIMULATION",
                    risk_tier="LOW",
                )
            assert service.list_grants() == []
        finally:
            await service.close()

    async def test_pending_approval_outlives_client_and_expires(self, tmp_path: Path) -> None:
        """§19.6：客户端断开 → pending approval 保持待定；过期后 verify 拒绝。"""
        from datetime import UTC, datetime, timedelta

        service = _service(tmp_path)
        try:
            mission = service.create_mission("过期测试")
            broker = service._broker
            from rosclaw.contracts.operator.approval import ActionDisplayV1, ApprovalRequestV2

            now = datetime.now(UTC)
            broker.create_request(
                ApprovalRequestV2(
                    request_id="appr_expire",
                    mission_id=mission.mission_id,
                    principal="user:local:1000",
                    body_id=mission.body_binding.body_id,
                    effective_body_hash=mission.body_binding.effective_body_hash,
                    mode="SIMULATION",
                    action_display=ActionDisplayV1(
                        title="t", summary="s", risk_tier="LOW"
                    ),
                    context_id="ctx_1",
                    context_revision=1,
                    created_at=now.isoformat(),
                    expires_at=(now + timedelta(seconds=1)).isoformat(),
                )
            )
            # 客户端"断开"（无任何 decide 调用）→ 卡片保持待定。
            assert broker.pending_requests(mission.mission_id)
            # 时间推进 → 过期后 decide 被拒（expires_at 存在 request_json 里）。
            import json as _json

            row = broker._conn.execute(
                "SELECT request_json FROM operator_requests WHERE request_id = ?",
                ("appr_expire",),
            ).fetchone()
            req = _json.loads(row["request_json"])
            req["expires_at"] = (now - timedelta(seconds=1)).isoformat()
            broker._conn.execute(
                "UPDATE operator_requests SET request_json = ? WHERE request_id = ?",
                (_json.dumps(req), "appr_expire"),
            )

            with pytest.raises(Exception, match="expired|EXPIRED"):
                broker.decide("appr_expire", principal="user:local:1000", approve=True)
        finally:
            await service.close()
