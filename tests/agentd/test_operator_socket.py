"""PR-11 Operator 安全通道测试（大纲 §14/§19.6 攻击回归）。

- peer identity 是唯一身份源（body 里的 principal 被忽略）
- display_hash 不匹配被拒
- approve → Broker 签发 single-use Grant（同一终端舒服确认）
- estop 无 daemon 时诚实不可用，不假装停止
- CSRF/Origin：外部 Origin 的变更请求被 403；pairing token 强制
- operator 方法永不出现在模型工具目录
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.operator_socket import (
    OperatorSocketServer,
    display_hash_for,
    operator_call,
)
from rosclaw.agentd.service import AgentService, create_app
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1


def _approval_turn(request) -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "d_appr",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "REQUEST_APPROVAL",
        "summary": "请求授权：播放提示音",
        "evidence_refs": [],
        "proposed_operation": {
            "type": "approval_request",
            "payload": {
                "title": "播放提示音",
                "summary": "speaker 660Hz 0.25s",
                "risk_tier": "LOW",
            },
        },
    }
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": "x"},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


async def _service_with_pending(tmp_path: Path):
    config = load_agent_config(tmp_path / "config.yaml")
    service = AgentService(
        config, tmp_path, gateway=MockModelGateway(mock_profile(), [_approval_turn] * 4)
    )
    mission = service.create_mission("operator socket 测试")
    await service.send_turn(mission.mission_id, "请求授权")
    pending = service.pending_approvals(mission.mission_id)
    assert len(pending) == 1
    return service, pending[0]


class TestOperatorSocket:
    async def test_peer_identity_and_approve_flow(self, tmp_path: Path) -> None:
        service, card = await _service_with_pending(tmp_path)
        sock_path = tmp_path / "run" / "operator.sock"
        server = OperatorSocketServer(service, sock_path)
        await server.start()
        try:
            assert (sock_path.stat().st_mode & 0o777) == 0o600
            # list：display_hash 可见。
            listed = await operator_call(sock_path, "approvals.list")
            assert listed["ok"]
            assert listed["principal"].startswith("user:local:")
            entry = listed["approvals"][0]
            assert entry["request_id"] == card.request_id
            assert entry["display_hash"] == display_hash_for(card)
            # 伪造 principal（body 里写 root）→ 被忽略，peer uid 胜出。
            decided = await operator_call(
                sock_path,
                "approvals.decide",
                {
                    "request_id": card.request_id,
                    "display_hash": entry["display_hash"],
                    "approve": True,
                    "principal": "user:local:0",  # 伪造字段必须被忽略
                },
            )
            assert decided["ok"]
            assert decided["principal"] != "user:local:0"
            grant = decided["grant_id"]
            assert grant
            # grant 的 principal 来自 peer identity，不是伪造值。
            stored = [g for g in service.list_grants() if g["grant_id"] == grant]
            assert stored and stored[0]["principal"] == decided["principal"]
            # EXACT_ACTION 单次性：重复 decide 同一 request → 拒绝。
            again = await operator_call(
                sock_path,
                "approvals.decide",
                {
                    "request_id": card.request_id,
                    "display_hash": entry["display_hash"],
                    "approve": True,
                },
            )
            assert not again["ok"]
        finally:
            await server.stop()
            await service.close()

    async def test_display_hash_mismatch_rejected(self, tmp_path: Path) -> None:
        service, card = await _service_with_pending(tmp_path)
        sock_path = tmp_path / "op.sock"
        server = OperatorSocketServer(service, sock_path)
        await server.start()
        try:
            result = await operator_call(
                sock_path,
                "approvals.decide",
                {
                    "request_id": card.request_id,
                    "display_hash": "0000000000000000",
                    "approve": True,
                },
            )
            assert not result["ok"] and result["error"] == "display_hash_mismatch"
            # 未批准：无 grant 产生。
            assert service.list_grants() == []
            # 空 hash 同样拒绝（fail closed）。
            empty = await operator_call(
                sock_path,
                "approvals.decide",
                {"request_id": card.request_id, "display_hash": "", "approve": True},
            )
            assert not empty["ok"]
        finally:
            await server.stop()
            await service.close()

    async def test_estop_honest_without_daemon(self, tmp_path: Path) -> None:
        service, _ = await _service_with_pending(tmp_path)
        sock_path = tmp_path / "op.sock"
        server = OperatorSocketServer(service, sock_path)
        await server.start()
        try:
            result = await operator_call(sock_path, "estop", {"reason": "test"})
            assert not result["ok"]
            assert "unavailable" in result["error"]
            assert "honest" in result["error"] or "not connected" in result["error"]
        finally:
            await server.stop()
            await service.close()

    async def test_unknown_method_rejected(self, tmp_path: Path) -> None:
        service, _ = await _service_with_pending(tmp_path)
        sock_path = tmp_path / "op.sock"
        server = OperatorSocketServer(service, sock_path)
        await server.start()
        try:
            result = await operator_call(sock_path, "permits.mint")
            assert not result["ok"] and "unknown method" in result["error"]
        finally:
            await server.stop()
            await service.close()


class TestCsrfOriginGuard:
    async def test_foreign_origin_rejected(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        service, card = await _service_with_pending(tmp_path)
        client = TestClient(create_app(service))
        try:
            # 网页跨源调用 approval endpoint → 403。
            hostile = client.post(
                f"/approvals/{card.request_id}/decide",
                json={"approve": True},
                headers={"Origin": "https://evil.example.com"},
            )
            assert hostile.status_code == 403
            # 本机来源放行。
            local = client.post(
                f"/approvals/{card.request_id}/decide",
                json={"approve": True},
                headers={"Origin": "http://localhost:3000"},
            )
            assert local.status_code == 200
        finally:
            await service.close()

    async def test_pairing_token_enforced_when_configured(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        service, card = await _service_with_pending(tmp_path)
        os.environ["ROSCLAW_CONSOLE_TOKEN"] = "pairing-test-token"
        try:
            client = TestClient(create_app(service))
            missing = client.post(
                f"/approvals/{card.request_id}/decide", json={"approve": True}
            )
            assert missing.status_code == 403
            wrong = client.post(
                f"/approvals/{card.request_id}/decide",
                json={"approve": True},
                headers={"X-Rosclaw-Token": "wrong"},
            )
            assert wrong.status_code == 403
            right = client.post(
                f"/approvals/{card.request_id}/decide",
                json={"approve": True},
                headers={"X-Rosclaw-Token": "pairing-test-token"},
            )
            assert right.status_code == 200
        finally:
            os.environ.pop("ROSCLAW_CONSOLE_TOKEN", None)
            await service.close()


class TestModelNeverReachesOperator:
    async def test_operator_methods_not_in_tool_catalog(self, tmp_path: Path) -> None:
        service, _ = await _service_with_pending(tmp_path)
        try:
            for descriptor in service.tool_catalog.list():
                assert "approve" not in descriptor.tool_id
                assert "estop" not in descriptor.tool_id
                assert "grant" not in descriptor.tool_id
                assert "revoke" not in descriptor.tool_id
        finally:
            await service.close()

    async def test_model_text_approve_does_not_decide(self, tmp_path: Path) -> None:
        """模型输出 '/approve xxx' 文本不得批准任何请求（§19.6）。"""
        service, card = await _service_with_pending(tmp_path)
        try:
            await service.send_turn(card.mission_id, "/approve " + card.request_id)
            # 卡片仍待定（命令只经 operator socket/HTTP 专用端点生效）。
            assert service.pending_approvals(card.mission_id)
        finally:
            await service.close()
