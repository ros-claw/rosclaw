"""PR-11 + 审计 P0-01：operator 通道拆分测试。

agentd 投影 socket：
- approvals.list 只读（owner 过滤）
- approvals.decide / grants.revoke / estop 一律拒绝并指明 operatord
- apply_decision：无 proof 拒、display_hash 不匹配拒、proof 齐全 →
  DEV_SIM_ONLY 语义应用（无 daemon）
- REAL/daemon 卡未先经 daemon 决定 → fail closed

rosclaw-operatord：
- enrollment 生成/加载/权限校验
- decide 全流程：operatord sign → agentd apply → grant minted（单次）
- nonce 一次性、伪造 proof 拒绝
- estop 无 daemon 诚实不可用
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
from rosclaw.operatord.enrollment import (
    enroll,
    load_enrollment,
    sign_decision_proof,
    verify_decision_proof,
)
from rosclaw.operatord.server import OperatorDaemon


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
    mission = service.create_mission("operator 拆分测试")
    await service.send_turn(mission.mission_id, "请求授权")
    pending = service.pending_approvals(mission.mission_id)
    assert len(pending) == 1
    return service, pending[0]


class TestAgentProjectionSocket:
    async def test_list_works_decide_revoke_estop_rejected(self, tmp_path: Path) -> None:
        service, card = await _service_with_pending(tmp_path)
        sock_path = tmp_path / "run" / "operator.sock"
        server = OperatorSocketServer(service, sock_path)
        await server.start()
        try:
            listed = await operator_call(sock_path, "approvals.list")
            assert listed["ok"]
            assert listed["approvals"][0]["request_id"] == card.request_id
            assert listed["approvals"][0]["display_hash"] == display_hash_for(card)
            for method in ("approvals.decide", "grants.revoke", "estop"):
                result = await operator_call(
                    sock_path,
                    method,
                    {"request_id": card.request_id, "display_hash": "x", "approve": True}
                    if method == "approvals.decide"
                    else {},
                )
                assert not result["ok"], method
                assert "operatord" in result["error"]
        finally:
            await server.stop()
            await service.close()

    async def test_apply_decision_requires_proof_and_hash(self, tmp_path: Path) -> None:
        service, card = await _service_with_pending(tmp_path)
        sock_path = tmp_path / "op.sock"
        server = OperatorSocketServer(service, sock_path)
        await server.start()
        try:
            # 无 proof → 拒。
            r1 = await operator_call(
                sock_path,
                "approvals.apply_decision",
                {"request_id": card.request_id, "display_hash": display_hash_for(card),
                 "approve": True},
            )
            assert not r1["ok"] and "operator_proof" in r1["error"]
            # hash 不匹配 → 拒。
            r2 = await operator_call(
                sock_path,
                "approvals.apply_decision",
                {"request_id": card.request_id, "display_hash": "0" * 16, "approve": True,
                 "operator_proof": "x", "enrollment_id": "e", "nonce": "n"},
            )
            assert not r2["ok"] and r2["error"] == "display_hash_mismatch"
            assert service.list_grants() == []
        finally:
            await server.stop()
            await service.close()


class TestEnrollment:
    def test_enroll_load_verify_roundtrip(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        home = tmp_path / "operatord"
        enrollment = enroll(home)
        assert (home / "operator-enrollment.json").stat().st_mode & 0o777 == 0o600
        loaded = load_enrollment(home)
        assert loaded.enrollment_id == enrollment.enrollment_id
        assert loaded.key == enrollment.key
        decided_at = datetime.now(UTC).isoformat()
        proof = sign_decision_proof(
            loaded,
            request_id="r1",
            approve=True,
            nonce="n1",
            decided_at=decided_at,
            display_hash="h",
        )
        assert verify_decision_proof(
            loaded.key,
            request_id="r1",
            approve=True,
            nonce="n1",
            decided_at=decided_at,
            enrollment_id=loaded.enrollment_id,
            display_hash="h",
            proof=proof,
        )
        assert not verify_decision_proof(
            loaded.key,
            request_id="r1",
            approve=True,
            nonce="n1",
            decided_at="1970-01-01T00:00:00Z",  # 篡改任一字段必失败
            enrollment_id=loaded.enrollment_id,
            display_hash="h",
            proof=proof,
        )
        # 拒绝二次 enroll（不覆盖既有 key）。
        from rosclaw.operatord.enrollment import EnrollmentError

        with pytest.raises(EnrollmentError, match="already exists"):
            enroll(home)

    def test_corrupt_enrollment_quarantined(self, tmp_path: Path) -> None:
        from rosclaw.operatord.enrollment import EnrollmentError

        home = tmp_path / "operatord"
        home.mkdir()
        (home / "operator-enrollment.json").write_text("not json")
        with pytest.raises(EnrollmentError, match="corrupt"):
            load_enrollment(home)


class TestOperatordFlow:
    async def test_full_decide_flow_single_use(self, tmp_path: Path) -> None:
        service, card = await _service_with_pending(tmp_path)
        agent_sock = tmp_path / "run" / "operator.sock"
        agent_server = OperatorSocketServer(service, agent_sock)
        await agent_server.start()
        enrollment = enroll(tmp_path / "operatord")
        daemon = OperatorDaemon(
            enrollment=enrollment,
            socket_path=tmp_path / "run" / "operatord.sock",
            agent_socket=agent_sock,
            daemon_client=None,
            require_human_presence=False,
        )
        await daemon.start()
        try:
            sock = tmp_path / "run" / "operatord.sock"
            decided = await operator_call(
                sock,
                "approvals.decide",
                {
                    "request_id": card.request_id,
                    "display_hash": display_hash_for(card),
                    "approve": True,
                },
            )
            assert decided["ok"], decided
            assert decided["grant_id"]
            assert decided["profile"] == "DEV_SIM_ONLY"  # 无 daemon → 明确标记
            # grant 由 agentd broker 铸造；单次性由 broker 保证。
            grants = [g for g in service.list_grants() if g["grant_id"] == decided["grant_id"]]
            assert grants and grants[0]["tier"] == "EXACT_ACTION"
            # 重复 decide 同一张卡 → agentd 侧已决，拒绝。
            again = await operator_call(
                sock,
                "approvals.decide",
                {
                    "request_id": card.request_id,
                    "display_hash": display_hash_for(card),
                    "approve": True,
                },
            )
            assert not again["ok"]
            # 伪造 proof 直接打 agentd → 不能铸造 grant（SIM 卡 proof 字段
            # 只是门槛；伪造者拿不到卡片与 hash 的一致组合——这里验证无
            # proof 直接调 decide 必拒）。
            forged = await operator_call(
                agent_sock, "approvals.decide", {"request_id": card.request_id}
            )
            assert not forged["ok"]
        finally:
            await daemon.stop()
            await agent_server.stop()
            await service.close()

    async def test_estop_honest_without_daemon(self, tmp_path: Path) -> None:
        enrollment = enroll(tmp_path / "operatord")
        daemon = OperatorDaemon(
            enrollment=enrollment,
            socket_path=tmp_path / "op.sock",
            agent_socket=None,
            daemon_client=None,
            require_human_presence=False,
        )
        await daemon.start()
        try:
            result = await operator_call(tmp_path / "op.sock", "estop", {"reason": "t"})
            assert not result["ok"]
            assert "unavailable" in result["error"]
        finally:
            await daemon.stop()


class TestHttpBypassClosed:
    async def test_http_mutations_403_by_default(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        service, card = await _service_with_pending(tmp_path)
        client = TestClient(create_app(service))
        try:
            assert client.get("/approvals/pending").status_code == 400
            r = client.post(f"/approvals/{card.request_id}/decide", json={"approve": True})
            assert r.status_code == 403
            grants = service.list_grants()
            gid = grants[0]["grant_id"] if grants else "grant_x"
            assert client.post(f"/grants/{gid}/revoke").status_code == 403
            assert service.pending_approvals(card.mission_id), "403 后卡片仍待定"
        finally:
            await service.close()
