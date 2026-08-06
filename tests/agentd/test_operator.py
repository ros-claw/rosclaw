"""Operator Broker / K5 authorization tests (PR-OP-060/061/062 exits).

- approval card → approve → grant minted (public only, signature private)
- verify fail-closed: unknown, revoked, expired, wrong principal, body hash
  drift, mode mismatch, risk above ceiling, replay (single-use), forged sig
- grant JSON contains no signature/permit material
- full loop: REQUEST_APPROVAL → WAIT_APPROVAL → /approve → REQUEST_ACTION →
  verified (EXACT_ACTION consumed; second use denied)
- HTTP approvals/grants endpoints
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.mission import MissionStore
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService, create_app
from rosclaw.contracts.agent.decision import DecisionV1
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.operator.approval import ActionDisplayV1, ApprovalRequestV2
from rosclaw.operator import GrantDeniedError, OperatorBroker
from tests.agentd.conftest import LOCAL_PRINCIPAL

NOW = datetime.now(UTC)


def _request(broker_body_hash: str = "body_abc", **overrides) -> ApprovalRequestV2:
    payload = {
        "request_id": "appr_test1",
        "mission_id": "mis_x",
        "principal": LOCAL_PRINCIPAL,
        "body_id": "sim/ur5e",
        "effective_body_hash": broker_body_hash,
        "mode": "SIMULATION",
        "action_display": ActionDisplayV1(
            title="移动仿真臂到初始位",
            summary="joints → home",
            risk_tier="LOW",
            expected_effect="关节回到零点",
            failure_handling="超时即停",
        ),
        "context_id": "ctx_1",
        "context_revision": 1,
        "created_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(minutes=10)).isoformat(),
    }
    payload.update(overrides)
    return ApprovalRequestV2(**payload)


@pytest.fixture
def broker(tmp_path: Path) -> OperatorBroker:
    store = MissionStore(tmp_path / "m.db")
    return OperatorBroker(store.connection, policy_hash="pol_test")


class TestGrantLifecycle:
    def test_approve_mints_public_grant(self, broker: OperatorBroker) -> None:
        broker.create_request(_request())
        grant = broker.decide("appr_test1", principal=LOCAL_PRINCIPAL, approve=True)
        assert grant is not None
        dumped = json.dumps(grant.model_dump(mode="json"))
        assert "signature" not in dumped
        assert "permit" not in dumped.lower()
        assert grant.public_hash.startswith("grantpub_")
        # Private signature exists but stays in the broker's store.
        row = broker._conn.execute(
            "SELECT private_signature FROM mission_grants WHERE grant_id = ?",
            (grant.grant_id,),
        ).fetchone()
        assert row and len(row["private_signature"]) == 64

    def test_deny_is_terminal(self, broker: OperatorBroker) -> None:
        broker.create_request(_request())
        assert broker.decide("appr_test1", principal=LOCAL_PRINCIPAL, approve=False) is None
        with pytest.raises(Exception, match="already"):
            broker.decide("appr_test1", principal=LOCAL_PRINCIPAL, approve=True)

    def test_verify_happy_path_single_use(self, broker: OperatorBroker) -> None:
        broker.create_request(_request())
        grant = broker.decide("appr_test1", principal=LOCAL_PRINCIPAL, approve=True)
        intent = broker.action_intent_for_grant(grant.grant_id)
        verified = broker.verify(
            grant.grant_id,
            principal=LOCAL_PRINCIPAL,
            body_hash="body_abc",
            mode="SIMULATION",
            risk_tier="LOW",
            action_intent=intent,
        )
        assert verified.grant_id == grant.grant_id
        # Replay: EXACT_ACTION is single-use.
        with pytest.raises(GrantDeniedError, match="grant_consumed"):
            broker.verify(
                grant.grant_id,
                principal=LOCAL_PRINCIPAL,
                body_hash="body_abc",
                mode="SIMULATION",
                risk_tier="LOW",
                action_intent=intent,
            )


class TestFailClosed:
    def _grant(self, broker: OperatorBroker) -> str:
        broker.create_request(_request())
        grant = broker.decide("appr_test1", principal=LOCAL_PRINCIPAL, approve=True)
        assert grant is not None
        return grant.grant_id

    def test_unknown_grant(self, broker: OperatorBroker) -> None:
        with pytest.raises(GrantDeniedError, match="unknown_grant"):
            broker.verify(
                "grant_ghost",
                principal=LOCAL_PRINCIPAL,
                body_hash="body_abc",
                mode="SIMULATION",
                risk_tier="LOW",
            )

    def test_revoked(self, broker: OperatorBroker) -> None:
        grant_id = self._grant(broker)
        broker.revoke(grant_id, principal=LOCAL_PRINCIPAL)
        with pytest.raises(GrantDeniedError, match="grant_revoked"):
            broker.verify(
                grant_id,
                principal=LOCAL_PRINCIPAL,
                body_hash="body_abc",
                mode="SIMULATION",
                risk_tier="LOW",
            )

    def test_expired(self, broker: OperatorBroker) -> None:
        grant_id = self._grant(broker)
        broker._conn.execute(
            "UPDATE mission_grants SET expires_at = '2000-01-01' WHERE grant_id = ?",
            (grant_id,),
        )
        with pytest.raises(GrantDeniedError, match="grant_expired"):
            broker.verify(
                grant_id,
                principal=LOCAL_PRINCIPAL,
                body_hash="body_abc",
                mode="SIMULATION",
                risk_tier="LOW",
            )

    def test_wrong_principal(self, broker: OperatorBroker) -> None:
        grant_id = self._grant(broker)
        with pytest.raises(GrantDeniedError, match="principal_mismatch"):
            broker.verify(
                grant_id,
                principal="user:local:evil",
                body_hash="body_abc",
                mode="SIMULATION",
                risk_tier="LOW",
            )

    def test_body_hash_drift(self, broker: OperatorBroker) -> None:
        grant_id = self._grant(broker)
        with pytest.raises(GrantDeniedError, match="body_hash_changed"):
            broker.verify(
                grant_id,
                principal=LOCAL_PRINCIPAL,
                body_hash="body_DIFFERENT",
                mode="SIMULATION",
                risk_tier="LOW",
            )

    def test_mode_mismatch(self, broker: OperatorBroker) -> None:
        grant_id = self._grant(broker)
        with pytest.raises(GrantDeniedError, match="mode_mismatch"):
            broker.verify(
                grant_id,
                principal=LOCAL_PRINCIPAL,
                body_hash="body_abc",
                mode="REAL",
                risk_tier="LOW",
            )

    def test_risk_above_ceiling(self, broker: OperatorBroker) -> None:
        grant_id = self._grant(broker)
        with pytest.raises(GrantDeniedError, match="risk_above_ceiling"):
            broker.verify(
                grant_id,
                principal=LOCAL_PRINCIPAL,
                body_hash="body_abc",
                mode="SIMULATION",
                risk_tier="HIGH",
            )

    def test_forged_signature_rejected(self, broker: OperatorBroker) -> None:
        grant_id = self._grant(broker)
        broker._conn.execute(
            "UPDATE mission_grants SET private_signature = 'forged' WHERE grant_id = ?",
            (grant_id,),
        )
        with pytest.raises(GrantDeniedError, match="forged_grant"):
            broker.verify(
                grant_id,
                principal=LOCAL_PRINCIPAL,
                body_hash="body_abc",
                mode="SIMULATION",
                risk_tier="LOW",
            )


def _approval_then_action(request) -> ModelTurnResultV1:
    """Main agent: first REQUEST_APPROVAL, then (next call) REQUEST_ACTION
    referencing the grant created out-of-band."""
    if not hasattr(_approval_then_action, "calls"):
        _approval_then_action.calls = 0
    _approval_then_action.calls += 1
    if _approval_then_action.calls == 1:
        decision = {
            "schema_version": "rosclaw.decision.v1",
            "decision_id": "dec_appr",
            "mission_id": request.mission_id,
            "context_id": request.context_id,
            "context_revision": request.context_revision,
            "next_intent": "REQUEST_APPROVAL",
            "summary": "请求授权：移动仿真臂到初始位",
            "evidence_refs": ["artifact://plan/1"],
            "proposed_operation": {
                "type": "approval_request",
                "payload": {
                    "title": "移动仿真臂到初始位",
                    "summary": "joints → home",
                    "risk_tier": "LOW",
                    "expected_effect": "关节回零",
                    "failure_handling": "超时即停",
                },
            },
            "verification": {
                "schema_version": "rosclaw.decision_verification.v1",
                "verifiers": ["deterministic:bounds"],
            },
        }
    else:
        grant_id = _approval_then_action.grant_id
        decision = {
            "schema_version": "rosclaw.decision.v1",
            "decision_id": "dec_act",
            "mission_id": request.mission_id,
            "context_id": request.context_id,
            "context_revision": request.context_revision,
            "next_intent": "REQUEST_ACTION",
            "summary": "按授权执行",
            "evidence_refs": ["artifact://plan/1"],
            "proposed_operation": {
                "type": "request_action",
                "payload": {"grant_id": grant_id, "risk_tier": "LOW"},
            },
            "verification": {
                "schema_version": "rosclaw.decision_verification.v1",
                "verifiers": ["deterministic:bounds"],
            },
        }
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="mock-model",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": None},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


class TestApprovalLoop:
    async def test_full_exact_action_flow(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        _approval_then_action.calls = 0
        gateway = MockModelGateway(mock_profile(), [_approval_then_action] * 4)
        service = AgentService(config, tmp_path, gateway=gateway)
        try:
            mission = service.create_mission("授权闭环")
            r1 = await service.send_turn(mission.mission_id, "请请求授权移动机械臂")
            assert "授权请求" in r1.reply
            assert r1.state.value == "WAIT_APPROVAL"
            pending = service.pending_approvals(mission.mission_id)
            assert len(pending) == 1
            grant = await service.decide_approval(
                pending[0].request_id, principal=LOCAL_PRINCIPAL, approve=True
            , _from_operatord=True)
            assert grant is not None
            consent = service._compiler._sources.consent.get_consent(mission.mission_id)
            assert consent is not None
            assert grant.grant_id in consent.public_scope_summary
            assert grant.public_hash in consent.public_scope_summary
            _approval_then_action.grant_id = grant.grant_id
            r2 = await service.send_turn(mission.mission_id, "我已批准，继续执行")
            assert "授权已验证" in r2.reply
            assert "已消费" in r2.reply
            # §5.6 新语义：本服务没有 daemon consent/action channel，
            # 无 verified terminal receipt → 诚实停留 MONITOR（提交≠完成）。
            assert r2.state.value == "MONITOR"
            # EXACT_ACTION consumed: a third attempt must fail closed.
            from rosclaw.operator import GrantDeniedError

            with pytest.raises(GrantDeniedError, match="grant_consumed"):
                service._broker.verify(
                    grant.grant_id,
                    principal=LOCAL_PRINCIPAL,
                    body_hash=grant.effective_body_hash,
                    mode="SIMULATION",
                    risk_tier="LOW",
                )
        finally:
            await service.close()

    async def test_real_approval_uses_daemon_ttl_and_exact_action_payload(
        self, tmp_path: Path
    ) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        service = AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), []))

        class FakeConsent:
            called: dict = {}

            async def create_proposal(self, **kwargs):
                self.called = kwargs
                return {"request_id": "proposal_real", "action_id": "action_real"}

        fake = FakeConsent()
        service._handlers._mode = "REAL"
        service._handlers._consent_channel = fake
        decision = DecisionV1.model_validate_contract(
            {
                "schema_version": "rosclaw.decision.v1",
                "decision_id": "dec_real_tone",
                "mission_id": "mis_real",
                "context_id": "ctx_real",
                "context_revision": 1,
                "next_intent": "REQUEST_APPROVAL",
                "summary": "play exact tone",
                "proposed_operation": {
                    "type": "approval_request",
                    "payload": {
                        "capability_id": "limo.play_tone",
                        "arguments": {
                            "schema_version": "limo.tone.v1",
                            "frequency_hz": 660,
                            "duration_sec": 0.25,
                            "volume_percent": 18,
                        },
                        "risk_tier": "LOW",
                    },
                },
            }
        )

        try:
            reply = await service._handlers.request_approval(decision)
            assert "5 分钟有效" in reply.text
            assert fake.called["ttl_sec"] == 300.0
            assert fake.called["capability_id"] == "limo.play_tone"
            assert fake.called["arguments"]["frequency_hz"] == 660
            pending = service.pending_approvals("mis_real")
            assert pending[0].action_display.parameters["duration_sec"] == 0.25
        finally:
            await service.close()

    async def test_real_approval_rejects_missing_daemon_action_payload(
        self, tmp_path: Path
    ) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        service = AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), []))
        service._handlers._mode = "REAL"
        decision = DecisionV1.model_validate_contract(
            {
                "schema_version": "rosclaw.decision.v1",
                "decision_id": "dec_empty_real",
                "mission_id": "mis_real",
                "context_id": "ctx_real",
                "context_revision": 1,
                "next_intent": "REQUEST_APPROVAL",
                "summary": "incomplete action",
                "proposed_operation": {
                    "type": "approval_request",
                    "payload": {"risk_tier": "LOW"},
                },
            }
        )

        try:
            reply = await service._handlers.request_approval(decision)
            assert "缺少 capability_id 或 arguments" in reply.text
            assert service.pending_approvals("mis_real") == []
        finally:
            await service.close()

    async def test_approvals_over_http(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        config = load_agent_config(tmp_path / "config.yaml")
        _approval_then_action.calls = 0
        gateway = MockModelGateway(mock_profile(), [_approval_then_action] * 4)
        service = AgentService(config, tmp_path, gateway=gateway)
        client = TestClient(create_app(service), headers={'x-rosclaw-token': service.control_token})
        try:
            mission = service.create_mission("HTTP 授权")
            await service.send_turn(mission.mission_id, "请求授权")
            # 审计 P0-01/B3：全局 pending 枚举不再提供；必须指定 mission_id。
            assert client.get("/approvals/pending").status_code == 400
            pending = client.get(f"/approvals/pending?mission_id={mission.mission_id}").json()
            assert len(pending) == 1
            rid = pending[0]["request_id"]
            # HTTP 决定旁路默认关闭（403）；DEV_SIM_ONLY 显式打开才可用。
            assert client.post(f"/approvals/{rid}/decide", json={"approve": True}).status_code == 403
            import os

            os.environ["ROSCLAW_DEV_HTTP_DECIDE"] = "1"
            try:
                r = client.post(f"/approvals/{rid}/decide", json={"approve": True})
            finally:
                os.environ.pop("ROSCLAW_DEV_HTTP_DECIDE", None)
            assert r.status_code == 200
            assert r.json()["grant_id"]
            assert r.json()["profile"] == "DEV_SIM_ONLY"
            grants = client.get("/grants").json()
            assert len(grants) == 1
            assert grants[0]["tier"] == "EXACT_ACTION"
            # revoke → HTTP 旁路 403；经 broker 直撤（测试即 operator）。
            gid = grants[0]["grant_id"]
            assert client.post(f"/grants/{gid}/revoke").status_code == 403
            service.revoke_grant(gid, principal=LOCAL_PRINCIPAL)
            from rosclaw.operator import GrantDeniedError

            with pytest.raises(GrantDeniedError, match="grant_revoked"):
                service._broker.verify(
                    gid,
                    principal=LOCAL_PRINCIPAL,
                    body_hash=grants[0].get("effective_body_hash", ""),
                    mode="SIMULATION",
                    risk_tier="LOW",
                )
        finally:
            await service.close()
