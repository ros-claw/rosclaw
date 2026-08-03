"""Daemon consent channel tests (ADR-0007, REAL 模式授权全链）。

- proposal public view: no challenge_nonce / no permit material
- ACCEPT → daemon 独立签 permit + 派发 + 终态 receipt + provenance
- DECLINE → no dispatch; wrong nonce → fail closed
- daemon restart invalidates pending proposals
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.agentd.consent_channel import (
    ConsentChannelError,
    DaemonConsentChannel,
)
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.client import DaemonClient, DaemonRequestError
from rosclaw.daemon.ledger import DaemonLedger
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.kernel.contracts import (
    ActionExecutionResult,
    ActionState,
    EvidenceLevel,
    ExecutionMode,
)
from tests.agentd.conftest import LOCAL_PRINCIPAL

REAL_CAPABILITY = "rh56.finger.move"


def _real_executor(action) -> ActionExecutionResult:
    return ActionExecutionResult(
        final_state=ActionState.COMPLETED,
        evidence_level=EvidenceLevel.DRIVER_CONFIRMED,
        driver_ack={"ack": True},
        verification_result={"verified": True},
    )


def _start_daemon(tmp_path: Path):
    runtime = Runtime(
        RuntimeConfig(
            robot_id="rh56-test",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=False,
        )
    )
    runtime.action_gateway.register_executor(REAL_CAPABILITY, ExecutionMode.REAL, _real_executor)
    ledger_ctx = DaemonLedger(
        tmp_path / "state" / "ledger.sqlite3", key_path=tmp_path / "state" / "ledger.key"
    )
    ledger = ledger_ctx.__enter__()
    service = DaemonControlPlane(runtime=runtime, ledger=ledger)
    socket_path = tmp_path / "run" / "rosclawd.sock"
    daemon = RosclawDaemon(service=service, socket_path=socket_path)
    daemon.start()
    return daemon, socket_path, ledger_ctx


@pytest.fixture
def daemon(tmp_path: Path):
    daemon, socket_path, ledger_ctx = _start_daemon(tmp_path)
    client = DaemonClient(socket_path=socket_path, timeout_sec=5.0)
    try:
        yield client, socket_path
    finally:
        daemon.stop()
        ledger_ctx.__exit__(None, None, None)


def _channel(client: DaemonClient) -> DaemonConsentChannel:
    return DaemonConsentChannel(client, actor_id="agent:test", body_id="rh56-test", body_hash="h")


class TestRealConsentFlow:
    async def test_accept_full_chain_with_provenance(self, daemon) -> None:
        client, _ = daemon
        channel = _channel(client)
        proposal = await channel.create_proposal(
            capability_id=REAL_CAPABILITY,
            arguments={"finger": "index", "delta_raw": 20},
            display={"title": "Move one finger", "risk_tier": "HIGH"},
            execution_mode="REAL",
            risk_class="high",
            ttl_sec=60.0,
        )
        # Agent 视角：无 challenge、无 permit（K5 核心安全语义）。
        assert "challenge_nonce" not in proposal
        assert "permit" not in str(proposal).lower()
        decided = await channel.decide(
            proposal["request_id"],
            principal_id=LOCAL_PRINCIPAL,
            accept=True,
            supervise_timeout_sec=15.0,
        )
        assert decided.get("command_dispatched") is True
        assert decided.get("permit_exposed") is False
        terminal = await channel.proposal(proposal["request_id"])
        assert terminal.get("state") == "TERMINAL"
        receipt = await channel.action_receipt(proposal["action_id"])
        inner = receipt.get("receipt", receipt)
        assert inner.get("final_state") == "COMPLETED"
        provenance = (inner.get("authorization_decision") or {}).get("provenance") or {}
        assert provenance.get("proposal_request_id") == proposal["request_id"]
        assert provenance.get("operator_principal") == LOCAL_PRINCIPAL

    async def test_decline_dispatches_nothing(self, daemon) -> None:
        client, _ = daemon
        channel = _channel(client)
        proposal = await channel.create_proposal(
            capability_id=REAL_CAPABILITY,
            arguments={"finger": "index", "delta_raw": 20},
            display={"title": "t", "risk_tier": "HIGH"},
            execution_mode="REAL",
            risk_class="high",
            ttl_sec=60.0,
        )
        await channel.decide(proposal["request_id"], principal_id=LOCAL_PRINCIPAL, accept=False)
        terminal = await channel.proposal(proposal["request_id"])
        assert terminal.get("state") == "DECLINED"

    async def test_wrong_nonce_fails_closed(self, daemon) -> None:
        client, _ = daemon
        channel = _channel(client)
        proposal = await channel.create_proposal(
            capability_id=REAL_CAPABILITY,
            arguments={"finger": "index", "delta_raw": 20},
            display={"title": "t", "risk_tier": "HIGH"},
            execution_mode="REAL",
            risk_class="high",
            ttl_sec=60.0,
        )
        pending = client.list_pending_operator_proposals()["proposals"]
        with pytest.raises(DaemonRequestError):
            client.decide_operator_proposal(
                proposal["request_id"],
                decision="ACCEPT",
                principal_id=LOCAL_PRINCIPAL,
                challenge_nonce="forged-nonce",
                action_intent_hash=pending[0]["action_intent_hash"],
                channel="test",
                reason="forgery attempt",
            )
        # 未被裁决，仍是 pending（被拒绝不会改变状态）。
        still = await channel.proposal(proposal["request_id"])
        assert still.get("state") in ("CREATED", "PRESENTED")

    async def test_pending_after_decide_is_gone(self, daemon) -> None:
        client, _ = daemon
        channel = _channel(client)
        proposal = await channel.create_proposal(
            capability_id=REAL_CAPABILITY,
            arguments={"finger": "index", "delta_raw": 20},
            display={"title": "t", "risk_tier": "HIGH"},
            execution_mode="REAL",
            risk_class="high",
            ttl_sec=60.0,
        )
        await channel.decide(proposal["request_id"], principal_id=LOCAL_PRINCIPAL, accept=True)
        with pytest.raises(ConsentChannelError, match="no pending proposal"):
            await channel.decide(
                proposal["request_id"], principal_id=LOCAL_PRINCIPAL, accept=True
            )


class TestRestartInvalidation:
    async def test_pending_invalidated_by_restart(self, tmp_path: Path) -> None:
        daemon, socket_path, ledger_ctx = _start_daemon(tmp_path)
        client = DaemonClient(socket_path=socket_path, timeout_sec=5.0)
        channel = _channel(client)
        proposal = await channel.create_proposal(
            capability_id=REAL_CAPABILITY,
            arguments={"finger": "index", "delta_raw": 20},
            display={"title": "t", "risk_tier": "HIGH"},
            execution_mode="REAL",
            risk_class="high",
            ttl_sec=300.0,
        )
        daemon.stop()
        ledger_ctx.__exit__(None, None, None)
        # Restart over the same ledger: the pending proposal is durably
        # invalidated — it cannot be decided anymore (fail closed).
        daemon2, socket_path2, ledger_ctx2 = _start_daemon(tmp_path)
        client2 = DaemonClient(socket_path=socket_path2, timeout_sec=5.0)
        channel2 = _channel(client2)
        with pytest.raises(ConsentChannelError, match="no pending proposal"):
            await channel2.decide(
                proposal["request_id"], principal_id=LOCAL_PRINCIPAL, accept=True
            )
        # …并且账本里记录了 INVALIDATED 事件（可追溯）。
        from rosclaw.daemon.ledger import DaemonLedger

        with DaemonLedger(
            tmp_path / "state" / "ledger.sqlite3",
            key_path=tmp_path / "state" / "ledger.key",
        ) as ledger:
            events = ledger.events(
                entity_kind="OPERATOR_PROPOSAL", entity_id=proposal["request_id"]
            )
        assert events[-1].event_type == "OPERATOR_PROPOSAL_INVALIDATED"
        daemon2.stop()
        ledger_ctx2.__exit__(None, None, None)
