"""Unit tests for exact, bounded Operator Broker proposals."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.kernel import ActionEnvelope, AuthorizationContext, ExecutionMode
from rosclaw.operator import OperatorProposalError, OperatorProposalStore, ProposalState


def _action(*, action_id: str = "action-proposal", delta: int = 20) -> ActionEnvelope:
    return ActionEnvelope(
        action_id=action_id,
        actor_id="codex-agent",
        agent_framework="codex",
        session_id=f"session-{action_id}",
        body_id="rh56-test",
        body_snapshot_hash="sha256:body",
        capability_id="rh56.finger.move",
        arguments={"finger": "index", "delta_raw": delta},
        execution_mode=ExecutionMode.REAL,
        deadline_at=datetime.now(UTC) + timedelta(minutes=2),
        authorization=AuthorizationContext(
            principal_id="forged",
            approved=True,
            approval_id="forged-permit",
            scopes=["*"],
        ),
    )


def test_proposal_strips_caller_approval_and_hides_operator_challenge() -> None:
    store = OperatorProposalStore()
    proposal = store.create(
        _action(),
        display={"title": "Move one finger", "risk_tier": "HIGH"},
        origin_peer=PeerCredentials(pid=123, uid=1001, gid=1001),
        daemon_instance_id="daemon-test",
        ttl_sec=60.0,
    )

    assert proposal.action.authorization == AuthorizationContext()
    assert "challenge_nonce" not in proposal.public_dict()
    assert proposal.operator_dict()["challenge_nonce"] == proposal.challenge_nonce
    assert proposal.public_dict()["origin"] == {"uid": 1001, "pid": 123}


def test_same_action_id_cannot_be_rebound_to_changed_intent() -> None:
    store = OperatorProposalStore()
    peer = PeerCredentials(pid=123, uid=1001, gid=1001)
    store.create(
        _action(delta=20),
        display={"title": "Move"},
        origin_peer=peer,
        daemon_instance_id="daemon-test",
        ttl_sec=60.0,
    )

    with pytest.raises(OperatorProposalError) as error:
        store.create(
            _action(delta=200),
            display={"title": "Move farther"},
            origin_peer=peer,
            daemon_instance_id="daemon-test",
            ttl_sec=60.0,
        )

    assert error.value.code == "PROPOSAL_ACTION_ID_CONFLICT"


def test_expired_proposal_can_never_return_to_pending() -> None:
    store = OperatorProposalStore()
    created = datetime.now(UTC)
    proposal = store.create(
        _action(),
        display={"title": "Move"},
        origin_peer=PeerCredentials(pid=123, uid=1001, gid=1001),
        daemon_instance_id="daemon-test",
        ttl_sec=5.0,
        now=created,
    )

    assert store.pending(now=created + timedelta(seconds=6)) == []
    assert proposal.state is ProposalState.EXPIRED
    assert proposal.failure_code == "PROPOSAL_EXPIRED"
