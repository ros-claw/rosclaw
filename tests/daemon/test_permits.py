"""Server-side permit tests: caller approval claims are never authoritative."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rosclaw.daemon.permits import (
    ExecutionPermit,
    PermitAuthority,
    action_intent_hash,
)
from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.kernel import (
    ActionEnvelope,
    AuthorizationContext,
    ExecutionMode,
)


def _action(
    *,
    action_id: str = "action-permit-test",
    approval_id: str = "permit-1",
) -> ActionEnvelope:
    return ActionEnvelope(
        action_id=action_id,
        actor_id="codex-agent",
        agent_framework="codex",
        session_id="session-1",
        body_id="rh56-test",
        body_snapshot_hash="sha256:body",
        capability_id="rh56.finger.move",
        arguments={"finger": "index", "delta_raw": 20},
        execution_mode=ExecutionMode.REAL,
        authorization=AuthorizationContext(
            principal_id="operator-1",
            approved=True,
            approval_id=approval_id,
            scopes=["*"],
        ),
    )


def test_forged_caller_approval_is_rejected_without_server_permit() -> None:
    authority = PermitAuthority()

    decision = authority.authorize(
        _action(),
        PeerCredentials(pid=101, uid=1001, gid=1001),
    )

    assert decision.allowed is False
    assert decision.code == "AUTHORIZATION_REQUIRED"
    assert decision.authorization.approved is False


def test_permit_is_bound_to_peer_body_snapshot_capability_and_single_use() -> None:
    authority = PermitAuthority()
    permitted_action = _action()
    authority.register(
        ExecutionPermit(
            permit_id="permit-1",
            principal_id="operator-1",
            peer_uid=1001,
            body_id="rh56-test",
            body_snapshot_hash="sha256:body",
            capabilities=("rh56.finger.move",),
            action_intent_hash=action_intent_hash(permitted_action),
            expires_at=datetime.now(UTC) + timedelta(minutes=1),
            max_uses=1,
        )
    )
    peer = PeerCredentials(pid=101, uid=1001, gid=1001)

    first = authority.authorize(permitted_action, peer)
    second = authority.authorize(_action(action_id="action-second"), peer)

    assert first.allowed is True
    assert first.authorization.approved is True
    assert first.authorization.scopes == ["rh56.finger.move"]
    assert second.allowed is False
    assert second.code == "PERMIT_EXHAUSTED"


def test_permit_rejects_wrong_peer_before_consuming_use() -> None:
    authority = PermitAuthority()
    permitted_action = _action()
    authority.register(
        ExecutionPermit(
            permit_id="permit-1",
            principal_id="operator-1",
            peer_uid=1001,
            body_id="rh56-test",
            body_snapshot_hash="sha256:body",
            capabilities=("rh56.finger.move",),
            action_intent_hash=action_intent_hash(permitted_action),
            expires_at=datetime.now(UTC) + timedelta(minutes=1),
        )
    )

    wrong_peer = authority.authorize(
        permitted_action,
        PeerCredentials(pid=202, uid=2002, gid=2002),
    )
    right_peer = authority.authorize(
        permitted_action,
        PeerCredentials(pid=101, uid=1001, gid=1001),
    )

    assert wrong_peer.allowed is False
    assert wrong_peer.code == "PERMIT_PEER_MISMATCH"
    assert right_peer.allowed is True


def test_permit_rejects_argument_substitution_before_consuming_use() -> None:
    authority = PermitAuthority()
    permitted_action = _action()
    authority.register(
        ExecutionPermit(
            permit_id="permit-1",
            principal_id="operator-1",
            peer_uid=1001,
            body_id="rh56-test",
            body_snapshot_hash="sha256:body",
            capabilities=("rh56.finger.move",),
            action_intent_hash=action_intent_hash(permitted_action),
            expires_at=datetime.now(UTC) + timedelta(minutes=1),
        )
    )
    substituted = _action(action_id="action-substituted")
    substituted.arguments["delta_raw"] = 200
    peer = PeerCredentials(pid=101, uid=1001, gid=1001)

    rejected = authority.authorize(substituted, peer)
    accepted = authority.authorize(permitted_action, peer)

    assert rejected.allowed is False
    assert rejected.code == "PERMIT_INTENT_MISMATCH"
    assert accepted.allowed is True


def test_permit_rejects_wildcard_capability() -> None:
    with pytest.raises(ValueError, match="wildcard"):
        ExecutionPermit(
            permit_id="permit-1",
            principal_id="operator-1",
            peer_uid=1001,
            body_id="rh56-test",
            body_snapshot_hash="sha256:body",
            capabilities=("*",),
            action_intent_hash=action_intent_hash(_action()),
            expires_at=datetime.now(UTC) + timedelta(minutes=1),
        )
