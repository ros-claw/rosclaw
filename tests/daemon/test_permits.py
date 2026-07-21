"""Server-side permit tests: caller approval claims are never authoritative."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.daemon.ledger import DaemonLedger, LedgerIntegrityError
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

_ACTION_DEADLINE = datetime(2099, 1, 1, tzinfo=UTC)


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
        deadline_at=_ACTION_DEADLINE,
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


def test_daemon_restart_invalidates_even_consumed_permit_generation(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    permitted_action = _action(action_id="action-before-restart")
    permit = ExecutionPermit(
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
    peer = PeerCredentials(pid=101, uid=1001, gid=1001)

    with DaemonLedger(database, key_path=key) as ledger:
        authority = PermitAuthority(ledger=ledger)
        authority.register(permit)
        assert authority.authorize(permitted_action, peer).allowed is True

    with DaemonLedger(database, key_path=key) as ledger:
        restored = PermitAuthority(ledger=ledger)
        restored.register(permit)
        same_action = restored.authorize(permitted_action, peer)
        replay = restored.authorize(_action(action_id="action-after-restart"), peer)

    assert same_action.allowed is False
    assert same_action.code == "PERMIT_REVOKED"
    assert replay.allowed is False
    assert replay.code == "PERMIT_REVOKED"


def test_concurrent_single_use_permit_has_one_durable_winner(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=1001, gid=1001)
    actions = [_action(action_id=f"action-race-{index}") for index in range(8)]
    permit = ExecutionPermit(
        permit_id="permit-1",
        principal_id="operator-1",
        peer_uid=peer.uid,
        body_id="rh56-test",
        body_snapshot_hash="sha256:body",
        capabilities=("rh56.finger.move",),
        action_intent_hash=action_intent_hash(actions[0]),
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )

    with DaemonLedger(database, key_path=key) as ledger:
        authority = PermitAuthority(ledger=ledger)
        authority.register(permit)
        with ThreadPoolExecutor(max_workers=len(actions)) as executor:
            decisions = list(
                executor.map(lambda action: authority.authorize(action, peer), actions)
            )

    with DaemonLedger(database, key_path=key) as ledger:
        restored = PermitAuthority(ledger=ledger)
        consumption_events = [
            event
            for event in ledger.events(entity_kind="PERMIT", entity_id=permit.permit_id)
            if event.event_type == "PERMIT_CONSUMED"
        ]

    assert sum(decision.allowed for decision in decisions) == 1
    assert {decision.code for decision in decisions if not decision.allowed} == {"PERMIT_EXHAUSTED"}
    assert len(consumption_events) == 1
    assert restored.status()["consumed_actions"] == 1


def test_restore_rejects_authenticated_permit_overconsumption(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    action = _action(action_id="action-legitimate")
    peer = PeerCredentials(pid=101, uid=1001, gid=1001)
    permit = ExecutionPermit(
        permit_id="permit-1",
        principal_id="operator-1",
        peer_uid=peer.uid,
        body_id="rh56-test",
        body_snapshot_hash="sha256:body",
        capabilities=("rh56.finger.move",),
        action_intent_hash=action_intent_hash(action),
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
        max_uses=1,
    )
    with DaemonLedger(database, key_path=key) as ledger:
        authority = PermitAuthority(ledger=ledger)
        authority.register(permit)
        assert authority.authorize(action, peer).allowed is True
        ledger.append(
            "PERMIT_CONSUMED",
            entity_kind="PERMIT",
            entity_id=permit.permit_id,
            payload={
                "permit_id": permit.permit_id,
                "action_id": "action-illegal-second-use",
                "peer_uid": peer.uid,
                "action_intent_hash": permit.action_intent_hash,
            },
        )

    with (
        DaemonLedger(database, key_path=key) as ledger,
        pytest.raises(LedgerIntegrityError, match="max_uses"),
    ):
        PermitAuthority(ledger=ledger)


def test_restore_rejects_coerced_permit_consumption_fields(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    action = _action(action_id="action-typed-consumption")
    permit = ExecutionPermit(
        permit_id="permit-1",
        principal_id="operator-1",
        peer_uid=1001,
        body_id="rh56-test",
        body_snapshot_hash="sha256:body",
        capabilities=("rh56.finger.move",),
        action_intent_hash=action_intent_hash(action),
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )
    with DaemonLedger(database, key_path=key) as ledger:
        authority = PermitAuthority(ledger=ledger)
        authority.register(permit)
        ledger.append(
            "PERMIT_CONSUMED",
            entity_kind="PERMIT",
            entity_id=permit.permit_id,
            payload={
                "permit_id": permit.permit_id,
                "action_id": 123,
                "peer_uid": permit.peer_uid,
                "action_intent_hash": permit.action_intent_hash,
            },
        )

    with (
        DaemonLedger(database, key_path=key) as ledger,
        pytest.raises(LedgerIntegrityError, match="consumption is invalid"),
    ):
        PermitAuthority(ledger=ledger)


def test_action_id_cannot_consume_a_second_permit() -> None:
    authority = PermitAuthority()
    peer = PeerCredentials(pid=101, uid=1001, gid=1001)
    first_action = _action(action_id="action-one-permit", approval_id="permit-1")
    second_action = _action(action_id="action-one-permit", approval_id="permit-2")
    for permit_id in ("permit-1", "permit-2"):
        authority.register(
            ExecutionPermit(
                permit_id=permit_id,
                principal_id="operator-1",
                peer_uid=peer.uid,
                body_id="rh56-test",
                body_snapshot_hash="sha256:body",
                capabilities=("rh56.finger.move",),
                action_intent_hash=action_intent_hash(first_action),
                expires_at=datetime.now(UTC) + timedelta(minutes=1),
            )
        )

    assert authority.authorize(first_action, peer).allowed is True
    conflict = authority.authorize(second_action, peer)

    assert conflict.allowed is False
    assert conflict.code == "PERMIT_ACTION_ID_CONFLICT"
    assert authority.status()["consumed_actions"] == 1


def test_restore_rejects_duplicate_authenticated_permit_registration(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    action = _action()
    permit = ExecutionPermit(
        permit_id="permit-1",
        principal_id="operator-1",
        peer_uid=1001,
        body_id="rh56-test",
        body_snapshot_hash="sha256:body",
        capabilities=("rh56.finger.move",),
        action_intent_hash=action_intent_hash(action),
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )
    with DaemonLedger(database, key_path=key) as ledger:
        for _index in range(2):
            ledger.append(
                "PERMIT_REGISTERED",
                entity_kind="PERMIT",
                entity_id=permit.permit_id,
                payload={"permit": permit.to_dict()},
            )

    with (
        DaemonLedger(database, key_path=key) as ledger,
        pytest.raises(LedgerIntegrityError, match="duplicate registration"),
    ):
        PermitAuthority(ledger=ledger)


def test_persisted_permit_parser_rejects_coerced_identifiers() -> None:
    payload = ExecutionPermit(
        permit_id="permit-1",
        principal_id="operator-1",
        peer_uid=1001,
        body_id="rh56-test",
        body_snapshot_hash="sha256:body",
        capabilities=("rh56.finger.move",),
        action_intent_hash=action_intent_hash(_action()),
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    ).to_dict()
    payload["permit_id"] = 123

    with pytest.raises(ValueError, match="permit_id"):
        ExecutionPermit.from_dict(payload)


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
