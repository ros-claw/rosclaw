"""Session revocation ordering, including actual authenticated UDS transport."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from rosclaw.daemon.client import DaemonRequestError
from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.daemon.session_manager import SessionError, SessionManager

pytest_plugins = ("tests.daemon.test_server",)


def create(manager, peer, session_id="late-session"):
    return manager.create_session(
        session_id=session_id,
        actor_id="fixture-owner",
        agent_framework="fixture",
        body_scope=["fixture-body"],
        capability_scope=["fixture.action"],
        ttl_ms=1000,
        peer=peer,
    )


def test_missing_close_stays_unknown_but_revokes_same_uid_late_creation():
    manager = SessionManager()
    peer = PeerCredentials(pid=11, uid=1001, gid=1001)
    with pytest.raises(SessionError) as missing:
        manager.close_session("late-session", peer)
    assert missing.value.code == "SESSION_NOT_FOUND"
    assert manager.status()["total"] == 0
    with pytest.raises(SessionError) as late:
        create(manager, peer)
    assert late.value.code == "SESSION_CREATION_REVOKED"
    with pytest.raises(SessionError) as missing:
        manager.get_session("late-session", peer)
    assert missing.value.code == "SESSION_NOT_FOUND"


def test_revoke_cannot_deny_other_uid_creation_or_close_its_session():
    manager = SessionManager()
    first = PeerCredentials(pid=11, uid=1001, gid=1001)
    other = PeerCredentials(pid=12, uid=1002, gid=1002)
    with pytest.raises(SessionError):
        manager.close_session("late-session", first)
    created = create(manager, other)
    with pytest.raises(SessionError) as foreign:
        manager.close_session("late-session", first)
    assert foreign.value.code == "SESSION_OWNERSHIP_MISMATCH"
    assert manager.get_session("late-session", other).state.value == "ACTIVE"
    assert create(manager, other) is created


def test_other_ids_and_existing_close_keep_original_contract():
    manager = SessionManager()
    peer = PeerCredentials(pid=11, uid=1001, gid=1001)
    with pytest.raises(SessionError):
        manager.close_session("late-session", peer)
    session = create(manager, peer, "another-session")
    assert create(manager, peer, "another-session") is session
    assert manager.close_session("another-session", peer).state.value == "CLOSED"
    with pytest.raises(SessionError) as terminal:
        create(manager, peer, "another-session")
    assert terminal.value.code == "SESSION_ID_CONFLICT"


def test_actual_uds_cleanup_precedes_delayed_creation(daemon_client, monkeypatch):
    client, _runtime, _permits, daemon = daemon_client
    entered = threading.Event()
    release = threading.Event()
    create_session = daemon.service.create_session

    def delayed_create(**kwargs):
        entered.set()
        assert release.wait(5), "Fixture creation was not released"
        return create_session(**kwargs)

    monkeypatch.setattr(daemon.service, "create_session", delayed_create)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            client.create_session,
            session_id="fixture-delayed-session",
            actor_id="fixture-owner",
            agent_framework="fixture",
            body_scope=["rh56-test"],
            capability_scope=["rh56.finger.move"],
        )
        try:
            assert entered.wait(1)
            with pytest.raises(DaemonRequestError) as missing:
                client.close_session("fixture-delayed-session", reason="fixture_cancel")
            assert missing.value.code == "SESSION_NOT_FOUND"
        finally:
            release.set()
        with pytest.raises(DaemonRequestError) as late:
            pending.result(timeout=3)
        assert late.value.code == "SESSION_CREATION_REVOKED"
    with pytest.raises(DaemonRequestError) as missing:
        client.get_session("fixture-delayed-session")
    assert missing.value.code == "SESSION_NOT_FOUND"
    assert client.get_runtime_status()["hardware_actions_executed"] == 0


def test_actual_uds_timed_out_create_cannot_arrive_after_cleanup(daemon_client, monkeypatch):
    from rosclaw.daemon.client import DaemonClient, DaemonUnavailableError
    from rosclaw.daemon.service import ControlPlaneError

    client, _runtime, _permits, daemon = daemon_client
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    errors = []
    create_session = daemon.service.create_session

    def delayed_create(**kwargs):
        entered.set()
        try:
            assert release.wait(5)
            return create_session(**kwargs)
        except ControlPlaneError as exc:
            errors.append(exc.code)
            raise
        finally:
            finished.set()

    monkeypatch.setattr(daemon.service, "create_session", delayed_create)
    slow = DaemonClient(socket_path=client.socket_path, timeout_sec=0.1)
    try:
        with pytest.raises(DaemonUnavailableError):
            slow.create_session(
                session_id="fixture-timed-out-session",
                actor_id="fixture-owner",
                agent_framework="fixture",
                body_scope=["rh56-test"],
                capability_scope=["rh56.finger.move"],
            )
        assert entered.wait(2)
        with pytest.raises(DaemonRequestError) as missing:
            client.close_session("fixture-timed-out-session", reason="fixture_timeout_cleanup")
        assert missing.value.code == "SESSION_NOT_FOUND"
    finally:
        release.set()
    assert finished.wait(2)
    assert errors == ["SESSION_CREATION_REVOKED"]
    with pytest.raises(DaemonRequestError) as missing:
        client.get_session("fixture-timed-out-session")
    assert missing.value.code == "SESSION_NOT_FOUND"
    assert client.get_runtime_status()["hardware_actions_executed"] == 0


def test_legacy_action_adoption_cannot_recreate_revoked_session():
    from rosclaw.kernel import ActionEnvelope, ExecutionMode

    manager = SessionManager()
    peer = PeerCredentials(pid=11, uid=1001, gid=1001)
    with pytest.raises(SessionError):
        manager.close_session("late-session", peer)
    action = ActionEnvelope(
        session_id="late-session",
        actor_id="fixture-owner",
        agent_framework="fixture",
        body_id="fixture-body",
        capability_id="fixture.action",
        arguments={},
        execution_mode=ExecutionMode.SIMULATION,
    )
    with pytest.raises(SessionError) as late:
        manager.adopt_action(action, peer)
    assert late.value.code == "SESSION_CREATION_REVOKED"
