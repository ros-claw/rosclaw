"""Agent Session, Action Lease, and watchdog fault behavior."""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.daemon.service import ControlPlaneError, DaemonControlPlane
from rosclaw.daemon.session_manager import SessionError, SessionManager
from rosclaw.kernel import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    EvidenceLevel,
    ExecutionMode,
    OrphanPolicy,
    VerificationPolicy,
)


def _runtime() -> Runtime:
    return Runtime(
        RuntimeConfig(
            robot_id="lease-test",
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


def _action(
    action_id: str,
    *,
    session_id: str = "session-lease",
    ttl_ms: int = 120,
    orphan_policy: OrphanPolicy = OrphanPolicy.STOP_ON_CLIENT_LOSS,
    deadline_sec: float = 5.0,
) -> ActionEnvelope:
    return ActionEnvelope(
        action_id=action_id,
        actor_id="codex-agent",
        agent_framework="codex",
        session_id=session_id,
        body_id="lease-test",
        body_snapshot_hash="sha256:lease-test",
        capability_id="fixture.long-action",
        arguments={},
        execution_mode=ExecutionMode.SHADOW,
        deadline_at=datetime.now(UTC) + timedelta(seconds=deadline_sec),
        lease_ttl_ms=ttl_ms,
        renew_interval_ms=max(100, ttl_ms - 20),
        orphan_policy=orphan_policy,
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.TASK_VERIFIED,
            timeout_sec=max(0.1, deadline_sec),
        ),
    )


def _wait_state(
    service: DaemonControlPlane,
    action_id: str,
    peer: PeerCredentials,
    state: str,
    *,
    timeout_sec: float = 2.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        status = service.get_action_status(action_id, peer)
        if status["state"] == state:
            return status
        time.sleep(0.01)
    pytest.fail(f"action {action_id} did not reach {state}")


def _blocking_executor(
    started: threading.Event,
    release: threading.Event,
):
    def execute(_action: ActionEnvelope) -> ActionExecutionResult:
        started.set()
        assert release.wait(timeout=3.0)
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
            dispatch_result={"accepted": True},
            driver_ack={"acknowledged": True},
            verification_result={"verified": True},
        )

    return execute


def test_session_manager_enforces_uid_scope_and_monotonic_expiry() -> None:
    manager = SessionManager()
    owner = PeerCredentials(pid=11, uid=1001, gid=1001)
    other = PeerCredentials(pid=12, uid=1002, gid=1002)
    session = manager.create_session(
        session_id="session-one",
        actor_id="agent-one",
        agent_framework="codex",
        body_scope=["body-one"],
        capability_scope=["capability.one"],
        ttl_ms=300,
        peer=owner,
    )

    with pytest.raises(SessionError, match="does not own"):
        manager.heartbeat(session.session_id, other)
    expired = manager.expire_sessions(now_monotonic=session.expires_monotonic + 0.001)

    assert [item.session_id for item in expired] == [session.session_id]
    assert manager.status() == {"total": 1, "active": 0, "lost": 1, "closed": 0}


def test_action_lease_expiry_requests_stop_and_overrides_late_success() -> None:
    runtime = _runtime()
    started = threading.Event()
    release = threading.Event()
    runtime.action_gateway.register_executor(
        "fixture.long-action",
        ExecutionMode.SHADOW,
        _blocking_executor(started, release),
    )
    service = DaemonControlPlane(runtime=runtime)
    peer = PeerCredentials(pid=11, uid=os.geteuid(), gid=os.getegid())
    action = _action("action-lease-expires")
    service.start()
    try:
        service.request_action(action, peer)
        assert started.wait(timeout=1.0)
        deadline = time.monotonic() + 2.0
        while not runtime.emergency_stop_latched and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runtime.emergency_stop_latched is True
        release.set()
        status = _wait_state(service, action.action_id, peer, "FINISHED")
    finally:
        release.set()
        service.close()

    receipt = status["receipt"]
    assert isinstance(receipt, dict)
    assert receipt["final_state"] == "TIMED_OUT"
    assert receipt["usable_for_real_execution"] is False
    assert receipt["errors"][-1]["code"] == "ACTION_LEASE_EXPIRED"
    assert isinstance(receipt["safety_stop"], dict)


def test_expired_action_is_rejected_before_executor_dispatch() -> None:
    runtime = _runtime()
    dispatched = threading.Event()
    runtime.action_gateway.register_executor(
        "fixture.long-action",
        ExecutionMode.SHADOW,
        lambda _action: dispatched.set(),
    )
    service = DaemonControlPlane(runtime=runtime)
    peer = PeerCredentials(pid=11, uid=os.geteuid(), gid=os.getegid())
    action = _action("action-already-expired", deadline_sec=-0.01)
    service.start()
    try:
        with pytest.raises(ControlPlaneError) as rejected:
            service.request_action(action, peer)
    finally:
        service.close()

    assert rejected.value.code == "ACTION_DEADLINE_EXPIRED"
    assert dispatched.is_set() is False


def test_action_expired_in_queue_never_reaches_executor() -> None:
    runtime = _runtime()
    first_started = threading.Event()
    release_first = threading.Event()
    dispatched: list[str] = []

    def execute(action: ActionEnvelope) -> ActionExecutionResult:
        dispatched.append(action.action_id)
        if action.action_id == "action-holds-worker":
            first_started.set()
            assert release_first.wait(timeout=3.0)
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
            dispatch_result={"accepted": True},
            driver_ack={"acknowledged": True},
            verification_result={"verified": True},
        )

    runtime.action_gateway.register_executor(
        "fixture.long-action",
        ExecutionMode.SHADOW,
        execute,
    )
    service = DaemonControlPlane(runtime=runtime, max_workers=1)
    peer = PeerCredentials(pid=11, uid=os.geteuid(), gid=os.getegid())
    first = _action("action-holds-worker", ttl_ms=1_000)
    queued = _action("action-expires-queued", ttl_ms=1_000, deadline_sec=0.08)
    service.start()
    try:
        service.request_action(first, peer)
        assert first_started.wait(timeout=1.0)
        service.request_action(queued, peer)
        time.sleep(0.12)
        release_first.set()
        status = _wait_state(service, queued.action_id, peer, "FINISHED")
    finally:
        release_first.set()
        service.close()

    assert dispatched == [first.action_id]
    receipt = status["receipt"]
    assert isinstance(receipt, dict)
    assert receipt["final_state"] == "TIMED_OUT"
    assert receipt["errors"][-1]["code"] == "ACTION_DEADLINE_EXPIRED"


def test_session_close_orphans_active_action_and_revokes_ownership() -> None:
    runtime = _runtime()
    started = threading.Event()
    release = threading.Event()
    runtime.action_gateway.register_executor(
        "fixture.long-action",
        ExecutionMode.SHADOW,
        _blocking_executor(started, release),
    )
    service = DaemonControlPlane(runtime=runtime)
    peer = PeerCredentials(pid=11, uid=os.geteuid(), gid=os.getegid())
    action = _action("action-session-close", ttl_ms=1_000)
    service.start()
    try:
        service.create_session(
            session_id=action.session_id,
            actor_id=action.actor_id,
            agent_framework=action.agent_framework,
            body_scope=[action.body_id],
            capability_scope=[action.capability_id],
            ttl_ms=1_000,
            peer=peer,
        )
        service.request_action(action, peer)
        assert started.wait(timeout=1.0)
        service.close_session(action.session_id, peer, reason="agent_process_exit")
        release.set()
        status = _wait_state(service, action.action_id, peer, "FINISHED")
    finally:
        release.set()
        service.close()

    receipt = status["receipt"]
    assert isinstance(receipt, dict)
    assert receipt["final_state"] == "ORPHANED"
    assert receipt["errors"][-1]["code"] == "AGENT_SESSION_LOST"
    assert runtime.emergency_stop_latched is True


def test_continue_until_deadline_does_not_claim_or_request_stop_on_client_loss() -> None:
    runtime = _runtime()
    started = threading.Event()
    release = threading.Event()
    runtime.action_gateway.register_executor(
        "fixture.long-action",
        ExecutionMode.SHADOW,
        _blocking_executor(started, release),
    )
    service = DaemonControlPlane(runtime=runtime)
    peer = PeerCredentials(pid=11, uid=os.geteuid(), gid=os.getegid())
    action = _action(
        "action-continue-bounded",
        ttl_ms=300,
        orphan_policy=OrphanPolicy.CONTINUE_UNTIL_DEADLINE,
    )
    service.start()
    try:
        service.request_action(action, peer)
        assert started.wait(timeout=1.0)
        service.close_session(action.session_id, peer, reason="agent_process_exit")
        time.sleep(0.35)
        assert service.get_action_status(action.action_id, peer)["state"] == "RUNNING"
        assert runtime.emergency_stop_latched is False
        release.set()
        status = _wait_state(service, action.action_id, peer, "FINISHED")
    finally:
        release.set()
        service.close()

    receipt = status["receipt"]
    assert isinstance(receipt, dict)
    assert receipt["final_state"] == "COMPLETED"


def test_action_lease_renewal_heartbeats_session_and_keeps_action_active() -> None:
    runtime = _runtime()
    started = threading.Event()
    release = threading.Event()
    runtime.action_gateway.register_executor(
        "fixture.long-action",
        ExecutionMode.SHADOW,
        _blocking_executor(started, release),
    )
    service = DaemonControlPlane(runtime=runtime)
    peer = PeerCredentials(pid=11, uid=os.geteuid(), gid=os.getegid())
    action = _action("action-renewed", ttl_ms=300)
    service.start()
    try:
        service.request_action(action, peer)
        assert started.wait(timeout=1.0)
        for _index in range(4):
            time.sleep(0.12)
            renewed = service.renew_action_lease(action.action_id, action.session_id, peer)
            assert renewed["action_lease"]["active"] is True
        assert runtime.emergency_stop_latched is False
        release.set()
        status = _wait_state(service, action.action_id, peer, "FINISHED")
    finally:
        release.set()
        service.close()

    receipt = status["receipt"]
    assert isinstance(receipt, dict)
    assert receipt["final_state"] == "COMPLETED"


def test_daemon_generation_starts_disarmed_and_requires_service_uid_to_arm() -> None:
    service = DaemonControlPlane(runtime=_runtime())
    daemon_peer = PeerCredentials(pid=11, uid=os.geteuid(), gid=os.getegid())
    untrusted_peer = PeerCredentials(pid=12, uid=os.geteuid() + 1, gid=os.getegid())
    service.start()
    try:
        assert service.get_runtime_status(daemon_peer)["supervision_state"] == "DISARMED"
        with pytest.raises(ControlPlaneError) as rejected:
            service.arm_runtime("untrusted request", untrusted_peer)
        armed = service.arm_runtime("operator preflight complete", daemon_peer)
    finally:
        service.close()

    assert rejected.value.code == "PERMISSION_DENIED"
    assert armed["supervision_state"] == "ARMED"
