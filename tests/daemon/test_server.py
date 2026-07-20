"""End-to-end Unix-socket tests for the rosclawd process boundary."""

from __future__ import annotations

import os
import socket
import stat
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.client import (
    DaemonClient,
    DaemonRequestError,
    DaemonSecurityError,
    DaemonUnavailableError,
)
from rosclaw.daemon.permits import (
    ExecutionPermit,
    PermitAuthority,
    action_intent_hash,
)
from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import ControlPlaneError, DaemonControlPlane
from rosclaw.kernel import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)


def _runtime() -> Runtime:
    return Runtime(
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


def _real_action(
    *,
    action_id: str,
    approval_id: str = "forged-permit",
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
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.DRIVER_CONFIRMED,
            timeout_sec=2.0,
        ),
    )


def _shadow_action(*, action_id: str) -> ActionEnvelope:
    return ActionEnvelope(
        action_id=action_id,
        actor_id="codex-agent",
        agent_framework="codex",
        session_id="session-1",
        body_id="rh56-test",
        body_snapshot_hash="sha256:body",
        capability_id="rh56.finger.move",
        arguments={"finger": "index", "delta_raw": 20},
        execution_mode=ExecutionMode.SHADOW,
    )


@pytest.fixture
def daemon_client(tmp_path: Path) -> tuple[DaemonClient, Runtime, PermitAuthority, RosclawDaemon]:
    runtime = _runtime()
    permits = PermitAuthority()
    service = DaemonControlPlane(runtime=runtime, permits=permits)
    socket_path = tmp_path / "run" / "rosclawd.sock"
    daemon = RosclawDaemon(service=service, socket_path=socket_path)
    daemon.start()
    client = DaemonClient(socket_path=socket_path, timeout_sec=2.0)
    try:
        yield client, runtime, permits, daemon
    finally:
        daemon.stop()


def test_status_proves_a_separate_kernel_authenticated_socket_boundary(
    daemon_client: tuple[DaemonClient, Runtime, PermitAuthority, RosclawDaemon],
) -> None:
    client, _runtime_instance, _permits, daemon = daemon_client

    status = client.get_runtime_status()
    socket_mode = stat.S_IMODE(daemon.socket_path.stat().st_mode)

    assert status["protocol_version"] == "rosclaw.daemon.v1"
    assert status["daemon_pid"] == os.getpid()
    assert status["client_peer"]["pid"] == os.getpid()
    assert status["southbound_owner"] == "rosclawd"
    assert status["hardware_actions_executed"] == 0
    assert socket_mode == 0o600


def test_forged_real_authorization_is_blocked_before_executor_dispatch(
    daemon_client: tuple[DaemonClient, Runtime, PermitAuthority, RosclawDaemon],
) -> None:
    client, runtime, _permits, _daemon = daemon_client
    dispatched: list[ActionEnvelope] = []

    def execute(action: ActionEnvelope) -> ActionExecutionResult:
        dispatched.append(action)
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.DRIVER_CONFIRMED,
        )

    runtime.action_gateway.register_executor(
        "rh56.finger.move",
        ExecutionMode.REAL,
        execute,
    )

    ticket = client.request_action(_real_action(action_id="action-forged"))
    status = client.wait_for_action(ticket["action_id"], timeout_sec=2.0)
    receipt = status["receipt"]

    assert status["state"] == "FINISHED"
    assert status["final_state"] == "BLOCKED"
    assert status["error_code"] == "AUTHORIZATION_REQUIRED"
    assert receipt["final_state"] == "BLOCKED"
    assert receipt["errors"][0]["code"] == "AUTHORIZATION_REQUIRED"
    assert dispatched == []
    assert client.get_runtime_status()["hardware_actions_executed"] == 0


def test_unknown_method_is_rejected_by_exact_allowlist(
    daemon_client: tuple[DaemonClient, Runtime, PermitAuthority, RosclawDaemon],
) -> None:
    client, _runtime_instance, _permits, _daemon = daemon_client

    with pytest.raises(DaemonRequestError) as error:
        client.call("driver.import", {"module": "serial"})

    assert error.value.code == "METHOD_NOT_ALLOWED"


def test_action_id_replay_is_idempotent_only_for_the_same_request(
    daemon_client: tuple[DaemonClient, Runtime, PermitAuthority, RosclawDaemon],
) -> None:
    client, _runtime_instance, _permits, _daemon = daemon_client
    original = _shadow_action(action_id="action-idempotency")

    first = client.request_action(original)
    repeated = client.request_action(original)
    conflicting = _shadow_action(action_id="action-idempotency")
    conflicting.arguments["delta_raw"] = 21

    assert repeated["action_id"] == first["action_id"]
    with pytest.raises(DaemonRequestError) as error:
        client.request_action(conflicting)
    assert error.value.code == "ACTION_ID_CONFLICT"


def test_action_status_receipt_and_cancel_are_bound_to_authenticated_uid(
    daemon_client: tuple[DaemonClient, Runtime, PermitAuthority, RosclawDaemon],
) -> None:
    client, _runtime_instance, _permits, daemon = daemon_client
    action_id = "action-owner-bound"
    client.request_action(_shadow_action(action_id=action_id))
    other_peer = PeerCredentials(pid=99999, uid=os.geteuid() + 1, gid=os.getegid())

    for operation in (
        daemon.service.get_action_status,
        daemon.service.get_execution_receipt,
        daemon.service.cancel_action,
    ):
        with pytest.raises(ControlPlaneError) as error:
            operation(action_id, other_peer)
        assert error.value.code == "ACTION_OWNERSHIP_MISMATCH"


def test_client_rejects_daemon_uid_outside_operator_expectation(
    daemon_client: tuple[DaemonClient, Runtime, PermitAuthority, RosclawDaemon],
) -> None:
    client, _runtime_instance, _permits, _daemon = daemon_client
    guarded_client = DaemonClient(
        socket_path=client.socket_path,
        expected_daemon_uid=os.geteuid() + 1,
    )

    with pytest.raises(DaemonSecurityError) as error:
        guarded_client.get_runtime_status()

    assert error.value.code == "UNEXPECTED_DAEMON_UID"


def test_client_rejects_socket_inside_writable_directory(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "attacker-controlled"
    runtime_dir.mkdir()
    runtime_dir.chmod(0o777)
    socket_path = runtime_dir / "rosclawd.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(socket_path))
        socket_path.chmod(0o600)

        with pytest.raises(DaemonSecurityError) as error:
            DaemonClient(socket_path=socket_path).get_runtime_status()
    finally:
        listener.close()

    assert error.value.code == "WRITABLE_SOCKET_DIRECTORY"


def test_daemon_side_single_use_permit_allows_exactly_one_dispatch(
    daemon_client: tuple[DaemonClient, Runtime, PermitAuthority, RosclawDaemon],
) -> None:
    client, runtime, permits, _daemon = daemon_client
    dispatched: list[str] = []

    def execute(action: ActionEnvelope) -> ActionExecutionResult:
        dispatched.append(action.action_id)
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.DRIVER_CONFIRMED,
            authorization_decision={"authorized": True},
            dispatch_result={"accepted": True},
            driver_ack={"acknowledged": True},
        )

    runtime.action_gateway.register_executor(
        "rh56.finger.move",
        ExecutionMode.REAL,
        execute,
    )
    first = _real_action(action_id="action-allowed", approval_id="permit-daemon")
    permits.register(
        ExecutionPermit(
            permit_id="permit-daemon",
            principal_id="operator-1",
            peer_uid=os.getuid(),
            body_id="rh56-test",
            body_snapshot_hash="sha256:body",
            capabilities=("rh56.finger.move",),
            action_intent_hash=action_intent_hash(first),
            expires_at=datetime.now(UTC) + timedelta(minutes=1),
            max_uses=1,
        )
    )

    second = _real_action(action_id="action-replay", approval_id="permit-daemon")
    first_receipt = client.wait_for_action(
        client.request_action(first)["action_id"],
        timeout_sec=2.0,
    )["receipt"]
    second_receipt = client.wait_for_action(
        client.request_action(second)["action_id"],
        timeout_sec=2.0,
    )["receipt"]

    assert first_receipt["final_state"] == "COMPLETED"
    assert second_receipt["final_state"] == "BLOCKED"
    assert second_receipt["errors"][0]["code"] == "PERMIT_EXHAUSTED"
    assert dispatched == ["action-allowed"]
    assert client.get_runtime_status()["hardware_actions_executed"] == 1


def test_emergency_stop_uses_daemon_driver_even_if_async_subsystems_are_absent(
    daemon_client: tuple[DaemonClient, Runtime, PermitAuthority, RosclawDaemon],
) -> None:
    client, runtime, _permits, _daemon = daemon_client

    class Driver:
        def emergency_stop(self) -> dict[str, Any]:
            return {
                "acknowledged": True,
                "physical_stop_observed": False,
                "execution_mode": "REAL",
            }

    runtime.register_driver("rh56", Driver())

    receipt = client.emergency_stop("operator halt", source="test")

    assert receipt["request_dispatched"] is True
    assert receipt["driver_acknowledged"] is True
    assert receipt["physical_stop_observed"] is False
    assert receipt["final_status"] == "ACKNOWLEDGED"
    assert receipt["authenticated_peer"]["uid"] == os.geteuid()
    assert receipt["requested_source"] == "test"


def test_unexpected_executor_failure_produces_terminal_failure_receipt(
    daemon_client: tuple[DaemonClient, Runtime, PermitAuthority, RosclawDaemon],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runtime, _permits, _daemon = daemon_client

    def fail(_action: ActionEnvelope) -> None:
        raise RuntimeError("executor implementation crashed")

    monkeypatch.setattr(runtime, "submit_action", fail)
    ticket = client.request_action(_shadow_action(action_id="action-daemon-failure"))
    status = client.wait_for_action(ticket["action_id"], timeout_sec=2.0)

    assert status["state"] == "FINISHED"
    assert status["receipt"]["final_state"] == "FAILED"
    assert status["receipt"]["errors"][0]["code"] == "DAEMON_ACTION_FAILED"


def test_stopping_daemon_makes_further_action_requests_impossible(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "run" / "rosclawd.sock"
    daemon = RosclawDaemon(
        service=DaemonControlPlane(runtime=_runtime()),
        socket_path=socket_path,
    )
    daemon.start()
    client = DaemonClient(socket_path=socket_path, timeout_sec=0.2)
    assert client.get_runtime_status()["running"] is True

    daemon.stop()

    with pytest.raises(DaemonUnavailableError):
        client.request_action(_real_action(action_id="action-after-stop"))


def test_terminal_action_history_is_bounded_and_releases_gateway_receipts(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    service = DaemonControlPlane(
        runtime=runtime,
        max_workers=1,
        max_queued_actions=1,
        max_retained_actions=2,
    )
    socket_path = tmp_path / "run" / "rosclawd.sock"
    daemon = RosclawDaemon(service=service, socket_path=socket_path)
    daemon.start()
    client = DaemonClient(socket_path=socket_path)
    action_ids = [f"action-retention-{index}" for index in range(3)]
    try:
        for action_id in action_ids:
            ticket = client.request_action(_shadow_action(action_id=action_id))
            client.wait_for_action(ticket["action_id"], timeout_sec=2.0)

        status = client.get_runtime_status()
        assert status["history"] == {
            "retained": 2,
            "capacity": 2,
            "evicted": 1,
        }
        with pytest.raises(DaemonRequestError) as error:
            client.get_action_status(action_ids[0])
        assert error.value.code == "ACTION_NOT_FOUND"
        assert runtime.action_gateway.get_receipt(action_ids[0]) is None
        assert client.get_action_status(action_ids[1])["state"] == "FINISHED"
        assert client.get_action_status(action_ids[2])["state"] == "FINISHED"
    finally:
        daemon.stop()


def test_queue_rejection_does_not_evict_retained_action_evidence(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    running = threading.Event()
    release = threading.Event()

    def execute(action: ActionEnvelope) -> ActionExecutionResult:
        if action.action_id == "action-running":
            running.set()
            assert release.wait(timeout=5.0)
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
        )

    runtime.action_gateway.register_executor(
        "rh56.finger.move",
        ExecutionMode.SHADOW,
        execute,
    )
    service = DaemonControlPlane(
        runtime=runtime,
        max_workers=1,
        max_queued_actions=1,
        max_retained_actions=2,
    )
    daemon = RosclawDaemon(
        service=service,
        socket_path=tmp_path / "run" / "rosclawd.sock",
    )
    daemon.start()
    client = DaemonClient(socket_path=daemon.socket_path)
    try:
        first = _shadow_action(action_id="action-retained")
        client.wait_for_action(
            client.request_action(first)["action_id"],
            timeout_sec=2.0,
        )
        client.request_action(_shadow_action(action_id="action-running"))
        assert running.wait(timeout=2.0)

        with pytest.raises(DaemonRequestError) as error:
            client.request_action(_shadow_action(action_id="action-rejected"))

        assert error.value.code == "ACTION_QUEUE_FULL"
        assert client.get_action_status(first.action_id)["state"] == "FINISHED"
        assert client.get_runtime_status()["history"] == {
            "retained": 2,
            "capacity": 2,
            "evicted": 0,
        }
    finally:
        release.set()
        daemon.stop()


def test_real_action_history_fails_closed_instead_of_reusing_action_ids(
    tmp_path: Path,
) -> None:
    service = DaemonControlPlane(
        runtime=_runtime(),
        max_workers=1,
        max_queued_actions=1,
        max_retained_actions=1,
    )
    daemon = RosclawDaemon(
        service=service,
        socket_path=tmp_path / "run" / "rosclawd.sock",
    )
    daemon.start()
    client = DaemonClient(socket_path=daemon.socket_path)
    try:
        first = _real_action(action_id="action-real-retained")
        status = client.wait_for_action(
            client.request_action(first)["action_id"],
            timeout_sec=2.0,
        )
        assert status["final_state"] == "BLOCKED"

        with pytest.raises(DaemonRequestError) as error:
            client.request_action(_shadow_action(action_id="action-after-real-history"))

        assert error.value.code == "ACTION_HISTORY_FULL"
        assert client.get_action_status(first.action_id)["final_state"] == "BLOCKED"
    finally:
        daemon.stop()


def test_daemon_constructor_rejects_world_accessible_socket_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="world"):
        RosclawDaemon(
            service=DaemonControlPlane(runtime=_runtime()),
            socket_path=tmp_path / "run" / "rosclawd.sock",
            socket_mode=0o666,
        )


def test_daemon_bounds_concurrent_client_connections(tmp_path: Path) -> None:
    daemon = RosclawDaemon(
        service=DaemonControlPlane(runtime=_runtime()),
        socket_path=tmp_path / "run" / "rosclawd.sock",
        request_timeout_sec=0.5,
        max_clients=1,
    )
    daemon.start()
    blocker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        blocker.connect(str(daemon.socket_path))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with daemon._lock:
                if len(daemon._client_threads) == 1:
                    break
            time.sleep(0.01)
        else:
            pytest.fail("blocking client was not accepted")

        with pytest.raises(DaemonUnavailableError):
            DaemonClient(socket_path=daemon.socket_path, timeout_sec=0.2).get_runtime_status()
    finally:
        blocker.close()
        daemon.stop()


def test_daemon_cleans_socket_when_service_start_fails(tmp_path: Path) -> None:
    service = DaemonControlPlane(runtime=_runtime())
    service.close()
    socket_path = tmp_path / "run" / "rosclawd.sock"
    daemon = RosclawDaemon(service=service, socket_path=socket_path)

    with pytest.raises(RuntimeError, match="cannot restart"):
        daemon.start()

    assert not socket_path.exists()


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_daemon_refuses_to_replace_non_socket_paths(tmp_path: Path, kind: str) -> None:
    socket_path = tmp_path / "run" / "rosclawd.sock"
    socket_path.parent.mkdir(parents=True)
    socket_path.parent.chmod(0o700)
    if kind == "file":
        socket_path.write_text("do not delete", encoding="utf-8")
    else:
        target = tmp_path / "target"
        target.write_text("do not delete", encoding="utf-8")
        socket_path.symlink_to(target)

    daemon = RosclawDaemon(
        service=DaemonControlPlane(runtime=_runtime()),
        socket_path=socket_path,
    )

    with pytest.raises(RuntimeError, match="refusing"):
        daemon.start()

    assert socket_path.exists()
