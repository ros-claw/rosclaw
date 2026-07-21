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
from rosclaw.daemon.ledger import DaemonLedger, LedgerError, LedgerIntegrityError
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

_ACTION_DEADLINE = datetime(2099, 1, 1, tzinfo=UTC)


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
        deadline_at=_ACTION_DEADLINE,
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
    client.arm_runtime("daemon test fixture preflight complete")
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


@pytest.mark.parametrize("invalid_action_id", ["x" * 257, "action-with-null\0byte"])
def test_invalid_action_id_does_not_poison_durable_ledger(
    tmp_path: Path,
    invalid_action_id: str,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    with DaemonLedger(database, key_path=key) as ledger:
        service = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        service.start()
        with pytest.raises(ControlPlaneError) as rejected:
            service.request_action(_shadow_action(action_id=invalid_action_id), peer)

        valid = _shadow_action(action_id="action-after-invalid-id")
        service.request_action(valid, peer)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            valid_status = service.get_action_status(valid.action_id, peer)
            if valid_status["state"] == "FINISHED":
                break
            time.sleep(0.01)
        else:
            pytest.fail("valid action did not finish after invalid action id")
        runtime_status = service.get_runtime_status(peer)
        service.close()

    assert rejected.value.code == "INVALID_ACTION"
    assert valid_status["state"] == "FINISHED"
    assert runtime_status["ledger"]["write_failed"] is False


def test_terminal_action_and_receipt_survive_daemon_restart(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    action = _shadow_action(action_id="action-durable-terminal")

    with DaemonLedger(database, key_path=key) as ledger:
        first_service = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        first_service.start()
        first_service.request_action(action, peer)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            first_status = first_service.get_action_status(action.action_id, peer)
            if first_status["state"] == "FINISHED":
                break
            time.sleep(0.01)
        else:
            pytest.fail("durable action did not finish")
        first_service.close()

    with DaemonLedger(database, key_path=key) as ledger:
        restored_service = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        restored_service.start()
        restored_status = restored_service.get_action_status(action.action_id, peer)
        repeated_status = restored_service.request_action(action, peer)
        restored_service.close()

    assert restored_status == first_status
    assert repeated_status == first_status


def test_authenticated_receipt_must_match_its_immutable_action(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    action = _shadow_action(action_id="action-invalid-durable-receipt")
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    runtime = _runtime()
    receipt = runtime.action_gateway.reject(
        action,
        code="TEST_REJECTION",
        message="test receipt",
    ).to_dict()
    receipt["mode"] = "REAL"
    receipt["execution_mode"] = "REAL"
    with DaemonLedger(database, key_path=key) as ledger:
        ledger.append(
            "ACTION_SUBMITTED",
            entity_kind="ACTION",
            entity_id=action.action_id,
            payload={
                "action": action.to_dict(),
                "peer": peer.to_dict(),
                "submitted_at": timestamp,
            },
        )
        ledger.append(
            "ACTION_TERMINAL",
            entity_kind="ACTION",
            entity_id=action.action_id,
            payload={
                "scheduler_state": "FINISHED",
                "finished_at": timestamp,
                "receipt": receipt,
            },
        )

    with (
        DaemonLedger(database, key_path=key) as ledger,
        pytest.raises(LedgerIntegrityError, match="immutable action"),
    ):
        DaemonControlPlane(runtime=_runtime(), ledger=ledger)


def test_incomplete_real_action_requires_operator_review_after_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    interrupted = _real_action(action_id="action-interrupted-real")
    submitted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    with DaemonLedger(database, key_path=key) as ledger:
        ledger.append(
            "ACTION_SUBMITTED",
            entity_kind="ACTION",
            entity_id=interrupted.action_id,
            payload={
                "action": interrupted.to_dict(),
                "peer": peer.to_dict(),
                "submitted_at": submitted_at,
            },
        )
        ledger.append(
            "ACTION_STARTED",
            entity_kind="ACTION",
            entity_id=interrupted.action_id,
            payload={"started_at": submitted_at},
        )

    with DaemonLedger(database, key_path=key) as ledger:
        runtime = _runtime()
        restored = DaemonControlPlane(runtime=runtime, ledger=ledger)
        restored.start()
        status = restored.get_action_status(interrupted.action_id, peer)
        runtime_status = restored.get_runtime_status(peer)

        with pytest.raises(ControlPlaneError) as error:
            restored.request_action(_real_action(action_id="action-new-real"), peer)
        restored.close()

    assert status["state"] == "FINISHED"
    assert status["final_state"] == "FAILED"
    assert status["error_code"] == "DAEMON_RESTART_OUTCOME_UNKNOWN"
    assert runtime_status["emergency_stop_latched"] is True
    assert runtime_status["recovery"] == {
        "required": True,
        "action_ids": [interrupted.action_id],
        "real_action_ids": [interrupted.action_id],
    }
    assert error.value.code == "RECOVERY_REVIEW_REQUIRED"


def test_queued_real_action_is_cancelled_without_claiming_restart_dispatch(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    queued = _real_action(action_id="action-queued-before-restart")
    submitted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    with DaemonLedger(database, key_path=key) as ledger:
        ledger.append(
            "ACTION_SUBMITTED",
            entity_kind="ACTION",
            entity_id=queued.action_id,
            payload={
                "action": queued.to_dict(),
                "peer": peer.to_dict(),
                "submitted_at": submitted_at,
            },
        )

    with DaemonLedger(database, key_path=key) as ledger:
        runtime = _runtime()
        service = DaemonControlPlane(runtime=runtime, ledger=ledger)
        service.start()
        status = service.get_action_status(queued.action_id, peer)
        runtime_status = service.get_runtime_status(peer)
        service.close()

    assert status["state"] == "CANCELLED"
    assert status["final_state"] == "CANCELLED"
    assert status["error_code"] == "DAEMON_RESTART_CANCELLED_BEFORE_DISPATCH"
    assert runtime_status["recovery"]["required"] is False
    assert runtime_status["emergency_stop_requests"] == 0


def test_only_daemon_uid_can_persistently_acknowledge_restart_recovery(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    daemon_peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    interrupted = _real_action(action_id="action-review-required")
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    with DaemonLedger(database, key_path=key) as ledger:
        ledger.append(
            "ACTION_SUBMITTED",
            entity_kind="ACTION",
            entity_id=interrupted.action_id,
            payload={
                "action": interrupted.to_dict(),
                "peer": daemon_peer.to_dict(),
                "submitted_at": started_at,
            },
        )
        ledger.append(
            "ACTION_STARTED",
            entity_kind="ACTION",
            entity_id=interrupted.action_id,
            payload={"started_at": started_at},
        )

    with DaemonLedger(database, key_path=key) as ledger:
        service = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        service.start()
        other_peer = PeerCredentials(
            pid=202,
            uid=os.geteuid() + 1,
            gid=os.getegid(),
        )
        with pytest.raises(ControlPlaneError) as denied:
            service.acknowledge_recovery("reviewed evidence", other_peer)
        with pytest.raises(ControlPlaneError) as invalid_reason:
            service.acknowledge_recovery("x" * 1025, daemon_peer)

        acknowledgement = service.acknowledge_recovery(
            "operator reviewed unknown physical outcome",
            daemon_peer,
        )
        new_action = _real_action(action_id="action-after-review")
        service.request_action(new_action, daemon_peer)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            new_status = service.get_action_status(new_action.action_id, daemon_peer)
            if new_status["state"] == "FINISHED":
                break
            time.sleep(0.01)
        else:
            pytest.fail("post-review action did not finish")
        service.close()

    with DaemonLedger(database, key_path=key) as ledger:
        restarted = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        restarted.start()
        recovery = restarted.get_runtime_status(daemon_peer)["recovery"]
        restarted.close()

    assert denied.value.code == "PERMISSION_DENIED"
    assert invalid_reason.value.code == "INVALID_ARGUMENT"
    assert acknowledgement["acknowledged"] is True
    assert acknowledgement["recovery_required"] is False
    assert new_status["error_code"] == "AUTHORIZATION_REQUIRED"
    assert recovery == {"required": False, "action_ids": [], "real_action_ids": []}


def test_daemon_client_acknowledges_recovery_through_authenticated_rpc(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=os.getpid(), uid=os.geteuid(), gid=os.getegid())
    interrupted = _real_action(action_id="action-rpc-recovery")
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    with DaemonLedger(database, key_path=key) as ledger:
        ledger.append(
            "ACTION_SUBMITTED",
            entity_kind="ACTION",
            entity_id=interrupted.action_id,
            payload={
                "action": interrupted.to_dict(),
                "peer": peer.to_dict(),
                "submitted_at": started_at,
            },
        )
        ledger.append(
            "ACTION_STARTED",
            entity_kind="ACTION",
            entity_id=interrupted.action_id,
            payload={"started_at": started_at},
        )

    with DaemonLedger(database, key_path=key) as ledger:
        daemon = RosclawDaemon(
            service=DaemonControlPlane(runtime=_runtime(), ledger=ledger),
            socket_path=tmp_path / "run" / "rosclawd.sock",
        )
        daemon.start()
        client = DaemonClient(socket_path=daemon.socket_path)
        try:
            assert client.get_runtime_status()["recovery"]["required"] is True
            result = client.acknowledge_recovery("operator reviewed RPC evidence")
            assert client.get_runtime_status()["recovery"]["required"] is False
        finally:
            daemon.stop()

    assert result["acknowledged"] is True
    assert result["action_ids"] == [interrupted.action_id]
    assert result["emergency_stop_latched"] is True


def test_pending_recovery_relatches_estop_on_every_daemon_restart(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    interrupted = _real_action(action_id="action-repeated-restart")
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    with DaemonLedger(database, key_path=key) as ledger:
        ledger.append(
            "ACTION_SUBMITTED",
            entity_kind="ACTION",
            entity_id=interrupted.action_id,
            payload={
                "action": interrupted.to_dict(),
                "peer": peer.to_dict(),
                "submitted_at": started_at,
            },
        )
        ledger.append(
            "ACTION_STARTED",
            entity_kind="ACTION",
            entity_id=interrupted.action_id,
            payload={"started_at": started_at},
        )

    with DaemonLedger(database, key_path=key) as ledger:
        first_runtime = _runtime()
        first = DaemonControlPlane(runtime=first_runtime, ledger=ledger)
        first.start()
        assert first.get_runtime_status(peer)["emergency_stop_latched"] is True
        first.close()

    with DaemonLedger(database, key_path=key) as ledger:
        second_runtime = _runtime()
        second = DaemonControlPlane(runtime=second_runtime, ledger=ledger)
        second.start()
        second_status = second.get_runtime_status(peer)
        second.close()

    assert second_status["recovery"]["required"] is True
    assert second_status["emergency_stop_latched"] is True
    assert second_status["emergency_stop_requests"] == 1


def test_repeated_restart_recovery_never_discards_pending_real_actions(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    actions = [
        _real_action(action_id="action-recovery-a"),
        _real_action(action_id="action-recovery-b"),
    ]
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    receipt_runtime = _runtime()
    with DaemonLedger(database, key_path=key) as ledger:
        for action in actions:
            ledger.append(
                "ACTION_SUBMITTED",
                entity_kind="ACTION",
                entity_id=action.action_id,
                payload={
                    "action": action.to_dict(),
                    "peer": peer.to_dict(),
                    "submitted_at": timestamp,
                },
            )
            ledger.append(
                "ACTION_STARTED",
                entity_kind="ACTION",
                entity_id=action.action_id,
                payload={"started_at": timestamp},
            )
        action_ids = [action.action_id for action in actions]
        ledger.append(
            "RECOVERY_REQUIRED",
            entity_kind="RECOVERY",
            entity_id="rosclawd",
            payload={
                "action_ids": action_ids,
                "real_action_ids": action_ids,
                "required_at": timestamp,
                "reason": "interrupted_real_action_outcome_unknown",
            },
        )
        first_receipt = receipt_runtime.action_gateway.reject(
            actions[0],
            code="DAEMON_RESTART_OUTCOME_UNKNOWN",
            message="first action was terminalized before another restart",
            state=ActionState.FAILED,
        ).to_dict()
        ledger.append(
            "ACTION_TERMINAL",
            entity_kind="ACTION",
            entity_id=actions[0].action_id,
            payload={
                "scheduler_state": "FINISHED",
                "finished_at": timestamp,
                "receipt": first_receipt,
            },
        )

    with DaemonLedger(database, key_path=key) as ledger:
        service = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        service.start()
        status = service.get_runtime_status(peer)
        second_action = service.get_action_status(actions[1].action_id, peer)
        service.close()

    assert status["recovery"] == {
        "required": True,
        "action_ids": action_ids,
        "real_action_ids": action_ids,
    }
    assert second_action["error_code"] == "DAEMON_RESTART_OUTCOME_UNKNOWN"


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


def test_action_is_not_dispatched_when_started_event_cannot_be_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    runtime = _runtime()
    dispatched: list[str] = []

    def execute(action: ActionEnvelope) -> ActionExecutionResult:
        dispatched.append(action.action_id)
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
        )

    runtime.action_gateway.register_executor(
        "rh56.finger.move",
        ExecutionMode.SHADOW,
        execute,
    )
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    with DaemonLedger(database, key_path=key) as ledger:
        original_append = ledger.append

        def fail_started_event(event_type: str, **kwargs: Any):
            if event_type == "ACTION_STARTED":
                raise LedgerError("injected durable write failure")
            return original_append(event_type, **kwargs)

        monkeypatch.setattr(ledger, "append", fail_started_event)
        service = DaemonControlPlane(runtime=runtime, ledger=ledger)
        service.start()
        action = _shadow_action(action_id="action-ledger-start-failure")
        service.request_action(action, peer)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            status = service.get_action_status(action.action_id, peer)
            if status["state"] == "FINISHED":
                break
            time.sleep(0.01)
        else:
            pytest.fail("ledger write failure did not produce a terminal action")

        with pytest.raises(ControlPlaneError) as unavailable:
            service.request_action(_shadow_action(action_id="action-after-ledger-failure"), peer)
        runtime_status = service.get_runtime_status(peer)
        service.close()

    assert dispatched == []
    assert status["final_state"] == "FAILED"
    assert status["error_code"] == "DAEMON_LEDGER_WRITE_FAILED"
    assert unavailable.value.code == "LEDGER_UNAVAILABLE"
    assert runtime_status["ledger"]["write_failed"] is True


def test_action_submission_fails_closed_when_ledger_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    with DaemonLedger(database, key_path=key) as ledger:
        original_append = ledger.append

        def fail_submission(event_type: str, **kwargs: Any):
            if event_type == "ACTION_SUBMITTED":
                raise LedgerError("injected submission write failure")
            return original_append(event_type, **kwargs)

        monkeypatch.setattr(ledger, "append", fail_submission)
        service = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        service.start()
        with pytest.raises(ControlPlaneError) as failure:
            service.request_action(_shadow_action(action_id="action-submit-write-failure"), peer)
        runtime_status = service.get_runtime_status(peer)
        service.close()

    assert failure.value.code == "LEDGER_UNAVAILABLE"
    assert runtime_status["ledger"]["write_failed"] is True
    assert runtime_status["queue"] == {
        "QUEUED": 0,
        "RUNNING": 0,
        "FINISHED": 0,
        "CANCELLED": 0,
    }


def test_executor_scheduling_failure_is_durable_and_never_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    action = _shadow_action(action_id="action-scheduling-failure")
    with DaemonLedger(database, key_path=key) as ledger:
        service = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        service.start()

        def fail_schedule(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected executor scheduling failure")

        monkeypatch.setattr(service._executor, "submit", fail_schedule)
        status = service.request_action(action, peer)
        service.close()

    with DaemonLedger(database, key_path=key) as ledger:
        restored = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        restored.start()
        restored_status = restored.get_action_status(action.action_id, peer)
        restored.close()

    assert status["state"] == "FINISHED"
    assert status["final_state"] == "FAILED"
    assert status["error_code"] == "DAEMON_SCHEDULING_FAILED"
    assert restored_status == status


def test_real_terminal_write_failure_revokes_success_and_requests_estop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    runtime = _runtime()
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
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    action = _real_action(
        action_id="action-terminal-ledger-failure",
        approval_id="permit-terminal-ledger-failure",
    )
    with DaemonLedger(database, key_path=key) as ledger:
        permits = PermitAuthority(ledger=ledger)
        permits.register(
            ExecutionPermit(
                permit_id="permit-terminal-ledger-failure",
                principal_id="operator-1",
                peer_uid=peer.uid,
                body_id="rh56-test",
                body_snapshot_hash="sha256:body",
                capabilities=("rh56.finger.move",),
                action_intent_hash=action_intent_hash(action),
                expires_at=datetime.now(UTC) + timedelta(minutes=1),
            )
        )
        original_append = ledger.append

        def fail_terminal_event(event_type: str, **kwargs: Any):
            if event_type == "ACTION_TERMINAL":
                raise LedgerError("injected terminal write failure")
            return original_append(event_type, **kwargs)

        monkeypatch.setattr(ledger, "append", fail_terminal_event)
        service = DaemonControlPlane(
            runtime=runtime,
            permits=permits,
            ledger=ledger,
        )
        service.start()
        service.arm_runtime("test preflight complete", peer)
        service.request_action(action, peer)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            status = service.get_action_status(action.action_id, peer)
            if status["state"] == "FINISHED":
                break
            time.sleep(0.01)
        else:
            pytest.fail("terminal ledger failure did not finish")
        runtime_status = service.get_runtime_status(peer)
        service.close()

    assert dispatched == [action.action_id]
    assert status["final_state"] == "FAILED"
    assert status["error_code"] == "DAEMON_LEDGER_TERMINAL_WRITE_FAILED"
    assert runtime_status["recovery"]["required"] is True
    assert runtime_status["emergency_stop_latched"] is True
    assert runtime_status["hardware_actions_executed"] == 0


def test_concurrent_real_terminal_failures_accumulate_unknown_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    runtime = _runtime()
    release = threading.Event()
    dispatched: list[str] = []
    dispatched_lock = threading.Lock()

    def execute(action: ActionEnvelope) -> ActionExecutionResult:
        with dispatched_lock:
            dispatched.append(action.action_id)
        assert release.wait(timeout=5.0)
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
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    actions = [
        _real_action(action_id="action-terminal-failure-a", approval_id="permit-two-uses"),
        _real_action(action_id="action-terminal-failure-b", approval_id="permit-two-uses"),
    ]
    with DaemonLedger(database, key_path=key) as ledger:
        permits = PermitAuthority(ledger=ledger)
        permits.register(
            ExecutionPermit(
                permit_id="permit-two-uses",
                principal_id="operator-1",
                peer_uid=peer.uid,
                body_id="rh56-test",
                body_snapshot_hash="sha256:body",
                capabilities=("rh56.finger.move",),
                action_intent_hash=action_intent_hash(actions[0]),
                expires_at=datetime.now(UTC) + timedelta(minutes=1),
                max_uses=2,
            )
        )
        original_append = ledger.append

        def fail_terminal_event(event_type: str, **kwargs: Any):
            if event_type == "ACTION_TERMINAL":
                raise LedgerError("injected concurrent terminal write failure")
            return original_append(event_type, **kwargs)

        monkeypatch.setattr(ledger, "append", fail_terminal_event)
        service = DaemonControlPlane(
            runtime=runtime,
            permits=permits,
            ledger=ledger,
            max_workers=2,
            max_queued_actions=2,
        )
        service.start()
        service.arm_runtime("test preflight complete", peer)
        for action in actions:
            service.request_action(action, peer)
        start_deadline = time.monotonic() + 2.0
        while time.monotonic() < start_deadline:
            started = [service.get_action_status(action.action_id, peer) for action in actions]
            if all(item["state"] == "RUNNING" for item in started):
                break
            time.sleep(0.01)
        else:
            pytest.fail("concurrent daemon jobs did not start")
        release.set()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            statuses = [service.get_action_status(action.action_id, peer) for action in actions]
            if all(item["state"] == "FINISHED" for item in statuses):
                break
            time.sleep(0.01)
        else:
            pytest.fail("concurrent terminal failures did not finish")
        runtime_status = service.get_runtime_status(peer)
        service.close()

    action_ids = sorted(action.action_id for action in actions)
    assert dispatched
    assert len(dispatched) == len(set(dispatched))
    assert set(dispatched).issubset(action_ids)
    assert all(item["error_code"] == "DAEMON_LEDGER_TERMINAL_WRITE_FAILED" for item in statuses)
    assert runtime_status["recovery"] == {
        "required": True,
        "action_ids": action_ids,
        "real_action_ids": action_ids,
    }
    assert runtime_status["hardware_actions_executed"] == 0


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


def test_durable_history_allows_memory_eviction_without_action_id_replay(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    first = _real_action(action_id="action-durable-real-history")
    with DaemonLedger(database, key_path=key) as ledger:
        service = DaemonControlPlane(
            runtime=_runtime(),
            ledger=ledger,
            max_workers=1,
            max_queued_actions=1,
            max_retained_actions=1,
        )
        service.start()
        service.request_action(first, peer)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            first_status = service.get_action_status(first.action_id, peer)
            if first_status["state"] == "FINISHED":
                break
            time.sleep(0.01)
        else:
            pytest.fail("durable REAL history action did not finish")

        second = _shadow_action(action_id="action-after-durable-real")
        service.request_action(second, peer)
        while time.monotonic() < deadline + 2.0:
            second_status = service.get_action_status(second.action_id, peer)
            if second_status["state"] == "FINISHED":
                break
            time.sleep(0.01)
        else:
            pytest.fail("action after durable REAL history did not finish")

        repeated = service.request_action(first, peer)
        terminal_cancel = service.cancel_action(first.action_id, peer)
        conflicting = _real_action(action_id=first.action_id)
        conflicting.arguments["delta_raw"] = 21
        with pytest.raises(ControlPlaneError) as conflict:
            service.request_action(conflicting, peer)
        service.close()

    assert repeated == first_status
    assert terminal_cancel == {
        "action_id": first.action_id,
        "cancelled": False,
        "state": "FINISHED",
        "message": "Action is already terminal.",
    }
    assert conflict.value.code == "ACTION_ID_CONFLICT"


def test_restart_bounds_memory_while_older_durable_actions_remain_queryable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    peer = PeerCredentials(pid=101, uid=os.geteuid(), gid=os.getegid())
    action_ids = [f"action-durable-restart-{index}" for index in range(3)]
    with DaemonLedger(database, key_path=key) as ledger:
        service = DaemonControlPlane(
            runtime=_runtime(),
            ledger=ledger,
            max_retained_actions=3,
        )
        service.start()
        for action_id in action_ids:
            service.request_action(_shadow_action(action_id=action_id), peer)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                status = service.get_action_status(action_id, peer)
                if status["state"] == "FINISHED":
                    break
                time.sleep(0.01)
            else:
                pytest.fail(f"durable action did not finish: {action_id}")
        service.close()

    with DaemonLedger(database, key_path=key) as ledger:
        restored = DaemonControlPlane(
            runtime=_runtime(),
            ledger=ledger,
            max_queued_actions=1,
            max_retained_actions=1,
        )
        restored.start()
        runtime_status = restored.get_runtime_status(peer)
        oldest = restored.get_action_status(action_ids[0], peer)
        restored.close()

    assert runtime_status["history"]["retained"] == 1
    assert runtime_status["history"]["evicted"] == 2
    assert oldest["action_id"] == action_ids[0]
    assert oldest["state"] == "FINISHED"


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
