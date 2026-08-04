#!/usr/bin/env python3
"""T1 角色驱动：每个子命令以特定 UID 运行（由 entrypoint.py 编排）。

环境：ROSCLAW_REPO 指向仓库根（pip install -e 已装 rosclaw）。
所有状态在 /tmp/e2e/<user>/。
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path("/tmp/e2e")


def _daemon_socket() -> Path:
    return ROOT / "runtime" / "rosclawd" / "rosclawd.sock"


def cmd_daemon() -> int:
    """rosclawd：最小 runtime + SHADOW executor + 控制组 socket。"""
    from rosclaw.core.runtime import Runtime, RuntimeConfig
    from rosclaw.daemon.ledger import DaemonLedger
    from rosclaw.daemon.server import RosclawDaemon
    from rosclaw.daemon.service import DaemonControlPlane
    from rosclaw.kernel.contracts import (
        ActionExecutionResult,
        ActionState,
        EvidenceDomain,
        EvidenceLevel,
        ExecutionMode,
    )

    def shadow_executor(action) -> ActionExecutionResult:
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.TASK_VERIFIED,
            evidence_domain=EvidenceDomain.SHADOW,
            simulation_result={"actuated": False, "usable_for_real_execution": False},
            verification_result={"verified": True, "actuation_gate": "hard_blocked"},
        )

    runtime = Runtime(
        RuntimeConfig(
            robot_id="e2e-cross-uid",
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
    runtime.action_gateway.register_executor("e2e.shadow.ping", ExecutionMode.SHADOW, shadow_executor)
    home = ROOT / "rcd"
    ledger = DaemonLedger(home / "state" / "ledger.sqlite3", key_path=home / "state" / "ledger.key")
    daemon = RosclawDaemon(
        service=DaemonControlPlane(runtime=runtime, ledger=ledger),
        socket_path=ROOT / "runtime" / "rosclawd" / "rosclawd.sock",
        socket_mode=0o660,
        socket_group="rccontrol",
    )
    daemon.start()
    print("DAEMON_READY", flush=True)
    try:
        signal_pause()
    finally:
        daemon.stop()
    return 0


def signal_pause() -> None:
    import signal
    import time

    stopped = False

    def _stop(signum, frame):  # noqa: ARG001
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, _stop)
    while not stopped:
        time.sleep(0.1)


def cmd_enroll() -> int:
    """operatord 侧：生成 Ed25519 身份并导出公钥到共享目录。"""
    from rosclaw.operatord.enrollment import enroll

    identity = enroll(ROOT / "rco" / "operatord")
    shared = ROOT / "shared" / "operator-pubkey.pem"
    shared.write_text(identity.public_key_pem, encoding="utf-8")
    (ROOT / "shared" / "operator-enrollment-id").write_text(identity.enrollment_id)
    print(json.dumps({"enrollment_id": identity.enrollment_id, "fingerprint": identity.fingerprint}))
    return 0


def cmd_register_operator() -> int:
    """daemon 管理员：从共享目录登记 operator 公钥。"""
    from rosclaw.daemon.client import DaemonClient

    client = DaemonClient(socket_path=_daemon_socket())
    pubkey = (ROOT / "shared" / "operator-pubkey.pem").read_text(encoding="utf-8")
    enrollment_id = (ROOT / "shared" / "operator-enrollment-id").read_text().strip()
    operator_uid = 2003
    result = client.register_operator_enrollment(
        enrollment_id,
        public_key_pem=pubkey,
        operator_uid=operator_uid,
    )
    print(json.dumps(result))
    return 0


def cmd_register_attacker() -> int:
    """agent 试图抢注 enrollment（必须 PERMISSION_DENIED）。"""
    from rosclaw.contracts.operator.decision import generate_ed25519_keypair
    from rosclaw.daemon.client import DaemonClient, DaemonClientError

    _, pem = generate_ed25519_keypair()
    client = DaemonClient(socket_path=_daemon_socket())
    try:
        client.register_operator_enrollment(
            "oen_attacker", public_key_pem=pem, operator_uid=2002
        )
    except DaemonClientError as exc:
        print(f"DENIED {exc.code}")
        return 0
    print("REGISTERED_UNEXPECTEDLY")
    return 1


def _e2e_action():
    from rosclaw.kernel.contracts import (
        ActionEnvelope,
        AuthorizationContext,
        EvidenceLevel,
        ExecutionMode,
        VerificationPolicy,
    )

    return ActionEnvelope(
        action_id="act-e2e-shadow",
        actor_id="e2e-agent",
        agent_framework="rosclaw-native",
        session_id="sess-e2e",
        body_id="e2e-body",
        body_snapshot_hash="sha256:e2e-body",
        capability_id="e2e.shadow.ping",
        arguments={"probe": True},
        execution_mode=ExecutionMode.SHADOW,
        deadline_at=datetime.now(UTC) + timedelta(minutes=2),
        authorization=AuthorizationContext(),
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.TASK_VERIFIED, timeout_sec=30.0
        ),
    )


def cmd_create_proposal() -> int:
    from rosclaw.daemon.client import DaemonClient

    client = DaemonClient(socket_path=_daemon_socket())
    created = client.create_operator_proposal(
        _e2e_action(),
        display={"title": "E2E SHADOW ping", "risk_tier": "LOW"},
        ttl_sec=60.0,
        client_reference={"agent_request_id": "req-e2e-1", "mission_id": "mis-e2e-1"},
    )
    proposal = created["proposal"]
    assert "challenge_nonce" not in proposal, "challenge leaked to agent view"
    (ROOT / "shared" / "action-id").write_text(str(proposal["action_id"]))
    print(json.dumps({"request_id": proposal["request_id"], "action_id": proposal["action_id"]}))
    return 0


def cmd_agent_self_decide(proposal_id: str) -> int:
    """agent 自签 key 直接 decide（必须 PERMISSION_DENIED）。"""
    from dataclasses import replace

    from rosclaw.contracts.operator.decision import (
        DecisionChallengeV1,
        OperatorDecisionProofV1,
        generate_ed25519_keypair,
        sign_b64,
    )
    from rosclaw.daemon.client import DaemonClient, DaemonClientError

    client = DaemonClient(socket_path=_daemon_socket())
    private, _pem = generate_ed25519_keypair()
    try:
        challenge = DecisionChallengeV1.from_dict(client.get_operator_challenge(proposal_id)["challenge"])
    except DaemonClientError as exc:
        # agent 连 challenge 都拿不到（更强拒绝）——同样算 fail closed。
        print(f"DENIED_AT_CHALLENGE {exc.code}")
        return 0
    proof = OperatorDecisionProofV1(
        enrollment_id="oen_agent_forged",
        challenge=challenge,
        decision="ACCEPT",
        decided_at=datetime.now(UTC).isoformat(),
        human_confirmation_method="forged",
    )
    proof = replace(proof, signature_b64=sign_b64(private, proof.signing_payload()))
    try:
        client.decide_operator_proposal(
            proposal_id,
            decision="ACCEPT",
            principal_id="user:local:2002",
            channel="agent-forgery",
            reason="agent attempted self-approval",
            proof=proof.to_dict(),
        )
    except DaemonClientError as exc:
        print(f"DENIED {exc.code}")
        return 0
    print("DECIDED_UNEXPECTEDLY")
    return 1


def cmd_operator_decide(proposal_id: str) -> int:
    """operatord 协议全链：challenge → Ed25519 sign → decide → receipt 验证。"""
    from dataclasses import replace

    from rosclaw.contracts.operator.decision import (
        DecisionChallengeV1,
        DecisionReceiptV1,
        OperatorDecisionProofV1,
    )
    from rosclaw.daemon.client import DaemonClient
    from rosclaw.operatord.enrollment import load_identity

    identity = load_identity(ROOT / "rco" / "operatord")
    client = DaemonClient(socket_path=_daemon_socket())
    # enrolled operator UID 可以 list pending（P0-4.1）。
    pending = client.list_pending_operator_proposals()
    assert any(p["request_id"] == proposal_id for p in pending["proposals"]), "not pending"
    challenge = DecisionChallengeV1.from_dict(
        client.get_operator_challenge(proposal_id)["challenge"]
    )
    proof = OperatorDecisionProofV1(
        enrollment_id=identity.enrollment_id,
        challenge=challenge,
        decision="ACCEPT",
        decided_at=datetime.now(UTC).isoformat(),
        human_confirmation_method="e2e-scripted",
    )
    proof = replace(proof, signature_b64=identity.sign(proof.signing_payload()))
    decided = client.decide_operator_proposal(
        proposal_id,
        decision="ACCEPT",
        principal_id="user:local:2003",
        channel="rosclaw_operatord",
        reason="e2e cross-uid decision",
        proof=proof.to_dict(),
    )
    receipt = decided["decision_receipt"]
    identity_info = client.daemon_identity()
    parsed = DecisionReceiptV1.from_dict(receipt)
    assert parsed.verify_signature(identity_info["public_key_pem"]), "receipt signature invalid"
    assert parsed.decision == "ACCEPT"
    assert parsed.agent_request_id == "req-e2e-1"
    assert parsed.mission_id == "mis-e2e-1"
    assert parsed.execution_mode == "SHADOW"
    assert parsed.capability_id == "e2e.shadow.ping"
    assert parsed.operator_enrollment_id == identity.enrollment_id
    print("RECEIPT_OK")
    return 0


def cmd_await_action() -> int:
    from rosclaw.daemon.client import DaemonClient

    client = DaemonClient(socket_path=_daemon_socket())
    action_id = (ROOT / "shared" / "action-id").read_text().strip()
    terminal = client.wait_for_action(action_id, timeout_sec=20.0)
    receipt = terminal["receipt"]
    assert receipt["final_state"] == "COMPLETED", receipt
    assert receipt["evidence_domain"].upper() == "SHADOW"
    assert (receipt.get("simulation_result") or {}).get("actuated") is False
    print("TERMINAL_OK")
    return 0


def cmd_probe_internal_methods() -> int:
    """内部 arm/permit 方法不得经 IPC 到达；arm 非管理员必拒。"""
    from rosclaw.daemon.client import DaemonClient, DaemonClientError

    client = DaemonClient(socket_path=_daemon_socket())
    failures = []
    try:
        client.call("_issue_permit_after_operator_decision", {})
        failures.append("internal_permit_reached")
    except DaemonClientError as exc:
        if exc.code != "METHOD_NOT_ALLOWED":
            failures.append(f"internal_permit:{exc.code}")
    try:
        client.call("_arm_after_operator_decision", {})
        failures.append("internal_arm_reached")
    except DaemonClientError as exc:
        if exc.code != "METHOD_NOT_ALLOWED":
            failures.append(f"internal_arm:{exc.code}")
    try:
        client.arm_runtime("agent attempts arm") if hasattr(client, "arm_runtime") else client.call(
            "runtime.arm", {"reason": "agent attempts arm"}
        )
        failures.append("agent_armed_runtime")
    except DaemonClientError as exc:
        if exc.code != "PERMISSION_DENIED":
            failures.append(f"arm:{exc.code}")
    if failures:
        print("INTERNAL_FAIL " + ",".join(failures))
        return 1
    print("INTERNAL_OK")
    return 0


def cmd_check_registry_after_restart() -> int:
    """重启后：enrollment 持久化 + 已焚毁 nonce 持久化。"""
    from rosclaw.daemon.client import DaemonClient
    from rosclaw.daemon.operator_registry import OperatorRegistry

    client = DaemonClient(socket_path=_daemon_socket())
    enrollment_id = (ROOT / "shared" / "operator-enrollment-id").read_text().strip()
    listed = client.list_operator_enrollments()
    assert any(
        e["enrollment_id"] == enrollment_id and e["status"] == "active"
        for e in listed["enrollments"]
    ), listed
    registry = OperatorRegistry(ROOT / "rcd" / "state" / "operator-enrollments.json")
    record = registry.active(enrollment_id)
    assert record is not None and record.operator_uid == 2003
    print("REGISTRY_OK")
    return 0


def cmd_probe_daemon_socket() -> int:
    try:
        import socket as _socket

        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.connect(str(_daemon_socket()))
    except PermissionError:
        print("EACCES")
        return 0
    except OSError as exc:
        print(f"OTHER:{exc}")
        return 1
    print("CONNECTED_UNEXPECTEDLY")
    return 1


def cmd_probe_operator_key() -> int:
    try:
        (ROOT / "rco" / "operatord" / "operator-identity.json").read_text()
    except PermissionError:
        print("EACCES")
        return 0
    except OSError as exc:
        print(f"OTHER:{exc}")
        return 1
    print("READ_UNEXPECTEDLY")
    return 1


COMMANDS = {
    "daemon": cmd_daemon,
    "enroll": cmd_enroll,
    "register_operator": cmd_register_operator,
    "register_attacker": cmd_register_attacker,
    "create_proposal": cmd_create_proposal,
    "agent_self_decide": cmd_agent_self_decide,
    "operator_decide": cmd_operator_decide,
    "await_action": cmd_await_action,
    "probe_internal_methods": cmd_probe_internal_methods,
    "check_registry_after_restart": cmd_check_registry_after_restart,
    "probe_daemon_socket": cmd_probe_daemon_socket,
    "probe_operator_key": cmd_probe_operator_key,
}

if __name__ == "__main__":
    name = sys.argv[1]
    args = sys.argv[2:]
    try:
        sys.exit(COMMANDS[name](*args))
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"DRIVER_ERROR {type(exc).__name__}: {exc}")
        sys.exit(1)
