#!/usr/bin/env python3
"""§5 authorized REAL agent blackbox — operator step (NOT the Agent).

Composes the exact one-step noop ActionEnvelope, issues the single-use daemon
ExecutionPermit through the operator channel, submits the authorized action,
and verifies the receipt + one-time semantics.  Run only AFTER the Agent's own
blocked attempt (AUTHORIZATION_REQUIRED) is on the record.

Checks written to ``<evidence>/operator_result.json``:
  * receipt final_state COMPLETED, evidence PHYSICALLY_OBSERVED or TASK_VERIFIED
  * driver_ack present with PROTOCOL_ACKNOWLEDGED / DELIVERY_INFERRED stage
  * daemon hardware_actions_executed == 1 (the blocked attempt dispatches nothing)
  * single-use: a second distinct action under the same permit is rejected
    (PERMIT_EXHAUSTED) without any further hardware action
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rosclaw.daemon.client import DaemonClient
from rosclaw.kernel import (
    ActionEnvelope,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)

SESSION_ID = "mcp-session"
PRINCIPAL_ID = "operator-blackbox"
PERMIT_TTL_SEC = 240.0


def _wait_terminal(client: DaemonClient, action_id: str, timeout: float = 30.0) -> dict[str, Any]:
    result = client.wait_for_action(action_id, timeout_sec=timeout)
    return result["receipt"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", required=True, help="daemon_bootstrap.json path")
    parser.add_argument("--evidence", required=True, help="evidence output directory")
    args = parser.parse_args()

    evidence = Path(args.evidence).resolve()
    bootstrap = json.loads(Path(args.bootstrap).read_text(encoding="utf-8"))
    client = DaemonClient(socket_path=bootstrap["socket"], timeout_sec=10.0)

    # 1. Session + arm (operator channel; daemon service uid required).
    client.create_session(
        session_id=SESSION_ID,
        actor_id="rosclaw-mcp",
        agent_framework="mcp",
        body_scope=[bootstrap["body_id"]],
        capability_scope=[bootstrap["capability_id"]],
        ttl_ms=3_600_000,
    )
    client.arm_runtime("§5 blackbox: operator preflight complete, E-Stop in reach")

    # 2. Compose the exact one-step noop envelope (hold open pose).
    deadline = datetime.now(UTC) + timedelta(seconds=PERMIT_TTL_SEC)
    action = ActionEnvelope(
        action_id=f"action_blackbox_noop_{int(time.time())}",
        actor_id="rosclaw-mcp",
        agent_framework="mcp",
        session_id=SESSION_ID,
        body_id=bootstrap["body_id"],
        body_snapshot_hash=bootstrap["hashes"]["body_hash"],
        capability_id=bootstrap["capability_id"],
        arguments={
            "permit_id": bootstrap["in_process_permit_id"],
            "names": bootstrap["names"],
            "values": [1000.0] * len(bootstrap["names"]),
            "representation": bootstrap["representation"],
            "units": bootstrap["units"],
            "hashes": bootstrap["hashes"],
            "speed": bootstrap["speed"],
            "force_limit_g": bootstrap["force_limit_g"],
            "observation_timestamp_ns": time.monotonic_ns(),
        },
        execution_mode=ExecutionMode.REAL,
        authorization=AuthorizationContext(
            principal_id=PRINCIPAL_ID,
            approved=False,
            approval_id=None,
            scopes=[],
        ),
        # The RH56 single-step executor observes the physical outcome but does
        # not claim task-level verification; requiring TASK_VERIFIED here makes
        # the gateway fail an otherwise-COMPLETED action (proven in run 1).
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.PHYSICALLY_OBSERVED,
            timeout_sec=30.0,
            fail_closed=True,
        ),
        deadline_at=deadline,
    )
    (evidence / "operator_proposal.json").write_text(
        json.dumps(action.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8"
    )

    # 3. Issue the single-use permit (daemon/operator channel — never the Agent).
    issued = client.issue_execution_permit(
        action,
        principal_id=PRINCIPAL_ID,
        target_peer_uid=os.geteuid(),
        expires_in_sec=PERMIT_TTL_SEC,
        reason=(
            "§5 authorized REAL agent blackbox: one noop single-step on "
            f"{bootstrap['body_id']}; workspace clear; E-Stop in reach"
        ),
    )
    permit = issued["permit"]
    authorized = issued["authorized_action"]
    (evidence / "operator_permit.json").write_text(
        json.dumps(issued, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # 4. Submit the authorized envelope verbatim and wait for the receipt.
    ticket = client.request_action(authorized)
    receipt = _wait_terminal(client, ticket["action_id"])
    (evidence / "receipt.json").write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # 5. Single-use proof: a second, distinct action under the same permit.
    replay = dict(authorized)
    replay["action_id"] = f"{ticket['action_id']}_replay"
    replay_ticket = client.request_action(replay)
    replay_receipt = _wait_terminal(client, replay_ticket["action_id"])
    (evidence / "receipt_replay.json").write_text(
        json.dumps(replay_receipt, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )

    status = client.get_runtime_status()
    checks = {
        "receipt_completed": receipt.get("final_state") == "COMPLETED",
        "evidence_level": receipt.get("evidence_level") in ("PHYSICALLY_OBSERVED", "TASK_VERIFIED"),
        "driver_ack": bool((receipt.get("driver_ack") or {}).get("acknowledged")),
        "ack_stage": (receipt.get("driver_ack") or {}).get("stage")
        in ("PROTOCOL_ACKNOWLEDGED", "DELIVERY_INFERRED", None)
        or receipt.get("acknowledgement_stage") in ("PROTOCOL_ACKNOWLEDGED", "DELIVERY_INFERRED"),
        "hardware_actions_executed_eq_1": status.get("hardware_actions_executed") == 1,
        "single_use_rejected": replay_receipt.get("final_state") == "BLOCKED"
        and any(e.get("code") == "PERMIT_EXHAUSTED" for e in replay_receipt.get("errors", [])),
        "permit_id": permit["permit_id"],
        "max_uses": permit["max_uses"],
    }
    checks["passed"] = all(v for k, v in checks.items() if isinstance(v, bool))
    (evidence / "operator_result.json").write_text(
        json.dumps(checks, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(checks, indent=1, ensure_ascii=False))
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
