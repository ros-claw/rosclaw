#!/usr/bin/env python3
"""§5 authorized REAL agent blackbox — operator-side rosclawd bootstrap (RH56).

Starts a real rosclawd control socket with the RH56 REAL single-step executor
registered against the physical hand, exactly mirroring the exp3/exp4
acceptance wiring (SerialModbusTransport -> RH56Executor -> SingleStepExecutor
-> RH56RealStepExecutor -> ActionGateway), plus the daemon-level permit gate.

Safety / boundary notes
-----------------------
* The daemon process owns the serial device; the Agent under test only ever
  sees the MCP tools (southbound_owner=rosclawd).
* The in-process SingleStepExecutor permit and the daemon ExecutionPermit are
  BOTH required for dispatch, same as exp3.
* ``max_action_age_ms`` is relaxed to 600 s here and DECLARED in the evidence:
  the canonical 300 ms freshness budget is incompatible with a human/operator
  permit-issuance step (compose -> permit-issue -> submit takes seconds).
  The strict 300 ms rejection is proven separately by Exp 8 S1 on this hand.
* The practice event chain (execution.armed / command.requested / sent /
  protocol_acknowledged / feedback.verified / step.completed) is recorded to a
  rollout trace and finalized into a practice session at shutdown.

Writes ``<evidence>/daemon_bootstrap.json`` with everything the operator step
needs to compose the exact one-step envelope (in-process permit_id, hashes,
names, body_id, socket).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rosclaw.body.execution.rh56_executor import RH56Executor
from rosclaw.body.rh56.calibration import RH56CalibrationGate, load_rh56_calibration
from rosclaw.body.rh56.resources import rh56_reference_policy_path
from rosclaw.body.rh56.transport import SerialModbusTransport
from rosclaw.body.rh56.transport_profile import (
    load_transport_profile,
    validate_transport_binding,
)
from rosclaw.daemon.cli import build_daemon_runtime
from rosclaw.daemon.ledger import (
    DaemonLedger,
    get_daemon_ledger_key_path,
    get_daemon_ledger_path,
)
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.integrations.lerobot.execution import (
    ArmingController,
    FeedbackVerifier,
    PermitManager,
    SingleStepExecutor,
)
from rosclaw.integrations.lerobot.execution.rh56_real_executor import (
    CAPABILITY_ID,
    RH56RealStepExecutor,
)
from rosclaw.integrations.lerobot.rollout.practice_bridge import (
    finalize_rollout_practice_session,
)
from rosclaw.integrations.lerobot.rollout.recorder import RolloutRecorder
from rosclaw.kernel.contracts import ExecutionMode

# Declared relaxation for the operator-in-the-loop permit flow (see docstring).
BLACKBOX_MAX_ACTION_AGE_MS = 600_000.0


def _compute_hashes(body_id: str, calibration: Any, profile: Any) -> dict[str, str]:
    """Same derivation as exp3_graded_execution.GradedDriver._compute_hashes."""
    policy_dir = Path(rh56_reference_policy_path())
    contract = policy_dir / "policy_contract.yaml"
    contract_hash = (
        f"sha256:{hashlib.sha256(contract.read_bytes()).hexdigest()}"
        if contract.exists()
        else "sha256:no_contract"
    )
    body_hash = hashlib.sha256(body_id.encode()).hexdigest()
    mapping_hash = hashlib.sha256((str(policy_dir) + body_id).encode()).hexdigest()
    return {
        "policy_contract_hash": contract_hash,
        "body_hash": f"sha256:{body_hash}",
        "calibration_hash": calibration.content_hash(),
        "mapping_hash": f"sha256:{mapping_hash}",
        "transport_profile_hash": profile.content_hash(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-profile", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--body-id", required=True)
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--socket", required=True, help="daemon unix socket path")
    parser.add_argument("--evidence", required=True, help="evidence output directory")
    args = parser.parse_args()

    robot_id = args.robot_id or args.body_id
    evidence = Path(args.evidence).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    socket_path = Path(args.socket).resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    profile = load_transport_profile(args.transport_profile)
    calibration = load_rh56_calibration(args.calibration)
    validate_transport_binding(
        profile,
        action_dim=len(profile.action_order),
        action_names=list(profile.action_order),
    )
    RH56CalibrationGate(calibration, profile).check()
    hashes = _compute_hashes(args.body_id, calibration, profile)

    transport = SerialModbusTransport(profile)
    transport.connect()
    start_position = list(transport.read_state().position)

    trace_path = evidence / "rollout_trace.jsonl"
    recorder = RolloutRecorder(
        trace_path=trace_path,
        robot_id=robot_id,
        body_id=args.body_id,
        policy_id=str(rh56_reference_policy_path()),
        task_id="agent_blackbox_real_noop",
    )

    def _event_sink(event_type: str, payload: dict[str, Any]) -> None:
        recorder._emit(
            "provider" if event_type.startswith("execution.command") else "runtime",
            event_type,
            payload,
        )

    permit_manager = PermitManager()
    arming = ArmingController(permit_manager)
    step = SingleStepExecutor(
        executor=RH56Executor(transport, profile),
        profile=profile,
        permit_manager=permit_manager,
        arming=arming,
        verifier=FeedbackVerifier(profile, calibration),
        execution_mode=ExecutionMode.REAL,
        event_sink=_event_sink,
        max_action_age_ms=BLACKBOX_MAX_ACTION_AGE_MS,
    )

    runtime = build_daemon_runtime(robot_id)
    runtime.action_gateway.register_executor(
        CAPABILITY_ID, ExecutionMode.REAL, RH56RealStepExecutor(step)
    )

    # In-process arming (second gate, same as exp3): preflight -> shadow hashes
    # -> issue -> arm.  This permit_id goes into the envelope arguments.
    arming.begin_preflight()
    arming.mark_shadow_validated(**hashes)
    permit = permit_manager.issue(
        body_id=args.body_id,
        **hashes,
        max_step_delta_raw=50.0,
        max_speed=400,
        max_force_g=400.0,
        expires_in_sec=7200.0,
        operator_armed=True,
        physical_estop_confirmed=True,
        task="agent_blackbox_noop",
        calibration_status=calibration.status,
        execution_mode="REAL",
    )
    arming.arm(permit.permit_id)
    _event_sink(
        "execution.armed",
        {"permit_id": permit.permit_id, "level": "blackbox_noop"},
    )

    bootstrap = {
        "schema_version": "rosclaw.acceptance.rh56_real_daemon.v1",
        "body_id": args.body_id,
        "robot_id": robot_id,
        "capability_id": CAPABILITY_ID,
        "socket": str(socket_path),
        "in_process_permit_id": permit.permit_id,
        "hashes": hashes,
        "names": list(profile.action_order),
        "units": "raw_device_unit",
        "representation": "joint_position",
        "speed": 400,
        "force_limit_g": 400.0,
        "max_action_age_ms": BLACKBOX_MAX_ACTION_AGE_MS,
        "max_action_age_note": (
            "relaxed for operator-in-loop permit issuance; strict 300 ms "
            "rejection proven by Exp 8 S1 on this hand"
        ),
        "start_position": start_position,
        "trace_path": str(trace_path),
        "practice_root": str(evidence / "practice"),
    }
    (evidence / "daemon_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    with DaemonLedger(
        get_daemon_ledger_path(evidence / "state" / "daemon" / "ledger.sqlite3"),
        key_path=get_daemon_ledger_key_path(evidence / "state" / "daemon" / "ledger.key"),
    ) as ledger:
        service = DaemonControlPlane(runtime=runtime, ledger=ledger)
        daemon = RosclawDaemon(
            service=service,
            socket_path=socket_path,
            socket_mode=0o600,
        )

        def _request_shutdown(_signum: int, _frame: Any) -> None:
            daemon.request_shutdown()

        previous = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, _request_shutdown)
        print(f"[rh56-real-daemon] serving {socket_path}", flush=True)
        try:
            daemon.serve_forever()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            service.close()

    # Finalize the practice event chain for verification.
    practice_id = None
    with contextlib.suppress(Exception) as finalize_error:
        practice_id = finalize_rollout_practice_session(
            trace_path,
            {"robot_id": robot_id, "task_id": "agent_blackbox_real_noop"},
            data_root=str(evidence / "practice"),
        )
    summary = {
        "practice_id": practice_id,
        "practice_finalize_error": str(finalize_error) if finalize_error else None,
        "hardware_actions_executed": step.hardware_actions_executed,
        "shutdown_at_monotonic_ns": time.monotonic_ns(),
    }
    (evidence / "daemon_shutdown.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[rh56-real-daemon] shutdown: {json.dumps(summary)}", flush=True)
    with contextlib.suppress(Exception):
        transport.close()
    with contextlib.suppress(Exception):
        runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
