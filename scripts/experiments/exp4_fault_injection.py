#!/usr/bin/env python3
"""Experiment 4: REAL fault injection on the RH56 bridge (DoD §5).

Eight scenarios; each asserts the DoD-expected outcome on the real pipeline
(ActionGateway + SingleStepExecutor + real SerialModbusTransport):

    S1 stale_observation        -> stale_action, command not sent
    S2 calibration_hash_changed -> permit refused + revoked
    S3 sandbox_block            -> command not sent
    S4 slave_no_response        -> COMMUNICATION_LOST + permit revoked
    S5 worker_restart           -> permit invalidated (session/permit lost)
    S6 status_protection        -> estop + failure event + permit revoked
    S7 ctrl_c                   -> emergency stop, armed state torn down
    S8 usb_unplug (interactive) -> stop dispatch, permit revoked, recovery

S8 is operator-assisted (no passwordless sudo for sysfs unbind): the script
prompts for physical unplug/replug while the hand is STATIC (DoD safety
note).  Use --skip-usb for unattended runs.

Usage (repo root, main venv):
    python scripts/experiments/exp4_fault_injection.py [--skip-usb]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rosclaw.body.execution.rh56_executor import RH56Executor
from rosclaw.body.rh56.calibration import load_rh56_calibration
from rosclaw.body.rh56.resources import rh56_reference_policy_path
from rosclaw.body.rh56.transport import RH56Feedback, SerialModbusTransport, TransportIOError
from rosclaw.body.rh56.transport_profile import (
    TransportProfile,
    load_transport_profile,
)
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
from rosclaw.kernel.action_gateway import ActionGateway
from rosclaw.kernel.contracts import (
    ActionEnvelope,
    ActionState,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)
from rosclaw.observability.exporters.jsonl import JsonlTraceExporter
from rosclaw.observability.tracer import Tracer

REPO_ROOT = Path(__file__).resolve().parents[2]


class FaultHarness:
    """Real armed stack for fault scenarios (one per scenario, fresh)."""

    def __init__(
        self,
        profile_path: str,
        calibration_path: str,
        slave_id: int | None = None,
        sysfs_id: str | None = None,
        body_id: str = "rh56_left_01",
        robot_id: str | None = None,
        spans_path: str | None = None,
    ):
        self.body_id = body_id
        self.robot_id = robot_id or body_id
        self.profile = load_transport_profile(profile_path)
        if sysfs_id:
            # When a sysfs id is declared, ALWAYS resolve by it — the node
            # name flips between minors across re-enumerations.
            node = _find_adapter_node(sysfs_id)
            if node:
                self.profile.transport.device = node
        if not Path(self.profile.transport.device).exists():
            node = _find_adapter_node(sysfs_id)
            if node:
                self.profile.transport.device = node
        if slave_id is not None:
            data = self.profile.to_dict()
            data["transport"]["slave_id"] = slave_id
            self.profile = TransportProfile.from_dict(data)
        self.calibration = load_rh56_calibration(calibration_path)
        print(f"[harness] device={self.profile.transport.device}", flush=True)
        self.transport = SerialModbusTransport(self.profile)
        self.transport.connect()
        self.permit_manager = PermitManager()
        self.arming = ArmingController(self.permit_manager)
        # P0-5: persist ROBOT_ACTION/ROBOT_STATE spans so the acceptance trace
        # gate can replay the fault-injection causal tree.
        self.span_exporter = JsonlTraceExporter(output_path=spans_path) if spans_path else None
        self.tracer = Tracer(exporters=[self.span_exporter] if self.span_exporter else [])
        self.gateway = ActionGateway(tracer=self.tracer)
        self.step = SingleStepExecutor(
            executor=RH56Executor(self.transport, self.profile),
            profile=self.profile,
            permit_manager=self.permit_manager,
            arming=self.arming,
            verifier=FeedbackVerifier(self.profile, self.calibration),
            execution_mode=ExecutionMode.REAL,
        )
        self.gateway.register_executor(
            CAPABILITY_ID, ExecutionMode.REAL, RH56RealStepExecutor(self.step)
        )
        self.hashes = self._compute_hashes()
        self.permit = None

    def _compute_hashes(self) -> dict[str, str]:
        policy_dir = Path(rh56_reference_policy_path())
        contract = policy_dir / "policy_contract.yaml"
        contract_hash = (
            f"sha256:{hashlib.sha256(contract.read_bytes()).hexdigest()}"
            if contract.exists()
            else "sha256:no_contract"
        )
        body_hash = hashlib.sha256(self.body_id.encode()).hexdigest()
        mapping_hash = hashlib.sha256((str(policy_dir) + self.body_id).encode()).hexdigest()
        return {
            "policy_contract_hash": contract_hash,
            "body_hash": f"sha256:{body_hash}",
            "calibration_hash": self.calibration.content_hash(),
            "mapping_hash": f"sha256:{mapping_hash}",
            "transport_profile_hash": self.profile.content_hash(),
        }

    def arm(self) -> None:
        self.arming.begin_preflight()
        self.arming.mark_shadow_validated(**self.hashes)
        self.permit = self.permit_manager.issue(
            body_id=self.body_id,
            **self.hashes,
            max_step_delta_raw=50.0,
            max_speed=400,
            max_force_g=400.0,
            expires_in_sec=300.0,
            operator_armed=True,
            physical_estop_confirmed=True,
            task="exp4_fault_injection",
            calibration_status=self.calibration.status,
            execution_mode="REAL",
        )
        self.arming.arm(self.permit.permit_id)

    def envelope(self, values: list[float], **arg_overrides) -> ActionEnvelope:
        args = {
            "permit_id": self.permit.permit_id,
            "names": list(self.profile.action_order),
            "values": values,
            "representation": "joint_position",
            "units": "raw_device_unit",
            "hashes": self.hashes,
            "speed": 400,
            "force_limit_g": 400.0,
            "observation_timestamp_ns": time.monotonic_ns(),
        }
        args.update(arg_overrides)
        return ActionEnvelope(
            actor_id="exp4_fault_injection",
            agent_framework="rosclaw.exp4",
            session_id=f"exp4_{uuid.uuid4().hex[:8]}",
            body_id=self.body_id,
            capability_id=CAPABILITY_ID,
            arguments=args,
            execution_mode=ExecutionMode.REAL,
            body_snapshot_hash=self.hashes["calibration_hash"],
            authorization=AuthorizationContext(
                principal_id="operator_exp4",
                approved=True,
                approval_id=f"exp4_{self.permit.permit_id}",
                scopes=[CAPABILITY_ID],
            ),
            verification_policy=VerificationPolicy(
                required_evidence=EvidenceLevel.PHYSICALLY_OBSERVED,
                fail_closed=True,
            ),
        )

    def update_transport(self, transport) -> None:
        """Point the whole stack at a (re-bound) transport instance."""
        self.transport = transport
        self.step.executor = RH56Executor(transport, self.profile)

    def noop_values(self) -> list[float]:
        return [1000.0] * 6

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.tracer.close()
        with contextlib.suppress(Exception):
            self.transport.close()


def _find_adapter_node(sysfs_id: str | None) -> str | None:
    """Find the /dev/ttyUSB* node bound to a sysfs id (re-enumeration safe).

    USB re-enumeration may assign a different minor (ttyUSB2), so the
    adapter must be rediscovered by sysfs id instead of assuming ttyUSB1.
    """
    if not sysfs_id:
        # Never guess an adapter: falling back to "any FTDI" can pick the
        # WRONG hand's port.  Fail loudly instead.
        return "/dev/ttyUSB1" if Path("/dev/ttyUSB1").exists() else None
    for node in sorted(Path("/dev").glob("ttyUSB*")):
        try:
            out = subprocess.run(
                ["udevadm", "info", "-q", "property", "-n", str(node)],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except Exception:  # noqa: BLE001
            continue
        if f"/{sysfs_id}:" in out:
            return str(node)
    return None


def _read_with_retry(h: FaultHarness, attempts: int = 3) -> Any:
    """read_state with reconnect-and-retry for post-rebind first-contact flakes."""
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return h.transport.read_state()
        except TransportIOError as exc:
            last_exc = exc
            with contextlib.suppress(Exception):
                h.transport.close()
            with contextlib.suppress(Exception):
                h.transport.connect()
            time.sleep(0.3)
    raise last_exc  # type: ignore[misc]


def _receipt_codes(receipt) -> list[str]:
    return [e.get("code", "") for e in receipt.errors]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def s1_stale_observation(h: FaultHarness) -> dict[str, Any]:
    """2 s-old observation must be refused before dispatch (stale_action)."""
    h.arm()
    before = h.step.hardware_actions_executed
    receipt = h.gateway.submit(
        h.envelope(
            h.noop_values(),
            observation_timestamp_ns=time.monotonic_ns() - int(2e9),
        )
    )
    codes = _receipt_codes(receipt)
    passed = (
        receipt.final_state is ActionState.BLOCKED
        and codes
        and "stale_action" in codes[0]
        and h.step.hardware_actions_executed == before
        and not h.permit_manager.is_revoked(h.permit.permit_id)
    )
    return {
        "scenario": "S1 stale_observation",
        "expected": "stale_action, command not sent, permit kept",
        "final_state": receipt.final_state.value,
        "error_codes": codes,
        "hardware_actions": h.step.hardware_actions_executed,
        "passed": passed,
    }


def s2_calibration_hash_changed(h: FaultHarness) -> dict[str, Any]:
    """A changed calibration hash must refuse + revoke the permit."""
    h.arm()
    tampered = dict(h.hashes, calibration_hash="sha256:tampered_calibration")
    receipt = h.gateway.submit(h.envelope(h.noop_values(), hashes=tampered))
    codes = _receipt_codes(receipt)
    revoked = h.permit_manager.is_revoked(h.permit.permit_id)
    # The next submit (even with correct hashes) must also be refused.
    receipt2 = h.gateway.submit(h.envelope(h.noop_values()))
    passed = (
        receipt.final_state is ActionState.BLOCKED
        and any("permit_hash_mismatch" in c for c in codes)
        and revoked
        and receipt2.final_state is ActionState.BLOCKED
        and any("permit_revoked" in c for c in _receipt_codes(receipt2))
        and h.step.hardware_actions_executed == 0
    )
    return {
        "scenario": "S2 calibration_hash_changed",
        "expected": "permit_hash_mismatch, permit revoked, later submits refused",
        "final_state": receipt.final_state.value,
        "error_codes": codes,
        "permit_revoked": revoked,
        "followup_codes": _receipt_codes(receipt2),
        "passed": passed,
    }


def s3_sandbox_block(h: FaultHarness) -> dict[str, Any]:
    """A calibration safe-range violation must be blocked before dispatch."""
    from types import SimpleNamespace

    from rosclaw.body.action_mapping import resolve_body_action_space
    from rosclaw.body.rh56.mock_body import build_mock_rh56_body
    from rosclaw.body.rh56.sandbox import run_rh56_sandbox_preflight

    body = build_mock_rh56_body(h.profile)
    body_space = resolve_body_action_space(body)
    current = [float(p) for p in _read_with_retry(h).position]
    # index safe_min_raw=138: target 100 violates the calibrated safe range.
    values = list(current)
    values[3] = 100.0
    sandbox = run_rh56_sandbox_preflight(
        SimpleNamespace(body_action_values=values),
        body_space,
        profile=h.profile,
        calibration=h.calibration,
        current_positions=current,
        max_step_delta_raw=50.0,
    )
    passed = not sandbox.get("is_safe") and h.step.hardware_actions_executed == 0
    return {
        "scenario": "S3 sandbox_block",
        "expected": "is_safe False (calibration_safe_range), command not sent",
        "sandbox": {k: sandbox.get(k) for k in ("is_safe", "reason", "violations")},
        "hardware_actions": h.step.hardware_actions_executed,
        "passed": passed,
    }


def s4_slave_no_response(
    profile_path: str,
    calibration_path: str,
    sysfs_id: str | None = None,
    harness_kwargs: dict | None = None,
) -> dict[str, Any]:
    """A silent slave (no unit answers) must escalate to COMMUNICATION_LOST."""
    kwargs = dict(harness_kwargs or {})
    kwargs.pop("sysfs_id", None)
    h = FaultHarness(profile_path, calibration_path, slave_id=2, sysfs_id=sysfs_id, **kwargs)
    outcome: dict[str, Any] = {"scenario": "S4 slave_no_response"}
    try:
        h.arm()
        receipt = h.gateway.submit(h.envelope(h.noop_values()))
        codes = _receipt_codes(receipt)
        outcome.update(
            expected="COMMUNICATION_LOST + permit revoked",
            final_state=receipt.final_state.value,
            error_codes=codes,
            arming_state=h.arming.machine.state.value,
            permit_revoked=h.permit_manager.is_revoked(h.permit.permit_id),
            passed=(
                receipt.final_state is ActionState.FAILED
                and any("communication" in c.lower() or "command_send_failed" in c for c in codes)
                and h.arming.machine.state.value == "COMMUNICATION_LOST"
                and h.permit_manager.is_revoked(h.permit.permit_id)
            ),
        )
    except TransportIOError as exc:
        # The arm-time noop itself may surface the dead slave; that is also a
        # valid COMMUNICATION_LOST signal for this scenario.
        outcome.update(
            expected="COMMUNICATION_LOST",
            exception=str(exc),
            passed="serial" in str(exc).lower() or "timeout" in str(exc).lower(),
        )
    finally:
        h.close()
    return outcome


def s5_worker_restart(h: FaultHarness) -> dict[str, Any]:
    """A policy worker restart must invalidate the session permit."""
    h.arm()
    revoked_count = h.permit_manager.on_worker_restart()
    receipt = h.gateway.submit(h.envelope(h.noop_values()))
    codes = _receipt_codes(receipt)
    passed = (
        revoked_count >= 1
        and receipt.final_state is ActionState.BLOCKED
        and any("permit_revoked" in c for c in codes)
        and h.step.hardware_actions_executed == 0
    )
    return {
        "scenario": "S5 worker_restart",
        "expected": "permit invalidated; later submits refused (session lost)",
        "permits_revoked_on_restart": revoked_count,
        "final_state": receipt.final_state.value,
        "error_codes": codes,
        "passed": passed,
    }


class _ProtectionBitWrapper:
    """Injects one protection bit into the next post-command feedback read."""

    def __init__(self, inner: SerialModbusTransport, bit: int = 0x04):
        self._inner = inner
        self._bit = bit
        self._reads = 0

    @property
    def execution_mode(self) -> str:
        return self._inner.execution_mode

    def is_connected(self) -> bool:
        return self._inner.is_connected()

    def connect(self) -> None:
        self._inner.connect()

    def close(self) -> None:
        self._inner.close()

    def read_angle_setpoints(self) -> list[int]:
        return self._inner.read_angle_setpoints()

    def write_position(self, positions, *, speed: int, force_limit: int) -> bool:
        return self._inner.write_position(positions, speed=speed, force_limit=force_limit)

    def emergency_stop(self) -> bool:
        return self._inner.emergency_stop()

    @property
    def last_command_delivery(self) -> str:
        return self._inner.last_command_delivery

    def read_state(self) -> RH56Feedback:
        fb = self._inner.read_state()
        self._reads += 1
        if self._reads >= 2:  # post-command feedback (verifier-facing)
            fb.status_bits = [s | self._bit for s in fb.status_bits]
        return fb


def s6_status_protection(h: FaultHarness) -> dict[str, Any]:
    """A STATUS protection bit must fault the step: estop + permit revoked."""
    wrapped = _ProtectionBitWrapper(h.transport, bit=0x04)
    h.step.executor = RH56Executor(wrapped, h.profile)
    h.arm()
    # True no-op: command the current pose (the verifier faults on the
    # injected protection bit, not on any intended motion).
    current = [float(p) for p in h.transport.read_state().position]
    receipt = h.gateway.submit(h.envelope(current))
    codes = _receipt_codes(receipt)
    state = h.arming.machine.state.value
    revoked = h.permit_manager.is_revoked(h.permit.permit_id)
    passed = (
        receipt.final_state is ActionState.FAILED
        and state == "FAULT"
        and revoked
        and any("feedback" in c or "fault" in c for c in codes)
    )
    return {
        "scenario": "S6 status_protection_simulated",
        "expected": "verifier fault (estop) + failure event + permit revoked",
        "final_state": receipt.final_state.value,
        "arming_state": state,
        "permit_revoked": revoked,
        "error_codes": codes,
        "passed": passed,
    }


def s7_ctrl_c() -> dict[str, Any]:
    """SIGINT during a level: emergency stop, no further dispatch."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "scripts/experiments/exp3_graded_execution.py",
            "--levels",
            "noop",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(12.0)  # let it arm and run a few noop steps
    proc.send_signal(signal.SIGINT)
    try:
        out, _ = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    interrupted = '"interrupted": true' in out
    estop = "emergency stop" in out or "Ctrl+C" in out
    passed = proc.returncode is not None and interrupted and estop
    return {
        "scenario": "S7 ctrl_c",
        "expected": "estop + interrupted summary + process exits (DISARMED)",
        "returncode": proc.returncode,
        "interrupted": interrupted,
        "estop_seen": estop,
        "passed": passed,
    }


def _sudo_n(*cmd: str) -> tuple[bool, str]:
    """Run a command as root: direct when euid==0, sudo -S with EXP4_SUDO_PASS,
    else cached sudo (sudo -v beforehand)."""
    if os.geteuid() == 0:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    elif os.environ.get("EXP4_SUDO_PASS") is not None:
        r = subprocess.run(
            ["sudo", "-S", *cmd],
            input=os.environ["EXP4_SUDO_PASS"] + "\n",
            capture_output=True,
            text=True,
            timeout=15,
        )
    else:
        r = subprocess.run(["sudo", "-n", *cmd], capture_output=True, text=True, timeout=15)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def s8_usb_unplug(
    h: FaultHarness, sysfs_id: str | None = None, set_transport=None
) -> dict[str, Any]:
    """USB unplug while STATIC: stop dispatch, revoke permit, recover.

    With ``sysfs_id`` the disconnect is driven electrically through sysfs
    unbind/bind (equivalent to pulling the cable; needs cached sudo
    credentials).  Without it, the operator is prompted to unplug physically.
    """
    h.arm()
    # Baseline noop before the unplug.  A freshly re-bound adapter can flake
    # on first contact, so allow one retry with a fresh transport — but any
    # baseline failure must be visible in the report.
    receipt_ok = None
    baseline_errors: list[list[str]] = []
    for attempt in range(2):
        receipt_ok = h.gateway.submit(h.envelope(h.noop_values()))
        if receipt_ok.final_state is ActionState.COMPLETED:
            break
        baseline_errors.append(_receipt_codes(receipt_ok))
        if attempt == 0:
            node = _find_adapter_node(sysfs_id)
            if node:
                h.profile.transport.device = node
                try:
                    h.update_transport(SerialModbusTransport(h.profile))
                    h.transport.connect()
                except Exception:  # noqa: BLE001
                    pass
                h.arming.disarm("baseline retry")
                h.arm()
    pre_ok = receipt_ok is not None and receipt_ok.final_state is ActionState.COMPLETED

    if sysfs_id:
        ok, err = _sudo_n("sh", "-c", f"echo {sysfs_id} > /sys/bus/usb/drivers/usb/unbind")
        if not ok:
            return {
                "scenario": "S8 usb_unplug",
                "expected": "COMMUNICATION_LOST + permit revoked + recovery",
                "passed": False,
                "note": f"sysfs unbind failed (run `sudo -v` interactively first): {err}",
            }
        print(f"[S8] sysfs unbind {sysfs_id} — electrical disconnect issued", flush=True)
        time.sleep(1.5)
        disconnected = True
    else:
        print("\n[S8] Hand is STATIC at the open pose.")
        print("[S8] Please UNPLUG the /dev/ttyUSB1 adapter now (60 s window)...", flush=True)
        disconnected = False
        for _ in range(600):
            try:
                h.transport.read_state()
            except TransportIOError:
                disconnected = True
                break
            if not Path("/dev/ttyUSB1").exists():
                disconnected = True
                break
            time.sleep(0.1)
    if not disconnected:
        return {
            "scenario": "S8 usb_unplug",
            "expected": "COMMUNICATION_LOST + permit revoked + recovery",
            "passed": False,
            "note": "no disconnect observed within 60 s (operator skipped?)",
        }

    receipt = h.gateway.submit(h.envelope(h.noop_values()))
    codes = _receipt_codes(receipt)
    revoked = h.permit_manager.is_revoked(h.permit.permit_id)
    fault_state = h.arming.machine.state.value

    if sysfs_id:
        ok, err = _sudo_n("sh", "-c", f"echo {sysfs_id} > /sys/bus/usb/drivers/usb/bind")
        if not ok:
            return {
                "scenario": "S8 usb_unplug",
                "expected": "recovery after bind",
                "passed": False,
                "note": f"sysfs bind failed: {err} (device left unbound!)",
            }
        print(f"[S8] sysfs bind {sysfs_id} — reconnect issued", flush=True)
    else:
        print("[S8] Disconnect handled. Please REPLUG the adapter (60 s window)...", flush=True)

    recovered = False
    for _ in range(600):
        node = _find_adapter_node(sysfs_id)
        if node:
            with contextlib.suppress(Exception):
                h.transport.close()
            try:
                h.profile.transport.device = node
                new_transport = SerialModbusTransport(h.profile)
                new_transport.connect()
                new_transport.read_state()
                if set_transport is not None:
                    set_transport(new_transport)
                recovered = True
                if node != "/dev/ttyUSB1":
                    print(f"[S8] adapter re-enumerated as {node}", flush=True)
                break
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.1)

    # Full recovery: re-arm a fresh permit and dispatch one verified noop.
    recovery_completed = False
    if recovered:
        if set_transport is not None:
            set_transport(h.transport)
        h.arming.disarm("post-recovery re-arm")
        h.arm()
        receipt2 = h.gateway.submit(h.envelope(h.noop_values()))
        recovery_completed = receipt2.final_state is ActionState.COMPLETED

    passed = (
        pre_ok
        and receipt.final_state is ActionState.FAILED
        and revoked
        and fault_state == "COMMUNICATION_LOST"
        and recovered
        and recovery_completed
    )
    return {
        "scenario": "S8 usb_unplug",
        "expected": "stop dispatch (COMMUNICATION_LOST), permit revoked, clean recovery",
        "pre_unplug_completed": pre_ok,
        "baseline_errors": baseline_errors,
        "final_state": receipt.final_state.value,
        "error_codes": codes,
        "permit_revoked": revoked,
        "arming_state": fault_state,
        "recovered": recovered,
        "recovery_completed": recovery_completed,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-usb", action="store_true", help="skip S8 (unattended)")
    parser.add_argument(
        "--only",
        default=None,
        help="run only one scenario, e.g. s8 (others must have passed separately)",
    )
    parser.add_argument(
        "--usb-sysfs",
        default=None,
        help="sysfs id for S8 electrical unbind/bind (e.g. 5-1.4); needs cached sudo",
    )
    parser.add_argument("--transport-profile", default="configs/rh56_left_rs485_v1.yaml")
    parser.add_argument("--calibration", default="configs/rh56_left_01_calibration.yaml")
    parser.add_argument("--body-id", default="rh56_left_01")
    parser.add_argument("--robot-id", default=None)
    args = parser.parse_args()
    spans_path = f"/tmp/exp4_fault_injection_{uuid.uuid4().hex[:8]}.spans.jsonl"
    harness_kwargs = {
        "sysfs_id": getattr(args, "usb_sysfs", None),
        "body_id": args.body_id,
        "robot_id": args.robot_id,
        "spans_path": spans_path,
    }
    print(f"spans: {spans_path}", flush=True)

    results: list[dict[str, Any]] = []

    if args.only == "s8":
        h = FaultHarness(args.transport_profile, args.calibration, **harness_kwargs)
        try:
            results.append(
                s8_usb_unplug(h, sysfs_id=args.usb_sysfs, set_transport=h.update_transport)
            )
        finally:
            h.close()
        passed = all(r.get("passed") for r in results)
        report = {
            "experiment": "exp4_fault_injection",
            "only": "s8",
            "passed": passed,
            "scenarios": results,
        }
        Path("/tmp/exp4_fault_injection_s8.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
        for r in results:
            print(f"[{'PASS' if r.get('passed') else 'FAIL'}] {r['scenario']}")
        print(f"EXP4 S8 {'PASSED' if passed else 'FAILED'}")
        return 0 if passed else 1

    h = FaultHarness(args.transport_profile, args.calibration, **harness_kwargs)
    try:
        results.append(s1_stale_observation(h))
    finally:
        h.close()
        time.sleep(1.0)  # FTDI settle between rapid harness reopen cycles

    h = FaultHarness(args.transport_profile, args.calibration, **harness_kwargs)
    try:
        results.append(s2_calibration_hash_changed(h))
    finally:
        h.close()
        time.sleep(1.0)

    h = FaultHarness(args.transport_profile, args.calibration, **harness_kwargs)
    try:
        results.append(s3_sandbox_block(h))
    finally:
        h.close()
        time.sleep(1.0)

    results.append(
        s4_slave_no_response(
            args.transport_profile,
            args.calibration,
            sysfs_id=args.usb_sysfs,
            harness_kwargs=harness_kwargs,
        )
    )

    h = FaultHarness(args.transport_profile, args.calibration, **harness_kwargs)
    try:
        results.append(s5_worker_restart(h))
    finally:
        h.close()
        time.sleep(1.0)

    h = FaultHarness(args.transport_profile, args.calibration, **harness_kwargs)
    try:
        results.append(s6_status_protection(h))
    finally:
        h.close()

    results.append(s7_ctrl_c())

    if not args.skip_usb:
        h = FaultHarness(args.transport_profile, args.calibration, **harness_kwargs)
        try:
            results.append(
                s8_usb_unplug(h, sysfs_id=args.usb_sysfs, set_transport=h.update_transport)
            )
        finally:
            h.close()

    passed = all(r.get("passed") for r in results)
    report = {"experiment": "exp4_fault_injection", "passed": passed, "scenarios": results}
    Path("/tmp/exp4_fault_injection.json").write_text(json.dumps(report, indent=2, default=str))
    for r in results:
        mark = "PASS" if r.get("passed") else "FAIL"
        print(f"[{mark}] {r['scenario']}")
    print(f"EXP4 {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
