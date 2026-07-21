#!/usr/bin/env python3
"""Experiment 3: graded REAL execution on the RH56 left hand (DoD Exp 2-7).

Levels run in order, never skipping:

    noop     20 trials  hold current pose (delta 0)
    micro    10 trials  index finger ±20 raw
    motion   10 trials  index finger ±50 raw
    gesture  10 trials  multi-finger half-close, non-contact
    ok       10 trials  index-thumb contact, FORCE_ACT criterion

Release criteria (DoD §4):

    noop 20/20, micro 10/10, motion 10/10, gesture 10/10, ok >= 9/10,
    hardware protection events == 0, emergency over-contact == 0.

Every motion command goes through the Runtime ActionGateway with an issued
REAL permit (one per level): observe → sandbox preflight → envelope →
gateway.submit → receipt.  One envelope = one single-step command; waypoints
walk at <= 40 raw per step, far under the permit's max_step_delta_raw.

Usage (repo root, main venv):
    python scripts/experiments/exp3_graded_execution.py [--levels noop,micro]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rosclaw.body.action_mapping import resolve_body_action_space
from rosclaw.body.execution.rh56_executor import RH56Executor
from rosclaw.body.rh56.calibration import RH56CalibrationGate, load_rh56_calibration
from rosclaw.body.rh56.mock_body import build_mock_rh56_body
from rosclaw.body.rh56.resources import rh56_reference_policy_path
from rosclaw.body.rh56.sandbox import run_rh56_sandbox_preflight
from rosclaw.body.rh56.transport import SerialModbusTransport, TransportIOError
from rosclaw.body.rh56.transport_profile import (
    load_transport_profile,
    validate_transport_binding,
)
from rosclaw.integrations.lerobot.execution import (
    ArmingController,
    FeedbackVerifier,
    PermitManager,
    SingleStepExecutor,
)
from rosclaw.integrations.lerobot.execution.feedback_verifier import (
    STATUS_PROTECTION_MASK,
)
from rosclaw.integrations.lerobot.execution.rh56_real_executor import (
    CAPABILITY_ID,
    RH56RealStepExecutor,
)
from rosclaw.integrations.lerobot.rollout.practice_bridge import (
    finalize_rollout_practice_session,
)
from rosclaw.integrations.lerobot.rollout.recorder import RolloutRecorder
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
PRACTICE_ROOT = Path.home() / ".rosclaw" / "practice" / "runs" / "lerobot_bridge"
REPORT_DIR = REPO_ROOT / "reports" / "lerobot_bridge"

WAYPOINT_DELTA = 10  # raw per single-step command (permit allows 50)
WALK_SPEED = 400
WALK_SETTLE_MS = 600.0
HOLD_FORCE_G = 400.0  # FORCE_SET needed to hold pose against gravity (exp1)
CONTACT_FORCE_G = 70.0  # FORCE_ACT contact rise over motion baseline (~49 g)
CONTACT_APPROACH_FORCE_G = 150.0  # gentle motor torque during OK approach
OVER_CONTACT_ABORT_G = 250.0  # controlled abort, back off immediately
FORCE_HARD_LIMIT_G = 300.0  # calibration hard limit — never to be reached

LEVELS = ("noop", "micro", "motion", "gesture", "ok")
REQUIRED = {"noop": 20, "micro": 10, "motion": 10, "gesture": 10, "ok": 9}
TRIALS = {"noop": 20, "micro": 10, "motion": 10, "gesture": 10, "ok": 10}


# OK contact geometry per hand.  Left: measured on this rig 2026-07-17
# (settle-wait grid search).  Right: promoted pose from the v2 OK contact
# experiments on this same physical hand (ok_contact_safe_v2_together_test:
# thumb 420 / index 410 / thumb_rot 300, force window thumb 80-180 g).
OK_GEOMETRY = {
    "left": {
        "coarse": [1000.0, 1000.0, 1000.0, 400.0, 400.0, 250.0],
        "floor": [1000.0, 1000.0, 1000.0, 400.0, 210.0, 250.0],
    },
    "right": {
        "coarse": [1000.0, 1000.0, 1000.0, 410.0, 700.0, 300.0],
        "floor": [1000.0, 1000.0, 1000.0, 410.0, 340.0, 300.0],
    },
}


class GradedDriver:
    def __init__(
        self,
        profile_path: str,
        calibration_path: str,
        trace_path: Path,
        *,
        body_id: str = "rh56_left_01",
        robot_id: str | None = None,
        hand: str = "left",
    ):
        self.body_id = body_id
        self.robot_id = robot_id or body_id
        if hand not in OK_GEOMETRY:
            raise ValueError(f"unknown hand {hand!r}; expected one of {sorted(OK_GEOMETRY)}")
        self.ok_geometry = OK_GEOMETRY[hand]
        self.profile = load_transport_profile(profile_path)
        self.calibration = load_rh56_calibration(calibration_path)
        validate_transport_binding(
            self.profile,
            action_dim=len(self.profile.action_order),
            action_names=list(self.profile.action_order),
        )
        RH56CalibrationGate(self.calibration, self.profile).check()

        self.transport = SerialModbusTransport(self.profile)
        self.transport.connect()
        self.body = build_mock_rh56_body(self.profile)
        self.body_space = resolve_body_action_space(self.body)

        self.permit_manager = PermitManager()
        self.arming = ArmingController(self.permit_manager)
        # P0-5: persist ROBOT_ACTION/ROBOT_STATE spans next to the rollout
        # JSONL so the acceptance trace gate can replay the causal tree.
        self.span_exporter = JsonlTraceExporter(output_path=trace_path.with_suffix(".spans.jsonl"))
        self.tracer = Tracer(exporters=[self.span_exporter])
        self.gateway = ActionGateway(tracer=self.tracer)
        self.step = SingleStepExecutor(
            executor=RH56Executor(self.transport, self.profile),
            profile=self.profile,
            permit_manager=self.permit_manager,
            arming=self.arming,
            verifier=FeedbackVerifier(self.profile, self.calibration),
            execution_mode=ExecutionMode.REAL,
            event_sink=self._event_sink,
        )
        self.gateway.register_executor(
            CAPABILITY_ID, ExecutionMode.REAL, RH56RealStepExecutor(self.step)
        )

        self.recorder = RolloutRecorder(
            trace_path=trace_path,
            robot_id=self.robot_id,
            body_id=self.body_id,
            policy_id=str(rh56_reference_policy_path()),
            task_id="exp3_graded_execution",
        )
        self.hashes = self._compute_hashes()
        self.protection_events = 0
        self.emergency_over_contact = 0
        self.aborted = False
        self.permit = None

    # ------------------------------------------------------------------
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

    def _event_sink(self, event_type: str, payload: dict[str, Any]) -> None:
        self.recorder._emit(
            "provider" if event_type.startswith("execution.command") else "runtime",
            event_type,
            payload,
        )

    # ------------------------------------------------------------------
    def arm_level(self, level: str, expires_in_sec: float = 900.0) -> None:
        if self.arming.machine.state.value != "DISARMED":
            self.arming.disarm(f"level {level} re-arm")
        self.arming.begin_preflight()
        self.arming.mark_shadow_validated(**self.hashes)
        self.permit = self.permit_manager.issue(
            body_id=self.body_id,
            **self.hashes,
            max_step_delta_raw=50.0,
            max_speed=WALK_SPEED,
            max_force_g=400.0,
            expires_in_sec=expires_in_sec,
            operator_armed=True,
            physical_estop_confirmed=True,
            task=f"exp3_{level}",
            calibration_status=self.calibration.status,
            execution_mode="REAL",
        )
        self.arming.arm(self.permit.permit_id)
        self._event_sink("execution.armed", {"permit_id": self.permit.permit_id, "level": level})

    def disarm(self, reason: str) -> None:
        self.arming.disarm(reason)
        self.permit = None

    # ------------------------------------------------------------------
    def _check_feedback_safety(self, obs: dict[str, Any], context: str) -> list[str]:
        """Scan one feedback observation for protection/over-contact evidence."""
        problems = []
        for i, status in enumerate(obs.get("status_bits", [])):
            if int(status) & STATUS_PROTECTION_MASK:
                self.protection_events += 1
                problems.append(f"status_protection[{i}]=0x{int(status):02x}")
        for i, force in enumerate(obs.get("force_g", [])):
            if float(force) >= FORCE_HARD_LIMIT_G:
                self.emergency_over_contact += 1
                problems.append(f"force_hard_limit[{i}]={force}")
        for i, temp in enumerate(obs.get("temperature_c", [])):
            if float(temp) >= self.calibration.feedback.temperature_stop_c:
                self.protection_events += 1
                problems.append(f"temperature_stop[{i}]={temp}")
        if problems:
            self._event_sink("exp3.safety_trip", {"context": context, "problems": problems})
        return problems

    def read_observation(self) -> tuple[Any, int]:
        fb = self.transport.read_state()
        return fb, time.monotonic_ns()

    def submit_step(
        self,
        values: list[float],
        *,
        obs_ts_ns: int,
        force_limit_g: float,
        settle_ms: float = WALK_SETTLE_MS,
        current: list[float] | None = None,
    ) -> Any:
        """Sandbox-preflight one waypoint and submit it through the gateway.

        ``current`` is the observation the waypoint was computed from; the
        sandbox must check against the SAME snapshot (a settling joint moves
        between reads and would falsely trip the step-delta guard).
        """
        if current is None:
            fb = self.transport.read_state()
            current = [float(p) for p in fb.position]
        sandbox = run_rh56_sandbox_preflight(
            SimpleNamespace(body_action_values=[float(v) for v in values]),
            self.body_space,
            profile=self.profile,
            calibration=self.calibration,
            current_positions=current,
            max_step_delta_raw=self.permit.max_step_delta_raw if self.permit else None,
        )
        if not sandbox.get("is_safe"):
            self._event_sink("exp3.sandbox_block", {"values": values, "sandbox": sandbox})
            raise RuntimeError(f"sandbox_block: {sandbox.get('reason')}")

        envelope = ActionEnvelope(
            actor_id="exp3_driver",
            agent_framework="rosclaw.exp3",
            session_id=f"exp3_{uuid.uuid4().hex[:8]}",
            body_id=self.body_id,
            capability_id=CAPABILITY_ID,
            arguments={
                "permit_id": self.permit.permit_id,
                "names": list(self.profile.action_order),
                "values": [float(v) for v in values],
                "representation": "joint_position",
                "units": "raw_device_unit",
                "hashes": self.hashes,
                "speed": WALK_SPEED,
                "force_limit_g": float(force_limit_g),
                "observation_timestamp_ns": obs_ts_ns,
                "settle_ms": settle_ms,
            },
            execution_mode=ExecutionMode.REAL,
            body_snapshot_hash=self.hashes["calibration_hash"],
            authorization=AuthorizationContext(
                principal_id="operator_exp3",
                approved=True,
                approval_id=f"exp3_{self.permit.permit_id}",
                scopes=[CAPABILITY_ID],
            ),
            verification_policy=VerificationPolicy(
                required_evidence=EvidenceLevel.PHYSICALLY_OBSERVED,
                fail_closed=True,
            ),
        )
        receipt = self.gateway.submit(envelope)
        self.recorder._emit(
            "runtime",
            "gateway.receipt",
            receipt.to_dict(),
        )
        return receipt

    def walk_to(
        self,
        target: list[float],
        *,
        force_limit_g: float,
        stop_on_contact: bool = False,
        abort_on_over_contact: bool = False,
        waypoint_delta: float = WAYPOINT_DELTA,
        contact_channel: int | None = None,
        convergence_channels: list[int] | None = None,
        settle_ms: float = WALK_SETTLE_MS,
    ) -> dict[str, Any]:
        """Walk to a target pose in <=waypoint_delta single-step commands.

        ``contact_channel`` limits contact/over-contact checks to one FORCE_ACT
        channel (default: max over all).  ``convergence_channels`` limits the
        reached check to those channels — auxiliary channels are held by the
        device (setpoint hysteresis) and their steady-state error must not
        block convergence of the channel being actively driven.
        """
        report: dict[str, Any] = {
            "reached": False,
            "contact": False,
            "over_contact_abort": False,
            "steps": 0,
            "problems": [],
            "final_position": None,
            "max_force": [0.0] * 6,
        }
        converging = (
            convergence_channels if convergence_channels is not None else list(range(len(target)))
        )
        last_waypoint: list[float] | None = None
        for _ in range(96):  # hard cap on waypoints per walk
            fb, obs_ts = self.read_observation()
            current = [float(p) for p in fb.position]
            if last_waypoint is None:
                last_waypoint = list(current)
            obs = {
                "status_bits": list(fb.status_bits),
                "force_g": list(fb.force_g),
                "temperature_c": list(fb.temperature_c),
            }
            problems = self._check_feedback_safety(obs, "walk")
            report["problems"].extend(problems)
            for i, f in enumerate(fb.force_g):
                report["max_force"][i] = max(report["max_force"][i], abs(float(f)))
            if problems:
                self.aborted = True
                return report
            forces = fb.force_g
            watched_force = (
                float(forces[contact_channel])
                if contact_channel is not None
                else max(float(f) for f in forces)
            )
            if stop_on_contact and watched_force >= CONTACT_FORCE_G:
                report["contact"] = True
                report["reached"] = True
                report["final_position"] = current
                return report
            if abort_on_over_contact and watched_force >= OVER_CONTACT_ABORT_G:
                report["over_contact_abort"] = True
                report["reached"] = True
                report["final_position"] = current
                return report

            tolerance = max(
                self.calibration.position_tolerance(self.profile.action_order[i])
                for i in converging
            )
            if all(abs(float(target[i]) - current[i]) <= tolerance for i in converging):
                report["reached"] = True
                report["final_position"] = current
                return report

            # Marching waypoints: advance from the last COMMANDED waypoint,
            # not from the (possibly lagging) actual — a slow joint must not
            # deadlock the walk against the setpoint hysteresis.  The command
            # stays within the permit's step window of the CURRENT actual:
            # a lagging joint shrinks the window, so progress waits for it.
            step_window = (self.permit.max_step_delta_raw if self.permit else 50.0) - 5.0
            waypoint = []
            for i in range(6):
                marched = last_waypoint[i] + max(
                    -waypoint_delta,
                    min(waypoint_delta, float(target[i]) - last_waypoint[i]),
                )
                waypoint.append(
                    max(current[i] - step_window, min(current[i] + step_window, marched))
                )
            receipt = self.submit_step(
                waypoint,
                obs_ts_ns=obs_ts,
                force_limit_g=force_limit_g,
                current=current,
                settle_ms=settle_ms,
            )
            last_waypoint = list(waypoint)
            report["steps"] += 1
            if receipt.final_state is not ActionState.COMPLETED:
                report["problems"].append(
                    f"receipt {receipt.final_state.value}: "
                    f"{receipt.errors[0]['code'] if receipt.errors else 'unknown'}"
                )
                return report
            for obs in receipt.observations:
                obs_forces = obs.get("force_g", [])
                for i, f in enumerate(obs_forces):
                    report["max_force"][i] = max(report["max_force"][i], abs(float(f)))
                self._check_feedback_safety(obs, "receipt")
                if obs_forces:
                    receipt_force = (
                        float(obs_forces[contact_channel])
                        if contact_channel is not None
                        else max(float(f) for f in obs_forces)
                    )
                    if stop_on_contact and receipt_force >= CONTACT_FORCE_G:
                        report["contact"] = True
                        report["reached"] = True
                        report["final_position"] = list(obs.get("position", current))
                        return report
                    if abort_on_over_contact and receipt_force >= OVER_CONTACT_ABORT_G:
                        report["over_contact_abort"] = True
                        report["reached"] = True
                        report["final_position"] = list(obs.get("position", current))
                        return report

        report["problems"].append("walk_waypoint_cap_exceeded")
        return report

    # ------------------------------------------------------------------
    # Trial runners
    # ------------------------------------------------------------------
    def restore_open_pose(self) -> dict[str, Any]:
        """Setup phase: walk to the open pose so every level starts known."""
        self.arm_level("restore_open")
        try:
            open_pose = [1000.0] * 6
            report = self.walk_to(open_pose, force_limit_g=HOLD_FORCE_G)
            report["phase"] = "restore_open"
            return report
        finally:
            self.disarm("restore_open done")

    def trial_noop(self, index: int) -> dict[str, Any]:
        # Zero intended motion from the (gravity-stable) open pose.  Note:
        # commanding the CURRENT mid-range position makes this firmware coast
        # for one servo cycle (~15-17 raw dip on gravity-loaded joints) — a
        # real device characteristic documented in the exp3 report; the open
        # pose is the honest zero-motion baseline.
        _, obs_ts = self.read_observation()
        target = [1000.0] * 6
        receipt = self.submit_step(target, obs_ts_ns=obs_ts, force_limit_g=HOLD_FORCE_G)
        passed = receipt.final_state is ActionState.COMPLETED
        return {
            "trial": index,
            "passed": passed,
            "target": target,
            "final_state": receipt.final_state.value,
            "errors": [e.get("code") for e in receipt.errors],
        }

    def trial_single_finger(self, index: int, delta: float) -> dict[str, Any]:
        fb, _ = self.read_observation()
        base = [float(p) for p in fb.position]
        flexed = list(base)
        flexed[3] = max(0.0, base[3] - delta)  # index flexes toward closed
        out = self.walk_to(flexed, force_limit_g=HOLD_FORCE_G)
        back = self.walk_to(base, force_limit_g=HOLD_FORCE_G)
        passed = out["reached"] and back["reached"] and not (out["problems"] or back["problems"])
        return {
            "trial": index,
            "passed": passed,
            "delta": delta,
            "out": _trim(out),
            "back": _trim(back),
        }

    def trial_gesture(self, index: int) -> dict[str, Any]:
        fb, _ = self.read_observation()
        base = [float(p) for p in fb.position]
        # Non-contact half-close: four fingers to ~55% close, thumb stays open.
        pose = list(base)
        for i, name in enumerate(self.profile.action_order):
            if name in ("little", "ring", "middle", "index"):
                pose[i] = max(450.0, base[i] - 450.0)
        out = self.walk_to(pose, force_limit_g=HOLD_FORCE_G)
        back = self.walk_to(base, force_limit_g=HOLD_FORCE_G)
        passed = out["reached"] and back["reached"] and not (out["problems"] or back["problems"])
        return {"trial": index, "passed": passed, "out": _trim(out), "back": _trim(back)}

    def trial_ok_contact(self, index: int) -> dict[str, Any]:
        fb, _ = self.read_observation()
        base = [float(p) for p in fb.position]
        # OK contact region: per-hand geometry from OK_GEOMETRY (left hand
        # measured 2026-07-17 on this rig; right hand from the promoted v2
        # contact pose on the same physical hand).  Either contact channel may register
        # the press (f_th 69-463 or f_idx 76-463 depending on micro-geometry);
        # contact is judged on the max FORCE_ACT of any channel.  Threshold
        # 70 g = clear rise over the ~49 g rest baseline / ~58 g motion
        # artifact.  Two-phase approach: coarse 40-raw steps to thumb=400,
        # then 10-raw steps (max ~120 g per step at ~12 g/raw gradient) so
        # the contact threshold is always caught before the >=250 g abort.
        coarse = self.walk_to(
            list(self.ok_geometry["coarse"]),
            force_limit_g=CONTACT_APPROACH_FORCE_G,
            settle_ms=800.0,
        )
        if not coarse["reached"]:
            return {
                "trial": index,
                "passed": False,
                "contact": False,
                "approach": _trim(coarse),
                "back": None,
                "reason": "coarse approach failed",
            }
        approach = self.walk_to(
            list(self.ok_geometry["floor"]),
            force_limit_g=CONTACT_APPROACH_FORCE_G,
            stop_on_contact=True,
            abort_on_over_contact=True,
            waypoint_delta=10.0,
            convergence_channels=[4],  # only the thumb must converge (contact floor)
            settle_ms=800.0,
        )
        contact = approach["contact"]
        back = self.walk_to(base, force_limit_g=HOLD_FORCE_G, settle_ms=800.0)
        passed = (
            contact
            and back["reached"]
            and not approach["over_contact_abort"]
            and not (coarse["problems"] or approach["problems"] or back["problems"])
        )
        return {
            "trial": index,
            "passed": passed,
            "contact": contact,
            "approach": _trim(approach),
            "back": _trim(back),
        }

    # ------------------------------------------------------------------
    def run_level(self, level: str) -> dict[str, Any]:
        self.arm_level(level, expires_in_sec=1800.0 if level == "ok" else 900.0)
        trials = []
        try:
            for i in range(TRIALS[level]):
                if self.aborted:
                    break
                if level == "noop":
                    result = self.trial_noop(i)
                elif level == "micro":
                    result = self.trial_single_finger(i, 20.0)
                elif level == "motion":
                    result = self.trial_single_finger(i, 50.0)
                elif level == "gesture":
                    result = self.trial_gesture(i)
                else:
                    result = self.trial_ok_contact(i)
                trials.append(result)
                self.recorder._emit("runtime", f"exp3.trial.{level}", result)
                print(
                    f"  {level}[{i}] {'PASS' if result['passed'] else 'FAIL'}",
                    flush=True,
                )
        finally:
            self.disarm(f"level {level} done")
        passed = sum(1 for t in trials if t["passed"])
        return {
            "level": level,
            "trials": trials,
            "passed": passed,
            "total": len(trials),
            "required": f"{REQUIRED[level]}/{TRIALS[level]}",
            "level_ok": passed >= REQUIRED[level] and len(trials) == TRIALS[level],
        }

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self.permit is not None:
                self.disarm("driver close")
        with contextlib.suppress(Exception):
            self.tracer.close()
        with contextlib.suppress(Exception):
            self.transport.close()


def _trim(walk: dict[str, Any]) -> dict[str, Any]:
    return {
        "reached": walk["reached"],
        "contact": walk.get("contact", False),
        "steps": walk["steps"],
        "problems": walk["problems"],
        "max_force": [round(f, 1) for f in walk["max_force"]],
        "final_position": walk.get("final_position"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--transport-profile", default="configs/rh56_left_rs485_v1.yaml")
    parser.add_argument("--calibration", default="configs/rh56_left_01_calibration.yaml")
    parser.add_argument("--body-id", default="rh56_left_01")
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--practice-root", default=None)
    args = parser.parse_args()
    levels = [lv for lv in LEVELS if lv in args.levels.split(",")]

    trace_path = Path(f"/tmp/rosclaw_exp3_graded_{uuid.uuid4().hex[:8]}.jsonl")
    driver = GradedDriver(
        args.transport_profile,
        args.calibration,
        trace_path,
        body_id=args.body_id,
        robot_id=args.robot_id,
        hand=args.hand,
    )
    practice_root = args.practice_root or str(PRACTICE_ROOT)
    summary: dict[str, Any] = {
        "experiment": "exp3_graded_execution",
        "levels": [],
        "protection_events": 0,
        "emergency_over_contact": 0,
    }
    print(f"trace: {trace_path}")
    try:
        print("=== setup: restore open pose ===", flush=True)
        restore = driver.restore_open_pose()
        summary["restore_open"] = _trim(restore)
        if not restore["reached"] or restore["problems"]:
            raise RuntimeError(f"restore_open failed: {restore['problems']}")
        for level in levels:
            print(f"=== level {level} ({TRIALS[level]} trials) ===", flush=True)
            report = driver.run_level(level)
            summary["levels"].append(report)
            if not report["level_ok"]:
                print(f"level {level} FAILED — stopping (no skipping allowed)")
                break
            if driver.aborted:
                print("ABORTED on safety trip — stopping")
                break
    except KeyboardInterrupt:
        print("Ctrl+C — emergency stop")
        driver.step.emergency_stop()
        summary["interrupted"] = True
    except (RuntimeError, TransportIOError) as exc:
        print(f"FATAL: {exc}")
        driver.step.emergency_stop()
        summary["fatal"] = str(exc)
    finally:
        summary["protection_events"] = driver.protection_events
        summary["emergency_over_contact"] = driver.emergency_over_contact
        summary["hardware_actions_executed"] = driver.step.hardware_actions_executed
        summary["permit_status"] = driver.permit_manager.status()
        try:
            summary["practice_id"] = finalize_rollout_practice_session(
                trace_path,
                {"robot_id": driver.robot_id, "task_id": "exp3_graded_execution"},
                data_root=practice_root,
            )
        except Exception as exc:  # noqa: BLE001
            summary["practice_error"] = str(exc)
        driver.close()

    gate_ok = (
        all(lv["level_ok"] for lv in summary["levels"])
        and len(summary["levels"]) == len(levels)
        and summary["protection_events"] == 0
        and summary["emergency_over_contact"] == 0
        and not summary.get("interrupted")
        and not summary.get("fatal")
    )
    summary["passed"] = gate_ok
    Path("/tmp/exp3_graded_execution.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "levels"}, indent=2, default=str))
    for lv in summary["levels"]:
        print(f"{lv['level']}: {lv['passed']}/{lv['total']} (required {lv['required']})")
    print(
        f"protection_events={summary['protection_events']} "
        f"emergency_over_contact={summary['emergency_over_contact']}"
    )
    print(f"EXP3 {'PASSED' if gate_ok else 'FAILED'}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
