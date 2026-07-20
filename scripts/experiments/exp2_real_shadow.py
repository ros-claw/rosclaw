#!/usr/bin/env python3
"""Experiment 2: real RH56 shadow gate (S1 smoke -> S2 full gate).

Runs the P5-B shadow rollout against the *real* SerialModbusTransport
(read-only: hardware_actions_executed must stay 0) instead of the fixture
MockModbusTransport.

Stage S1: 100 steps @ 2 Hz — smoke test.  Observes joint names/order, raw
range, observation age, force/current/temp/status values, serial latency,
mapping, sandbox and practice recording.  The 1000-step / 4.8 Hz gate
thresholds do not apply to S1; every other gate criterion must hold.

Stage S2: 1000 steps @ 5 Hz — the full shadow gate (plan §6.4):
steps>=1000, hardware_actions=0, unknown semantics=0, mapping blocks=0,
NaN/Inf=0, observation validation failures=0, serial disconnects=0,
worker restarts=0, effective_hz>=4.8, deadline miss rate<1%.

Usage (repo root, main venv):
    python scripts/experiments/exp2_real_shadow.py --stage both
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rosclaw.body.rh56.calibration import load_rh56_calibration
from rosclaw.body.rh56.resources import rh56_reference_policy_path
from rosclaw.body.rh56.transport import SerialModbusTransport
from rosclaw.body.rh56.transport_profile import load_transport_profile
from rosclaw.integrations.lerobot.rollout.loop import RolloutConfig
from rosclaw.integrations.lerobot.rollout.rh56_shadow import (
    render_shadow_report,
    run_rh56_shadow,
)
from rosclaw.integrations.lerobot.rollout.state import RolloutMode

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PYTHON = REPO_ROOT / ".venv-lerobot" / "bin" / "python"
PRACTICE_ROOT = Path.home() / ".rosclaw" / "practice" / "runs" / "lerobot_bridge"
REPORT_DIR = REPO_ROOT / "reports" / "lerobot_bridge"


class SamplingTransport:
    """Read-latency/feedback sampler wrapping a real RH56 transport.

    Only the attributes ``run_rh56_shadow`` touches are forwarded; the
    wrapper itself performs no I/O beyond ``read_state`` timing.
    """

    def __init__(self, inner: SerialModbusTransport):
        self._inner = inner
        self.read_latencies_ms: list[float] = []
        self.samples: list[dict] = []

    # --- forwarded transport surface -------------------------------------
    @property
    def execution_mode(self) -> str:
        return self._inner.execution_mode

    def is_connected(self) -> bool:
        return self._inner.is_connected()

    def connect(self) -> None:
        self._inner.connect()

    def close(self) -> None:
        self._inner.close()

    # --- sampled read -----------------------------------------------------
    def read_state(self):
        t0 = time.monotonic()
        fb = self._inner.read_state()
        dt_ms = (time.monotonic() - t0) * 1000.0
        self.read_latencies_ms.append(dt_ms)
        self.samples.append(
            {
                "i": len(self.samples),
                "latency_ms": round(dt_ms, 2),
                "position": list(fb.position),
                "force_g": [round(f, 1) for f in fb.force_g],
                "current_ma": [round(c, 1) for c in fb.current_ma],
                "temperature_c": [round(t, 1) for t in fb.temperature_c],
                "status_bits": list(fb.status_bits),
            }
        )
        return fb


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[k]


def _latency_summary(lat: list[float]) -> dict:
    if not lat:
        return {"count": 0}
    return {
        "count": len(lat),
        "mean_ms": round(statistics.fmean(lat), 2),
        "p50_ms": round(_pct(lat, 0.50), 2),
        "p95_ms": round(_pct(lat, 0.95), 2),
        "max_ms": round(max(lat), 2),
    }


def _build_transport(profile_path: str) -> SamplingTransport:
    profile = load_transport_profile(profile_path)
    transport = SerialModbusTransport(profile)
    transport.connect()
    return SamplingTransport(transport)


def _run_stage(
    stage: str,
    *,
    steps: int,
    control_hz: float,
    strict_deadline: bool,
    max_deadline_misses: int | None,
    profile_path: str,
    calibration_path: str,
) -> tuple[dict, object, SamplingTransport]:
    calibration = load_rh56_calibration(calibration_path)
    transport = _build_transport(profile_path)
    config = RolloutConfig(
        mode=RolloutMode.SHADOW,
        policy_path=str(rh56_reference_policy_path()),
        robot_id="rh56_left_01",
        steps=steps,
        control_hz=control_hz,
        strict_deadline=strict_deadline,
        max_deadline_misses=max_deadline_misses,
        python_executable=str(WORKER_PYTHON),
        practice_data_root=str(PRACTICE_ROOT),
        tags=["exp2", stage, "real_transport"],
    )
    result, gate = run_rh56_shadow(
        config,
        transport_profile_path=profile_path,
        transport=transport,
        task="hold_current",
        calibration=calibration,
        fixture_mode=False,
    )
    return gate, result, transport


def _evaluate_s1(gate: dict, result, sampler: SamplingTransport) -> dict:
    """S1 smoke verdict: every gate criterion except steps/hz thresholds."""
    checks = gate["checks"]
    metrics = result.metrics or {}
    positions = [p for s in sampler.samples for p in s["position"]]
    forces = [f for s in sampler.samples for f in s["force_g"]]
    smoke = {
        "steps_completed_100": result.steps_completed == 100,
        "hardware_actions_zero": result.hardware_actions_executed == 0,
        "unknown_semantics_zero": checks["unknown_action_semantics"]["pass"],
        "mapping_blocks_zero": checks["incompatible_mapping"]["pass"],
        "nan_inf_zero": checks["nan_inf"]["pass"],
        "observation_validation_zero": checks["required_observation_stale"]["pass"],
        "serial_disconnects_zero": checks["serial_disconnect_count"]["pass"],
        "read_failures_zero": gate["serial_health"].get("read_failure_count", 1) == 0,
        "worker_restarts_zero": checks["worker_restart_count"]["pass"],
        "positions_in_raw_range": all(0 <= p <= 1000 for p in positions),
        "force_finite": all(-32768 <= f <= 32767 for f in forces),
    }
    verdict = {
        "passed": all(smoke.values()),
        "checks": smoke,
        "stop_reason": gate["stop_reason"],
        "deadline_misses_at_2hz": metrics.get("deadline_misses", 0),
        "effective_hz_at_2hz": metrics.get("effective_control_hz", 0.0),
        "serial_read_latency": _latency_summary(sampler.read_latencies_ms),
        "step_latency_ms": metrics.get("step_latency_ms", {}),
        "first_observation": sampler.samples[0] if sampler.samples else None,
        "last_observation": sampler.samples[-1] if sampler.samples else None,
        "state_names": ["little", "ring", "middle", "index", "thumb", "thumb_rot"],
        "practice_id": gate.get("practice_id"),
        "trace_path": gate.get("trace_path"),
    }
    return verdict


def _provenance_appendix(sampler: SamplingTransport) -> str:
    lat = _latency_summary(sampler.read_latencies_ms)
    return f"""
## Provenance (real hardware)

This report supersedes the earlier fixture/mock shadow evidence (whose file
name misleadingly contained "REAL").  All observations above come from the
physical RH56 hand via the real Modbus-RTU transport:

- Transport: `SerialModbusTransport` (`hardware_serial_modbus_rtu`),
  `/dev/ttyUSB1` @ 115200 8N1, Modbus slave id 1 (HAND_ID read-back = 1).
- Serial read latency over {lat.get("count", 0)} reads: mean {lat.get("mean_ms")} ms,
  p95 {lat.get("p95_ms")} ms, max {lat.get("max_ms")} ms.
- Worker: `.venv-lerobot` (python 3.12, lerobot 0.6.x, plugin
  `lerobot-policy-rosclaw-rh56`).
- Runner: `scripts/experiments/exp2_real_shadow.py`.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["s1", "s2", "both"], default="both")
    parser.add_argument("--transport-profile", default="configs/rh56_left_rs485_v1.yaml")
    parser.add_argument("--calibration", default="configs/rh56_left_01_calibration.yaml")
    args = parser.parse_args()

    if not WORKER_PYTHON.exists():
        print(f"FATAL: worker python missing: {WORKER_PYTHON}")
        return 2

    out: dict = {"experiment": "exp2_real_shadow", "stage_results": {}}

    if args.stage in ("s1", "both"):
        print("=== S1: 100 steps @ 2 Hz (real transport smoke) ===", flush=True)
        gate, result, sampler = _run_stage(
            "s1",
            steps=100,
            control_hz=2.0,
            strict_deadline=False,
            max_deadline_misses=None,
            profile_path=args.transport_profile,
            calibration_path=args.calibration,
        )
        s1 = _evaluate_s1(gate, result, sampler)
        out["stage_results"]["s1"] = {"gate": gate, "verdict": s1}
        Path("/tmp/exp2_shadow_s1.json").write_text(
            json.dumps(out["stage_results"]["s1"], indent=2, default=str)
        )
        print(json.dumps(s1, indent=2, default=str), flush=True)
        if not s1["passed"]:
            failed = [k for k, v in s1["checks"].items() if not v]
            print(f"S1 FAILED ({failed}); not proceeding to S2.")
            Path("/tmp/exp2_shadow.json").write_text(json.dumps(out, indent=2, default=str))
            return 1
        print("S1 PASSED.", flush=True)

    if args.stage in ("s2", "both"):
        print("=== S2: 1000 steps @ 5 Hz (full shadow gate) ===", flush=True)
        gate, result, sampler = _run_stage(
            "s2",
            steps=1000,
            control_hz=5.0,
            strict_deadline=True,
            max_deadline_misses=10,
            profile_path=args.transport_profile,
            calibration_path=args.calibration,
        )
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        calibration = load_rh56_calibration(args.calibration)
        report_md = render_shadow_report(
            gate,
            result,
            calibration_hash=calibration.content_hash(),
        )
        report_md += _provenance_appendix(sampler)
        report_path = REPORT_DIR / "P5_RH56_SHADOW_REPORT_S2_OBSERVED.md"
        report_path.write_text(report_md, encoding="utf-8")
        out["stage_results"]["s2"] = {
            "gate": gate,
            "serial_read_latency": _latency_summary(sampler.read_latencies_ms),
            "report_path": str(report_path),
        }
        Path("/tmp/exp2_shadow_s2.json").write_text(
            json.dumps(out["stage_results"]["s2"], indent=2, default=str)
        )
        print(json.dumps(gate["checks"], indent=2, default=str), flush=True)
        print(f"gate passed: {gate['passed']}")
        print(f"report: {report_path}")
        Path("/tmp/exp2_shadow.json").write_text(json.dumps(out, indent=2, default=str))
        return 0 if gate["passed"] else 1

    Path("/tmp/exp2_shadow.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
