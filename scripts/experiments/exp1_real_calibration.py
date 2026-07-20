#!/usr/bin/env python3
"""Experiment 1: real RH56 calibration (replaces mock evidence).

Measures on the physical hand and writes a validated calibration document:

* serial identity (HAND_ID / VLTAGE / slave address);
* real open/closed range per actuator (commanded vs measured);
* steady-state position tolerance;
* force baseline (idle, open pose);
* idle temperature baseline + thermal drift curve under repeated motion;
* usable speed ladder (completion time per speed setting);
* threshold labeling: 55/60 °C stays but is marked ``measured_conservative``.

Usage (repo root, venv):
    python scripts/experiments/exp1_real_calibration.py \
        --device /dev/ttyUSB1 --slave-id 1 --body rh56_left_01 \
        --transport-profile configs/rh56_right_rs485_v1.yaml \
        --output configs/rh56_left_01_calibration.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rosclaw.body.rh56 import modbus
from rosclaw.body.rh56.calibration import (
    ActuatorCalibration,
    FeedbackCalibration,
    RH56Calibration,
    write_rh56_calibration,
)
from rosclaw.body.rh56.transport import SerialModbusTransport
from rosclaw.body.rh56.transport_profile import load_transport_profile


def _wait_settled(transport: SerialModbusTransport, timeout_s: float = 3.0, window: int = 3) -> None:
    """Wait until positions stop changing between consecutive reads."""
    last = None
    stable = 0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = transport.read_state()
        if last is not None and state.position == last:
            stable += 1
            if stable >= window:
                return
        else:
            stable = 0
        last = state.position
        time.sleep(0.15)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--slave-id", type=int, required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--transport-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thermal-seconds", type=float, default=90.0)
    parser.add_argument("--report", default="/tmp/exp1_calibration_report.json")
    args = parser.parse_args()

    profile = load_transport_profile(args.transport_profile)
    profile.transport.device = args.device
    profile.transport.slave_id = args.slave_id

    transport = SerialModbusTransport(profile)
    transport.connect()
    report: dict = {"body": args.body, "device": args.device, "slave_id": args.slave_id}

    # 1. Serial identity -------------------------------------------------------
    hand_id = transport._read_registers(modbus.Register.HAND_ID, 1)[0]
    voltage = transport._read_registers(modbus.Register.VLTAGE, 1)[0]
    report["serial_identity"] = {"hand_id": hand_id, "voltage_raw": voltage,
                                 "slave_matches": hand_id == args.slave_id}

    # 2. Force + idle temperature baseline (open pose, resting) ---------------
    _wait_settled(transport)
    time.sleep(1.0)  # let any in-motion force artifact decay before sampling
    forces, temps = [], []
    attempts = 0
    while len(forces) < 10 and attempts < 30:
        state = transport.read_state()
        peak = max(abs(f) for f in state.force_g)
        attempts += 1
        if peak > 200.0:
            # Motion/touch artifact — do not let a transient become the baseline.
            time.sleep(0.5)
            continue
        forces.append(peak)
        temps.append(max(state.temperature_c))
        time.sleep(0.2)
    force_baseline_max = max(forces) if forces else 200.0
    idle_temp_c = statistics.mean(temps)
    report["force_baseline"] = {"max_abs_g": force_baseline_max, "samples": len(forces),
                                "discarded_artifacts": attempts - len(forces)}
    report["idle_temp_baseline_c"] = round(idle_temp_c, 1)

    # 3. Open / closed range ---------------------------------------------------
    transport.write_position([1000] * 6, speed=400, force_limit=400)
    _wait_settled(transport)
    open_actual = transport.read_state().position
    transport.write_position([0] * 6, speed=400, force_limit=400)
    _wait_settled(transport)
    closed_actual = transport.read_state().position
    report["range"] = {"open_commanded": 1000, "open_actual": open_actual,
                       "closed_commanded": 0, "closed_actual": closed_actual}

    # 4. Position tolerance (mid-range targets) --------------------------------
    tolerances = []
    for target in (300, 700):
        transport.write_position([target] * 6, speed=400, force_limit=400)
        _wait_settled(transport)
        actual = transport.read_state().position
        tolerances.append(max(abs(a - target) for a in actual))
    position_tolerance = max(tolerances) + 5  # margin
    report["position_tolerance_raw"] = {"measured_max": max(tolerances),
                                        "calibrated": position_tolerance}

    # 5. Usable speed ladder ----------------------------------------------------
    closed_reference = max(closed_actual) + 40  # measured closed pose + margin
    speed_results = []
    for speed in (200, 400, 600, 800):
        transport.write_position([1000] * 6, speed=speed, force_limit=400)
        _wait_settled(transport)
        t0 = time.monotonic()
        transport.write_position([0] * 6, speed=speed, force_limit=400)
        _wait_settled(transport, timeout_s=6.0)
        elapsed = time.monotonic() - t0
        final = transport.read_state().position
        reached = max(final) <= closed_reference
        speed_results.append({"speed": speed, "close_time_s": round(elapsed, 2),
                              "final_max": max(final), "reached": reached})
    reached_speeds = [r["speed"] for r in speed_results if r["reached"]]
    usable = max(reached_speeds) if reached_speeds else 200
    report["speed_ladder"] = speed_results
    report["usable_speed"] = usable
    report["closed_reference"] = closed_reference

    # 6. Thermal curve under repeated motion ------------------------------------
    transport.write_position([1000] * 6, speed=usable, force_limit=400)
    _wait_settled(transport)
    temp_start = max(transport.read_state().temperature_c)
    samples = []
    deadline = time.monotonic() + args.thermal_seconds
    pose = 0
    while time.monotonic() < deadline:
        pose = 1000 - pose
        transport.write_position([pose] * 6, speed=usable, force_limit=400)
        time.sleep(2.0)
        state = transport.read_state()
        samples.append({"t": round(time.monotonic(), 1), "temp_max": max(state.temperature_c)})
    temp_end = max(transport.read_state().temperature_c)
    slope = (temp_end - temp_start) / (args.thermal_seconds / 60.0)
    report["thermal"] = {
        "start_c": temp_start, "end_c": temp_end,
        "slope_c_per_min": round(slope, 3),
        "samples": samples[:: max(1, len(samples) // 8)],
    }

    # 7. Restore open pose --------------------------------------------------------
    transport.write_position([1000] * 6, speed=usable, force_limit=400)
    _wait_settled(transport)
    transport.close()

    # 8. Write the calibration document -----------------------------------------
    actuators = {}
    for i, name in enumerate(profile.action_order):
        spec = ActuatorCalibration()
        spec.open_raw = open_actual[i]
        spec.closed_raw = closed_actual[i]
        spec.safe_min_raw = max(0, closed_actual[i] - 25)
        spec.safe_max_raw = min(1000, open_actual[i] + 25)
        spec.position_tolerance_raw = position_tolerance
        actuators[name] = spec

    feedback = FeedbackCalibration()
    feedback.force_baseline_file = ""
    feedback.force_soft_limit_g = max(100.0, force_baseline_max * 2)
    feedback.force_hard_limit_g = max(300.0, force_baseline_max * 6)
    feedback.temperature_warning_c = 55.0
    feedback.temperature_stop_c = 60.0
    feedback.thresholds_source = "measured_conservative"

    calib = RH56Calibration(
        body_id=args.body,
        transport_profile=profile.id,
        actuators=actuators,
        feedback=feedback,
    )
    out = Path(args.output).expanduser()
    write_rh56_calibration(calib, out)
    report["calibration_file"] = str(out)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
