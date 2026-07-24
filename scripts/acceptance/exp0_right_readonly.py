#!/usr/bin/env python3
"""Exp 0: right-hand transport READONLY verification (清单 §1).

Gates:
  * device opens, slave id 2 answers;
  * 6 actuators POS/FORCE/CURRENT/STATUS/TEMP readable with stable dims;
  * 1000 consecutive reads, success rate >= 99.9%, no device reset;
  * STATUS/TEMP register widths CONFIRMED (left=3/3 grouped; right must
    be measured, then declared explicitly in the transport profile).

Output: /tmp/rh56_exp0_notes.md + /tmp/rh56_exp0_readonly.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rosclaw.body.rh56.transport import SerialModbusTransport
from rosclaw.body.rh56.transport_profile import load_transport_profile

READS = 1000
PROFILE = str(Path(__file__).resolve().parents[2] / "configs" / "rh56_right_rs485_v1.yaml")


def main() -> int:
    profile = load_transport_profile(PROFILE)
    transport = SerialModbusTransport(profile)
    transport.connect()
    report: dict = {
        "device": profile.transport.device,
        "slave_id": profile.transport.slave_id,
        "reads": READS,
    }
    try:
        positions: dict[str, list[float]] = {}
        forces: dict[str, list[float]] = {}
        temps: dict[str, list[float]] = {}
        failures: list[str] = []
        widths: dict[str, int] = {}
        t0 = time.monotonic()
        ok = 0
        for i in range(READS):
            try:
                state = transport.read_state()
                ok += 1
                widths["position"] = len(state.position)
                widths["force"] = len(state.force_g)
                widths["current"] = len(state.current_ma)
                widths["status"] = len(state.status_bits)
                widths["temperature"] = len(state.temperature_c)
                names = list(profile.action_order)
                for j, name in enumerate(names):
                    positions.setdefault(name, []).append(state.position[j])
                    forces.setdefault(name, []).append(state.force_g[j])
                    temps.setdefault(name, []).append(state.temperature_c[j])
            except Exception as exc:  # noqa: BLE001
                failures.append(f"read {i}: {type(exc).__name__}: {exc}")
                if len(failures) >= 5:
                    break
            if i % 200 == 199:
                time.sleep(0.002)
        elapsed = time.monotonic() - t0
        report["ok_reads"] = ok
        report["success_rate"] = round(ok / READS, 5)
        report["elapsed_s"] = round(elapsed, 1)
        report["effective_hz"] = round(ok / elapsed, 1)
        report["widths"] = widths
        report["failures"] = failures
        report["baseline"] = {
            name: {
                "position_median": statistics.median(values),
                "position_min": min(values),
                "position_max": max(values),
                "force_median_g": statistics.median(forces[name]),
                "force_min_g": min(forces[name]),
                "force_max_g": max(forces[name]),
                "temp_max_c": max(temps[name]),
            }
            for name, values in positions.items()
        }
        gates = {
            "device_open_slave_answer": ok > 0,
            "six_actuators_readable": widths.get("position") == 6 and widths.get("force") == 6,
            "success_rate_ge_99.9pct": ok / READS >= 0.999,
            "no_failures": not failures,
            "status_width_confirmed": widths.get("status") in (3, 6),
            "temp_width_confirmed": widths.get("temperature") in (3, 6),
        }
        report["gates"] = gates
        report["passed"] = all(gates.values())
    finally:
        transport.close()
    Path("/tmp/rh56_exp0_readonly.json").write_text(json.dumps(report, indent=1))
    lines = [
        "# RH56 right hand — Exp 0 readonly baseline",
        "",
        f"- device: {report['device']} slave {report['slave_id']}",
        f"- reads: {report['ok_reads']}/{READS} ok (success {report['success_rate'] * 100:.2f}%), {report['effective_hz']} Hz over {report['elapsed_s']}s",
        f"- register widths: {widths}",
        f"- failures: {failures or 'none'}",
        "",
        "| actuator | pos median | pos range | force median (g) | force range (g) | temp max (°C) |",
        "|---|---|---|---|---|---|",
    ]
    for name, b in report["baseline"].items():
        lines.append(
            f"| {name} | {b['position_median']:.0f} | {b['position_min']:.0f}-{b['position_max']:.0f} "
            f"| {b['force_median_g']:.0f} | {b['force_min_g']:.0f}..{b['force_max_g']:.0f} | {b['temp_max_c']:.0f} |"
        )
    Path("/tmp/rh56_exp0_notes.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"widths": widths, "success_rate": report["success_rate"], "passed": report["passed"]}, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
