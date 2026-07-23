#!/usr/bin/env python3
"""§2 helper: measure per-actuator position tolerance on the RIGHT hand.

For each actuator, in free space, at calibration-safe speed/force:
  1. command current - STEP raw, settle, read actual -> err_down
  2. command back to start, settle, read actual -> err_up
tolerance = ceil(max(err_down, err_up) + MARGIN)

Also prints temp rise evidence for thresholds_source=measured_conservative.

Writes /tmp/rh56_right_tolerance_measured.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src")

from rosclaw.body.rh56.transport import SerialModbusTransport
from rosclaw.body.rh56.transport_profile import load_transport_profile

STEP = 300
SETTLE_S = 1.6
SPEED = 400
FORCE_LIMIT = 100
MARGIN = 5
PROFILE = "/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/configs/rh56_right_rs485_v1.yaml"


def main() -> int:
    profile = load_transport_profile(PROFILE)
    transport = SerialModbusTransport(profile)
    transport.connect()
    names = list(profile.action_order)
    out: dict = {"step_raw": STEP, "speed": SPEED, "force_limit_g": FORCE_LIMIT, "settle_s": SETTLE_S}
    try:
        start = list(transport.read_state().position)
        out["start_position"] = start
        temps0 = list(transport.read_state().temperature_c)
        errors: dict[str, dict] = {}
        for j, name in enumerate(names):
            down_target = [int(p) for p in start]
            down_target[j] = max(0, int(start[j]) - STEP)
            if not transport.write_position(down_target, speed=SPEED, force_limit=FORCE_LIMIT):
                raise RuntimeError(f"write_position down rejected on {name}")
            time.sleep(SETTLE_S)
            actual_down = transport.read_state().position[j]
            back_target = [int(p) for p in start]
            if not transport.write_position(back_target, speed=SPEED, force_limit=FORCE_LIMIT):
                raise RuntimeError(f"write_position back rejected on {name}")
            time.sleep(SETTLE_S)
            actual_back = transport.read_state().position[j]
            err_down = abs(actual_down - down_target[j])
            err_back = abs(actual_back - back_target[j])
            errors[name] = {
                "err_down_raw": err_down,
                "err_back_raw": err_back,
                "measured_raw": max(err_down, err_back),
                "tolerance_raw": int(max(err_down, err_back)) + MARGIN,
            }
        temps1 = list(transport.read_state().temperature_c)
        out["errors"] = errors
        out["temps_before_c"] = temps0
        out["temps_after_c"] = temps1
        out["passed"] = all(e["tolerance_raw"] <= 80 for e in errors.values())
    finally:
        transport.close()
    Path("/tmp/rh56_right_tolerance_measured.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["errors"], indent=1))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
