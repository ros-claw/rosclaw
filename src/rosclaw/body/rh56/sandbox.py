"""RH56-aware sandbox preflight.

The generic MuJoCo firewall uses hardcoded humanoid joint limits
(-6.28..6.28 rad) and a heuristic self-collision model, which is meaningless
for an RH56 raw-device-unit hand (0-1000).  This module provides the correct
fail-closed checks for raw-device-unit bodies:

- every target inside the transport profile position range
- every target inside the calibrated per-actuator safe range (when calibration
  is provided)
- per-step delta from the current position bounded by ``max_step_delta_raw``
- no NaN/Inf values
"""

from __future__ import annotations

import math
import time
from typing import Any

from rosclaw.body.action_mapping.schema import BodyActionSpace, MappedAction
from rosclaw.body.rh56.calibration import RH56Calibration
from rosclaw.body.rh56.transport_profile import TransportProfile


def run_rh56_sandbox_preflight(
    mapped_action: MappedAction,
    body_action_space: BodyActionSpace,
    *,
    profile: TransportProfile,
    calibration: RH56Calibration | None = None,
    current_positions: list[float] | None = None,
    max_step_delta_raw: float | None = None,
) -> dict[str, Any]:
    """Validate a mapped RH56 action against range/calibration/step limits."""
    t0 = time.perf_counter()
    violations: list[str] = []
    values = [float(v) for v in mapped_action.body_action_values]
    names = list(body_action_space.joint_names)

    lo, hi = profile.position_range
    for i, value in enumerate(values):
        if math.isnan(value) or math.isinf(value):
            violations.append(f"nan_inf:{names[i] if i < len(names) else i}")
            continue
        if value < lo or value > hi:
            violations.append(
                f"profile_range:{names[i] if i < len(names) else i}={value} outside [{lo}, {hi}]"
            )

    if calibration is not None:
        for i, value in enumerate(values):
            name = names[i] if i < len(names) else ""
            spec = calibration.actuators.get(name)
            if spec is None:
                violations.append(f"calibration_missing:{name or i}")
                continue
            if value < spec.safe_min_raw or value > spec.safe_max_raw:
                violations.append(
                    f"calibration_safe_range:{name}={value} outside "
                    f"[{spec.safe_min_raw}, {spec.safe_max_raw}]"
                )

    if current_positions is not None and max_step_delta_raw is not None:
        for i, value in enumerate(values):
            if i < len(current_positions):
                delta = abs(value - float(current_positions[i]))
                if delta > max_step_delta_raw:
                    violations.append(
                        f"step_delta:{names[i] if i < len(names) else i} delta {delta:.1f} "
                        f"> {max_step_delta_raw}"
                    )

    is_safe = not violations
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "decision": "ALLOW" if is_safe else "BLOCK",
        "is_safe": is_safe,
        "risk_score": 0.0 if is_safe else 0.9,
        "reason": "rh56_range_check" if is_safe else f"rh56_sandbox: {'; '.join(violations[:3])}",
        "violations": violations,
        "replay_id": None,
        "latency_ms": round(latency_ms, 3),
        "trajectory": [values],
    }
