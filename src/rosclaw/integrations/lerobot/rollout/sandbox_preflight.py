"""Sandbox preflight for a mapped body action candidate.

P4 policy rollouts are proposal-only / shadow: the sandbox validates the mapped
action sequence but never executes on hardware.  This module wires the existing
``SandboxRuntimeAdapter.validate_trajectory`` path so body-mapped candidates are
checked before any (simulated) motion.
"""

from __future__ import annotations

import time
from typing import Any

from rosclaw.body.action_mapping.schema import BodyActionSpace, MappedAction
from rosclaw.core.event_bus import EventBus


def run_sandbox_preflight(
    mapped_action: MappedAction,
    body_action_space: BodyActionSpace,
    *,
    robot_id: str = "rosclaw_default",
    world_id: str = "empty",
    safety_level: str = "MODERATE",
    rh56_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a mapped action candidate through the sandbox firewall.

    Chunked actions are expanded into a trajectory so each chunk waypoint can be
    checked.  If the sandbox is unavailable, the candidate is blocked to honor
    the fail-closed contract.

    When ``rh56_context`` is provided (keys: ``profile``, optional
    ``calibration``, ``current_positions``, ``max_step_delta_raw``), the
    RH56-aware range/calibration checker runs instead of the generic humanoid
    MuJoCo firewall, whose hardcoded radian limits are meaningless for a
    raw-device-unit hand.
    """
    if rh56_context is not None:
        from rosclaw.body.rh56.sandbox import run_rh56_sandbox_preflight

        return run_rh56_sandbox_preflight(
            mapped_action,
            body_action_space,
            profile=rh56_context["profile"],
            calibration=rh56_context.get("calibration"),
            current_positions=rh56_context.get("current_positions"),
            max_step_delta_raw=rh56_context.get("max_step_delta_raw"),
        )

    from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter

    t0 = time.perf_counter()
    trajectory = _mapped_action_to_trajectory(mapped_action, body_action_space)

    event_bus = EventBus()
    adapter = SandboxRuntimeAdapter(
        config={"robot_id": robot_id, "world_id": world_id, "engine": "mujoco"},
        event_bus=event_bus,
    )
    adapter.initialize()
    adapter.start()
    try:
        decision = adapter.validate_trajectory(trajectory, safety_level=safety_level)
    except Exception as exc:  # noqa: BLE001
        decision = {"is_safe": False, "reason": f"sandbox_error: {exc}"}
    finally:
        adapter.stop()

    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "decision": "ALLOW" if decision.get("is_safe") else "BLOCK",
        "is_safe": bool(decision.get("is_safe")),
        "risk_score": decision.get("risk_score", 1.0),
        "reason": decision.get("reason", ""),
        "violations": decision.get("violations", []),
        "replay_id": decision.get("replay_id"),
        "latency_ms": round(latency_ms, 3),
        "trajectory": trajectory,
    }


def _mapped_action_to_trajectory(
    mapped_action: MappedAction,
    body_action_space: BodyActionSpace,
) -> list[list[float]]:
    """Convert a mapped action into a trajectory for the sandbox."""
    dim = len(body_action_space.joint_names)
    chunks = mapped_action.chunk_size or 1
    values = mapped_action.body_action_values
    trajectory: list[list[float]] = []
    for i in range(chunks):
        start = i * dim
        end = start + dim
        trajectory.append([float(v) for v in values[start:end]])
    return trajectory
