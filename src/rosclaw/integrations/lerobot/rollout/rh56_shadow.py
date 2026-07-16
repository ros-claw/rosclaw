"""RH56 shadow gate runner (P5-B).

Wires a transport profile + RH56 transport (mock or serial) + the RH56
observation source + a mock/resolved body into the standard rollout loop, then
evaluates the Shadow Gate acceptance criteria (plan §6.4) and renders the
shadow report (plan §6.5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rosclaw.body.rh56.mock_body import build_mock_rh56_body
from rosclaw.body.rh56.transport import MockModbusTransport, RH56Transport
from rosclaw.body.rh56.transport_profile import (
    TransportProfile,
    load_transport_profile,
    validate_transport_binding,
)
from rosclaw.integrations.lerobot.rollout.loop import RolloutConfig, _run_loop
from rosclaw.integrations.lerobot.rollout.rh56_observation_source import (
    RH56ObservationSource,
)
from rosclaw.integrations.lerobot.rollout.state import RolloutMode, RolloutResult

# Shadow Gate acceptance thresholds (plan §6.4).
SHADOW_GATE_THRESHOLDS = {
    "min_steps": 1000,
    "min_effective_hz": 4.8,
    "max_deadline_miss_rate": 0.01,
}


def run_rh56_shadow(
    config: RolloutConfig,
    *,
    transport_profile_path: str | Path,
    transport: RH56Transport | None = None,
    task: str = "hold_current",
    body: Any | None = None,
    calibration: Any | None = None,
) -> tuple[RolloutResult, dict[str, Any]]:
    """Run an RH56 shadow rollout and evaluate the gate.

    Returns ``(result, gate_report)``.  Hardware execution is impossible in
    this path: the transport is only ever *read*.
    """
    profile = load_transport_profile(transport_profile_path)

    # Binding gate: policy/body action space must match the profile exactly.
    if body is None:
        body = build_mock_rh56_body(profile)
    validate_transport_binding(
        profile,
        action_dim=len(profile.action_order),
        action_names=list(profile.action_order),
    )

    if transport is None:
        transport = MockModbusTransport(profile)
    if not transport.is_connected():
        transport.connect()

    source = RH56ObservationSource(transport, profile, task=task)
    config = RolloutConfig(**{**config.__dict__, "mode": RolloutMode.SHADOW})
    config.rh56_context = {
        "profile": profile,
        "calibration": calibration,
    }
    config.body_override = body

    try:
        result = _run_loop(config, source)
    finally:
        source.close()

    gate_report = evaluate_shadow_gate(result, profile, source.serial_health())
    return result, gate_report


def evaluate_shadow_gate(
    result: RolloutResult,
    profile: TransportProfile,
    serial_health: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the Shadow Gate acceptance criteria (plan §6.4)."""
    metrics = result.metrics or {}
    steps = result.steps_completed
    hardware_actions = result.hardware_actions_executed

    mapping_blocks = metrics.get("mapping_blocks", 0)
    nan_inf_blocks = metrics.get("nan_inf_blocks", 0)
    obs_failures = metrics.get("observation_validation_failures", 0)

    deadline_misses = metrics.get("deadline_misses", 0)
    miss_rate = (deadline_misses / steps) if steps else 1.0
    effective_hz = metrics.get("effective_control_hz", 0.0)

    thresholds = SHADOW_GATE_THRESHOLDS
    checks = {
        "steps_completed": {
            "value": steps,
            "required": f">= {thresholds['min_steps']}",
            "pass": steps >= thresholds["min_steps"],
        },
        "hardware_actions_executed": {
            "value": hardware_actions,
            "required": "== 0",
            "pass": hardware_actions == 0,
        },
        "unknown_action_semantics": {
            "value": _count_unknown_semantics(result),
            "required": "== 0",
            "pass": _count_unknown_semantics(result) == 0,
        },
        "incompatible_mapping": {
            "value": mapping_blocks,
            "required": "== 0",
            "pass": mapping_blocks == 0,
        },
        "nan_inf": {
            "value": nan_inf_blocks,
            "required": "== 0",
            "pass": nan_inf_blocks == 0,
        },
        "required_observation_stale": {
            "value": obs_failures,
            "required": "== 0",
            "pass": obs_failures == 0,
        },
        "serial_disconnect_count": {
            "value": serial_health.get("disconnect_count", 0),
            "required": "== 0",
            "pass": serial_health.get("disconnect_count", 0) == 0,
        },
        "worker_restart_count": {
            "value": metrics.get("worker_restart_count", 0),
            "required": "== 0",
            "pass": metrics.get("worker_restart_count", 0) == 0,
        },
        "effective_hz": {
            "value": round(effective_hz, 3),
            "required": f">= {thresholds['min_effective_hz']}",
            "pass": effective_hz >= thresholds["min_effective_hz"],
        },
        "deadline_miss_rate": {
            "value": round(miss_rate, 5),
            "required": f"< {thresholds['max_deadline_miss_rate']}",
            "pass": miss_rate < thresholds["max_deadline_miss_rate"],
        },
    }
    passed = all(c["pass"] for c in checks.values())
    return {
        "gate": "rh56_shadow_gate",
        "passed": passed,
        "transport_profile": profile.id,
        "checks": checks,
        "serial_health": serial_health,
        "stop_reason": result.stop_reason.value,
        "practice_id": result.practice_id,
        "trace_path": result.trace_path,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
    }


def _count_unknown_semantics(result: RolloutResult) -> int:
    count = 0
    for proposal in result.proposals:
        safety = proposal.get("safety", {}) if isinstance(proposal, dict) else {}
        if safety.get("error_code") in {"unknown_action_semantics", "missing_action_names"}:
            count += 1
    return count


def render_shadow_report(
    gate_report: dict[str, Any],
    result: RolloutResult,
    *,
    calibration_hash: str = "",
    policy_contract: dict[str, Any] | None = None,
) -> str:
    """Render the P5 shadow report (plan §6.5) as Markdown."""
    metrics = result.metrics or {}
    lines = [
        "# P5 RH56 Real Shadow Report",
        "",
        f"- Transport profile: `{gate_report['transport_profile']}`",
        f"- Calibration hash: `{calibration_hash or 'n/a (mock)'}`",
        f"- Policy contract: `{(policy_contract or {}).get('policy_id', 'rh56_reference_policy_v1')}`",
        f"- Practice id: `{gate_report.get('practice_id')}`",
        f"- Stop reason: `{gate_report['stop_reason']}`",
        f"- Gate passed: **{gate_report['passed']}**",
        "",
        "## 1000-step metrics",
        "",
        "| Check | Value | Required | Pass |",
        "|---|---|---|---|",
    ]
    for name, check in gate_report["checks"].items():
        lines.append(
            f"| {name} | {check['value']} | {check['required']} | "
            f"{'✅' if check['pass'] else '❌'} |"
        )
    lines += [
        "",
        "## Latency distribution",
        "",
        "```json",
        json.dumps(
            {
                "step_latency_ms": metrics.get("step_latency_ms", {}),
                "inference_latency_ms": metrics.get("inference_latency_ms", {}),
                "mapping_latency_ms": metrics.get("mapping_latency_ms", {}),
                "sandbox_latency_ms": metrics.get("sandbox_latency_ms", {}),
                "deadline_miss_ms": metrics.get("deadline_miss_ms", {}),
                "effective_control_hz": metrics.get("effective_control_hz", 0.0),
            },
            indent=2,
        ),
        "```",
        "",
        "## Serial health",
        "",
        "```json",
        json.dumps(gate_report["serial_health"], indent=2),
        "```",
        "",
    ]
    if gate_report.get("errors"):
        lines += ["## Errors", "", "```", *gate_report["errors"], "```", ""]
    return "\n".join(lines)
