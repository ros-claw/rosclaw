"""Reader-facing explanations of canonical execution receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast


def explain_receipt(receipt: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    """Convert a receipt into the five questions a user or Agent needs answered."""

    policy = _mapping(receipt.get("policy_decision"))
    simulation = _mapping(receipt.get("simulation_result"))
    verification = _mapping(receipt.get("verification_result"))
    lease = _mapping(receipt.get("resource_lease"))
    errors = receipt.get("errors")
    error_list = errors if isinstance(errors, list) else []
    collision = simulation.get("collision")
    physics_executed = simulation.get("has_physics") is True
    task_verified = receipt.get("evidence_level") == "TASK_VERIFIED" and bool(
        verification.get("success")
    )

    contributions: list[str] = []
    body_hash = str(receipt.get("body_snapshot_hash", ""))
    if body_hash:
        contributions.append("Bound the action to an immutable body snapshot.")
    if policy:
        if policy.get("allowed") is True:
            contributions.append("Applied static policy and allowed the declared target.")
        elif policy.get("allowed") is False:
            contributions.append("Applied static policy and blocked the action before dispatch.")
    if lease:
        contributions.append("Acquired an exclusive resource lease for the body.")
    if physics_executed:
        contributions.append("Executed and observed the action in the MuJoCo physics engine.")
    if task_verified:
        contributions.append("Evaluated the task predicate and issued TASK_VERIFIED evidence.")

    return {
        "schema_version": "rosclaw.run_explanation.v1",
        "run_id": receipt.get("action_id"),
        "trace_id": receipt.get("trace_id"),
        "receipt_path": str(receipt_path),
        "requested": {
            "robot": receipt.get("body_id"),
            "body_snapshot_hash": body_hash,
            "capability": receipt.get("capability_id"),
            "mode": receipt.get("execution_mode") or receipt.get("mode"),
        },
        "policy": {
            "allowed": policy.get("allowed"),
            "reason": policy.get("reason"),
            "violations": policy.get("violations", []),
        },
        "execution": {
            "final_state": receipt.get("final_state"),
            "physics_executed": physics_executed,
            "engine": simulation.get("engine"),
            "engine_version": simulation.get("engine_version"),
            "world": simulation.get("world_id"),
            "steps": simulation.get("steps", 0),
            "resource_lease_id": lease.get("lease_id"),
        },
        "observation": {
            "collision_free": collision is None if physics_executed else None,
            "collision": collision,
            "final_error_m": verification.get("final_error_m"),
            "target": verification.get("target"),
            "final_end_effector": verification.get("final_end_effector"),
        },
        "verification": {
            "task_verified": task_verified,
            "success": verification.get("success"),
            "condition": verification.get("condition"),
            "tolerance_m": verification.get("tolerance_m"),
            "evidence_level": receipt.get("evidence_level"),
            "trust_level": receipt.get("trust_level"),
        },
        "rosclaw_contribution": contributions,
        "artifacts": receipt.get("artifacts", []),
        "errors": error_list,
        "transitions": receipt.get("transitions", []),
    }


def format_explanation(explanation: dict[str, Any]) -> str:
    """Render a concise beginner view without weakening evidence semantics."""

    requested = _mapping(explanation.get("requested"))
    policy = _mapping(explanation.get("policy"))
    execution = _mapping(explanation.get("execution"))
    observation = _mapping(explanation.get("observation"))
    verification = _mapping(explanation.get("verification"))
    allowed = policy.get("allowed")
    policy_result = "ALLOW" if allowed is True else "BLOCK" if allowed is False else "N/A"
    collision_free = observation.get("collision_free")
    collision_result = (
        "PASS" if collision_free is True else "FAIL" if collision_free is False else "N/A"
    )
    final_error = observation.get("final_error_m")
    final_error_text = (
        f"{float(final_error):.6f} m" if isinstance(final_error, int | float) else "N/A"
    )

    lines = [
        "ROSClaw Run Explanation",
        f"Run:              {explanation.get('run_id')}",
        f"Execution mode:   {requested.get('mode')}",
        f"Robot:            {requested.get('robot')}",
        f"Capability:       {requested.get('capability')}",
        f"Body snapshot:    {requested.get('body_snapshot_hash') or 'not recorded'}",
        f"Policy:           {policy_result} ({policy.get('reason') or 'no reason recorded'})",
        (
            f"Physics:          {'yes' if execution.get('physics_executed') else 'no'}"
            f" ({execution.get('engine') or 'N/A'} {execution.get('engine_version') or ''})"
        ).rstrip(),
        f"Steps:            {execution.get('steps', 0)}",
        f"Collision check:  {collision_result}",
        f"Final distance:   {final_error_text}",
        f"Final state:      {execution.get('final_state')}",
        (
            "Task verified:    "
            f"{'YES' if verification.get('task_verified') is True else 'NO'} "
            f"({verification.get('evidence_level')})"
        ),
        f"Trace:            {explanation.get('trace_id')}",
        f"Receipt:          {explanation.get('receipt_path')}",
    ]
    contributions = explanation.get("rosclaw_contribution")
    if isinstance(contributions, list) and contributions:
        lines.append("ROSClaw contribution:")
        lines.extend(f"  - {item}" for item in contributions)
    errors = explanation.get("errors")
    if isinstance(errors, list) and errors:
        lines.append("Errors:")
        for error in errors:
            if isinstance(error, dict):
                lines.append(f"  - {error.get('code')}: {error.get('message')}")
            else:
                lines.append(f"  - {error}")
    return "\n".join(lines)


def _mapping(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


__all__ = ["explain_receipt", "format_explanation"]
