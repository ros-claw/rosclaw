"""RH56 REAL single-step executor for the Runtime ActionGateway (P5-D).

This adapter is the only sanctioned REAL dispatch path for the RH56 bridge:
an :class:`~rosclaw.kernel.contracts.ActionEnvelope` arrives through
``Runtime.submit_action()`` / ``ActionGateway.submit()`` and is executed by the
:class:`SingleStepExecutor` pipeline (permit → watchdog → freshness → one
command → feedback verify).  One envelope = one single-step position command;
open-loop chunks and multi-step dispatch are impossible by construction.

Envelope ``arguments`` contract::

    permit_id: str                 # issued, non-expired REAL permit
    names: list[str]               # actuator names (profile order)
    values: list[float]            # absolute raw targets, one per name
    representation: str            # "joint_position"
    units: str                     # "raw_device_unit"
    hashes: {policy_contract_hash, body_hash, calibration_hash,
             mapping_hash, transport_profile_hash}
    speed: int                     # 0..1000, clamped by permit
    force_limit_g: float           # clamped by permit
    observation_timestamp_ns: int  # monotonic ns of the source observation
    settle_ms: float               # optional post-command settle

Evidence mapping (truthful, fail-closed):

* ``completed`` → COMPLETED + PHYSICALLY_OBSERVED (feedback verified on the
  physical device within calibrated tolerance)
* ``blocked`` / ``stale_action`` → BLOCKED + REQUESTED (no command sent)
* ``fault`` / ``aborted`` → FAILED; evidence DISPATCH_CONFIRMED when a
  command was actually sent, else REQUESTED
"""

from __future__ import annotations

from typing import Any

from rosclaw.integrations.lerobot.execution.executor import SingleStepExecutor
from rosclaw.integrations.lerobot.execution.schema import (
    ActionExecutionResult as StepResult,
)
from rosclaw.kernel.contracts import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    EvidenceLevel,
)

CAPABILITY_ID = "rh56.single_step"

_REQUIRED_ARGUMENTS = (
    "permit_id",
    "names",
    "values",
    "representation",
    "units",
    "hashes",
    "speed",
    "force_limit_g",
    "observation_timestamp_ns",
)


class RH56RealStepExecutor:
    """Gateway-callable REAL executor: one envelope → one verified step."""

    def __init__(self, step_executor: SingleStepExecutor):
        self._step = step_executor

    def __call__(self, action: ActionEnvelope) -> ActionExecutionResult:
        args = dict(action.arguments)
        missing = [k for k in _REQUIRED_ARGUMENTS if k not in args]
        if missing:
            return ActionExecutionResult(
                final_state=ActionState.BLOCKED,
                evidence_level=EvidenceLevel.REQUESTED,
                errors=[
                    {
                        "code": "ENVELOPE_ARGUMENTS_MISSING",
                        "message": f"missing arguments: {missing}",
                    }
                ],
            )

        result = self._step.execute_candidate(
            permit_id=str(args["permit_id"]),
            proposal_id=str(args.get("proposal_id") or action.action_id),
            names=[str(n) for n in args["names"]],
            values=[float(v) for v in args["values"]],
            representation=str(args["representation"]),
            units=str(args["units"]),
            hashes={str(k): str(v) for k, v in dict(args["hashes"]).items()},
            speed=int(args["speed"]),
            force_limit_g=float(args["force_limit_g"]),
            observation_timestamp_ns=int(args["observation_timestamp_ns"]),
            settle_ms=float(args.get("settle_ms", 0.0)),
        )
        return self._to_kernel_result(result)

    # ------------------------------------------------------------------

    @staticmethod
    def _to_kernel_result(result: StepResult) -> ActionExecutionResult:
        verification = result.verification.to_dict()
        observations: list[dict[str, Any]] = []
        if result.command_sent:
            observations.append(
                {
                    "position": list(result.actual),
                    "force_g": list(result.force),
                    "current_ma": list(result.current),
                    "temperature_c": list(result.temperature),
                    "status_bits": list(result.status_bits),
                    "position_error": list(result.position_error),
                }
            )

        if result.status == "completed":
            return ActionExecutionResult(
                final_state=ActionState.COMPLETED,
                evidence_level=EvidenceLevel.PHYSICALLY_OBSERVED,
                dispatch_result={"accepted": True, "command_sent": True},
                driver_ack={"acknowledged": bool(result.command_acknowledged)},
                observations=observations,
                verification_result=verification,
            )

        if result.status in ("blocked", "stale_action"):
            return ActionExecutionResult(
                final_state=ActionState.BLOCKED,
                evidence_level=EvidenceLevel.REQUESTED,
                dispatch_result={"accepted": False, "command_sent": False},
                verification_result=verification if result.command_sent else None,
                errors=[
                    {
                        "code": result.error_code or result.status,
                        "message": result.message or result.status,
                    }
                ],
            )

        # fault / aborted
        return ActionExecutionResult(
            final_state=ActionState.FAILED,
            evidence_level=(
                EvidenceLevel.DISPATCH_CONFIRMED if result.command_sent else EvidenceLevel.REQUESTED
            ),
            dispatch_result={"accepted": bool(result.command_sent)},
            driver_ack=(
                {"acknowledged": bool(result.command_acknowledged)} if result.command_sent else None
            ),
            observations=observations,
            verification_result=verification if result.command_sent else None,
            errors=[
                {
                    "code": result.error_code or "execution_fault",
                    "message": result.message or result.status,
                }
            ],
        )


__all__ = ["CAPABILITY_ID", "RH56RealStepExecutor"]
