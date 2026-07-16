"""Arming controller: DISARMED → PREFLIGHT → SHADOW_VALIDATED → ARMED.

Arming requires, in order:

1. preflight checks pass (transport binding, calibration validated, mapping
   compatible, sandbox reachable)
2. a shadow run has been validated for this exact hash set
3. an execution permit has been issued (operator-armed, estop confirmed)

Disarming is always allowed.  Re-arming after a fault requires going through
the full sequence again — there is no automatic recovery.
"""

from __future__ import annotations

from typing import Any

from rosclaw.integrations.lerobot.execution.permit import PermitManager
from rosclaw.integrations.lerobot.execution.state import (
    ExecutionState,
    ExecutionStateMachine,
)


class ArmingError(RuntimeError):
    """Arming sequence failure; message starts with a code."""


class ArmingController:
    """Drive the execution state machine through the arming sequence."""

    def __init__(self, permit_manager: PermitManager | None = None):
        self.machine = ExecutionStateMachine()
        self.permit_manager = permit_manager or PermitManager()
        self._shadow_validated_hashes: set[tuple[str, ...]] = set()
        self._armed_permit_id: str | None = None

    # ------------------------------------------------------------------

    def begin_preflight(self) -> ExecutionState:
        return self.machine.transition(ExecutionState.PREFLIGHT, "preflight begin")

    def preflight_failed(self, reason: str) -> ExecutionState:
        return self.machine.fault(ExecutionState.BLOCKED, reason)

    def mark_shadow_validated(
        self,
        *,
        policy_contract_hash: str,
        body_hash: str,
        calibration_hash: str,
        mapping_hash: str,
        transport_profile_hash: str,
    ) -> ExecutionState:
        """Record that a shadow run validated this exact hash set."""
        self._shadow_validated_hashes.add(
            (
                policy_contract_hash,
                body_hash,
                calibration_hash,
                mapping_hash,
                transport_profile_hash,
            )
        )
        if self.machine.state == ExecutionState.PREFLIGHT:
            return self.machine.transition(
                ExecutionState.SHADOW_VALIDATED, "shadow gate passed"
            )
        return self.machine.state

    def shadow_validated_for(
        self,
        *,
        policy_contract_hash: str,
        body_hash: str,
        calibration_hash: str,
        mapping_hash: str,
        transport_profile_hash: str,
    ) -> bool:
        return (
            policy_contract_hash,
            body_hash,
            calibration_hash,
            mapping_hash,
            transport_profile_hash,
        ) in self._shadow_validated_hashes

    def arm(self, permit_id: str) -> ExecutionState:
        """Arm for execution with an issued permit."""
        if self.machine.state == ExecutionState.DISARMED:
            raise ArmingError("arming_sequence_violation: must pass preflight first")
        permit = self.permit_manager.get(permit_id)
        if permit is None:
            raise ArmingError(f"arming_permit_invalid: {permit_id}")
        if not self.shadow_validated_for(
            policy_contract_hash=permit.policy_contract_hash,
            body_hash=permit.body_hash,
            calibration_hash=permit.calibration_hash,
            mapping_hash=permit.mapping_hash,
            transport_profile_hash=permit.transport_profile_hash,
        ):
            raise ArmingError(
                "arming_shadow_not_validated: shadow gate required for this hash set"
            )
        if self.machine.state != ExecutionState.SHADOW_VALIDATED:
            raise ArmingError(
                f"arming_sequence_violation: state is {self.machine.state.value}, "
                "expected SHADOW_VALIDATED"
            )
        self._armed_permit_id = permit_id
        return self.machine.transition(ExecutionState.ARMED, f"armed with {permit_id}")

    @property
    def armed_permit_id(self) -> str | None:
        return self._armed_permit_id

    def disarm(self, reason: str = "operator") -> ExecutionState:
        self._armed_permit_id = None
        return self.machine.disarm(reason)

    def fault(self, state: ExecutionState, reason: str) -> ExecutionState:
        """Enter a fault state and drop the armed permit reference."""
        self._armed_permit_id = None
        return self.machine.fault(state, reason)

    def status(self) -> dict[str, Any]:
        return {
            "state": self.machine.state.value,
            "armed_permit_id": self._armed_permit_id,
            "shadow_validated_sets": len(self._shadow_validated_hashes),
            "history": [(s.value, r) for s, r in self.machine.history[-10:]],
        }
