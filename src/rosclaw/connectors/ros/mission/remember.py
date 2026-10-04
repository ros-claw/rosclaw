"""Existing Practice/Memory commit after canonical mission verification."""

import hashlib
import json
from pathlib import Path

from rosclaw.connectors.ros.verification.mission import verify_mission
from rosclaw.kernel import (
    ActionExecutionResult,
    ActionState,
    EvidenceDomain,
    EvidenceLevel,
    ExecutionMode,
)
from rosclaw.runtime.event import RuntimeEvent


class VerifiedMissionMemoryExecutor:
    def __init__(self, runtime, *, evidence_directory, recorder_bus=None):
        self.runtime = runtime
        self.directory = Path(evidence_directory).resolve()
        self.recorder_bus = recorder_bus

    def get_execution_receipt(self, action_id):
        receipt = self.runtime.action_gateway.get_receipt(action_id)
        return {"receipt": receipt.to_dict() if receipt else {}}

    def __call__(self, action):
        try:
            if action.execution_mode is not ExecutionMode.SIMULATION:
                raise ValueError("this configured memory adapter accepts simulation evidence only")
            response = self.get_execution_receipt(action.arguments["coverage_action_id"])
            receipt = response["receipt"]
            if (
                receipt.get("body_id") != action.body_id
                or receipt.get("body_snapshot_hash") != action.body_snapshot_hash
            ):
                raise ValueError("receipt Body mismatch")
            artifact = receipt["verification_result"]["evidence_artifact"]
            path = Path(artifact["path"]).resolve()
            if (
                path.parent != self.directory
                or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]
            ):
                raise ValueError("daemon evidence artifact path/hash mismatch")
            evidence = json.loads(path.read_text())
            verification = verify_mission(evidence, daemon=self, event_bus=self.runtime.event_bus)
            if not verification["success"]:
                raise ValueError("unverified mission cannot be remembered as success")
            stored = (
                self.runtime.memory.get_experience(evidence["mission_id"])
                if self.runtime.memory
                else None
            )
            if not stored or stored.get("outcome") != "success":
                raise ValueError("verified mission did not persist into existing Memory")
            target = self.directory / (path.stem + ".verification.json")
            target.write_text(json.dumps(verification, indent=2) + "\n")
            if self.recorder_bus:
                self.recorder_bus.publish(
                    RuntimeEvent(
                        type="practice.stop",
                        source="runtime",
                        body_id=action.body_id,
                        robot=action.body_id,
                        payload={
                            "outcome": "SUCCESS",
                            "metadata": {
                                "verification_artifact": str(target),
                                "evidence_domain": "SIMULATION",
                            },
                        },
                    )
                )
            return ActionExecutionResult(
                final_state=ActionState.COMPLETED,
                evidence_level=EvidenceLevel.TASK_VERIFIED,
                evidence_domain=EvidenceDomain.SIMULATION,
                policy_decision={"allowed": True},
                dispatch_result={"accepted": True},
                artifacts=[str(target)],
                verification_result={
                    "mission_verification_artifact": str(target),
                    "success": True,
                    "memory_id": evidence["mission_id"],
                    "memory_outcome": stored["outcome"],
                },
            )
        except Exception as exc:
            return ActionExecutionResult(
                final_state=ActionState.BLOCKED,
                evidence_level=EvidenceLevel.REQUESTED,
                evidence_domain=EvidenceDomain.SIMULATION,
                policy_decision={"allowed": False},
                errors=[{"code": "MISSION_MEMORY_REJECTED", "message": str(exc)}],
            )
