"""Body query engine — answer questions from body state without LLM dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rosclaw.body.schema import BodyYaml, CalibrationYaml, EffectiveBody, MaintenanceEvent


@dataclass
class BodyQueryResult:
    """Result of a body query."""

    question: str
    answer: str
    evidence: dict[str, Any] = field(default_factory=dict)
    actionable_policy: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence": self.evidence,
            "actionable_policy": self.actionable_policy,
        }


class BodyQueryEngine:
    """Keyword-based question answering against body state."""

    def __init__(
        self,
        effective: EffectiveBody,
        body_yaml: BodyYaml,
        calibration: CalibrationYaml,
        maintenance: list[MaintenanceEvent],
    ):
        self.effective = effective
        self.body_yaml = body_yaml
        self.calibration = calibration
        self.maintenance = maintenance

    def answer(self, question: str) -> BodyQueryResult:
        """Return an answer plus evidence and policy notes."""
        q = question.lower()

        identity = self.body_yaml.get_identity()

        # Identity / what robot
        if any(kw in q for kw in ("what robot", "which robot", "what body", "who is this", "what is this")):
            return BodyQueryResult(
                question=question,
                answer=(
                    f"This is {identity.get('nickname') or identity.get('robot_instance_id')} "
                    f"({identity.get('robot_model') or 'unknown model'})."
                ),
                evidence={
                    "robot_instance_id": identity.get("robot_instance_id"),
                    "robot_model": identity.get("robot_model"),
                    "robot_vendor": identity.get("robot_vendor"),
                },
                actionable_policy=[],
            )

        # Sandbox bypass
        if any(kw in q for kw in ("bypass sandbox", "skip sandbox", "run without sandbox", "disable sandbox")):
            return BodyQueryResult(
                question=question,
                answer="No. Sandbox validation is mandatory for physical execution on this body.",
                evidence={
                    "policy": "physical_execution_requires_sandbox",
                    "value": True,
                },
                actionable_policy=[
                    "Refused: sandbox bypass is not allowed.",
                    "To execute physically, provide validation evidence via 'capability enable <id> --after-validation <run_id>'.",
                ],
            )

        # Can it walk / locomotion
        if any(kw in q for kw in ("walk", "run", "locomotion", "move around")):
            caps = self.effective.capabilities
            if "walk" in caps.get("enabled", []):
                return BodyQueryResult(
                    question=question,
                    answer="Yes, walking/locomotion is enabled.",
                    evidence={"capabilities": caps},
                    actionable_policy=[],
                )
            if "walk" in caps.get("degraded", []):
                return BodyQueryResult(
                    question=question,
                    answer="Walking is degraded; it may only be allowed under reduced limits or in simulation.",
                    evidence={"capabilities": caps, "open_faults": self._open_fault_ids()},
                    actionable_policy=["Resolve open faults and re-validate before full operation."],
                )
            return BodyQueryResult(
                question=question,
                answer="No, walking/locomotion is not enabled for this body.",
                evidence={"capabilities": caps},
                actionable_policy=["Enable the capability explicitly if the hardware supports it."],
            )

        # Can it see / vision
        if any(kw in q for kw in ("see", "vision", "camera", "navigate visually", "visual")):
            caps = self.effective.capabilities
            if "visual_navigation" in caps.get("enabled", []):
                return BodyQueryResult(
                    question=question,
                    answer="Yes, visual sensing/navigation is enabled.",
                    evidence={"capabilities": caps, "sensors": list(self.effective.sensors.keys())},
                    actionable_policy=[],
                )
            if "visual_navigation" in caps.get("degraded", []):
                return BodyQueryResult(
                    question=question,
                    answer="Visual capabilities are degraded, likely due to calibration or sensor state.",
                    evidence={
                        "capabilities": caps,
                        "calibration_status": self.calibration.overall_status(),
                    },
                    actionable_policy=["Validate calibration and check camera availability."],
                )
            return BodyQueryResult(
                question=question,
                answer="No, visual navigation/sensing is not enabled.",
                evidence={"capabilities": caps},
                actionable_policy=[],
            )

        # Calibration
        if any(kw in q for kw in ("calibration", "calibrated")):
            status = self.calibration.overall_status()
            if status in ("valid", "validated"):
                return BodyQueryResult(
                    question=question,
                    answer=f"Calibration status is '{status}'.",
                    evidence={"calibration_status": status},
                    actionable_policy=[],
                )
            return BodyQueryResult(
                question=question,
                answer=f"Calibration status is '{status}'; precision capabilities may be degraded.",
                evidence={"calibration_status": status},
                actionable_policy=["Run 'rosclaw body calibration update --file <path>' to update calibration."],
            )

        # Faults
        if any(kw in q for kw in ("fault", "problem", "issue", "broken", "wrong")):
            open_faults = self._open_fault_ids()
            if open_faults:
                return BodyQueryResult(
                    question=question,
                    answer=f"There are {len(open_faults)} open fault(s): {', '.join(open_faults)}.",
                    evidence={"open_faults": open_faults},
                    actionable_policy=["Resolve faults via 'rosclaw body fault resolve <fault_id>'."],
                )
            return BodyQueryResult(
                question=question,
                answer="No open faults are recorded.",
                evidence={"open_faults": []},
                actionable_policy=[],
            )

        # Capabilities list
        if any(kw in q for kw in ("capability", "capabilities", "can it do", "what can")):
            caps = self.effective.capabilities
            enabled = caps.get("enabled", [])
            return BodyQueryResult(
                question=question,
                answer=f"Enabled capabilities: {', '.join(enabled) if enabled else 'none'}.",
                evidence={"capabilities": caps, "forbidden": self.effective.forbidden_capabilities},
                actionable_policy=["Use 'rosclaw body state --json' for the full capability matrix."],
            )

        # Safety
        if any(kw in q for kw in ("safe", "safety", "limit", "emergency stop", "e-stop")):
            safety = self.effective.safety
            return BodyQueryResult(
                question=question,
                answer=f"Overall safety status is '{self.body_yaml.get_safety_status()}'.",
                evidence={
                    "safety_status": self.body_yaml.get_safety_status(),
                    "global_limits": safety.get("safety_limits") or safety.get("global_limits"),
                },
                actionable_policy=["Review EMBODIMENT.md section 7 for full safety limits."],
            )

        # Default fallback
        return BodyQueryResult(
            question=question,
            answer="I don't have a specific answer for that question. Try asking about identity, capabilities, calibration, faults, safety, or sandbox policy.",
            evidence={"capabilities": self.effective.capabilities},
            actionable_policy=["Run 'rosclaw body state --json' for structured body state."],
        )

    def _open_fault_ids(self) -> list[str]:
        ids: list[str] = []
        for fault in self.effective.known_faults:
            if fault.get("status") == "open":
                ids.append(fault.get("id") or "unknown")
        return ids
