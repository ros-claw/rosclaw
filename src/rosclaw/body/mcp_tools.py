"""P0 MCP body tools.

These helpers live in ``src/rosclaw/body/`` so the MCP adapter stays a thin
wrapper.  Every tool reads the current Effective Body through ``BodyResolver``
and respects the P0 safety contract: enabled ≠ executable, forbidden/blocked
actions are refused, and high/critical capabilities require sandbox validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rosclaw.body.compatibility import SkillCompatibilityChecker
from rosclaw.body.fleet import discover_skill_manifests
from rosclaw.body.query import BodyQueryEngine
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import EffectiveBody


class BodyMcpTools:
    """Read-only body tools consumed by the MCP RuntimeClient."""

    def __init__(self, workspace: Path | str | None = None):
        self.resolver = BodyResolver(workspace=Path(workspace) if workspace else None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_body(self) -> EffectiveBody:
        """Return the current effective body, recompiling if stale."""
        return self.resolver.get_effective_body(recompile_if_stale=True)

    @staticmethod
    def _forbidden_ids(body: EffectiveBody) -> set[str]:
        """Collect forbidden capability IDs from the effective body."""
        ids: set[str] = set()
        for item in body.forbidden_capabilities or []:
            if isinstance(item, dict):
                ids.add(item.get("id") or item.get("capability", "unknown"))
            else:
                ids.add(str(item))
        return ids

    @staticmethod
    def _is_high_risk(capability_id: str, body: EffectiveBody) -> bool:
        """Conservative heuristic: consider locomotion/manipulation high risk."""
        high_risk_keywords = {"walk", "run", "locomotion", "grasp", "pick", "place", "manipulate", "arm"}
        return any(kw in capability_id.lower() for kw in high_risk_keywords)

    # ------------------------------------------------------------------
    # P0 tools
    # ------------------------------------------------------------------

    def get_body_profile(self) -> dict[str, Any]:
        """Return a static profile summary of the current body."""
        body = self._require_body()
        body_yaml = self.resolver.get_current_body_yaml()
        identity = body_yaml.get_identity()

        body_parts = []
        if body_yaml.body_structure:
            body_parts = [
                {
                    "id": part.get("id", "unknown"),
                    "name": part.get("name", part.get("id", "unknown")),
                    "type": part.get("type", ""),
                    "links": part.get("links", []),
                    "joints": part.get("joints", []),
                }
                for part in body_yaml.body_structure.get("body_parts", [])
            ]

        sensors = [
            {"id": name, **sensor}
            for name, sensor in sorted(body.sensors.items())
        ]
        actuators = [
            {"id": name, **actuator}
            for name, actuator in sorted(body.actuators.items())
        ]

        return {
            "body_instance_id": body.body_instance_id,
            "robot_model": identity.get("robot_model") or body_yaml.body_instance.get("robot_model", "unknown"),
            "robot_vendor": identity.get("robot_vendor", "unknown"),
            "eurdf_uri": body.eurdf_uri,
            "effective_body_hash": body.effective_body_hash,
            "body_parts": body_parts,
            "sensors": sensors,
            "actuators": actuators,
        }

    def get_body_state(self, *, include_runtime: bool = True) -> dict[str, Any]:
        """Return the current body safety state."""
        body = self._require_body()
        body_yaml = self.resolver.get_current_body_yaml()
        calibration = self.resolver.get_calibration()

        forbidden = [
            item.get("id", item.get("capability", "unknown"))
            for item in (body.forbidden_capabilities or body_yaml.forbidden_capabilities or [])
        ]
        open_faults = [
            fault.get("id", "unknown")
            for fault in body.known_faults
            if fault.get("status") == "open"
        ]

        state: dict[str, Any] = {
            "body_instance_id": body.body_instance_id,
            "effective_body_hash": body.effective_body_hash,
            "safety_status": body_yaml.get_safety_status(),
            "calibration_status": calibration.overall_status(),
            "enabled_capabilities": body.capabilities.get("enabled", []),
            "degraded_capabilities": body.capabilities.get("degraded", []),
            "disabled_capabilities": body.capabilities.get("blocked", []),
            "forbidden_capabilities": forbidden,
            "open_faults": open_faults,
        }
        if include_runtime:
            state["runtime_overlay"] = body_yaml.runtime_state or {}
        return state

    def list_body_capabilities(self, *, status: str = "all") -> dict[str, Any]:
        """List capabilities grouped by status."""
        body = self._require_body()
        caps = body.capabilities
        groups = {
            "enabled": list(caps.get("enabled", [])),
            "degraded": list(caps.get("degraded", [])),
            "disabled": list(caps.get("blocked", [])),
            "forbidden": [
                item.get("id", item.get("capability", "unknown"))
                for item in (body.forbidden_capabilities or [])
            ],
        }

        status = status.lower()
        if status == "all":
            return groups
        if status in groups:
            return {status: groups[status]}
        return {"error": f"unknown status filter: {status}"}

    def query_body(self, question: str) -> dict[str, Any]:
        """Answer a natural-language question about the body."""
        body = self._require_body()
        body_yaml = self.resolver.get_current_body_yaml()
        calibration = self.resolver.get_calibration()
        maintenance = self.resolver.get_maintenance_events()

        engine = BodyQueryEngine(body, body_yaml, calibration, maintenance)
        result = engine.answer(question)

        decision = "unknown"
        q = question.lower()
        forbidden = self._forbidden_ids(body)

        if "bypass" in q or "skip" in q or any(cap in forbidden for cap in result.evidence.get("capabilities", {}).get("enabled", [])) or "no" in result.answer.lower() and "sandbox" in q:
            decision = "blocked"
        elif "yes" in result.answer.lower() and not forbidden:
            decision = "allowed_to_propose"
        elif "degraded" in result.answer.lower():
            decision = "requires_validation"
        else:
            decision = "unknown"

        return {
            "answer": result.answer,
            "decision": decision,
            "evidence": [
                {"key": key, "value": value}
                for key, value in result.evidence.items()
            ],
            "next_steps": result.actionable_policy
            or [
                "Verify capability through 'rosclaw body state --json'.",
                "Run sandbox validation before physical execution.",
            ],
        }

    def validate_body_action(
        self,
        action: str,
        capability_id: str,
        *,
        risk: str = "medium",
    ) -> dict[str, Any]:
        """Check whether a proposed action is allowed on the current body."""
        body = self._require_body()
        forbidden = self._forbidden_ids(body)
        caps = body.capabilities

        reasons: list[str] = []
        next_steps = ["get_robot_state", "sandbox_validate", "firewall_validate"]

        if capability_id in forbidden:
            return {
                "body_check": "blocked",
                "allowed_to_propose": False,
                "allowed_to_execute_real_robot": False,
                "reasons": [f"Capability '{capability_id}' is forbidden for this body."],
                "next_steps": [],
            }

        # Run a skill-compatibility check if a matching manifest exists.
        skill_check = "unknown"
        manifests = discover_skill_manifests(self.resolver.workspace)
        matching = [
            m for m in manifests
            if m.skill_id == capability_id or capability_id in m.requirement_ids()
        ]
        if matching:
            report = SkillCompatibilityChecker().check_all(matching, body)
            if report.skills:
                # Use the first matching result as representative.
                first = next(iter(report.skills.values()))
                skill_check = first.status
                reasons.append(first.reason)

        if capability_id in caps.get("blocked", []):
            skill_check = "blocked"
            reasons.append(f"Capability '{capability_id}' is disabled on this body.")
        elif capability_id in caps.get("degraded", []):
            skill_check = "degraded"
            reasons.append(f"Capability '{capability_id}' is degraded; use conservative constraints.")
        elif capability_id in caps.get("enabled", []):
            if skill_check == "unknown":
                skill_check = "compatible"
        else:
            reasons.append(f"Capability '{capability_id}' is not declared on this body.")

        risk = risk.lower()
        sandbox_required = risk in ("high", "critical") or self._is_high_risk(capability_id, body)
        human_approval_required = risk == "critical"

        allowed_to_propose = skill_check in ("compatible", "degraded")
        allowed_real = (
            allowed_to_propose
            and skill_check != "degraded"
            and capability_id not in forbidden
            and not sandbox_required
            and not human_approval_required
        )

        if sandbox_required:
            reasons.append("Sandbox validation is required for this action.")
            next_steps.append("sandbox_validate")
        if human_approval_required:
            reasons.append("Critical-risk action requires explicit human approval.")
            next_steps.append("human_approval")
        if forbidden:
            reasons.append(f"Forbidden capabilities on this body: {sorted(forbidden)}.")

        return {
            "body_check": skill_check,
            "allowed_to_propose": allowed_to_propose,
            "allowed_to_execute_real_robot": allowed_real,
            "reasons": reasons,
            "next_steps": list(dict.fromkeys(next_steps)),
        }

    def get_calibration_status(self, *, component: str | None = None) -> dict[str, Any]:
        """Return calibration status for the whole body or one component."""
        calibration = self.resolver.get_calibration()

        if component is None:
            return {
                "component": "*",
                "status": calibration.overall_status(),
                "confidence": 0.0,
                "blocks": [],
            }

        # Look for component-specific calibration data in spec-aligned fields.
        component_data: dict[str, Any] = {}
        for container in (calibration.sensors, calibration.joints, calibration.frames):
            if isinstance(container, dict) and component in container:
                component_data = container[component]
                break

        status = component_data.get("status") or calibration.overall_status()
        confidence = float(component_data.get("confidence", 0.0))
        blocks = component_data.get("blocks", [])
        if not blocks and status in ("missing", "expired", "unknown"):
            blocks.append(f"{component} calibration is {status}")

        return {
            "component": component,
            "status": status,
            "confidence": confidence,
            "blocks": blocks,
        }

    def get_calibration_status_from_body(self, component: str) -> dict[str, Any]:
        """Backward-compatible alias used by older MCP adapters."""
        return self.get_calibration_status(component=component)
