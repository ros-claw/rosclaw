"""Safety invariant engine — derive capability modifications from safety rules."""

from __future__ import annotations

from typing import Any

from rosclaw.body.schema import BodyYaml, CalibrationYaml, MaintenanceEvent


class SafetyInvariantEngine:
    """Apply mandatory safety invariants to capability derivation.

    These invariants are *additional* safety gates on top of explicit
    user configuration. They only restrict capabilities, never expand them.
    """

    def apply(
        self,
        body: BodyYaml,
        maintenance_events: list[MaintenanceEvent] | None,
        calibration: CalibrationYaml | None,
        base_capabilities: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Return capability modifications derived from safety invariants.

        Returns a dict with keys:
          - disabled: set[str]  capabilities that must be blocked
          - degraded: set[str]  capabilities that must be degraded
          - warnings: list[str] human-readable invariant messages
        """
        maintenance_events = maintenance_events or []
        calibration = calibration or CalibrationYaml()
        caps = base_capabilities or {"enabled": [], "degraded": [], "disabled": []}
        enabled = set(caps.get("enabled", []))
        degraded = set(caps.get("degraded", []))
        disabled = set(caps.get("disabled", []))
        warnings: list[str] = []

        # Invariant 1: high/critical risk capabilities must require sandbox validation.
        high_critical_capabilities = self._high_critical_capabilities(body)
        for cap in high_critical_capabilities:
            if cap in enabled and not self._has_sandbox_requirement(body, cap):
                degraded.add(cap)
                warnings.append(
                    f"Invariant: capability '{cap}' is high/critical risk but lacks "
                    "sandbox_required; degraded until validation policy is added."
                )

        # Invariant 2: critical forbidden capabilities must block real-robot execution.
        for forbidden in body.forbidden_capabilities or []:
            sev = forbidden.get("severity", "critical")
            if sev == "critical":
                enforcement = forbidden.get("enforcement", {})
                if not enforcement.get("real_robot_block", False):
                    cap_id = forbidden.get("id") or forbidden.get("capability")
                    warnings.append(
                        f"Invariant: forbidden capability '{cap_id}' is critical but "
                        "real_robot_block is not enforced; this is a configuration error."
                    )

        # Invariant 3: open critical fault degrades/blocks safety-sensitive capabilities.
        open_critical_faults = [
            f for f in (body.known_faults.get("faults", []) if body.known_faults else [])
            if f.get("status") == "open" and f.get("severity") in ("high", "critical")
        ]
        if open_critical_faults:
            safety_caps = self._safety_sensitive_capabilities(body)
            for cap in safety_caps:
                if cap in enabled:
                    disabled.add(cap)
                    enabled.discard(cap)
            for fault in open_critical_faults:
                for cap in fault.get("disables", []) or []:
                    disabled.add(cap)
                    enabled.discard(cap)
                for cap in fault.get("degrades", []) or []:
                    degraded.add(cap)
            warnings.append(
                f"Invariant: {len(open_critical_faults)} open critical fault(s) detected; "
                "safety-sensitive capabilities disabled until resolved."
            )

        # Invariant 4: missing/invalid calibration degrades dependent precision capabilities.
        cal_status = calibration.overall_status()
        if cal_status not in ("valid", "validated"):
            for cap in self._calibration_dependent_capabilities():
                if cap in enabled:
                    degraded.add(cap)
            warnings.append(
                f"Invariant: calibration status is '{cal_status}'; precision capabilities degraded."
            )

        return {
            "disabled": sorted(disabled),
            "degraded": sorted(degraded),
            "warnings": warnings,
        }

    def _high_critical_capabilities(self, body: BodyYaml) -> set[str]:
        """Collect capabilities marked high/critical risk in body or forbidden list."""
        high_critical: set[str] = set()
        for cap in body.capabilities.get("enabled", []):
            if isinstance(cap, dict) and cap.get("risk_level") in ("high", "critical"):
                high_critical.add(str(cap.get("id") or cap.get("name") or ""))
            elif isinstance(cap, str) and any(
                kw in cap.lower() for kw in ("run", "jump", "climb", "throw", "lift", "force", "fast")
            ):
                # Heuristic: certain capability names imply high risk
                high_critical.add(cap)
        for item in body.forbidden_capabilities or []:
            cap = item.get("id") or item.get("capability")
            if cap and item.get("severity") == "critical":
                high_critical.add(cap)
        return high_critical

    def _has_sandbox_requirement(self, body: BodyYaml, cap_id: str) -> bool:
        """Check whether capability explicitly requires sandbox validation."""
        for cap in body.capabilities.get("enabled", []):
            cid = cap.get("id") if isinstance(cap, dict) else cap
            if cid == cap_id and isinstance(cap, dict):
                validation = cap.get("validation", {})
                if validation.get("sandbox_required", False):
                    return True
        # Default: assume sandbox is required if agent_policy enforces it globally.
        return bool(body.agent_policy.get("physical_execution_requires_sandbox", True))

    def _safety_sensitive_capabilities(self, body: BodyYaml) -> set[str]:
        """Return capabilities that are safety-sensitive when critical faults are open."""
        # Explicit list from body policy
        explicit = set(body.agent_policy.get("safety_sensitive_capabilities", []))
        # Heuristic fallback
        heuristic = {
            "walk", "run", "jump", "climb_stairs", "balance", "locomotion",
            "dual_arm_coordination", "reach", "manipulate", "grasp", "precision_grasp",
        }
        return explicit | heuristic

    def _calibration_dependent_capabilities(self) -> set[str]:
        """Capabilities that should be degraded when calibration is not valid."""
        return {
            "precision_grasp",
            "visual_navigation",
            "scan_workspace",
            "force_guided_manipulation",
            "hand_eye_coordination",
        }
