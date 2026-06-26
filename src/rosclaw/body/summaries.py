"""Generated summary generators for machine-readable body views.

These summaries live under ``refs/generated/`` and are consumed by MCP tools,
the Dashboard, and Agent context compression.  They are derived from the
Effective Body Model and must not become a second source of truth.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.body.schema import (
    BodyYaml,
    CalibrationYaml,
    EffectiveBody,
    MaintenanceEvent,
    SkillCompatibilityReport,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class BodySummaryGenerator:
    """Generate compact JSON summaries from an Effective Body."""

    def __init__(self, workspace: Path | None = None, body_id: str | None = None):
        from rosclaw.body.resolver import BodyResolver

        self.resolver = BodyResolver(workspace=workspace, body_id=body_id)

    def generate_all(
        self,
        effective: EffectiveBody,
        body: BodyYaml,
        calibration: CalibrationYaml,
        report: SkillCompatibilityReport,
        maintenance: list[MaintenanceEvent],
        write: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Generate all three summary dicts and optionally write them to disk."""
        summaries = {
            "body.summary.json": self.generate_body_summary(effective, body, calibration, maintenance),
            "embodiment.agent.json": self.generate_agent_summary(effective, body, report, maintenance),
            "safety.summary.json": self.generate_safety_summary(effective, body, calibration, maintenance),
        }
        if write:
            self._write_summaries(summaries)
        return summaries

    def generate_body_summary(
        self,
        effective: EffectiveBody,
        body: BodyYaml,
        calibration: CalibrationYaml,
        maintenance: list[MaintenanceEvent] | None = None,
    ) -> dict[str, Any]:
        identity = body.get_identity()
        forbidden = [
            item.get("id", item.get("capability", "unknown"))
            for item in (effective.forbidden_capabilities or body.forbidden_capabilities or [])
        ]
        open_faults = [f.get("id", "unknown") for f in effective.known_faults if f.get("status") == "open"]
        return {
            "schema": "rosclaw.generated.body_summary.v1",
            "generated_at": _utc_now(),
            "robot_instance_id": effective.body_instance_id,
            "robot_model": identity.get("robot_model") or body.body_instance.get("robot_model", "unknown"),
            "eurdf_profile": body.model_ref.get("profile_id", "unknown"),
            "safety_status": body.get_safety_status(),
            "calibration_status": calibration.overall_status(),
            "capabilities": effective.capabilities,
            "forbidden": forbidden,
            "known_faults_open": open_faults,
            "effective_body_hash": effective.effective_body_hash,
            "generation": effective.generation,
        }

    def generate_agent_summary(
        self,
        effective: EffectiveBody,
        body: BodyYaml,
        report: SkillCompatibilityReport,
        maintenance: list[MaintenanceEvent] | None = None,
    ) -> dict[str, Any]:
        identity = body.get_identity()
        forbidden = [
            item.get("id", item.get("capability", "unknown"))
            for item in (effective.forbidden_capabilities or body.forbidden_capabilities or [])
        ]
        open_faults = [f.get("id", "unknown") for f in effective.known_faults if f.get("status") == "open"]
        return {
            "schema": "rosclaw.generated.embodiment_agent.v1",
            "generated_at": _utc_now(),
            "identity": identity,
            "capabilities": effective.capabilities,
            "forbidden": forbidden,
            "open_faults": open_faults,
            "compatibility_summary": report.to_dict() if report else {},
            "agent_policy": {
                "physical_execution_requires_sandbox": True,
                "direct_real_robot_execution_allowed": False,
                "human_approval_required_for_high_risk": True,
            },
        }

    def generate_safety_summary(
        self,
        effective: EffectiveBody,
        body: BodyYaml,
        calibration: CalibrationYaml,
        maintenance: list[MaintenanceEvent] | None = None,
    ) -> dict[str, Any]:
        open_faults = [f.get("id", "unknown") for f in effective.known_faults if f.get("status") == "open"]
        return {
            "schema": "rosclaw.generated.safety_summary.v1",
            "generated_at": _utc_now(),
            "safety_status": body.get_safety_status(),
            "global_limits": effective.safety.get("safety_limits") or effective.safety.get("global_limits") or {},
            "workspace_limits": effective.safety.get("workspace_boundaries") or effective.safety.get("workspace_limits") or [],
            "contact_limits": effective.safety.get("contact_limits") or [],
            "open_faults": open_faults,
            "calibration_status": calibration.overall_status(),
        }

    def _write_summaries(self, summaries: dict[str, dict[str, Any]]) -> None:
        out_dir = self.resolver.generated_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for filename, data in summaries.items():
            (out_dir / filename).write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
