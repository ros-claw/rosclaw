"""Body Agent View layer.

Responsible for rendering the Agent-readable artifacts derived from the
Effective Body Model:

- EMBODIMENT.md
- BODY.md alias
- refs/generated/*.summary.json

This module does not define body truth; it only renders views of the already
compiled Effective Body.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rosclaw.body.renderer import EmbodimentRenderer
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import (
    BodyYaml,
    CalibrationYaml,
    EffectiveBody,
    MaintenanceEvent,
    SkillCompatibilityReport,
)
from rosclaw.body.summaries import BodySummaryGenerator


class BodyAgentViewRenderer:
    """Render EMBODIMENT.md and related agent-readable artifacts."""

    REQUIRED_SECTIONS = [
        "1. Identity",
        "2. e-URDF Profile Reference",
        "3. Effective Body Hash",
        "4. Body Structure",
        "5. Important Frames",
        "6. Installed Sensors",
        "7. Installed Actuators / Tools",
        "8. Current Capabilities",
        "9. Degraded Capabilities",
        "10. Disabled / Forbidden Capabilities",
        "11. Safety Limits",
        "12. Calibration Summary",
        "13. Known Faults",
        "14. Skill Compatibility Summary",
        "15. Recent Maintenance Events",
        "16. Known Successful Experiences",
        "17. Known Failed Experiences",
        "18. Agent Operating Instructions",
        "19. Machine-readable Summary",
        "20. Source Files",
        "21. Regeneration Commands",
    ]

    def __init__(self, workspace: Path | None = None, body_id: str | None = None):
        self.resolver = BodyResolver(workspace=workspace, body_id=body_id)
        self._renderer = EmbodimentRenderer()

    def render_all(
        self,
        effective: EffectiveBody | None = None,
        body: BodyYaml | None = None,
        calibration: CalibrationYaml | None = None,
        maintenance: list[MaintenanceEvent] | None = None,
        report: SkillCompatibilityReport | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Render EMBODIMENT.md, BODY.md alias, and generated summaries.

        Returns a dict with paths to the generated artifacts.
        """
        effective = effective or self.resolver.get_effective_body()
        body = body or self.resolver.get_current_body_yaml()
        calibration = calibration or self.resolver.get_calibration()
        maintenance = maintenance if maintenance is not None else self.resolver.get_maintenance_events()
        if report is None:
            report = self.resolver.get_skill_compatibility()

        md = self.render_embodiment(
            effective, body, calibration, maintenance, report, preserve_human_notes=True
        )
        self.resolver.embodiment_md_path.write_text(md, encoding="utf-8")
        alias_path = self._refresh_body_md_alias()

        generator = BodySummaryGenerator(
            workspace=self.resolver.workspace, body_id=self.resolver.body_id
        )
        summaries = generator.generate_all(effective, body, calibration, report, maintenance, write=True)

        # Record render event unless explicitly disabled.
        if reason != "no_event":
            from rosclaw.body.notes import MaintenanceLog

            MaintenanceLog(self.resolver.maintenance_log_path).write_render_event(
                body_instance_id=effective.body_instance_id,
                reason=reason or "agent_view.render_all",
            )

        return {
            "embodiment_md": self.resolver.embodiment_md_path,
            "body_md": alias_path,
            "summaries": {name: self.resolver.generated_dir / name for name in summaries},
        }

    def render_embodiment(
        self,
        effective_body: EffectiveBody,
        body_yaml: BodyYaml,
        calibration: CalibrationYaml,
        maintenance_events: list[MaintenanceEvent],
        compatibility_report: SkillCompatibilityReport | None,
        preserve_human_notes: bool = True,
    ) -> str:
        """Render EMBODIMENT.md content satisfying all required sections."""
        human_notes = ""
        existing_md = None
        if preserve_human_notes and self.resolver.embodiment_md_path.exists():
            existing_md = self.resolver.embodiment_md_path.read_text(encoding="utf-8")
            human_notes = self._extract_human_notes(existing_md)

        rendered = self._renderer.render(
            effective_body,
            body_yaml,
            compatibility_report or SkillCompatibilityReport(
                body_instance_id=effective_body.body_instance_id,
                effective_body_hash=effective_body.effective_body_hash,
            ),
            maintenance_events,
            calibration,
            preserve_human_notes=False,
        )

        # Ensure every required section appears.  If the underlying renderer is
        # missing a section, inject a placeholder so the contract is satisfied.
        for section_header in self.REQUIRED_SECTIONS:
            if section_header not in rendered:
                rendered = self._inject_section(rendered, section_header, effective_body, body_yaml)

        if human_notes:
            rendered = (
                f"{rendered}\n"
                f"{self._renderer.HUMAN_NOTES_START}\n"
                f"{human_notes}\n"
                f"{self._renderer.HUMAN_NOTES_END}\n"
            )
        return rendered

    def _extract_human_notes(self, existing_md: str) -> str:
        return self._renderer._extract_human_notes(existing_md)  # noqa: SLF001

    def _inject_section(
        self,
        rendered: str,
        section_header: str,
        effective: EffectiveBody,
        body_yaml: BodyYaml,
    ) -> str:
        """Inject a minimal placeholder for a missing required section."""
        section_number, title = section_header.split(". ", 1)
        if title == "Effective Body Hash":
            content = f"`{effective.effective_body_hash}`"
        elif title == "e-URDF Profile Reference":
            content = (
                f"Profile ID: `{body_yaml.model_ref.get('profile_id', 'unknown')}`\n"
                f"Version: `{body_yaml.model_ref.get('profile_version', 'unknown')}`\n"
                f"Checksum: `{body_yaml.model_ref.get('profile_checksum', '')}`"
            )
        elif title == "Skill Compatibility Summary":
            content = (
                "See `skill_compatibility.yaml` and the Machine-readable Summary section.\n"
                "All skill compatibility checks must be up to date before physical execution."
            )
        elif title == "Degraded Capabilities":
            degraded = effective.capabilities.get("degraded", [])
            content = "No degraded capabilities." if not degraded else f"Degraded: {degraded}"
        else:
            content = f"_{title} not provided by renderer._"
        marker = self._renderer.GENERATED_END
        insert_after = rendered.rfind(marker)
        if insert_after == -1:
            return rendered + f"\n## {section_header}\n\n{content}\n"
        end = insert_after + len(marker)
        section_md = f"\n\n## {section_header}\n\n{content}\n"
        return rendered[:end] + section_md + rendered[end:]

    def _refresh_body_md_alias(self) -> Path:
        """Create or refresh BODY.md as a pointer to EMBODIMENT.md."""
        self.resolver.generated_dir.parent.mkdir(parents=True, exist_ok=True)
        body_md_content = (
            "# BODY.md\n\n"
            "> BODY.md is an alias copy of EMBODIMENT.md.\n"
            "> Canonical file: EMBODIMENT.md.\n"
            "> Do not edit this copy directly.\n\n"
            f"See: [{self.resolver.embodiment_md_path.name}]({self.resolver.embodiment_md_path.name})\n"
        )
        self.resolver.body_md_path.write_text(body_md_content, encoding="utf-8")
        return self.resolver.body_md_path

    @classmethod
    def required_sections(cls) -> list[str]:
        return list(cls.REQUIRED_SECTIONS)


class BodyAgentView:
    """High-level read-only view of the body for Agents / MCP / Dashboard."""

    def __init__(self, workspace: Path | None = None, body_id: str | None = None):
        self.resolver = BodyResolver(workspace=workspace, body_id=body_id)

    def get_state_json(self) -> str:
        """Return a JSON string with the current body safety state."""
        effective = self.resolver.get_effective_body()
        body = self.resolver.get_current_body_yaml()
        calibration = self.resolver.get_calibration()
        state = {
            "body_instance_id": effective.body_instance_id,
            "effective_body_hash": effective.effective_body_hash,
            "safety_status": body.get_safety_status(),
            "calibration_status": calibration.overall_status(),
            "enabled_capabilities": effective.capabilities.get("enabled", []),
            "degraded_capabilities": effective.capabilities.get("degraded", []),
            "disabled_capabilities": effective.capabilities.get("blocked", []),
            "forbidden_capabilities": [
                item.get("id", item.get("capability", "unknown"))
                for item in (effective.forbidden_capabilities or body.forbidden_capabilities or [])
            ],
            "open_faults": [f.get("id", "unknown") for f in effective.known_faults if f.get("status") == "open"],
            "runtime_overlay": body.runtime_state or {},
            "agent_policy": {
                "physical_execution_requires_sandbox": True,
                "direct_real_robot_execution_allowed": False,
                "human_approval_required_for_high_risk": True,
            },
        }
        return json.dumps(state, indent=2, default=str)

    def get_agent_summary(self) -> str:
        """Return a short Agent-readable text summary."""
        effective = self.resolver.get_effective_body()
        body = self.resolver.get_current_body_yaml()
        caps = effective.capabilities
        forbidden = body.forbidden_capabilities or effective.forbidden_capabilities or []
        lines = [
            f"Body: {effective.body_instance_id} ({body.body_instance.get('robot_model', 'unknown')})",
            f"Effective hash: {effective.effective_body_hash}",
            f"Safety status: {body.get_safety_status()}",
            f"Enabled capabilities: {caps.get('enabled', [])}",
            f"Degraded capabilities: {caps.get('degraded', [])}",
            f"Disabled capabilities: {caps.get('blocked', [])}",
            f"Forbidden capabilities: {[i.get('id', i.get('capability', 'unknown')) for i in forbidden]}",
            "",
            "All physical execution requires sandbox validation and is not allowed on the real robot without policy approval.",
        ]
        return "\n".join(lines)
