"""EMBODIMENT.md renderer — turns Effective Body into an Agent-readable manual."""

from __future__ import annotations

from typing import Any

from rosclaw.body.schema import (
    BodyYaml,
    CalibrationYaml,
    EffectiveBody,
    MaintenanceEvent,
    SkillCompatibilityReport,
)


class EmbodimentRenderer:
    """Render EMBODIMENT.md from Effective Body Model."""

    GENERATED_START = "<!-- ROSCLAW-GENERATED-START -->"
    GENERATED_END = "<!-- ROSCLAW-GENERATED-END -->"
    HUMAN_NOTES_START = "<!-- HUMAN-NOTES-START -->"
    HUMAN_NOTES_END = "<!-- HUMAN-NOTES-END -->"

    def render(
        self,
        body: EffectiveBody,
        body_yaml: BodyYaml,
        compatibility: SkillCompatibilityReport,
        maintenance_events: list[MaintenanceEvent],
        calibration: CalibrationYaml | None = None,
        preserve_human_notes: bool = True,
    ) -> str:
        """Render full EMBODIMENT.md content."""
        sections = [
            self._render_frontmatter(body, body_yaml),
            self._render_header(),
            self._render_identity(body, body_yaml),
            self._render_eurdf_profile_reference(body, body_yaml),
            self._render_effective_body_hash(body),
            self._render_body_structure(body, body_yaml),
            self._render_important_frames(body),
            self._render_sensors(body),
            self._render_actuators_and_tools(body),
            self._render_capabilities(body),
            self._render_degraded_capabilities(body),
            self._render_forbidden_capabilities(body, body_yaml),
            self._render_safety(body),
            self._render_calibration(body_yaml, calibration),
            self._render_known_faults(body),
            self._render_skill_compatibility(compatibility),
            self._render_maintenance_history(maintenance_events),
            self._render_known_successes(body),
            self._render_known_failures(body),
            self._render_agent_instructions(body_yaml),
            self._render_machine_readable_summary(body, body_yaml, calibration),
            self._render_source_files(body),
            self._render_regeneration(),
        ]

        generated = "\n".join(sections)

        human_notes = ""
        if preserve_human_notes:
            human_notes = f"\n{self.HUMAN_NOTES_START}\n\n{self.HUMAN_NOTES_END}\n"

        return f"{self.GENERATED_START}\n{generated}\n{self.GENERATED_END}\n{human_notes}"

    def render_into_existing(
        self,
        existing_md: str,
        body: EffectiveBody,
        body_yaml: BodyYaml,
        compatibility: SkillCompatibilityReport,
        maintenance_events: list[MaintenanceEvent],
        calibration: CalibrationYaml | None = None,
    ) -> str:
        """Re-render generated region while preserving human notes."""
        human_notes = self._extract_human_notes(existing_md)
        new_generated = self.render(
            body,
            body_yaml,
            compatibility,
            maintenance_events,
            calibration,
            preserve_human_notes=False,
        )
        if human_notes:
            return f"{new_generated}\n{self.HUMAN_NOTES_START}\n{human_notes}\n{self.HUMAN_NOTES_END}\n"
        return new_generated

    def _extract_human_notes(self, existing_md: str) -> str:
        """Extract content between human notes markers."""
        start = existing_md.find(self.HUMAN_NOTES_START)
        end = existing_md.find(self.HUMAN_NOTES_END)
        if start == -1 or end == -1 or end <= start:
            return ""
        return existing_md[start + len(self.HUMAN_NOTES_START) : end].strip()

    # ── Sections ──

    def _render_frontmatter(self, body: EffectiveBody, body_yaml: BodyYaml) -> str:
        identity = body_yaml.get_identity()
        safety_status = body_yaml.get_safety_status()
        body_yaml_path = "./body.yaml"
        calibration_yaml_path = "./calibration.yaml"
        maintenance_log_path = "./maintenance.log"
        eurdf_profile_path = body.source_trace.get("eurdf", "refs/eurdf.profile.yaml")
        eurdf_checksum = body_yaml.model_ref.get("profile_checksum", "")
        lines = [
            "---",
            "schema: rosclaw.embodiment.v1",
            "generated_by: rosclaw body init",
            f'generated_at: "{body.compiled_at}"',
            f'robot_instance_id: "{body.body_instance_id}"',
            f'robot_model: "{identity.get("robot_model") or body_yaml.body_instance.get("robot_model", "unknown")}"',
            f'robot_vendor: "{identity.get("robot_vendor") or "unknown"}"',
            f'eurdf_profile: "{body_yaml.model_ref.get("profile_id", "unknown")}"',
            f'eurdf_profile_path: "{eurdf_profile_path}"',
            f'eurdf_checksum: "{eurdf_checksum}"',
            f'body_yaml: "{body_yaml_path}"',
            f'calibration_yaml: "{calibration_yaml_path}"',
            f'maintenance_log: "{maintenance_log_path}"',
            f"body_state_generation: {body.generation}",
            f'safety_status: "{safety_status}"',
            "agent_readability: true",
            "do_not_edit_generated_sections: true",
            "---",
        ]
        return "\n".join(lines)

    def _render_header(self) -> str:
        return (
            "# EMBODIMENT.md — ROSClaw Body Context\n\n"
            "> This file tells an Agent what physical body it is dealing with.\n"
            "> It is generated from e-URDF, body.yaml, calibration.yaml, and maintenance.log.\n"
            "> It is an Agent-readable body context, not a permission bypass.\n"
            "> All physical actions still require sandbox / firewall / policy validation.\n"
        )

    def _render_identity(self, body: EffectiveBody, body_yaml: BodyYaml) -> str:
        identity = body_yaml.get_identity()
        safety_status = body_yaml.get_safety_status()
        lines = [
            "## 1. Identity",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Robot Instance ID | `{body.body_instance_id}` |",
            f"| Robot Model | {identity.get('robot_model') or body_yaml.body_instance.get('robot_model', 'unknown')} |",
            f"| Vendor | {identity.get('robot_vendor') or 'unknown'} |",
            f"| Nickname | {identity.get('nickname') or body_yaml.body_instance.get('nickname', 'N/A')} |",
            f"| Site / Lab | {identity.get('site') or body_yaml.body_instance.get('deployment_site', 'unknown')} |",
            f"| Owner / Operator | {identity.get('operator') or body_yaml.body_instance.get('owner', 'unknown')} |",
            f"| e-URDF Profile | {body_yaml.model_ref.get('profile_id', 'unknown')} |",
            f"| e-URDF Checksum | `{body_yaml.model_ref.get('profile_checksum', '')}` |",
            f"| Current Body State Generation | {body.generation} |",
            f"| Current Safety Status | {safety_status} |",
            "",
            "### Agent Summary",
            "",
            "You are interacting with a physical robot body described by this file.",
            "",
            "Before proposing or executing any physical action:",
            "",
            "1. Read **Capabilities**.",
            "2. Read **Forbidden Capabilities**.",
            "3. Read **Safety Limits**.",
            "4. Read **Known Faults**.",
            "5. Check current runtime state through `rosclaw body state` or MCP `get_robot_state`.",
            "6. Run sandbox / firewall validation before execution.",
            "",
        ]

        real_allowed = body.safety.get("environment", {}).get("real_robot_execution_allowed", True)
        if not real_allowed:
            lines.extend(
                [
                    "> **⚠️ Real-robot execution is disabled for this asset.**",
                    "> All manipulation and motion capabilities are simulation/sandbox-only until the safety policy is explicitly updated and clearance calibration is validated.",
                    "",
                ]
            )

        env = body.safety.get("environment", {})
        if env.get("perception_only") or env.get("no_actuation"):
            lines.extend(
                [
                    "> **📷 Perception-only body.**",
                    "> This device has no actuators. Proposing or executing motion, gripper, or actuator commands is forbidden. Only sensor capture and perception actions are allowed.",
                    "",
                ]
            )

        return "\n".join(lines)

    def _render_eurdf_profile_reference(self, body: EffectiveBody, body_yaml: BodyYaml) -> str:
        return (
            "## 2. e-URDF Profile Reference\n\n"
            f"Profile ID: `{body_yaml.model_ref.get('profile_id', 'unknown')}`\n\n"
            f"Version: `{body_yaml.model_ref.get('profile_version', 'unknown')}`\n\n"
            f"e-URDF URI: `{body.eurdf_uri}`\n\n"
            f"Checksum: `{body_yaml.model_ref.get('profile_checksum', '')}`\n\n"
            "This profile defines the static Physical DNA of the robot model. "
            "Instance-specific changes are recorded in `body.yaml`, not here.\n"
        )

    def _render_effective_body_hash(self, body: EffectiveBody) -> str:
        return (
            "## 3. Effective Body Hash\n\n"
            f"`{body.effective_body_hash}`\n\n"
            "This hash covers the compiled Effective Body Model (e-URDF + body.yaml + calibration + maintenance). "
            "Any module consuming body state should verify it is reading the same hash.\n"
        )

    def _render_body_structure(self, body: EffectiveBody, body_yaml: BodyYaml) -> str:
        lines = [
            "## 4. Body Structure",
            "",
            "### 4.1 Kinematic Tree Summary",
            "",
            "| Body Part | Links | Joints | Main Function | Notes |",
            "|---|---:|---:|---|---|",
        ]
        parts = body_yaml.body_structure.get("body_parts", []) if body_yaml.body_structure else []
        if not parts:
            parts = self._infer_body_parts(body)
        for part in parts:
            links = part.get("links", [])
            joints = part.get("joints", [])
            lines.append(
                f"| {part.get('name', part.get('id', 'unknown'))} "
                f"| {len(links)} | {len(joints)} "
                f"| {part.get('type', '')} | {part.get('notes') or ''} |"
            )
        if not parts:
            lines.append("| _No body parts defined._ | | | | |")
        lines.append("")

        # Joint groups
        lines.extend(
            [
                "### 4.2 Joint Groups",
                "",
                "| Group | Joints | Status | Notes |",
                "|---|---|---|---|",
            ]
        )
        groups = (
            body_yaml.body_structure.get("joint_groups", []) if body_yaml.body_structure else []
        )
        if not groups:
            groups = self._infer_joint_groups(body)
        for group in groups:
            joints = group.get("joints", [])
            joints_str = ", ".join(str(j) for j in joints[:5])
            if len(joints) > 5:
                joints_str += f", ... ({len(joints)} total)"
            lines.append(
                f"| {group.get('id', 'unknown')} | {joints_str or '_none_'} "
                f"| {group.get('status', 'unknown')} | {group.get('notes') or ''} |"
            )
        if not groups:
            lines.append("| _No joint groups defined._ | | | |")
        lines.append("")
        return "\n".join(lines)

    def _render_important_frames(self, body: EffectiveBody) -> str:
        lines = [
            "## 5. Important Frames",
            "",
            "| Frame | Parent | Type | Purpose | Source |",
            "|---|---|---|---|---|",
        ]
        frames = body.frames or {}
        if frames:
            for name, purpose in sorted(frames.items()):
                lines.append(f"| `{name}` | — | — | {purpose} | e-URDF |")
        else:
            lines.append("| _No frames defined._ | | | | |")
        lines.append("")
        return "\n".join(lines)

    def _render_sensors(self, body: EffectiveBody) -> str:
        lines = [
            "## 6. Installed Sensors",
            "",
            "| Sensor ID | Type | Mounted On | Frame | Status | Calibration | Notes |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, sensor in sorted(body.sensors.items()):
            sensor_type = sensor.get("type", "unknown")
            frame = sensor.get("parent_link") or sensor.get("frame", "unknown")
            status = sensor.get("status", "unknown")
            calibrated = "valid" if sensor.get("extrinsics") else "missing"
            mounted_on = sensor.get("mounted_on", "unknown")
            notes = sensor.get("notes", "")
            lines.append(
                f"| {name} | {sensor_type} | {mounted_on} | {frame} | {status} | {calibrated} | {notes} |"
            )
        if not body.sensors:
            lines.append("| _none_ | | | | | | |")
        lines.append("")

        # Sensor readiness
        readiness = self._sensor_readiness(body)
        lines.extend(
            [
                "### Sensor Readiness",
                "",
                f"- Vision: `{readiness.get('vision', 'unknown')}`",
                f"- Depth: `{readiness.get('depth', 'unknown')}`",
                f"- IMU: `{readiness.get('imu', 'unknown')}`",
                f"- Force / Torque: `{readiness.get('force_torque', 'unknown')}`",
                f"- Audio: `{readiness.get('audio', 'unknown')}`",
                f"- Proprioception: `{readiness.get('proprioception', 'unknown')}`",
                "",
            ]
        )
        return "\n".join(lines)

    def _render_actuators_and_tools(self, body: EffectiveBody) -> str:
        lines = [
            "## 7. Installed Actuators / Tools",
            "",
            "| Tool / Actuator | Type | Mounted On | Frame | Status | Safety Class | Notes |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, actuator in sorted(body.actuators.items()):
            act_type = actuator.get("type", "unknown")
            status = actuator.get("status", "unknown")
            mounted_on = actuator.get("mounted_on", "unknown")
            frame = actuator.get("frame", "unknown")
            safety_class = actuator.get("safety_class", "medium")
            notes = actuator.get("notes", "")
            lines.append(
                f"| {name} | {act_type} | {mounted_on} | {frame} | {status} | {safety_class} | {notes} |"
            )
        if not body.actuators:
            lines.append("| _none_ | | | | | | |")
        lines.append("")
        return "\n".join(lines)

    def _render_capabilities(self, body: EffectiveBody) -> str:
        lines = [
            "## 8. Current Capabilities",
            "",
            "Capabilities listed here are **declared capabilities**, not automatic permission to execute.",
            "",
            "Every physical action still requires policy, sandbox, and runtime validation.",
            "",
            "| Capability ID | Name | Body Parts | Required Sensors | Status | Risk | Validation Required |",
            "|---|---|---|---|---|---|---|",
        ]
        caps = body.capabilities
        enabled = caps.get("enabled", [])
        degraded = caps.get("degraded", [])
        blocked = caps.get("blocked", [])

        all_caps: list[tuple[str, str, str]] = []
        for cap in enabled:
            all_caps.append((cap, "enabled", "yes"))
        for cap in degraded:
            all_caps.append((cap, "degraded", "yes"))
        for cap in blocked:
            all_caps.append((cap, "disabled", "yes"))

        for cap_id, status, validation in sorted(all_caps):
            lines.append(f"| {cap_id} | {cap_id} | — | — | {status} | medium | {validation} |")
        if not all_caps:
            lines.append("| _none_ | | | | | | |")
        lines.append("")

        # Detailed YAML blocks
        lines.extend(
            [
                "### 8.1 Enabled Capabilities",
                "",
                "```yaml",
                "enabled:",
            ]
        )
        for cap in sorted(enabled):
            lines.append(f'  - id: "{cap}"')
            lines.append(f'    name: "{cap}"')
            lines.append('    scope: "unknown"')
            lines.append("    preconditions: []")
            lines.append("    validation:")
            lines.append("      sandbox_required: true")
            lines.append("      human_approval_required: false")
            lines.append('      max_risk: "medium"')
        if not enabled:
            lines.append("  []")
        lines.append("```")
        lines.append("")

        lines.extend(
            [
                "### 8.2 Degraded Capabilities",
                "",
                "```yaml",
                "degraded:",
            ]
        )
        for cap in sorted(degraded):
            lines.append(f'  - id: "{cap}"')
            lines.append('    reason: "calibration/maintenance/runtime"')
            lines.append('    allowed_mode: "slow"')
            lines.append("    blocked_subactions: []")
        if not degraded:
            lines.append("  []")
        lines.append("```")
        lines.append("")

        lines.extend(
            [
                "### 8.3 Disabled Capabilities",
                "",
                "```yaml",
                "disabled:",
            ]
        )
        for cap in sorted(blocked):
            lines.append(f'  - id: "{cap}"')
            lines.append('    reason: "prohibited or unavailable"')
            lines.append('    since: ""')
            lines.append("    reenable_requires: []")
        if not blocked:
            lines.append("  []")
        lines.append("```")
        lines.append("")
        return "\n".join(lines)

    def _render_degraded_capabilities(self, body: EffectiveBody) -> str:
        degraded = body.capabilities.get("degraded", [])
        lines = [
            "## 9. Degraded Capabilities",
            "",
            "Degraded capabilities are still declared but must be treated conservatively.",
            "",
            "```yaml",
            "degraded:",
        ]
        for cap in sorted(degraded):
            lines.append(f'  - id: "{cap}"')
            lines.append('    reason: "calibration / maintenance / runtime"')
            lines.append('    allowed_mode: "slow / constrained"')
            lines.append("    requires_human_approval: true")
        if not degraded:
            lines.append("  []")
        lines.extend(
            [
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    def _render_forbidden_capabilities(self, body: EffectiveBody, body_yaml: BodyYaml) -> str:
        lines = [
            "## 10. Disabled / Forbidden Capabilities",
            "",
            "The following actions must not be proposed or executed by an Agent unless a future explicitly approved body update removes them.",
            "",
            "```yaml",
            "forbidden:",
        ]
        forbidden = body.forbidden_capabilities or body_yaml.forbidden_capabilities or []
        for item in forbidden:
            lines.append(f'  - id: "{item.get("id", item.get("capability", "unknown"))}"')
            lines.append(f'    description: "{item.get("description", item.get("reason", ""))}"')
            lines.append(f'    reason: "{item.get("reason", "safety")}"')
            lines.append(f'    severity: "{item.get("severity", "critical")}"')
            enforcement = item.get("enforcement", {})
            lines.append("    enforcement:")
            lines.append(f"      policy_block: {enforcement.get('policy_block', True)}")
            lines.append(f"      sandbox_block: {enforcement.get('sandbox_block', True)}")
            lines.append(f"      real_robot_block: {enforcement.get('real_robot_block', True)}")
        if not forbidden:
            lines.append("  []")
        lines.extend(
            [
                "```",
                "",
                "Examples of forbidden capability classes:",
                "",
                "- Running or jumping without validated locomotion policy.",
                "- High-speed motion near humans.",
                "- Forceful manipulation without force feedback.",
                "- Stair climbing without validated stair profile.",
                "- Lifting objects above rated payload.",
                "- Using disabled limbs or known faulty joints.",
                "- Executing uncalibrated camera-guided grasping.",
                "- Bypassing sandbox / firewall validation.",
                "",
            ]
        )
        return "\n".join(lines)

    def _render_safety(self, body: EffectiveBody) -> str:
        lines = [
            "## 11. Safety Limits",
            "",
            "### 11.1 Global Safety Envelope",
            "",
            "```yaml",
        ]
        safety = body.safety or {}
        global_limits = safety.get("global_limits", {}) or {}
        if not global_limits and safety.get("safety_limits"):
            global_limits = safety.get("safety_limits", {})
        defaults = {
            "max_linear_speed_mps": None,
            "max_angular_speed_radps": None,
            "max_joint_speed_scale": 0.5,
            "max_joint_torque_scale": 0.5,
            "min_battery_percent_for_motion": 30,
            "human_distance_min_m": None,
            "require_estop_ready": True,
            "require_sandbox_validation": True,
            "require_runtime_monitor": True,
        }
        for key, default in defaults.items():
            value = global_limits.get(key, default)
            if value is None:
                value = "null"
            elif isinstance(value, bool):
                value = str(value).lower()
            lines.append(f"  {key}: {value}")
        lines.extend(
            [
                "```",
                "",
                "### 11.2 Workspace Limits",
                "",
                "| Workspace | Frame | Bounds | Allowed | Notes |",
                "|---|---|---|---|---|",
            ]
        )
        workspaces = safety.get("workspace_limits", []) or []
        for ws in workspaces:
            lines.append(
                f"| {ws.get('id', 'unknown')} | {ws.get('frame', '')} | {ws.get('bounds', '')} "
                f"| {ws.get('allowed', 'no')} | {ws.get('notes', '')} |"
            )
        if not workspaces:
            lines.append("| _No workspace limits defined._ | | | | |")
        lines.append("")

        lines.extend(
            [
                "### 11.3 Contact / Force Limits",
                "",
                "| Body Part | Max Force | Max Torque | Contact Allowed | Notes |",
                "|---|--:|--:|---|---|",
            ]
        )
        contacts = safety.get("contact_limits", []) or []
        for contact in contacts:
            lines.append(
                f"| {contact.get('body_part', 'unknown')} | {contact.get('max_force', '')} "
                f"| {contact.get('max_torque', '')} | {contact.get('contact_allowed', 'no')} "
                f"| {contact.get('notes', '')} |"
            )
        if not contacts:
            lines.append("| _No contact limits defined._ | | | | |")
        lines.append("")

        lines.extend(
            [
                "### 11.4 Safety Gates",
                "",
                "Before physical execution, the Agent must pass:",
                "",
                "1. Body capability check.",
                "2. Forbidden capability check.",
                "3. Fault check.",
                "4. Calibration validity check.",
                "5. Runtime state check.",
                "6. Sandbox validation.",
                "7. Firewall / risk policy validation.",
                "8. Human approval when required.",
                "9. Practice Timeline logging.",
                "",
            ]
        )
        return "\n".join(lines)

    def _render_known_faults(self, body: EffectiveBody) -> str:
        lines = [
            "## 13. Known Faults",
            "",
            "| Fault ID | Component | Severity | Status | First Seen | Last Seen | Impact | Required Action |",
            "|---|---|---|---|---|---|---|---|",
        ]
        faults = body.known_faults or []
        open_faults = []
        for fault in faults:
            sev = fault.get("severity", "low")
            status = fault.get("status", "open")
            if status == "open":
                open_faults.append(fault)
            lines.append(
                f"| {fault.get('id', 'unknown')} | {fault.get('component', '')} | {sev} "
                f"| {status} | {fault.get('first_seen', '')} | {fault.get('last_seen', '')} "
                f"| {fault.get('impact', '')} | {fault.get('required_action', '')} |"
            )
        if not faults:
            lines.append("| _No known faults._ | | | | | | | |")
        lines.append("")

        lines.extend(
            [
                "### Fault-derived Capability Changes",
                "",
                "```yaml",
                "fault_capability_overrides:",
            ]
        )
        if open_faults:
            for fault in open_faults:
                lines.append(f'  - fault_id: "{fault.get("id", "unknown")}"')
                lines.append(f"    disables: {fault.get('disables', []) or []}")
                lines.append(f"    degrades: {fault.get('degrades', []) or []}")
        else:
            lines.append("  []")
        lines.extend(
            [
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    def _render_skill_compatibility(self, compatibility: SkillCompatibilityReport) -> str:
        summary = compatibility.summary if compatibility else {}
        skills = compatibility.skills if compatibility else {}
        lines = [
            "## 14. Skill Compatibility Summary",
            "",
            "Skill compatibility is derived from the Effective Body Model and each skill manifest.",
            "A skill listed as compatible here still requires sandbox validation before physical execution.",
            "",
            "| Status | Count |",
            "|---|---:|",
            f"| compatible | {summary.get('compatible', 0)} |",
            f"| degraded | {summary.get('degraded', 0)} |",
            f"| blocked | {summary.get('blocked', 0)} |",
            f"| unknown | {summary.get('unknown', 0)} |",
            "",
            "```yaml",
            "skills:",
        ]
        for skill_id, result in sorted(skills.items()):
            lines.append(f"  {skill_id}:")
            lines.append(f"    status: {result.status}")
            if result.reason:
                lines.append(f'    reason: "{result.reason}"')
        if not skills:
            lines.append("  {}")
        lines.extend(
            [
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    def _render_known_successes(self, body: EffectiveBody) -> str:
        lines = [
            "## 16. Known Successful Experiences",
            "",
            "These are not generic truths. They are references to practice / memory records that were successful on this body or a sufficiently similar body.",
            "",
            "| Experience ID | Task | Body Parts | Conditions | Outcome | Reuse Scope |",
            "|---|---|---|---|---|---|",
        ]
        for exp in body.known_successes or []:
            lines.append(
                f"| {exp.get('id', 'unknown')} | {exp.get('task', '')} "
                f"| {', '.join(exp.get('body_parts', []))} | {', '.join(exp.get('conditions', []))} "
                f"| {exp.get('outcome', '')} | {exp.get('reuse_scope', 'same_body')} |"
            )
        if not body.known_successes:
            lines.append("| _No recorded successes._ | | | | | |")
        lines.append("")

        lines.extend(
            [
                "### Recommended Reuse Policy",
                "",
                "```yaml",
                "experience_reuse:",
                '  same_body: "can_reuse_after_state_check"',
                '  same_model: "verify_in_sandbox_first"',
                '  different_model_same_profile_family: "require_transfer_validation"',
                '  different_morphology: "do_not_reuse_directly"',
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    def _render_known_failures(self, body: EffectiveBody) -> str:
        lines = [
            "## 17. Known Failed Experiences",
            "",
            "| Failure ID | Task | Body Parts | Failure Mode | Root Cause | Avoidance |",
            "|---|---|---|---|---|---|",
        ]
        for failure in body.known_failures or []:
            lines.append(
                f"| {failure.get('id', 'unknown')} | {failure.get('task', '')} "
                f"| {', '.join(failure.get('body_parts', []))} | {failure.get('failure_mode', '')} "
                f"| {failure.get('root_cause', '')} | {failure.get('avoidance', '')} |"
            )
        if not body.known_failures:
            lines.append("| _No recorded failures._ | | | | | |")
        lines.append("")
        return "\n".join(lines)

    def _render_calibration(self, body_yaml: BodyYaml, calibration: CalibrationYaml | None) -> str:
        cal = calibration or CalibrationYaml()
        overall = cal.overall_status()
        lines = [
            "## 12. Calibration Summary",
            "",
            "| Calibration Item | Status | Last Updated | Confidence | Source |",
            "|---|---|---|---|---|",
        ]
        items = [
            (
                "Joint zero offsets",
                cal.joints if cal.joints else body_yaml.calibration.get("joint_offsets", {}),
            ),
            ("Camera intrinsics", cal.sensors if cal.sensors else {}),
            ("Camera extrinsics", cal.sensor_extrinsics if cal.sensor_extrinsics else {}),
            ("IMU bias", cal.sensors if cal.sensors else {}),
            ("Tool frames", cal.frames if cal.frames else {}),
        ]
        for item_name, source in items:
            has_data = bool(source)
            status = "valid" if has_data else "missing"
            lines.append(f"| {item_name} | {status} | — | — | calibration.yaml |")
        lines.append("")

        warnings = []
        if overall not in ("valid", "validated"):
            warnings.append(f"Calibration overall status is '{overall}'.")
        lines.extend(
            [
                "### Calibration Warnings",
                "",
                "```yaml",
                "warnings:",
            ]
        )
        for warning in warnings:
            lines.append(f'  - "{warning}"')
        if not warnings:
            lines.append("  []")
        lines.extend(
            [
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    def _render_maintenance_history(self, maintenance_events: list[MaintenanceEvent]) -> str:
        lines = [
            "## 15. Recent Maintenance Events",
            "",
            "Recent maintenance events are summarized here. Full append-only history is in `maintenance.log`.",
            "",
            "| Time | Event Type | Component | Summary | Operator | Result |",
            "|---|---|---|---|---|---|",
        ]
        recent = sorted(maintenance_events, key=lambda e: e.ts or e.time, reverse=True)[:20]
        for event in recent:
            component = event.component or ", ".join(event.affects) or "—"
            lines.append(
                f"| {event.time or event.ts} | {event.type} | {component} "
                f"| {event.summary or event.message} | {event.operator or event.author} | {event.result.get('status') if event.result else '—'} |"
            )
        if not recent:
            lines.append("| _No maintenance events recorded._ | | | | | |")
        lines.append("")
        return "\n".join(lines)

    def _render_agent_instructions(self, body_yaml: BodyYaml) -> str:
        policy = body_yaml.agent_policy if body_yaml.agent_policy else {}
        lines = [
            "## 18. Agent Operating Instructions",
            "",
            "### 18.1 Must Do",
            "",
            "- Read this file before planning physical action.",
            "- Query current runtime state before action.",
            "- Respect disabled and forbidden capabilities.",
            "- Treat degraded capability as requiring conservative execution.",
            "- Use sandbox / firewall validation before physical execution.",
            "- Log physical attempts into Practice Timeline.",
            "- Write failures into Memory / Know / How / Auto loop.",
            "",
            "### 18.2 Must Not Do",
            "",
            "- Do not assume a capability is safe because it appears in this file.",
            "- Do not bypass sandbox validation.",
            "- Do not command disabled joints or faulty components.",
            "- Do not reuse a skill from another robot body without transfer validation.",
            "- Do not edit generated sections directly.",
            "- Do not overwrite maintenance history.",
            "- Do not treat stale calibration as valid.",
            "",
            "### 18.3 When Unsure",
            "",
            "If body state, calibration, or capability status is unknown:",
            "",
            "```text",
            "Prefer: ask for body state / run sandbox / require human approval.",
            "Avoid: direct physical execution.",
            "```",
            "",
        ]
        if policy:
            lines.extend(
                [
                    "### 18.4 Instance Policy",
                    "",
                    "```yaml",
                ]
            )
            for key, value in sorted(policy.items()):
                lines.append(f"  {key}: {value}")
            lines.extend(
                [
                    "```",
                    "",
                ]
            )
        return "\n".join(lines)

    def _render_machine_readable_summary(
        self,
        body: EffectiveBody,
        body_yaml: BodyYaml,
        calibration: CalibrationYaml | None,
    ) -> str:
        cal = calibration or CalibrationYaml()
        identity = body_yaml.get_identity()
        caps = body.capabilities
        forbidden = [
            item.get("id", item.get("capability", "unknown"))
            for item in (body.forbidden_capabilities or body_yaml.forbidden_capabilities or [])
        ]
        open_faults = [
            f.get("id", "unknown") for f in (body.known_faults or []) if f.get("status") == "open"
        ]
        lines = [
            "## 19. Machine-readable Summary",
            "",
            "```yaml",
            f'robot_instance_id: "{body.body_instance_id}"',
            f'robot_model: "{identity.get("robot_model") or body_yaml.body_instance.get("robot_model", "unknown")}"',
            f'eurdf_profile: "{body_yaml.model_ref.get("profile_id", "unknown")}"',
            f'safety_status: "{body_yaml.get_safety_status()}"',
            "capabilities:",
            f"  enabled: {caps.get('enabled', [])}",
            f"  degraded: {caps.get('degraded', [])}",
            f"  disabled: {caps.get('blocked', [])}",
            f"forbidden: {forbidden}",
            f"known_faults_open: {open_faults}",
            "calibration:",
            f'  overall_status: "{cal.overall_status()}"',
            "agent_action_policy:",
            "  physical_execution_requires_sandbox: true",
            "  direct_real_robot_execution_allowed: false",
            "  human_approval_required_for_high_risk: true",
            "```",
            "",
        ]
        return "\n".join(lines)

    def _render_source_files(self, body: EffectiveBody) -> str:
        lines = [
            "## 20. Source Files",
            "",
            "| Source | Path | Role |",
            "|---|---|---|",
            f"| e-URDF profile | `{body.source_trace.get('eurdf', 'refs/eurdf.profile.yaml')}` | Static physical body definition |",
            "| body.yaml | `./body.yaml` | Instance body state |",
            "| calibration.yaml | `./calibration.yaml` | Calibration parameters |",
            "| maintenance.log | `./maintenance.log` | Append-only history |",
            "| generated JSON summary | `./refs/generated/body.summary.json` | Agent / MCP compact summary |",
            "",
        ]
        return "\n".join(lines)

    def _render_regeneration(self) -> str:
        return (
            "## 21. Regeneration Commands\n\n"
            "This file can be regenerated by:\n\n"
            "```bash\n"
            "rosclaw body render\n"
            "```\n\n"
            "Validate all body files by:\n\n"
            "```bash\n"
            "rosclaw body validate\n"
            "```\n\n"
            "Show Agent-readable summary by:\n\n"
            "```bash\n"
            "rosclaw body show --agent\n"
            "```\n"
        )

    # ── Helpers ──

    def _infer_body_parts(self, body: EffectiveBody) -> list[dict[str, Any]]:
        """Infer body parts from joint names when not explicitly defined."""
        parts: dict[str, dict[str, Any]] = {}
        for joint_name in body.joints:
            name = str(joint_name).lower()
            if "left" in name and "arm" in name:
                key = "left_arm"
            elif "right" in name and "arm" in name:
                key = "right_arm"
            elif "left" in name and "leg" in name:
                key = "left_leg"
            elif "right" in name and "leg" in name:
                key = "right_leg"
            elif "head" in name or "neck" in name:
                key = "head"
            elif "torso" in name or "waist" in name:
                key = "torso"
            else:
                key = "base"
            parts.setdefault(
                key,
                {
                    "id": key,
                    "name": key.replace("_", " ").title(),
                    "type": key,
                    "links": [],
                    "joints": [],
                },
            )
            parts[key]["joints"].append(joint_name)
        return list(parts.values())

    def _infer_joint_groups(self, body: EffectiveBody) -> list[dict[str, Any]]:
        """Infer joint groups from joint names."""
        groups: dict[str, list[str]] = {}
        for joint_name in body.joints:
            name = str(joint_name).lower()
            if "arm" in name:
                key = "arms"
            elif "leg" in name:
                key = "locomotion"
            elif "head" in name or "neck" in name:
                key = "head"
            elif "torso" in name or "waist" in name:
                key = "torso"
            else:
                key = "other"
            groups.setdefault(key, []).append(joint_name)
        return [{"id": k, "joints": v, "status": "enabled"} for k, v in groups.items()]

    def _sensor_readiness(self, body: EffectiveBody) -> dict[str, str]:
        """Summarize sensor readiness by category."""
        readiness: dict[str, str] = {
            "vision": "unavailable",
            "depth": "unavailable",
            "imu": "unavailable",
            "force_torque": "unavailable",
            "audio": "unavailable",
            "proprioception": "unavailable",
        }
        has_available = False
        for name, sensor in body.sensors.items():
            status = sensor.get("status", "unknown")
            stype = str(sensor.get("type", "")).lower()
            if status == "available":
                has_available = True
            if "camera" in name.lower() or stype == "camera":
                readiness["vision"] = self._best_readiness(readiness["vision"], status)
            if "depth" in name.lower() or stype in ("depth", "lidar"):
                readiness["depth"] = self._best_readiness(readiness["depth"], status)
            if "imu" in name.lower() or stype == "imu":
                readiness["imu"] = self._best_readiness(readiness["imu"], status)
            if "force" in name.lower() or "ft" in name.lower() or stype == "force_torque":
                readiness["force_torque"] = self._best_readiness(readiness["force_torque"], status)
            if "audio" in name.lower() or "mic" in name.lower() or stype == "microphone":
                readiness["audio"] = self._best_readiness(readiness["audio"], status)
        if body.joints:
            readiness["proprioception"] = "ready" if has_available else "degraded"
        return readiness

    def _best_readiness(self, current: str, status: str) -> str:
        order = ["unavailable", "degraded", "ready"]
        idx_current = order.index(current) if current in order else 0
        idx_status = order.index(status) if status in order else 1
        if status == "available":
            idx_status = order.index("ready")
        return order[max(idx_current, idx_status)]
