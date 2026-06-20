# EMBODIMENT.md format

`~/.rosclaw/body/EMBODIMENT.md` is the **Agent-readable compiled body manual** in the ROSClaw three-layer body model.
It is rendered from the `EffectiveBody` by `EmbodimentRenderer` and kept in sync by body commands.

## Purpose

- Tell an Agent what physical body it is operating.
- Summarize identity, structure, sensors, actuators, capabilities, limits, faults, and policy.
- Provide a stable, human-readable view of the body state without exposing internal YAML structures.
- Preserve hand-written notes between regenerations.

## Conventions

- The file is Markdown.
- Generated sections are wrapped in `<!-- ROSCLAW-GENERATED-START -->` and `<!-- ROSCLAW-GENERATED-END -->`.
- Do not edit generated sections by hand. They are overwritten by `rosclaw body render`, `update-state`, `note`, etc.
- Hand-written notes go between `<!-- HUMAN-NOTES-START -->` and `<!-- HUMAN-NOTES-END -->` and are preserved across renders.

## Frontmatter

The generated YAML frontmatter identifies the file schema, generation time, and source files.

| Field | Type | Description |
|---|---|---|
| `schema` | string | `rosclaw.embodiment.v1` |
| `generated_by` | string | Tool that produced the file (e.g., `rosclaw body init`). |
| `generated_at` | ISO-8601 timestamp | Render time in UTC. |
| `robot_instance_id` | string | Unique body instance identifier. |
| `robot_model` | string | Robot model / e-URDF profile ID. |
| `robot_vendor` | string | Vendor name, if known. |
| `eurdf_profile` | string | Linked e-URDF profile ID. |
| `eurdf_profile_path` | string | Relative path to the normalized e-URDF profile. |
| `eurdf_checksum` | string | Checksum of the pinned e-URDF profile. |
| `body_yaml` | string | Relative path to `body.yaml`. |
| `calibration_yaml` | string | Relative path to `calibration.yaml`. |
| `maintenance_log` | string | Relative path to `maintenance.log`. |
| `body_state_generation` | integer | Effective body generation counter. |
| `safety_status` | string | Current safety status. |
| `agent_readability` | boolean | Always `true`. |
| `do_not_edit_generated_sections` | boolean | Always `true`. |

## Sections

The rendered file contains the following sections in order.

| # | Section | Source data | Purpose |
|---|---------|-------------|---------|
| 1 | **Identity** | `body.yaml` instance identity | Who/what/where this robot is. |
| 2 | **Body Structure** | `body.yaml` body parts + `EffectiveBody.frames` | Kinematic tree, frames, joint groups. |
| 3 | **Installed Sensors** | `EffectiveBody.sensors` | Sensor inventory and readiness. |
| 4 | **Installed Actuators and Tools** | `EffectiveBody.actuators` | Actuator / tool inventory. |
| 5 | **Current Capabilities** | `EffectiveBody.capabilities` | Enabled, degraded, blocked capabilities. |
| 6 | **Forbidden Capabilities** | `body.yaml.forbidden_capabilities` | Capabilities that must never be used. |
| 7 | **Safety Limits** | `EffectiveBody.safety` | Global envelope, workspace, contact limits, gates. |
| 8 | **Known Faults** | `EffectiveBody.known_faults` | Open faults and capability overrides. |
| 9 | **Known Successful Experiences** | `EffectiveBody.known_successes` | Reusable memory references. |
| 10 | **Known Failed Experiences** | `EffectiveBody.known_failures` | Failures to avoid. |
| 11 | **Calibration Summary** | `calibration.yaml` + `body.yaml.calibration` | Calibration status per category. |
| 12 | **Maintenance and Modification History** | `maintenance.log` | Recent maintenance events. |
| 13 | **Agent Operating Instructions** | `body.yaml.agent_policy` | Must do / must not do / when unsure. |
| 14 | **Machine-readable Summary** | `EffectiveBody` + calibration | Compact YAML block for agents / MCP. |
| 15 | **Source Files** | `EffectiveBody.source_trace` | Where each piece of information came from. |
| 16 | **Regeneration** | hard-coded | Commands to refresh this file. |

## Capability rendering

Capabilities are listed twice:

1. A summary table with `enabled` / `degraded` / `disabled` status.
2. Separate YAML blocks under **5.1 Enabled**, **5.2 Degraded**, and **5.3 Disabled**.

The YAML blocks include validation requirements. A capability appearing in this file is **not** permission to execute; it is a declaration that the body is physically capable. Execution still requires skill compatibility, sandbox validation, and policy checks.

## Human notes

Add free-form Markdown between the human-notes markers:

```markdown
<!-- HUMAN-NOTES-START -->

## Operator notes

- Camera cable tends to slip after long runs. Check before vision tasks.
- Left gripper has reduced force; use precision_grasp only in supervised mode.

<!-- HUMAN-NOTES-END -->
```

`rosclaw body render` and other update commands preserve this block.

## Regeneration

```bash
# Re-render EMBODIMENT.md from the current effective body
rosclaw body render

# Validate body files
rosclaw body validate

# Print compact agent summary
rosclaw body show --agent
```

## Files

- `src/rosclaw/body/renderer.py` — `EmbodimentRenderer`
- `src/rosclaw/body/schema.py` — `EffectiveBody` schema
- `~/.rosclaw/body/EMBODIMENT.md` — rendered output for the linked body
