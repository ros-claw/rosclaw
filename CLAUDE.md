# CLAUDE.md — ROSClaw Project Onboarding

This file tells Claude Code how to work with the ROSClaw physical-AI runtime.
It is safe to edit the human sections below. Managed blocks are updated by
`rosclaw agent init claude-code` and should not be hand-edited.

<!-- ROSCLAW-MANAGED-BEGIN -->
## ROSClaw Runtime Boundary (managed)

- Project: `rosclaw-v1.0`
- Project root: `/home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0`
- Default robot: (none detected)
- MCP transport: `stdio`

This project exposes a P0 ROSClaw MCP server. Connect via the configured `stdio` transport defined in `.mcp.json`.

### Safety contract (P0)

Read-only / simulation / emergency tools only:

| Tool | Safety level | Purpose |
|------|--------------|---------|
| `get_robot_state` | S0 read-only | Current body state and readiness |
| `list_skills` | S0 read-only | Skills available to the runtime |
| `query_memory` | S0 read-only | Retrieve similar past experiences |
| `practice_query` | S0 read-only | Query practice episodes |
| `validate_trajectory` | S2 validated-plan | Plan validation, never real motion |
| `sandbox_run` | S1 simulation-only | MuJoCo simulation only |
| `emergency_stop` | S4 emergency | Halt all motion immediately |
| `list_bodies` | S0 read-only | List registered bodies in the workspace |
| `get_body` | S0 read-only | Get registry entry and effective body snapshot |
| `switch_body` | S0 config | Change the active body pointer (no motion) |
| `list_body_history` | S0 read-only | List body snapshot history |
| `check_skill_compatibility` | S0 read-only | Check skill compatibility for the current body |
| `fleet_skill_compatibility` | S0 read-only | Aggregate skill compatibility across all bodies |

There is **no real-execution tool** in P0. Any request to move the real robot
must be refused or routed through `validate_trajectory` + operator confirmation.
<!-- ROSCLAW-MANAGED-END -->

## Human notes

### Body module architecture

ROSClaw uses a three-layer body model so the Agent always knows what physical
robot it is running on and which skills are safe to execute.

| Layer | File / concept | Purpose |
|-------|----------------|---------|
| Physical DNA | `e-urdf-zoo/*` → `EurdfProfile` | Model-level definition: joints, frames, sensors, actuators, capability hints, safety limits. |
| Body Instance Ledger | `~/.rosclaw/body/body.yaml` | This robot's instance state: installed components, overrides, notes, forbidden capabilities, identity. |
| Compiled Body Manual | `~/.rosclaw/body/EMBODIMENT.md` | Agent-readable rendered summary generated from the **Effective Body Model**. |

The **Effective Body Model** is the single source of truth. It is produced by
`EffectiveBodyCompiler` by merging e-URDF, `body.yaml`, `calibration.yaml`, and
`maintenance.log` events. Do not read `body.yaml` directly from other modules;
consume body state through `BodyResolver` or `rosclaw://` URIs.

### Cross-module body references

Always use the resolver API or URIs:

```python
from rosclaw.body.resolver import BodyResolver

resolver = BodyResolver()
body = resolver.get_effective_body()
```

Supported `rosclaw://` URIs:

| URI | Returns |
|-----|---------|
| `rosclaw://body/current` | current `BodyYaml` |
| `rosclaw://body/current/effective` | current `EffectiveBody` |
| `rosclaw://body/current/calibration` | `CalibrationYaml` |
| `rosclaw://body/current/maintenance` | list of `MaintenanceEvent` |
| `rosclaw://body/current/capabilities` | capability dict |
| `rosclaw://eurdf/{id}@{version}` | `EurdfProfile` for the linked body |

### Skill / body compatibility safety contract

`SkillExecutor` checks every skill against the current effective body before
execution. Status semantics:

- `compatible` — execute normally.
- `degraded` — execute only if the skill manifest declares the degradation
  acceptable (e.g. `allow_uncalibrated_camera`).
- `blocked` — refuse execution.
- `unknown` — **fail-closed**: refuse execution until the skill manifest is
  available and compatibility is resolved.

If `BodyResolver` raises an exception during the check, execution is blocked.
Do not change this to fail-open. If no body is linked, execution proceeds for
backward compatibility but logs a warning.
