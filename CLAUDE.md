# CLAUDE.md — ROSClaw Project Onboarding

This file tells Claude Code how to work with the ROSClaw physical-AI runtime.
It is safe to edit the human sections below. Managed blocks are updated by
`rosclaw agent init claude-code` and should not be hand-edited.

<!-- ROSCLAW-MANAGED-BEGIN -->
## ROSClaw Runtime Boundary (managed)

- Project: `rosclaw_repo`
- Project root: `.`
- Default robot: (none detected)
- MCP transport: `stdio`
- Pinned ROSClaw CLI: `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint`

This project exposes a P0 ROSClaw MCP server. Connect via the configured `stdio` transport defined in `.mcp.json`.

Claude Code must approve the project-scoped `rosclaw` server from `.mcp.json`
before it can use these tools. Approval is owned by Claude Code and is not
bypassed by the ROSClaw installer.

### Safety contract (P0)

Read-only, body-context, simulation, guarded-action, and emergency tools:

| Tool | Safety level | Purpose |
|------|--------------|---------|
| `get_robot_state` | S0 | Current body state and readiness |
| `list_skills` | S0 | Skills available to the runtime |
| `query_memory` | S0 | Retrieve similar past experiences |
| `validate_trajectory` | S2 | Plan validation, never real motion |
| `sandbox_run` | S1 | MuJoCo simulation preview only |
| `practice_query` | S0 | Query practice episodes |
| `emergency_stop` | S4 | Request daemon E-Stop; verify physical-stop evidence |
| `get_body_profile` | S0 | Static effective body profile |
| `get_body_state` | S0 | Body safety state and capability matrix |
| `list_body_capabilities` | S0 | Capabilities grouped by status |
| `query_body` | S0 | Answer questions about the current body |
| `validate_body_action` | S0 | Validate proposed body-level action |
| `get_calibration_status` | S0 | Calibration status for body components |
| `get_runtime_status` | S0 | rosclawd health and privilege-boundary status |
| `request_action` | S3 | Submit a guarded SHADOW or REAL action to rosclawd |
| `get_action_status` | S0 | Read daemon queue state and terminal receipt |
| `cancel_action` | S3 | Cancel queued work; active motion requires E-Stop |
| `get_product_status` | S0 | Canonical release and evidence boundary |
| `list_product_demos` | S0 | Official evidence-bearing simulation demos |
| `run_product_demo` | S1 | Run an official simulation and persist its receipt |
| `get_execution_receipt` | S0 | Read and integrity-check a receipt |
| `explain_execution` | S0 | Explain policy, execution, observation, and evidence |

`request_action` is a request to `rosclawd`, not execution authority. A REAL
request succeeds only when the daemon independently matches a server-issued,
body- and action-intent-bound permit. Never instantiate a local Runtime,
register a driver, or use ROS, DDS, serial, CAN, or a vendor SDK as an
alternate motion path.

### Robot Integration setup

Robot Integration installation and configuration are operator CLI workflows,
separate from the MCP tool surface. It is safe to inspect discovery and signed contracts
with `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint robot discover --json`, `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint robot install realsense --json`, and
`/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint robot verify realsense --stage contract --json`. Installing native
adapter dependencies, binding a live serial number, and running read-only
hardware verification require an explicit operator request. A local successful
check is candidate evidence only and must not be reported as canonical support.

### Capability Apps

Apps are capability-only task manifests, not drivers or permissions. Inspect
them with `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint app list` and `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint app validate <APP> --json`. Run an App
only through `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint app run <APP> --body <BODY> --mode SHADOW --json` unless the
operator explicitly establishes every REAL prerequisite. App installation does
not install hardware, arm rosclawd, issue a Permit, or prove execution.

## LeRobot Bridge v1.0.1

LeRobot is a policy backend inside ROSClaw. Do not operate its worker,
robot transport, executor, serial device, or vendor SDK directly.

### Discovery

1. Call `get_product_status`.
2. Call `get_runtime_status`.
3. Call `get_body_profile` and `get_body_state`.
4. Call `get_calibration_status`.
5. Check that `rh56.single_step` is available before requesting RH56 motion.

### Supported Reference Path

- Policy: `rosclaw_rh56_reference`
- Bodies: `inspire_rh56_left`, `inspire_rh56_right`
- Modes: proposal-only, SHADOW, single-step REAL
- REAL is submitted only through MCP `request_action`.

### Safety

- Use SHADOW before REAL.
- Never run `rosclaw lerobot rollout execute` from an Agent process.
- Never open serial/CAN devices directly.
- Never create or approve a Permit.
- Read `get_execution_receipt` and `explain_execution` before claiming success.

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
