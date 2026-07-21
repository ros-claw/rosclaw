---
name: rosclaw
description: Use when operating, validating, or changing ROSClaw physical-AI runtime workflows, especially CLI smoke tests, Practice evidence loops, body/runtime checks, MCP integration, MuJoCo sandbox verification, and safe ROS or hardware boundaries.
---

# ROSClaw Agent Skill

Use this exact launcher for every ROSClaw CLI command: `rosclaw`. Do not replace
it with a different `rosclaw` found on `PATH`.

## Safety

- Treat ROSClaw as physical-AI infrastructure. Do not publish ROS topics,
  actuate hardware, run real robot skills, or mutate a live workspace directly,
  even when a user asks for that specific action.
- Prefer dry-run, read-only, mock, fixture, simulation, or temp-workspace
  commands for validation.
- Use a temporary `ROSCLAW_HOME` for CLI smoke tests that write persistent state.
- Prefer `--json` for machine checks, then validate the JSON parses.
- The only Agent-side action entry is MCP `request_action`, backed by
  `rosclawd`. Never instantiate a local Runtime or register a hardware
  driver/executor from the Agent process.
- REAL actions require an immutable body snapshot, a daemon-issued
  capability- and action-intent-bound permit, and a registered verified REAL
  executor. Refuse direct ROS, DDS, serial, CAN, SDK, or motor commands.

## First Checks

```bash
rosclaw doctor --json
rosclaw status capabilities
rosclaw daemon status --json
rosclaw demo list
rosclaw demo run ur5e-reach
rosclaw explain latest
rosclaw agent doctor universal --project-root .
rosclaw agent test universal --project-root . --quick --mcp-probe
```

For `daemon status`, require `running` and `ledger.integrity_verified` to be
`true`; require `ledger.write_failed`, `recovery.required`, and
`emergency_stop_latched` to be `false`. Production REAL work also requires
`supervision_state=ARMED`, `privilege_separated=true`, and
`rosclaw daemon security-check --json` with
`boundary_ready=true`, `daemon_uid_pinned=true`, and
`ledger_state_private=true`; same-UID development proves only process separation.
Read an existing durable result with `rosclaw daemon action-status <ACTION_ID>
--json` and `rosclaw daemon receipt <ACTION_ID> --json`. Do not use
`daemon acknowledge-recovery` as Agent automation: it is an operator
incident-review command and does not clear E-stop or prove physical state.

## Robot Integration Workflow

Robot Integration installation/configuration is a CLI lifecycle around the MCP
runtime. It is not an additional MCP tool surface.

```bash
rosclaw robot discover --json
rosclaw robot install realsense --json
rosclaw robot verify realsense --stage contract --json
```

- `robot install` verifies and installs the signed contract. Pass
  `--install-adapter` only when the operator explicitly authorizes dependency
  installation in the selected Python environment.
- Bind a discovered device with `robot configure` only when an exact model and
  stable serial are available. `--allow-offline` creates configuration, never
  hardware evidence.
- A read-only hardware check requires a configured instance and canonical
  `rosclawd` receipt. Treat local success as candidate evidence; never promote
  product support or claim physical success from CLI text alone.
- Do not call the vendor SDK directly. Runtime device access stays behind MCP
  `request_action` and the daemon boundary.

## Capability App Workflow

```bash
rosclaw app list
rosclaw app validate realsense-inspect --json
```

- Apps call named Capabilities only. They do not install drivers, issue
  permits, arm rosclawd, or authorize direct hardware access.
- Run SHADOW first with `app run <APP> --body <BODY> --mode SHADOW --json`.
  Treat a local success as component evidence, never an independent hardware
  or Agent verification claim.

## Practice Evidence Loop

Use fixture-based Practice workflows before touching real robots:

```bash
TMP=$(mktemp -d /tmp/rosclaw-practice.XXXXXX)
export ROSCLAW_HOME="$TMP/home"
rosclaw practice record --fixture tests/fixtures/practice/rh56_minimal_loop.json --out "$TMP/practice" --json
rosclaw practice verify practice_rh56_minimal_loop --data-root "$TMP/practice" --strict --json
rosclaw practice distill practice_rh56_minimal_loop --data-root "$TMP/practice" --json
```

## MCP Contract

- Expected P0 tools: `get_robot_state`, `list_skills`, `query_memory`, `validate_trajectory`, `sandbox_run`, `practice_query`, `emergency_stop`, `get_body_profile`, `get_body_state`, `list_body_capabilities`, `query_body`, `validate_body_action`, `get_calibration_status`, `get_runtime_status`, `request_action`, `get_action_status`, `cancel_action`, `get_product_status`, `list_product_demos`, `run_product_demo`, `get_execution_receipt`, `explain_execution`.
- `run_product_demo` executes an official MuJoCo path and persists an
  integrity-checked `ExecutionReceipt`; it never commands real hardware.
- Use `get_execution_receipt` and `explain_execution` instead of inferring
  success from text output.
- `sandbox_run` is simulation-only. If no physics state is available, treat the
  result as degraded rather than live.
- `validate_trajectory` can approve a plan, but it does not authorize direct
  hardware execution.
- `request_action` sends an unapproved envelope to `rosclawd`; only the daemon
  may turn a matching server-issued permit into authorization.
- `cancel_action` never claims that active physical motion stopped. Use
  `emergency_stop` and the certified hardware E-stop when motion may be active.
