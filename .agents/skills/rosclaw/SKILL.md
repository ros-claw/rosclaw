---
name: rosclaw
description: Use when operating, validating, or changing ROSClaw physical-AI runtime workflows, especially CLI smoke tests, Practice evidence loops, body/runtime checks, MCP integration, MuJoCo sandbox verification, and safe ROS or hardware boundaries.
---

# ROSClaw Agent Skill

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
rosclaw demo list
rosclaw demo run ur5e-reach
rosclaw explain latest
rosclaw agent doctor universal --project-root .
rosclaw agent test universal --project-root . --quick --mcp-probe
```

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
