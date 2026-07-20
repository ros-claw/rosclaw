# ROSClaw Agent Instructions

ROSClaw is physical-AI runtime infrastructure. Treat robot, ROS, actuator,
motor, and hardware commands as safety-sensitive. This file is safe for Codex,
Claude Code, OpenClaw, and other agent frameworks that read project guidance.

<!-- ROSCLAW-MANAGED-BEGIN -->
## Runtime boundary

- Project root: `.`
- Default robot: (none detected)
- MCP transport: `stdio`
- One-line setup: `rosclaw agent install --project-root . --skip-secrets`
- Codex activation: trust this exact repository, then run
  `rosclaw agent doctor codex --project-root .`.
- Claude Code activation: approve the project `rosclaw` MCP server.

## Tool policy

The ROSClaw MCP server exposes 22 read-only, body-context,
simulation, validation, guarded-action, and emergency tools. It exposes no raw
hardware primitive and gives the Agent no authority to approve its own REAL
request.

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

## Safety

- Do not publish ROS topics, actuate hardware, run real robot skills, or mutate
  a live robot workspace directly. A user request or human confirmation does
  not make a shell/DDS/vendor command an approved execution path.
- Prefer fixture, mock, simulation, read-only, dry-run, or temporary
  `ROSCLAW_HOME` workflows for validation.
- Submit SHADOW or REAL work only through the MCP `request_action` tool backed
  by `rosclawd`. REAL requires an immutable body snapshot, a daemon-issued
  capability- and action-intent-bound permit, and a verified REAL executor. If
  any are unavailable, refuse the action and explain the missing prerequisite.
- Never instantiate `Runtime`, register a driver/executor, or open ROS, DDS,
  serial, CAN, or a vendor SDK from the Agent process.
<!-- ROSCLAW-MANAGED-END -->

## Repository checks

```bash
python -m compileall -q src tests
ruff check .
ruff format --check .
mypy --config-file .github/mypy-ci.ini src/rosclaw/daemon src/rosclaw/mcp/adapters src/rosclaw/mcp/onboarding src/rosclaw/agent_runtime/mcp_hub.py src/rosclaw/connectors/ros/mcp/tools.py src/rosclaw/core/runtime.py src/rosclaw/cli.py src/rosclaw/body src/rosclaw/firstboot src/rosclaw/hub
pytest tests/practice -q
```

Use the repo skill at `.agents/skills/rosclaw/SKILL.md` for deeper CLI,
Practice evidence, MCP, and agent-framework workflows.
