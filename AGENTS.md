# ROSClaw Agent Instructions

ROSClaw is physical-AI runtime infrastructure. Treat robot, ROS, actuator,
motor, and hardware commands as safety-sensitive. This file is safe for Codex,
Claude Code, OpenClaw, and other agent frameworks that read project guidance.

<!-- ROSCLAW-MANAGED-BEGIN -->
## Runtime boundary

- Project root: `.`
- Default robot: (none detected)
- MCP transport: `stdio`
- Pinned ROSClaw CLI: `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint`; use it instead of another `rosclaw` on `PATH`.
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

## Robot Integration setup

- Robot Integration setup is an operator CLI workflow, not an MCP tool. Use
  `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint robot discover --json`, `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint robot install realsense --json`, and
  `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint robot verify realsense --stage contract --json` for read-only
  discovery and signed-contract checks.
- Install native adapter dependencies or bind a live device only when the
  operator explicitly requests it. Offline configuration is not physical
  evidence, and local verification never promotes canonical support status.

## Capability Apps

- Apps are capability-only task manifests. Use `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint app list` and
  `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint app validate <APP> --json` for inspection.
- App installation grants no hardware access. Every step remains subject to
  rosclawd Session, Lease, Permit, policy, executor, and Receipt checks.

## rosclawd read-only inspection

- Inspect daemon health and the durable control ledger with
  `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint daemon status --json`. Require `running` and
  `ledger.integrity_verified` to be `true`; require `ledger.write_failed`,
  `recovery.required`, and `emergency_stop_latched` to be `false` before
  relying on the control plane.
- Production REAL work also requires `supervision_state=ARMED`,
  `privilege_separated=true`, and `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint daemon security-check --json` with `boundary_ready=true`,
  `daemon_uid_pinned=true`, and `ledger_state_private=true`. Same-UID
  development proves only a process boundary.
- Inspect an existing action with `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint daemon action-status <ACTION_ID>
  --json` and `/code/rosclaw/rosclaw_lerobot/rosclaw_repo/.venv/bin/python -m rosclaw.entrypoint daemon receipt <ACTION_ID> --json`.
- `daemon acknowledge-recovery` is an operator incident-review command. Do
  not invoke it as routine Agent automation; it does not clear E-stop or prove
  physical state.
- Session or Action Lease loss is terminal according to orphan policy. Do not
  recreate direct motion or reuse a Session from an earlier daemon generation.

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
