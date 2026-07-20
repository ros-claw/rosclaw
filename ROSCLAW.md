# ROSCLAW.md — Physical AI Runtime Manifest

This file is the authoritative boundary description for the ROSClaw runtime in
this project. It is read by Claude Code on every session start. Human edits
outside managed blocks are preserved by `rosclaw agent init claude-code`.

<!-- ROSCLAW-MANAGED-BEGIN -->
## Runtime profile (managed)

- **Project root:** `.`
- **MCP transport:** `stdio`
- **Robot ID:** (none detected)

## Agent harness activation

- Codex loads `.codex/config.toml` only after this exact repository root is
  trusted. Reopen the repository, accept workspace trust, and verify with
  `rosclaw agent doctor codex --project-root .`.
- Claude Code requires approval of the project-scoped `rosclaw` server from
  `.mcp.json`. Open the project once, approve it, and verify with
  `claude mcp get rosclaw`.
- OpenClaw discovers the workspace skill in `.agents/skills`; native OpenClaw
  MCP registration remains operator-owned.

## Agent tool surface

The MCP server exposes 22 tools: `get_robot_state`, `list_skills`, `query_memory`, `validate_trajectory`, `sandbox_run`, `practice_query`, `emergency_stop`, `get_body_profile`, `get_body_state`, `list_body_capabilities`, `query_body`, `validate_body_action`, `get_calibration_status`, `get_runtime_status`, `request_action`, `get_action_status`, `cancel_action`, `get_product_status`, `list_product_demos`, `run_product_demo`, `get_execution_receipt`, `explain_execution`.
Core safety tools: `get_robot_state`, `list_skills`, `query_memory`, `validate_trajectory`, `sandbox_run`, `practice_query`, `emergency_stop`.
Body context tools: `get_body_profile`, `get_body_state`, `list_body_capabilities`, `query_body`, `validate_body_action`, `get_calibration_status`.
Product workflow tools: `get_product_status`, `list_product_demos`, `run_product_demo`, `get_execution_receipt`, `explain_execution`.
Control-plane tools: `get_runtime_status`, `request_action`, `get_action_status`, `cancel_action`.

## Validate-before-motion workflow

1. Call `get_runtime_status` and inspect the effective Body and Capability.
2. Propose and validate motion via `validate_trajectory`.
3. `validate_trajectory` returns `{"is_safe": true}` **only** when the plan
   passes the firewall gate and sandbox simulation.
4. A safe validation result and conversational human confirmation are not
   execution authorization. Submit SHADOW or REAL work only with
   `request_action`; `rosclawd` independently validates a peer-, body-,
   snapshot-, capability-, and action-intent-bound permit.
5. Use `get_action_status` for the terminal `ExecutionReceipt`. `cancel_action`
   cancels only queued work; active motion requires `emergency_stop`.
6. `sandbox_run` may be used to preview physics in MuJoCo; it never commands
   real hardware.
7. On unexpected behavior, call `emergency_stop` and follow local E-stop
   procedures.
8. Never instantiate `Runtime`, register a driver/executor, publish a command
   topic, or open a device/SDK from the Agent process.

## Deny rules

Claude Code must never run these commands directly:

- `rostopic pub /cmd_vel ...`
- `ros2 topic pub /cmd_vel ...`
- Any direct motor/DDS/hardware write, including after operator confirmation
- Any `sudo` command on the robot host without explicit justification
<!-- ROSCLAW-MANAGED-END -->

## Maintainer notes

_Add operational notes here. They will be preserved across init runs._
