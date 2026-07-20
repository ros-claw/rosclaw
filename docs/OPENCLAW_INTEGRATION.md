# OpenClaw Integration

ROSClaw integrates with OpenClaw in two distinct ways:

1. a workspace skill teaches the Agent the ROSClaw safety and evidence workflow;
2. OpenClaw's operator-owned MCP registry exposes the bounded ROSClaw tool
   surface to OpenClaw-managed runs.

The project installer creates the workspace files. It deliberately does not
edit the user's global OpenClaw configuration.

## Safety Boundary

The current P0 Agent surface is read-only, validation, simulation, product
evidence, and emergency control. It contains no real-execution tool.

The legacy `rosclaw-ur5-mcp` entry point remains available for compatibility,
but its motion handlers fail closed behind the runtime action gateway. Do not
use it as a direct hardware integration path. A real action must enter through
`Runtime.submit_action()` with an immutable body snapshot, scoped
authorization, safety policy, a registered verified executor, feedback, and an
execution receipt.

## 1. Install ROSClaw Into an OpenClaw Workspace

Run this from the OpenClaw workspace:

```bash
rosclaw agent install openclaw --project-root . --skip-secrets
```

This writes:

- `.agents/skills/rosclaw/SKILL.md`, which OpenClaw discovers as a workspace
  skill;
- `ROSCLAW.md` and `AGENTS.md`, which state the runtime and deny boundaries;
- `.rosclaw/agent/context.snapshot.json`, which records the advertised tools
  and safety levels;
- cross-agent Codex and Claude Code files for the same project.

Start a new OpenClaw session after installation so the workspace skill is
included in the session's skill snapshot.

Validate the generated project files and the ROSClaw stdio server:

```bash
rosclaw agent doctor openclaw --project-root .
rosclaw agent test openclaw --project-root . --quick --mcp-probe
```

The second command runs a protocol-level probe and a real MuJoCo
`ur5e-reach` evidence workflow. It never contacts physical hardware.

## 2. Register the Native MCP Server

OpenClaw stores outbound MCP servers in its global operator configuration.
Review the command and run it explicitly from the intended workspace:

```bash
openclaw mcp add rosclaw \
  --command rosclaw \
  --arg mcp \
  --arg serve \
  --arg=--project \
  --arg "$PWD" \
  --arg=--log-level \
  --arg ERROR \
  --cwd "$PWD" \
  --env ROSCLAW_AGENT_CLIENT=openclaw \
  --env ROSCLAW_MCP_AUDIT=1 \
  --timeout 300 \
  --include \
  'get_robot_state,list_skills,query_memory,validate_trajectory,sandbox_run,practice_query,emergency_stop,get_body_profile,get_body_state,list_body_capabilities,query_body,validate_body_action,get_calibration_status,get_product_status,list_product_demos,run_product_demo,get_execution_receipt,explain_execution'
openclaw mcp doctor rosclaw --probe
```

`openclaw mcp doctor rosclaw --probe` must report a live server with exactly
the 18 tools listed above. If it reports more tools, remove the server and
investigate before giving it to an Agent:

```bash
openclaw mcp unset rosclaw
```

The project installer does not run these global commands because it cannot
know which OpenClaw agent, workspace, or security policy should own the server.

## 3. Agent Evidence Workflow

Ask the OpenClaw Agent to use the ROSClaw tools, not shell commands or direct
device APIs:

1. call `get_product_status`;
2. call `list_product_demos`;
3. call `run_product_demo` with `demo_id="ur5e-reach"`;
4. call `get_execution_receipt` with the returned run ID;
5. call `explain_execution` with the same run ID.

A valid result reports:

- execution mode `SIMULATION`;
- policy decision `ALLOW`;
- MuJoCo physics steps greater than zero;
- a verified task result;
- an integrity-checked receipt;
- `usable_for_real_execution: false`.

Do not infer success from conversational text. The receipt and its integrity
check are the product evidence.

## Tool Boundary

| Category | Tools | Meaning |
|---|---|---|
| Read-only | `get_robot_state`, `list_skills`, `query_memory`, `practice_query` | Inspect runtime state and evidence |
| Validation/simulation | `validate_trajectory`, `sandbox_run` | Validate or simulate; never authorize real motion |
| Emergency | `emergency_stop` | Request an immediate stop |
| Body context | `get_body_profile`, `get_body_state`, `list_body_capabilities`, `query_body`, `validate_body_action`, `get_calibration_status` | Inspect and validate the effective body |
| Product evidence | `get_product_status`, `list_product_demos`, `run_product_demo`, `get_execution_receipt`, `explain_execution` | Run and explain official evidence workflows |

OpenClaw's MCP registry and skill behavior can change between releases. Check
the current OpenClaw MCP and Skills documentation before automating global
installation.

## Current Verification Status

- The workspace skill and ROSClaw MCP protocol probe are covered by automated
  tests.
- The full product evidence workflow has been exercised through an independent
  local Codex CLI process.
- OpenClaw native registry configuration is documented but has not been
  independently verified in this repository's CI because OpenClaw is not a
  test dependency.
- Agent-controlled real actuation is not verified and remains unavailable from
  the P0 MCP surface.
