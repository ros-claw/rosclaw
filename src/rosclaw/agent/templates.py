"""Template generators for project-level agent onboarding files."""

from __future__ import annotations

import json
from typing import Any

from rosclaw.agent.detectors import ProjectProfile
from rosclaw.agent.tool_catalog import (
    P0_AGENT_MCP_TOOLS,
    P0_BODY_CONTEXT_TOOLS,
    P0_CONTROL_PLANE_TOOLS,
    P0_CORE_TOOLS,
    P0_PRODUCT_TOOLS,
    compact_safety_level,
)

# Markers used by the managed-block merge engine.
MANAGED_BEGIN = "<!-- ROSCLAW-MANAGED-BEGIN -->"
MANAGED_END = "<!-- ROSCLAW-MANAGED-END -->"
JSON_MANAGED_BEGIN = "/* ROSCLAW-MANAGED-BEGIN */"
JSON_MANAGED_END = "/* ROSCLAW-MANAGED-END */"
CODEX_MANAGED_BEGIN = "# ROSCLAW-MANAGED-BEGIN"
CODEX_MANAGED_END = "# ROSCLAW-MANAGED-END"


def _tool_table(tool_names: tuple[str, ...]) -> str:
    purposes = {
        "get_robot_state": "Current body state and readiness",
        "list_skills": "Skills available to the runtime",
        "query_memory": "Retrieve similar past experiences",
        "practice_query": "Query practice episodes",
        "validate_trajectory": "Plan validation, never real motion",
        "sandbox_run": "MuJoCo simulation preview only",
        "emergency_stop": "Request daemon E-Stop; verify physical-stop evidence",
        "get_runtime_status": "rosclawd health and privilege-boundary status",
        "request_action": "Submit a guarded SHADOW or REAL action to rosclawd",
        "get_action_status": "Read daemon queue state and terminal receipt",
        "cancel_action": "Cancel queued work; active motion requires E-Stop",
        "get_body_profile": "Static effective body profile",
        "get_body_state": "Body safety state and capability matrix",
        "list_body_capabilities": "Capabilities grouped by status",
        "query_body": "Answer questions about the current body",
        "validate_body_action": "Validate proposed body-level action",
        "get_calibration_status": "Calibration status for body components",
        "get_product_status": "Canonical release and evidence boundary",
        "list_product_demos": "Official evidence-bearing simulation demos",
        "run_product_demo": "Run an official simulation and persist its receipt",
        "get_execution_receipt": "Read and integrity-check a receipt",
        "explain_execution": "Explain policy, execution, observation, and evidence",
    }
    lines = ["| Tool | Safety level | Purpose |", "|------|--------------|---------|"]
    for tool in tool_names:
        lines.append(f"| `{tool}` | {compact_safety_level(tool)} | {purposes[tool]} |")
    return "\n".join(lines)


def render_mcp_json(
    profile: ProjectProfile,
    check: bool = False,
    agent_client: str = "claude-code",
) -> dict[str, Any]:
    """Render the .mcp.json configuration object.

    Aligns with the Claude Code project-level MCP format documented in
    ``rosclaw_agent接入优化.md``.
    """
    server_name = "rosclaw"
    transport = profile.default_transport
    if transport == "stdio":
        server = {
            "type": "stdio",
            "command": "rosclaw",
            "args": [
                "mcp",
                "serve",
                "--profile",
                "${ROSCLAW_PROFILE:-default}",
                "--project",
                "${PWD}",
                "--log-level",
                "${ROSCLAW_LOG_LEVEL:-ERROR}",
            ],
            "env": {
                "ROSCLAW_HOME": "${ROSCLAW_HOME:-~/.rosclaw}",
                "ROSCLAW_PROFILE": "${ROSCLAW_PROFILE:-default}",
                "ROSCLAW_AGENT_CLIENT": agent_client,
                "ROSCLAW_MCP_AUDIT": "1",
            },
            "timeout": 300000,
        }
    else:
        host = profile.runtime_profile.get("mcp", {}).get("host", "127.0.0.1")
        port = profile.runtime_profile.get("mcp", {}).get("port", 9090)
        server = {
            "type": "http",
            "url": "${ROSCLAW_MCP_URL:-http://" + f"{host}:{port}" + "/mcp}",
            "headers": {
                "X-ROSClaw-Agent": agent_client,
                "X-ROSClaw-Profile": "${ROSCLAW_PROFILE:-default}",
            },
            "timeout": 300000,
        }

    mcp_config: dict[str, Any] = {
        "mcpServers": {
            server_name: server,
        },
        "rosclaw": {
            "schema_version": "rosclaw.agent.context.v2",
            "robot_id": profile.robot_id,
            "transport": transport,
            "project_root": str(profile.project_root),
        },
    }
    if check:
        mcp_config["rosclaw"]["check_mode"] = True
    return mcp_config


def render_codex_config_toml(profile: ProjectProfile) -> str:
    """Render the project-scoped Codex MCP configuration."""
    enabled_tools = "\n".join(f'  "{tool}",' for tool in P0_AGENT_MCP_TOOLS)
    common = f"""enabled = true
required = false
startup_timeout_sec = 30.0
tool_timeout_sec = 300.0
default_tools_approval_mode = "approve"
enabled_tools = [
{enabled_tools}
]"""

    if profile.default_transport == "stdio":
        server = f"""[mcp_servers.rosclaw]
command = "rosclaw"
args = ["mcp", "serve", "--project", ".", "--log-level", "ERROR"]
env_vars = ["ROSCLAW_HOME", "ROSCLAW_PROFILE", "ROSCLAW_LOG_LEVEL"]
{common}

[mcp_servers.rosclaw.env]
ROSCLAW_AGENT_CLIENT = "codex"
ROSCLAW_MCP_AUDIT = "1"
"""
    else:
        host = profile.runtime_profile.get("mcp", {}).get("host", "127.0.0.1")
        port = profile.runtime_profile.get("mcp", {}).get("port", 9090)
        server = f"""[mcp_servers.rosclaw]
url = "http://{host}:{port}/mcp"
http_headers = {{ X-ROSClaw-Agent = "codex" }}
{common}"""

    return f"""{CODEX_MANAGED_BEGIN}
# Project-scoped ROSClaw MCP configuration for trusted Codex workspaces.
{server}
{CODEX_MANAGED_END}
"""


def render_claude_md(profile: ProjectProfile) -> str:
    """Render the project-level CLAUDE.md onboarding file."""
    robot_line = (
        f"- Default robot: `{profile.robot_id}`"
        if profile.robot_id
        else "- Default robot: (none detected)"
    )
    transport = profile.default_transport
    mcp_section = (
        "This project exposes a P0 ROSClaw MCP server. Connect via the configured "
        f"`{transport}` transport defined in `.mcp.json`."
    )
    return f"""# CLAUDE.md — ROSClaw Project Onboarding

This file tells Claude Code how to work with the ROSClaw physical-AI runtime.
It is safe to edit the human sections below. Managed blocks are updated by
`rosclaw agent init claude-code` or `rosclaw agent install` and should not be
hand-edited.

{MANAGED_BEGIN}
## ROSClaw Runtime Boundary (managed)

- Project: `{profile.project_root.name}`
- Project root: `.`
{robot_line}
- MCP transport: `{transport}`

{mcp_section}

Claude Code must approve the project-scoped `rosclaw` server from `.mcp.json`
before it can use these tools. Approval is owned by Claude Code and is not
bypassed by the ROSClaw installer.

### Safety contract (P0)

Read-only, body-context, simulation, guarded-action, and emergency tools:

{_tool_table(P0_AGENT_MCP_TOOLS)}

`request_action` is a request to `rosclawd`, not execution authority. A REAL
request succeeds only when the daemon independently matches a server-issued,
body- and action-intent-bound permit. Never instantiate a local Runtime,
register a driver, or use ROS, DDS, serial, CAN, or a vendor SDK as an
alternate motion path.
{MANAGED_END}

## Human notes

_Add your own project-specific instructions here. They will be preserved across
`rosclaw agent init` runs._
"""


def render_rosclaw_md(profile: ProjectProfile) -> str:
    """Render the ROSCLAW.md file with runtime instructions and guardrails."""
    robot_line = (
        f"- **Robot ID:** `{profile.robot_id}`"
        if profile.robot_id
        else "- **Robot ID:** (none detected)"
    )
    return f"""# ROSCLAW.md — Physical AI Runtime Manifest

This file is the authoritative boundary description for the ROSClaw runtime in
this project. It is read by agent clients on session start. Human edits outside
managed blocks are preserved by `rosclaw agent install`.

{MANAGED_BEGIN}
## Runtime profile (managed)

- **Project root:** `.`
- **MCP transport:** `{profile.default_transport}`
{robot_line}

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

The MCP server exposes {len(P0_AGENT_MCP_TOOLS)} tools: {", ".join(f"`{t}`" for t in P0_AGENT_MCP_TOOLS)}.
Core safety tools: {", ".join(f"`{t}`" for t in P0_CORE_TOOLS)}.
Body context tools: {", ".join(f"`{t}`" for t in P0_BODY_CONTEXT_TOOLS)}.
Product workflow tools: {", ".join(f"`{t}`" for t in P0_PRODUCT_TOOLS)}.
Control-plane tools: {", ".join(f"`{t}`" for t in P0_CONTROL_PLANE_TOOLS)}.

## Validate-before-motion workflow

1. Call `get_runtime_status` and inspect the effective Body and Capability.
2. Propose and validate motion via `validate_trajectory`.
3. `validate_trajectory` returns `{{"is_safe": true}}` **only** when the plan
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
{MANAGED_END}

## Maintainer notes

_Add operational notes here. They will be preserved across init runs._
"""


def render_agents_md(profile: ProjectProfile) -> str:
    """Render cross-agent instructions for Codex and other AGENTS.md readers."""
    robot_line = (
        f"- Default robot: `{profile.robot_id}`"
        if profile.robot_id
        else "- Default robot: (none detected)"
    )
    return f"""# ROSClaw Agent Instructions

ROSClaw is physical-AI runtime infrastructure. Treat robot, ROS, actuator,
motor, and hardware commands as safety-sensitive. This file is safe for Codex,
Claude Code, OpenClaw, and other agent frameworks that read project guidance.

{MANAGED_BEGIN}
## Runtime boundary

- Project root: `.`
{robot_line}
- MCP transport: `{profile.default_transport}`
- One-line setup: `rosclaw agent install --project-root . --skip-secrets`
- Codex activation: trust this exact repository, then run
  `rosclaw agent doctor codex --project-root .`.
- Claude Code activation: approve the project `rosclaw` MCP server.

## Tool policy

The ROSClaw MCP server exposes {len(P0_AGENT_MCP_TOOLS)} read-only, body-context,
simulation, validation, guarded-action, and emergency tools. It exposes no raw
hardware primitive and gives the Agent no authority to approve its own REAL
request.

{_tool_table(P0_AGENT_MCP_TOOLS)}

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
{MANAGED_END}

## Human notes

Add project-specific notes here. They will be preserved across install runs.
"""


def render_rosclaw_skill_md(profile: ProjectProfile) -> str:
    """Render a repo-local Codex/agent skill for ROSClaw projects."""
    return f"""---
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

- Expected P0 tools: {", ".join(f"`{tool}`" for tool in P0_AGENT_MCP_TOOLS)}.
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
"""


def render_claude_settings_json(profile: ProjectProfile) -> dict[str, Any]:
    """Render the .claude/settings.json object with safety deny rules."""
    return {
        "version": "1.0.0",
        "rosclaw": {
            "schema_version": "p0.2025-06-19",
            "robot_id": profile.robot_id,
        },
        "permissions": {
            "deny": [
                "Bash(rostopic pub *)",
                "Bash(ros2 topic pub *)",
                "Bash(rosrun * cmd_vel*)",
                "Bash(roslaunch *teleop*)",
                "Bash(sudo *motor*)",
                "Bash(sudo *ros*)",
            ],
        },
        "autoMemoryEnabled": True,
    }


def render_context_snapshot(profile: ProjectProfile) -> dict[str, Any]:
    """Render the machine-readable .rosclaw/agent/context.snapshot.json."""
    return {
        "schema_version": "rosclaw.agent.context.v2",
        "project": {
            "name": profile.project_root.name,
            "root": str(profile.project_root),
            "has_pyproject": profile.has_pyproject,
            "has_rosclaw_source": profile.has_rosclaw_src,
        },
        "runtime": {
            "robot_id": profile.robot_id,
            "profile_path": str(profile.profile_path) if profile.profile_path else None,
            "transport": profile.default_transport,
            "mcp": profile.runtime_profile.get("mcp", {}),
        },
        "tools": {
            "available": list(P0_AGENT_MCP_TOOLS),
            "safety_levels": {tool: compact_safety_level(tool) for tool in P0_AGENT_MCP_TOOLS},
        },
        "policies": {
            "direct_hardware_access": False,
            "real_execution_requires_rosclawd_permit": True,
            "agent_may_self_authorize": False,
            "fixture_allowed_for_real": False,
            "validate_before_motion": True,
            "emergency_stop_available": True,
        },
    }


def json_to_string(data: dict[str, Any], indent: int = 2) -> str:
    """Serialize a dict to a stable JSON string."""
    return json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=True) + "\n"
