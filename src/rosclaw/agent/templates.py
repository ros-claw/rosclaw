"""Template generators for project-level agent onboarding files."""

from __future__ import annotations

import json
from typing import Any

from rosclaw.agent.detectors import ProjectProfile
from rosclaw.agent.tool_catalog import (
    P0_AGENT_MCP_TOOLS,
    P0_BODY_CONTEXT_TOOLS,
    P0_CORE_TOOLS,
    compact_safety_level,
)

# Markers used by the managed-block merge engine.
MANAGED_BEGIN = "<!-- ROSCLAW-MANAGED-BEGIN -->"
MANAGED_END = "<!-- ROSCLAW-MANAGED-END -->"
JSON_MANAGED_BEGIN = "/* ROSCLAW-MANAGED-BEGIN */"
JSON_MANAGED_END = "/* ROSCLAW-MANAGED-END */"


def _tool_table(tool_names: tuple[str, ...]) -> str:
    purposes = {
        "get_robot_state": "Current body state and readiness",
        "list_skills": "Skills available to the runtime",
        "query_memory": "Retrieve similar past experiences",
        "practice_query": "Query practice episodes",
        "validate_trajectory": "Plan validation, never real motion",
        "sandbox_run": "MuJoCo simulation preview only",
        "emergency_stop": "Halt all motion immediately",
        "get_body_profile": "Static effective body profile",
        "get_body_state": "Body safety state and capability matrix",
        "list_body_capabilities": "Capabilities grouped by status",
        "query_body": "Answer questions about the current body",
        "validate_body_action": "Validate proposed body-level action",
        "get_calibration_status": "Calibration status for body components",
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
            "schema_version": "rosclaw.agent.context.v1",
            "robot_id": profile.robot_id,
            "transport": transport,
            "project_root": str(profile.project_root),
        },
    }
    if check:
        mcp_config["rosclaw"]["check_mode"] = True
    return mcp_config


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
- Project root: `{profile.project_root}`
{robot_line}
- MCP transport: `{transport}`

{mcp_section}

### Safety contract (P0)

Read-only / body-context / simulation / emergency tools only:

{_tool_table(P0_AGENT_MCP_TOOLS)}

There is **no real-execution tool** in P0. Any request to move the real robot
must be refused or routed through `validate_trajectory` + operator confirmation.
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

- **Project root:** `{profile.project_root}`
- **MCP transport:** `{profile.default_transport}`
{robot_line}

## Agent tool surface

The MCP server exposes {len(P0_AGENT_MCP_TOOLS)} tools: {", ".join(f"`{t}`" for t in P0_AGENT_MCP_TOOLS)}.
Core safety tools: {", ".join(f"`{t}`" for t in P0_CORE_TOOLS)}.
Body context tools: {", ".join(f"`{t}`" for t in P0_BODY_CONTEXT_TOOLS)}.

## Validate-before-motion workflow

1. Agent proposes motion via `validate_trajectory`.
2. `validate_trajectory` returns `{{"is_safe": true}}` **only** when the plan
   passes the firewall gate and sandbox simulation.
3. If safe, the agent **must** ask a human operator before sending any real
   command to ROS/hardware.
4. `sandbox_run` may be used to preview physics in MuJoCo; it never commands
   real hardware.
5. On unexpected behavior, call `emergency_stop` and follow local E-stop
   procedures.

## Deny rules

Claude Code must never run these commands directly:

- `rostopic pub /cmd_vel ...`
- `ros2 topic pub /cmd_vel ...`
- Any direct motor/DDS/hardware write without operator confirmation
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

- Project root: `{profile.project_root}`
{robot_line}
- MCP transport: `{profile.default_transport}`
- One-line setup: `rosclaw agent install --project-root . --skip-secrets`

## Tool policy

The ROSClaw MCP server exposes {len(P0_AGENT_MCP_TOOLS)} read-only, body-context,
simulation, validation, and emergency tools. It exposes no real robot execution
tool in P0.

{_tool_table(P0_AGENT_MCP_TOOLS)}

## Safety

- Do not publish ROS topics, actuate hardware, run real robot skills, or mutate
  a live robot workspace unless the user explicitly requests that exact action.
- Prefer fixture, mock, simulation, read-only, dry-run, or temporary
  `ROSCLAW_HOME` workflows for validation.
- For any motion-related request, validate through `validate_trajectory` and
  require operator confirmation before real execution.
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
  actuate hardware, run real robot skills, or mutate a live workspace unless
  the user explicitly asks for that specific action.
- Prefer dry-run, read-only, mock, fixture, simulation, or temp-workspace
  commands for validation.
- Use a temporary `ROSCLAW_HOME` for CLI smoke tests that write persistent state.
- Prefer `--json` for machine checks, then validate the JSON parses.

## First Checks

```bash
rosclaw doctor --json
rosclaw runtime backends
rosclaw agent doctor claude-code --project-root {profile.project_root}
rosclaw agent test claude-code --project-root {profile.project_root} --quick --mcp-probe
rosclaw sandbox verify --case ur5e-joint-preview --json
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
- `sandbox_run` is simulation-only. If no physics state is available, treat the
  result as degraded rather than live.
- `validate_trajectory` can approve a plan, but it does not authorize direct
  hardware execution.
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
        "schema_version": "rosclaw.agent.context.v1",
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
            "no_real_execution": True,
            "validate_before_motion": True,
            "emergency_stop_available": True,
        },
    }


def json_to_string(data: dict[str, Any], indent: int = 2) -> str:
    """Serialize a dict to a stable JSON string."""
    return json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=True) + "\n"
