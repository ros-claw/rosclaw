"""Template generators for project-level agent onboarding files."""

from __future__ import annotations

import json
from typing import Any

from rosclaw.agent.detectors import ProjectProfile

# Markers used by the managed-block merge engine.
MANAGED_BEGIN = "<!-- ROSCLAW-MANAGED-BEGIN -->"
MANAGED_END = "<!-- ROSCLAW-MANAGED-END -->"
JSON_MANAGED_BEGIN = "/* ROSCLAW-MANAGED-BEGIN */"
JSON_MANAGED_END = "/* ROSCLAW-MANAGED-END */"


def render_mcp_json(profile: ProjectProfile, check: bool = False) -> dict[str, Any]:
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
            ],
            "env": {
                "ROSCLAW_HOME": "${ROSCLAW_HOME:-~/.rosclaw}",
                "ROSCLAW_PROFILE": "${ROSCLAW_PROFILE:-default}",
                "ROSCLAW_AGENT_CLIENT": "claude-code",
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
                "X-ROSClaw-Agent": "claude-code",
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
    robot_line = f"- Default robot: `{profile.robot_id}`" if profile.robot_id else "- Default robot: (none detected)"
    transport = profile.default_transport
    mcp_section = (
        "This project exposes a P0 ROSClaw MCP server. Connect via the configured "
        f"`{transport}` transport defined in `.mcp.json`."
    )
    return f"""# CLAUDE.md — ROSClaw Project Onboarding

This file tells Claude Code how to work with the ROSClaw physical-AI runtime.
It is safe to edit the human sections below. Managed blocks are updated by
`rosclaw agent init claude-code` and should not be hand-edited.

{MANAGED_BEGIN}
## ROSClaw Runtime Boundary (managed)

- Project: `{profile.project_root.name}`
- Project root: `{profile.project_root}`
{robot_line}
- MCP transport: `{transport}`

{mcp_section}

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

There is **no real-execution tool** in P0. Any request to move the real robot
must be refused or routed through `validate_trajectory` + operator confirmation.
{MANAGED_END}

## Human notes

_Add your own project-specific instructions here. They will be preserved across
`rosclaw agent init` runs._
"""


def render_rosclaw_md(profile: ProjectProfile) -> str:
    """Render the ROSCLAW.md file with runtime instructions and guardrails."""
    robot_line = f"- **Robot ID:** `{profile.robot_id}`" if profile.robot_id else "- **Robot ID:** (none detected)"
    return f"""# ROSCLAW.md — Physical AI Runtime Manifest

This file is the authoritative boundary description for the ROSClaw runtime in
this project. It is read by Claude Code on every session start. Human edits
outside managed blocks are preserved by `rosclaw agent init claude-code`.

{MANAGED_BEGIN}
## Runtime profile (managed)

- **Project root:** `{profile.project_root}`
- **MCP transport:** `{profile.default_transport}`
{robot_line}

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
            "available": [
                "get_robot_state",
                "list_skills",
                "query_memory",
                "validate_trajectory",
                "sandbox_run",
                "practice_query",
                "emergency_stop",
            ],
            "safety_levels": {
                "get_robot_state": "S0",
                "list_skills": "S0",
                "query_memory": "S0",
                "practice_query": "S0",
                "validate_trajectory": "S2",
                "sandbox_run": "S1",
                "emergency_stop": "S4",
            },
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
