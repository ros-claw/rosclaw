# CLAUDE.md — ROSClaw Project Onboarding

This file tells Claude Code how to work with the ROSClaw physical-AI runtime.
It is safe to edit the human sections below. Managed blocks are updated by
`rosclaw agent init claude-code` and should not be hand-edited.

<!-- ROSCLAW-MANAGED-BEGIN -->
## ROSClaw Runtime Boundary (managed)

- Project: `rosclaw-v1.0`
- Project root: `/home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0`
- Default robot: (none detected)
- MCP transport: `stdio`

This project exposes a P0 ROSClaw MCP server. Connect via the configured `stdio` transport defined in `.mcp.json`.

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
<!-- ROSCLAW-MANAGED-END -->

## Human notes

_Add your own project-specific instructions here. They will be preserved across
`rosclaw agent init` runs._
