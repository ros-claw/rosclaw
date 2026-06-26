# MCP Body Tools

This document describes the P0 MCP body tools exposed by `rosclaw-mcp-server`.
All tools are read-only or configuration-only; no tool performs real motion.

## Tool inventory

| Tool | Safety level | Purpose |
|------|--------------|---------|
| `list_bodies` | S0 read-only | List registered bodies in the workspace. |
| `get_body` | S0 read-only | Get registry entry and effective body snapshot. |
| `switch_body` | S0 config | Change the active body pointer. No motion. |
| `list_body_history` | S0 read-only | List body snapshot history. |
| `check_skill_compatibility` | S0 read-only | Check skill compatibility for the current body. |
| `fleet_skill_compatibility` | S0 read-only | Aggregate skill compatibility across all bodies. |

## list_bodies

Returns all bodies registered in the current workspace.

```json
{
  "workspace": "/home/user/.rosclaw",
  "bodies": [
    {"body_id": "g1-a", "nickname": "G1 Alpha", "profile_id": "unitree-g1"},
    {"body_id": "g1-b", "nickname": "G1 Beta", "profile_id": "unitree-g1"}
  ]
}
```

## get_body

Returns the active body registry entry plus an Agent View snapshot.

```json
{
  "body_id": "g1-a",
  "effective_body_hash": "abc123...",
  "readiness": "ready",
  "capabilities": [...],
  "summary": "Unitree G1 humanoid, ready for locomotion and manipulation."
}
```

## switch_body

Changes the active body pointer. Accepts an optional `strict` flag.

- `strict=false` (default): runtime hooks are fired but failures are logged.
- `strict=true`: switch fails if any runtime hook fails.

```json
{"body_id": "g1-b", "strict": false}
```

The tool never moves the robot; it only changes which body configuration the
runtime reads.

## list_body_history

Returns recent maintenance events for the active body.

```json
{
  "body_instance_id": "g1-a",
  "count": 5,
  "events": [...]
}
```

## check_skill_compatibility

Returns compatibility for a single skill against the active body.

```json
{
  "skill_id": "walk_forward",
  "status": "compatible",
  "details": {...}
}
```

## fleet_skill_compatibility

Returns aggregated compatibility across all registered bodies, using the
`FleetCompatibilityCache` to avoid recomputation when state is unchanged.

```json
{
  "fleet_hash": "fleet-abc123...",
  "per_body": {
    "g1-a": {"walk_forward": "compatible"},
    "g1-b": {"walk_forward": "blocked"}
  }
}
```

## Implementation notes

- All tools consume the Effective Body Model via `BodyResolver`.
- `switch_body` dispatches `BodySwitchHooks` events.
- `fleet_skill_compatibility` invalidates its cache on body changes, active
  body switches, skill manifest changes, sense-body updates, and provider
  health safety events.

## Safety contract

- No MCP body tool performs real motion.
- `switch_body` only changes configuration pointers.
- Unknown skill compatibility is reported as `blocked`; execution is refused.

## See also

- [BODY_RUNTIME_OVERVIEW.md](BODY_RUNTIME_OVERVIEW.md)
- [SKILL_COMPATIBILITY.md](SKILL_COMPATIBILITY.md)
- [FLEET_OPERATIONS.md](FLEET_OPERATIONS.md)
