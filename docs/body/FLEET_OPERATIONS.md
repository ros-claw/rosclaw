# Fleet Operations

ROSClaw can host multiple physical or simulated robots in the same workspace.
Fleet operations let you observe and safely coordinate all of them without
issuing any real-motion commands.

## Scope

All fleet commands are read-only or emergency-only:

- `rosclaw body fleet-compat` — aggregate skill compatibility across bodies
- `rosclaw fleet status` — list bodies and their readiness
- `rosclaw fleet stop` — broadcast an emergency-stop event for every body

There is no fleet-wide motion or execution tool.

## Aggregate skill compatibility

`rosclaw body fleet-compat` discovers every registered body and every skill
manifest under `workspace/skills/`, then runs the body compatibility checker
against each body. The output shows:

- per-body compatible / degraded / blocked / unknown counts
- fleet-level totals
- skills that are `compatible_with_all` bodies
- skills that are `blocked_on_any` body

```bash
rosclaw body create --robot unitree-g1 --name g1-sim
rosclaw body create --robot unitree-g1 --name g1-real
rosclaw body fleet-compat
rosclaw body fleet-compat --json
```

The JSON report uses schema `rosclaw.fleet_compatibility.v1`.

## Fleet status

`rosclaw fleet status` prints a tabular summary of every registered body,
which body is current, and the latest readiness value from each body's
effective body snapshot.

```bash
rosclaw fleet status
rosclaw fleet status --json
```

## Fleet emergency stop

`rosclaw fleet stop` publishes a `robot.emergency_stop` event once per body.
If the runtime is not running, the command still acknowledges the request and
advises activating each robot's physical E-stop.

```bash
rosclaw fleet stop --reason "maintenance window"
```

## MCP tools

The same operations are exposed through the P0 MCP server:

- `list_bodies`
- `get_body`
- `switch_body`
- `list_body_history`
- `check_skill_compatibility`
- `fleet_skill_compatibility`

See `src/rosclaw/mcp/tools/__init__.py` and `CLAUDE.md` for the safety
classification of each tool.

## Dashboard

The dashboard adds a `/body` page that polls `/api/body` and renders the
current body, registry list, and compatibility summary. The WebSocket snapshot
also includes a `body` section.
