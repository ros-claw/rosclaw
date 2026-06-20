# ROSClaw Body Registry

A single ROSClaw workspace can host more than one physical or simulated robot
instance. The **Body Registry** tracks every registered body, which body is
currently active, and how to reach each body's storage directory.

## When to use the registry

- You run a fleet of identical robots (e.g. `g1-01`, `g1-02`).
- You keep a simulation body and a real robot body in the same workspace.
- You want to A/B test two configurations of the same hardware.
- You are migrating from the older single-body `workspace/body/` layout.

## Storage layout

A multi-body workspace looks like this:

```text
~/.rosclaw/
├── body_registry.yaml          # registry metadata and current body pointer
├── body/                       # legacy single-body directory (read-only after migration)
└── bodies/
    ├── g1-sim/
    │   ├── body.yaml
    │   ├── calibration.yaml
    │   ├── maintenance.log
    │   ├── refs/
    │   ├── snapshots/
    │   └── generated/
    └── g1-real/
        └── ...
```

The registry file is intentionally small: it stores only metadata, not the
full body state. Each body keeps its own `body.yaml`, `calibration.yaml`,
`maintenance.log`, snapshots, and generated artifacts under
`bodies/<body_id>/`.

## Registry schema

`body_registry.yaml` uses the `rosclaw.body_registry.v1` schema:

```yaml
schema: rosclaw.body_registry.v1
current_body_id: g1-sim
bodies:
  g1-sim:
    body_id: g1-sim
    nickname: "G1 Simulation"
    profile_id: unitree-g1
    profile_version: latest
    created_at: "2026-06-20T12:34:56.789012"
    updated_at: "2026-06-20T12:34:56.789012"
    path: bodies/g1-sim
    tags: ["sim", "lab"]
    metadata: {}
  g1-real:
    body_id: g1-real
    nickname: "G1 Real"
    profile_id: unitree-g1
    profile_version: "1.0.0"
    created_at: "2026-06-20T12:35:01.234567"
    updated_at: "2026-06-20T12:35:01.234567"
    path: bodies/g1-real
    tags: ["real", "production"]
    metadata: {}
metadata: {}
```

Body IDs are case-insensitive and must match `^[a-zA-Z0-9][a-zA-Z0-9_-]*$`.

## CLI commands

### Initialize or create a body

`body init` is the backward-compatible entry point. If a registry already
exists, `--name` creates a named body; otherwise it initializes the legacy
single-body layout.

```bash
# Legacy single-body flow (backward compatible)
rosclaw body init --robot unitree-g1 --workspace ~/.rosclaw

# Multi-body flow
rosclaw body create --robot unitree-g1 --name g1-sim --nickname "G1 Simulation" --workspace ~/.rosclaw
rosclaw body create --robot unitree-g1 --name g1-real --nickname "G1 Real" --workspace ~/.rosclaw
```

### List bodies

```bash
rosclaw body list --workspace ~/.rosclaw
rosclaw body list --workspace ~/.rosclaw --json
```

Text output marks the current body with `*`:

```text
============================================================
ROSClaw Bodies
============================================================
Current: g1-sim   Total: 2

  [*] g1-sim               G1 Simulation        unitree-g1@latest
  [ ] g1-real              G1 Real              unitree-g1@1.0.0
============================================================
```

### Switch the active body

```bash
rosclaw body switch g1-real --workspace ~/.rosclaw
```

All subsequent commands that do not pass `--body` operate on `g1-real`.

### Target a specific body for one command

Most body subcommands accept `--body`:

```bash
rosclaw body show --body g1-real --workspace ~/.rosclaw
rosclaw body state --body g1-real --json --workspace ~/.rosclaw
rosclaw body fault add --body g1-real --component left_knee --severity high \
  --summary "overheating" --workspace ~/.rosclaw
rosclaw body validate --body g1-sim --workspace ~/.rosclaw
```

### Remove a body

```bash
# Delete a body and all its data
rosclaw body remove g1-sim --workspace ~/.rosclaw

# Move data to bodies/_archive/<body_id>-<timestamp>/
rosclaw body remove g1-sim --archive --workspace ~/.rosclaw
```

Removing the active body automatically points `current_body_id` at another
registered body, or back to `default` if none remain.

## Programmatic usage

### BodyRegistryManager

```python
from pathlib import Path
from rosclaw.body.registry import BodyRegistryManager

manager = BodyRegistryManager(Path.home() / ".rosclaw")

# List bodies, current body first
for entry in manager.list_bodies():
    print(entry.body_id, entry.nickname)

# Switch current body
manager.set_current_body_id("g1-real")

# Create a new body entry and directory scaffold
entry = manager.create_body(
    body_id="g1-test",
    profile_id="unitree-g1",
    nickname="G1 Test Bench",
    tags=["test"],
)

# Statistics
print(manager.stats())
```

### BodyResolver

```python
from rosclaw.body.resolver import BodyResolver

# Current body
resolver = BodyResolver()
print(resolver.body_id, resolver.body_dir)

# Named body
resolver = BodyResolver(body_id="g1-real")
effective = resolver.get_effective_body()
```

`BodyResolver` is the single entry point used by the rest of the system. Other
modules should not construct paths to `body.yaml` manually.

## Backward compatibility

If `body_registry.yaml` does not exist but `workspace/body/` does, the
workspace is treated as a legacy single-body workspace. The first mutating
operation (create, switch, remove, etc.) automatically imports the legacy body
as registry entry `default` with `path: body`. After that, the workspace behaves
like a multi-body workspace.

To explicitly migrate without creating another body:

```bash
rosclaw body list --workspace ~/.rosclaw
```

This loads the registry and materializes the legacy entry on first read.

## Files

- `src/rosclaw/body/registry.py` — `BodyRegistryManager`, schema load/save,
  create/switch/remove/list/stats, legacy migration.
- `src/rosclaw/body/schema.py` — `BodyRegistry` and `BodyRegistryEntry`
  dataclasses.
- `src/rosclaw/body/resolver.py` — `BodyResolver`, body-aware path routing.
- `src/rosclaw/body/cli.py` — `list`, `create`, `switch`, `remove`, `--body`
  flag integration.

## See also

- [URI_SCHEME.md](URI_SCHEME.md) — stable `rosclaw://` references to body
  resources.
- [SKILL_COMPATIBILITY.md](SKILL_COMPATIBILITY.md) — how skills are checked
  against the current effective body.
- [BODY_HISTORY_EXPORT.md](BODY_HISTORY_EXPORT.md) — snapshots, history, and
  export of a body instance.
