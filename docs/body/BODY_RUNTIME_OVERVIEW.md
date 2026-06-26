# Body Runtime Overview

This document describes the unified body runtime in ROSClaw. It is the reference
for how the three-layer body model is compiled, consumed, and kept consistent
across the codebase.

## Three-layer body model

| Layer | Source | Purpose |
|-------|--------|---------|
| Physical DNA | `e-urdf-zoo/*` / `RobotCompleteProfile` | Model-level robot definition: links, joints, sensors, actuators, capabilities, safety limits. |
| Body Instance Ledger | `~/.rosclaw/body/body.yaml` | This robot's instance state: identity, installed components, overrides, forbidden capabilities, notes. |
| Compiled Body Manual | `~/.rosclaw/body/EMBODIMENT.md` | Agent-readable rendered summary generated from the **Effective Body Model**. |

## Effective Body Model

The **Effective Body Model** is the single source of truth. It is produced by
`EffectiveBodyCompiler` by merging, in order of increasing priority:

1. e-URDF profile
2. `body.yaml`
3. `calibration.yaml`
4. `maintenance.log` events

The result is a dataclass (`EffectiveBody`) with:

- `body_instance_id`
- `effective_body_hash` (sha256 of canonical merged state)
- `eurdf_uri`
- `frames`, `joints`, `sensors`, `actuators`
- `capabilities` (enabled / degraded / blocked)
- `safety` limits and emergency contacts
- `readiness` summary

Every consumer of body state reads the Effective Body Model. Do not read
`body.yaml` directly from other modules.

## Key entry points

### CLI

```bash
rosclaw body init --robot unitree-g1 --name g1-a
rosclaw body create --robot unitree-g1 --name g1-a
rosclaw body switch g1-a
rosclaw body show --agent
rosclaw body state --json
```

### Python API

```python
from rosclaw.body.resolver import BodyResolver

resolver = BodyResolver()
body = resolver.get_effective_body()
print(body.effective_body_hash)
```

### URI API

```python
body = resolver.resolve("rosclaw://body/current/effective")
calibration = resolver.resolve("rosclaw://body/current/calibration")
```

## Change propagation

When body state changes, the following sequence is guaranteed:

1. `body.yaml` is updated.
2. A `maintenance.log` event is appended.
3. `EffectiveBodyCompiler` recompiles the Effective Body Model.
4. `EMBODIMENT.md` is re-rendered.
5. `BodySwitchHooks` dispatches lifecycle events:
   - `BODY_EFFECTIVE_CHANGED`
   - `BODY_SKILL_COMPATIBILITY_CHANGED`
6. `FleetCompatibilityCache` invalidates affected entries.
7. Optional memory event is written (`BodyMemoryEventWriter`).

## Fail-closed skill compatibility

`SkillExecutor` checks every skill against the current effective body before
execution. Unknown compatibility is treated as **blocked**. See
[SKILL_COMPATIBILITY.md](SKILL_COMPATIBILITY.md) for status semantics.

## Runtime hooks

`BodySwitchHooks` is the extension point for runtime modules that need to react
to body changes. Subscribers are called with a `BodyHookContext`. Failures are
logged by default; with `strict=True` they propagate and block the operation.

```python
from rosclaw.body.hooks import BodyHookEvent, get_default_hooks

def on_switch(event_type, context):
    print(f"Switched to {context.body_instance_id}")

get_default_hooks().subscribe(BodyHookEvent.BODY_ACTIVE_SWITCHED, on_switch)
```

## Dashboard integration

The dashboard exposes body state via HTTP and WebSocket:

- `/api/body` — registry summary + active body snapshot
- `/api/body/effective` — full Effective Body Model
- `/api/body/skills` — skill compatibility report
- `/api/body/history` — recent maintenance events
- `/api/body/provider-health` — provider interface diagnosis
- `/ws` — live snapshot broadcast

## Multi-body registry

`BodyRegistryManager` supports multiple body instances in a single workspace.
The active body pointer is stored in the registry. Fleet compatibility can be
aggregated across all registered bodies. See
[FLEET_OPERATIONS.md](FLEET_OPERATIONS.md).

## See also

- [EURDF_BODY_CONTRACT.md](EURDF_BODY_CONTRACT.md)
- [AGENT_VIEW.md](AGENT_VIEW.md)
- [MCP_BODY_TOOLS.md](MCP_BODY_TOOLS.md)
- [SANDBOX_PROVIDER_MEMORY_INTEGRATION.md](SANDBOX_PROVIDER_MEMORY_INTEGRATION.md)
