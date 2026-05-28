# Event Flow Map

> **Author**: Release Integrator  
> **Date**: 2026-05-28  
> **Status**: DRAFT  
> **RFC Reference**: RFC-0001 Architecture Freeze §2.2 EventBus

---

## Executive Summary

Scanned all `bus.publish()` and `bus.subscribe()` calls in `src/rosclaw/` to map the complete event topology.

**Findings:**
- **24 publish sites** across 9 modules
- **23 subscribe sites** across 8 modules
- **21 unique event topics** in use
- **1 P0 gap**: `praxis.completed` / `praxis.failed` have subscribers but NO publisher — orphan subscribers
- **1 P0 violation**: `mcp/ur5_server.py` bypasses EventBus for firewall (see dependency-map.md)

---

## Event Topology Table

| Topic | Publisher(s) | Subscriber(s) | Flow Status |
|-------|-------------|---------------|-------------|
| `agent.command` | agent_runtime/mcp_hub | core/runtime, practice/recorder, practice/timeline, firewall/validator | OK |
| `agent.response` | firewall/validator | agent_runtime/mcp_hub | OK |
| `firewall.action_blocked` | firewall/validator | core/runtime | OK (KNOW/HOW integration) |
| `firewall.status` | firewall/validator | (none) | Orphan pub |
| `heuristic.recovery_suggested` | core/runtime | (none) | Orphan pub (KNOW/HOW integration) |
| `memory.experience.stored` | memory/interface | (none) | Orphan pub |
| `memory.status` | memory/interface | (none) | Orphan pub |
| `praxis.completed` | **(NONE)** | skill_manager/registry, practice/timeline | **P0: Orphan sub** |
| `praxis.failed` | **(NONE)** | skill_manager/registry, practice/timeline | **P0: Orphan sub** |
| `praxis.recorded` | practice/timeline | memory/interface, skill_manager/registry, practice/timeline | OK |
| `robot.emergency_stop` | core/runtime, agent_runtime/mcp_hub | core/runtime | OK |
| `robot.end_effector_pose` | (via mcp_hub dynamic) | agent_runtime/mcp_hub | OK |
| `robot.joint_states` | (via mcp_hub dynamic) | agent_runtime/mcp_hub | OK |
| `runtime.status` | core/runtime | (none) | Orphan pub |
| `safety.violation` | firewall/validator | core/runtime | OK |
| `skill.execution.complete` | skill_manager/executor | skill_manager/registry, practice/recorder, practice/timeline | OK |
| `skill.execution.start` | skill_manager/executor | practice/recorder, practice/timeline | OK |
| `skill.registered` | skill_manager/registry | (none) | Orphan pub |
| `swarm.agent_registered` | swarm/manager | (none) | Orphan pub |
| `swarm.allocate_result` | swarm/manager | (none) | Orphan pub |
| `swarm.status_result` | swarm/manager | (none) | Orphan pub |
| `timeline.status` | practice/timeline | (none) | Orphan pub |

---

## Module-by-Module Event Map

### core/runtime.py

| Direction | Topic | Trigger |
|-----------|-------|---------|
| PUBLISH | `runtime.status` | On start (state: "running") and stop (state: "shutting_down") |
| PUBLISH | `robot.emergency_stop` | On receiving `safety.violation` |
| SUBSCRIBE | `safety.violation` | Routes to emergency stop |
| SUBSCRIBE | `agent.command` | Logs command, routes to modules |
| SUBSCRIBE | `robot.emergency_stop` | Stops all modules |

### practice/timeline.py (UnifiedTimeline)

| Direction | Topic | Trigger |
|-----------|-------|---------|
| PUBLISH | `timeline.status` | On start |
| PUBLISH | `praxis.recorded` | After assembling timeline entries into PraxisEvent |
| SUBSCRIBE | `agent.command` | Records to AGENT_COMMAND channel |
| SUBSCRIBE | `praxis.completed` | Triggers praxis assembly + MCAP export |
| SUBSCRIBE | `praxis.failed` | Records failure to timeline |
| SUBSCRIBE | `skill.execution.start` | Records skill start event |
| SUBSCRIBE | `skill.execution.complete` | Records skill completion event |
| SUBSCRIBE | `swarm.message` | Records inter-robot messages |

### practice/recorder.py

| Direction | Topic | Trigger |
|-----------|-------|---------|
| SUBSCRIBE | `agent.command` | Records command for replay |
| SUBSCRIBE | `skill.execution.start` | Records skill start |
| SUBSCRIBE | `skill.execution.complete` | Records skill completion |

### memory/interface.py (MemoryInterface)

| Direction | Topic | Trigger |
|-----------|-------|---------|
| PUBLISH | `memory.status` | On start (experience count, embodied status) |
| PUBLISH | `memory.experience.stored` | After storing experience to SeekDB |
| SUBSCRIBE | `praxis.recorded` | Stores PraxisEvent to experience graph |

### firewall/validator.py (FirewallValidator)

| Direction | Topic | Trigger |
|-----------|-------|---------|
| PUBLISH | `firewall.status` | On start |
| PUBLISH | `agent.response` | After successful validation |
| PUBLISH | `safety.violation` | On safety violation (joint limits, collision) |
| SUBSCRIBE | `agent.command` | Intercepts commands for safety check |

### skill_manager/registry.py (SkillRegistry)

| Direction | Topic | Trigger |
|-----------|-------|---------|
| PUBLISH | `skill.registered` | After registering a new skill |
| SUBSCRIBE | `praxis.completed` | Auto-updates skill success rate |
| SUBSCRIBE | `praxis.failed` | Auto-updates skill failure rate |
| SUBSCRIBE | `skill.execution.complete` | Tracks execution counts |

### skill_manager/executor.py (SkillExecutor)

| Direction | Topic | Trigger |
|-----------|-------|---------|
| PUBLISH | `skill.execution.start` | Before executing skill |
| PUBLISH | `skill.execution.complete` | After skill execution (success or fail) |

### agent_runtime/mcp_hub.py (MCP Hub)

| Direction | Topic | Trigger |
|-----------|-------|---------|
| PUBLISH | `agent.command` | On receiving MCP tool call |
| PUBLISH | `robot.emergency_stop` | On agent-initiated emergency |
| PUBLISH | (dynamic topics via `topic=topic`) | Robot state forwarding |
| SUBSCRIBE | `robot.joint_states` | Forwards to agent via MCP |
| SUBSCRIBE | `robot.end_effector_pose` | Forwards to agent via MCP |
| SUBSCRIBE | `agent.response` | Routes response back to agent |

### swarm/manager.py (SwarmRuntimeManager)

| Direction | Topic | Trigger |
|-----------|-------|---------|
| PUBLISH | `swarm.allocate_result` | After allocating agents |
| PUBLISH | `swarm.status_result` | After status query |
| PUBLISH | `swarm.agent_registered` | After agent registration |
| SUBSCRIBE | `swarm.register` | Agent registration request |
| SUBSCRIBE | `swarm.allocate` | Agent allocation request |
| SUBSCRIBE | `swarm.status` | Status query request |

---

## PraxisEvent Flow Verification

### Intended flow (per RFC-0001 §2.2):

```
Agent gives goal
    |
    v
agent_runtime/mcp_hub --[agent.command]--> EventBus
    |
    +---> firewall/validator (subscribes, validates safety)
    |         |
    |         +--[agent.response]--> mcp_hub (if approved)
    |         +--[safety.violation]--> runtime (if blocked)
    |
    +---> practice/recorder (subscribes, records command)
    +---> practice/timeline (subscribes, records to AGENT_COMMAND channel)
    |
    v
skill_manager/executor --[skill.execution.start]--> EventBus
    |                                                    |
    |                                                    +--> practice/recorder
    |                                                    +--> practice/timeline
    v
skill_manager/executor --[skill.execution.complete]--> EventBus
    |                                                      |
    |                                                      +--> skill_manager/registry
    |                                                      +--> practice/recorder
    |                                                      +--> practice/timeline
    v
practice/timeline assembles PraxisEvent
    |
    v
practice/timeline --[praxis.recorded]--> EventBus
    |                                         |
    |                                         +--> memory/interface (stores to SeekDB)
    |                                         +--> skill_manager/registry
    |                                         +--> practice/timeline (self)
    v
[COMPLETE]
```

### Flow gaps found:

**P0 — Orphan subscribers: `praxis.completed` / `praxis.failed`**

| Event | Subscribers | Publisher |
|-------|------------|-----------|
| `praxis.completed` | skill_manager/registry, practice/timeline | **NONE** |
| `praxis.failed` | skill_manager/registry, practice/timeline | **NONE** |

**Impact**: SkillRegistry's auto-update of success/failure rates never fires. Timeline's praxis assembly for completed/failed flows is dead code.

**Root cause**: The executor publishes `skill.execution.complete` but never promotes it to `praxis.completed`. The timeline publishes `praxis.recorded` but not `praxis.completed`.

**Suggested fix**: In `skill_manager/executor.py`, after `skill.execution.complete`, promote to `praxis.completed` / `praxis.failed` based on outcome:

```python
# After skill.execution.complete publish:
self.event_bus.publish(Event(
    topic="praxis.completed" if success else "praxis.failed",
    payload={...skill result...},
    source="skill_executor",
))
```

---

## EventBus Bypass Verification

### Modules that communicate via EventBus (compliant):

- core/runtime: publish + subscribe
- practice/timeline: publish + subscribe
- practice/recorder: subscribe only
- memory/interface: publish + subscribe
- firewall/validator: publish + subscribe
- skill_manager/registry: publish + subscribe
- skill_manager/executor: publish only
- swarm/manager: publish + subscribe
- agent_runtime/mcp_hub: publish + subscribe

### Modules that bypass EventBus (non-compliant):

**P0: mcp/ur5_server.py -> firewall/decorator.py**

```python
# src/rosclaw/mcp/ur5_server.py:8
from rosclaw.firewall.decorator import firewall_protected, SafetyViolation
```

Direct function call, not EventBus. See dependency-map.md for details.

---

## Orphan Events (published but no subscriber)

| Topic | Publisher | Severity |
|-------|-----------|----------|
| `firewall.status` | firewall/validator | P2 |
| `memory.experience.stored` | memory/interface | P2 |
| `memory.status` | memory/interface | P2 |
| `runtime.status` | core/runtime | P2 |
| `skill.registered` | skill_manager/registry | P2 |
| `swarm.agent_registered` | swarm/manager | P2 |
| `swarm.allocate_result` | swarm/manager | P2 |
| `swarm.status_result` | swarm/manager | P2 |
| `timeline.status` | practice/timeline | P2 |

These are status/informational events. Having no subscriber is acceptable for v1.0 — they exist for dashboard/observability integration in v1.1.

---

## Verification Commands

```bash
# List all publish sites
grep -rn "\.publish(" src/rosclaw/ --include="*.py" | grep -v test

# List all subscribe sites
grep -rn "\.subscribe(" src/rosclaw/ --include="*.py" | grep -v test

# Extract all unique topics
grep -rh 'topic="' src/rosclaw/ --include="*.py" | \
  grep -oP 'topic="[^"]+"' | sort -u

# Verify PraxisEvent flow
grep -rn "praxis\." src/rosclaw/ --include="*.py" | grep -v test
```

---

## Recommendations

### Immediate (P0)

1. **Fix orphan `praxis.completed`/`praxis.failed`**: Add publisher in `skill_manager/executor.py` after skill execution completes
2. **Refactor mcp/ur5_server.py firewall bypass**: Move to EventBus-based validation (see dependency-map.md)

### Short-Term (P1)

3. **Add contract tests** for the PraxisEvent flow: verify that `agent.command` -> `skill.execution.*` -> `praxis.recorded` -> `memory.experience.stored` completes end-to-end
4. **Document orphan pub/sub topics** in module docstrings so future integrators know which events are dashboard-reserved

---

## Related Documents

- [RFC-0001: Architecture Freeze](../RFC-0001-architecture-freeze.md)
- [RFC-0005: Acceptance Gates](../RFC-0005-acceptance-gates.md)
- [Dependency Map](./dependency-map.md)
- [Audit Report: Runtime](../audits/audit-runtime.md)
