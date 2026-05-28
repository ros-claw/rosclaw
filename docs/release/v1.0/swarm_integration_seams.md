# Swarm Integration Seams

> **Owner**: Swarm Domain (rosclaw-swarm)
> **Date**: 2026-05-28
> **Status**: v1.0 FROZEN — integration seams for v1.1 consumers
> **RFC Reference**: RFC-0001 Architecture Freeze §2.2 EventBus

---

## Purpose

This document tells other ROSClaw modules **how to integrate with Swarm in v1.1** without breaking v1.0 contracts. Swarm passed the v1.0 audit — no code changes needed. These seams are the extension points.

---

## 1. Metadata Extension Fields

Swarm models expose `metadata: Dict[str, Any]` on three core types. These are the **open extension points** for cross-module integration.

### SwarmContext.metadata

```python
class SwarmContext(BaseModel):
    swarm_session_id: str
    topology: List[AgentCapabilities]
    current_token: Optional[TaskToken]
    task_graph: Optional[TaskGraph]
    dds_domain_id: int = 42
    shared_world_frame: str = "swarm_world_001"
    metadata: Dict[str, Any] = Field(default_factory=dict)  # <-- EXTENSION
```

**v1.1 usage by module:**

| Consumer | Key | Value | Purpose |
|----------|-----|-------|---------|
| Practice | `practice.mcap_path` | `str` | MCAP recording path for this swarm session |
| Practice | `practice.timeline_id` | `str` | UnifiedTimeline ID binding |
| Memory | `memory.experience_graph_id` | `str` | SeekDB experience graph for collective memory |
| Dashboard | `dashboard.session_url` | `str` | Live status URL for dashboard rendering |
| Firewall | `firewall.safety_level` | `str` | Safety validation level for reflex mode |

### Task.metadata

```python
class Task(BaseModel):
    id: str
    dependencies: List[str]
    capability: Optional[str]
    status: TaskStatus
    token: Optional[TaskToken]
    metadata: Dict[str, Any] = Field(default_factory=dict)  # <-- EXTENSION
```

**v1.1 usage by module:**

| Consumer | Key | Value | Purpose |
|----------|-----|-------|---------|
| Practice | `practice.praxis_id` | `str` | Link task execution to PraxisEvent |
| Memory | `memory.experience_id` | `str` | Link task outcome to experience graph node |
| SkillManager | `skill_manager.skill_id` | `str` | Bound skill for execution |
| How | `how.heuristic_applied` | `str` | Recovery heuristic used on failure |

### AgentCapabilities.metadata

```python
class AgentCapabilities(BaseModel):
    agent_id: str
    hardware_type: str
    capabilities: List[Capability]
    metadata: Dict[str, Any] = Field(default_factory=dict)  # <-- EXTENSION
```

**v1.1 usage by module:**

| Consumer | Key | Value | Purpose |
|----------|-----|-------|---------|
| e-URDF | `eurdf.zoo_path` | `str` | Path to e-URDF robot profile |
| e-URDF | `eurdf.capabilities_yaml` | `str` | Source file for capabilities |
| Memory | `memory.agent_profile_id` | `str` | Agent's experience profile in SeekDB |
| Dashboard | `dashboard.agent_status_url` | `str` | Per-agent status endpoint |

---

## 2. EventBus Topic Structure

### Swarm topics in the monorepo EventBus

These topics are used by `src/rosclaw/swarm/manager.py` in the monorepo:

| Topic | Direction | Publisher/Subscriber | Payload |
|-------|-----------|---------------------|---------|
| `swarm.register` | IN | Subscribe | Agent registration request |
| `swarm.allocate` | IN | Subscribe | Agent allocation request |
| `swarm.status` | IN | Subscribe | Status query request |
| `swarm.agent_registered` | OUT | Publish | Agent registration confirmation |
| `swarm.allocate_result` | OUT | Publish | Allocation result with assignments |
| `swarm.status_result` | OUT | Publish | Status query result |
| `swarm.message` | OUT | Publish | Inter-robot message (consumed by practice/timeline) |

### Internal events (standalone package)

The `rosclaw-swarm` standalone package uses internal Python events (not EventBus topics):

| Event Name | Payload | When Fired |
|------------|---------|------------|
| `SwarmSessionCreatedEvent` | `SwarmContext` | After `create_session()` |
| `SwarmTaskAssignedEvent` | `{session_id, assignments}` | After `schedule_session()` |
| `SwarmContextActivatedEvent` | `SwarmContext` | After `activate_reflex()` |

### Topic namespace — no conflicts detected

| Namespace | Owner | Conflicts with `swarm.*`? |
|-----------|-------|--------------------------|
| `agent.*` | agent_runtime | ❌ No |
| `firewall.*` | firewall | ❌ No |
| `memory.*` | memory | ❌ No |
| `praxis.*` | practice | ❌ No |
| `robot.*` | core/runtime | ❌ No |
| `runtime.*` | core/runtime | ❌ No |
| `safety.*` | firewall | ❌ No |
| `skill.*` | skill_manager | ❌ No |
| `swarm.*` | **swarm** | — (owner) |
| `timeline.*` | practice | ❌ No |
| `knowledge.*` | know (future) | ❌ No |
| `heuristic.*` | how (future) | ❌ No |

**All swarm topics are namespaced under `swarm.*`.** No conflicts with any existing or planned topics.

---

## 3. Pydantic Contract Freeze List

These models are **frozen for v1.0**. Changes require an RFC amendment.

| Model | File | Fields | Freeze Status |
|-------|------|--------|---------------|
| `TaskStatus` | models.py:15-21 | Enum: PENDING, ASSIGNED, RUNNING, COMPLETED, FAILED, CANCELLED | FROZEN |
| `Capability` | models.py:24-31 | name, skill, success_rate, latency_ms, metadata | FROZEN |
| `AgentCapabilities` | models.py:34-47 | agent_id, hardware_type, dof, payload_limit_kg, capabilities, active_topics, pose, battery_level, load, risk_score, metadata | FROZEN |
| `TaskToken` | models.py:50-57 | task_id, action_type, target_object_id, parameters, assigned_agent_id | FROZEN |
| `Task` | models.py:60-77 | id, parent_id, dependencies, skill, capability, description, token, status, priority, estimated_duration_ms, metadata | FROZEN |
| `TaskGraph` | models.py:80-99 | graph_id, goal, tasks, root_task_id | FROZEN |
| `SwarmContext` | models.py:102-111 | swarm_session_id, topology, current_token, task_graph, dds_domain_id, shared_world_frame, metadata | FROZEN |
| `SwarmReflexMessage` | models.py:114-125 | stamp_ns, sender_agent_id, expected_tf_offsets, current_tcp_pose, current_tcp_velocity, actual_wrench, joint_torques, intent_phase, confidence | FROZEN |

### Breaking change protocol

If a module needs to change a frozen model:

1. File an RFC amendment against RFC-0001
2. Get sign-off from Swarm domain owner
3. Add new fields as `Optional` with defaults — never remove or rename existing fields
4. The `metadata` dict is the escape hatch for non-breaking extensions

---

## 4. How Other Modules Integrate (v1.1 Guide)

### Practice -> Swarm

Practice records swarm sessions for replay and analysis.

```python
# practice/timeline.py subscribes to swarm events:
self.bus.subscribe("swarm.message", self._on_swarm_message)

# On swarm session creation, bind MCAP recording:
def on_swarm_session_created(self, ctx: SwarmContext):
    mcap_path = self.start_recording(
        session_id=ctx.swarm_session_id,
        channels=["swarm.reflex", "swarm.intent"],
    )
    ctx.metadata["practice.mcap_path"] = mcap_path
```

### Memory -> Swarm

Memory stores collective experience from swarm sessions.

```python
# memory/interface.py subscribes to swarm completion:
self.bus.subscribe("swarm.allocate_result", self._on_swarm_result)

# Store team pattern:
def on_swarm_complete(self, session_id, result):
    self.store_team_pattern(
        agents=result["agents"],
        task=result["task_graph"]["goal"],
        success=result["all_succeeded"],
    )
```

### Dashboard -> Swarm

Dashboard shows live swarm status.

```python
# Dashboard subscribes to status events:
self.bus.subscribe("swarm.status_result", self._update_swarm_panel)
self.bus.subscribe("swarm.agent_registered", self._update_agent_list)
```

### Know -> Swarm

Knowledge graph informs swarm planning.

```python
# Before scheduling, query knowledge graph:
def enrich_task_graph(self, task_graph: TaskGraph):
    for task in task_graph.tasks:
        hints = self.know.get_heuristics(task.capability)
        task.metadata["knowledge.hints"] = hints
```

### How -> Swarm

Heuristic recovery on swarm task failure.

```python
# On task failure, query How for recovery:
self.bus.subscribe("swarm.allocate_result", self._check_failures)

def on_task_failed(self, task_id, session_id):
    heuristic = self.how.find_recovery(task_id)
    if heuristic:
        task.metadata["how.heuristic_applied"] = heuristic.id
```

---

## 5. Integration Contract

Any module integrating with Swarm MUST:

1. **Never import directly** from `rosclaw_swarm.*` — use EventBus topics
2. **Use metadata dicts** for module-specific data — never add fields to frozen models
3. **Namespace EventBus topics** — use `<module>.<event>` format (e.g., `memory.swarm.experience`)
4. **Handle optional fields** — all metadata values are `Optional`; check before use
5. **Respect the freeze** — propose changes via RFC, not pull requests

---

## Related Documents

- [Audit Report: Swarm](audits/audit-swarm.md)
- [Event Flow Map](integration/event-flow-map.md)
- [Dependency Map](integration/dependency-map.md)
- [RFC-0001: Architecture Freeze](RFC-0001-architecture-freeze.md)
- [v1.1 Integration Checklist](swarm_v11_checklist.md)
