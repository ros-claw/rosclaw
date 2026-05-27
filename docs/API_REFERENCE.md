# ROSClaw v1.0 API Reference

> **Quick Start** — All public APIs with correct import paths and usage examples.

---

## Table of Contents

1. [Core Types](#core-types)
2. [EventBus](#eventbus)
3. [Lifecycle](#lifecycle)
4. [Runtime](#runtime)
5. [MCP Hub (Agent Runtime)](#mcp-hub)
6. [e-URDF Parser](#e-urdf-parser)
7. [Firewall](#firewall)
8. [Practice (Timeline)](#practice-timeline)
9. [Memory (SeekDB)](#memory-seekdb)
10. [Skill Manager](#skill-manager)

---

## 1. Core Types

```python
from rosclaw.core.types import RobotState, PraxisEvent, PraxisEventType

# PraxisEventType enum
PraxisEventType.SUCCESS   # "success"
PraxisEventType.FAILURE   # "failure"
PraxisEventType.EMERGENCY # "emergency"
PraxisEventType.MOVE      # "move"
PraxisEventType.GRASP     # "grasp"
```

---

## 2. EventBus

```python
from rosclaw.core.event_bus import EventBus, Event, EventPriority

bus = EventBus()

# Subscribe
def handler(event):
    print(event.payload)
bus.subscribe("agent.command", handler)

# Publish
bus.publish(Event(
    topic="agent.command",
    payload={"action": "move_joints"},
    priority=EventPriority.HIGH,
))

# Await response (async)
import asyncio
response = await bus.await_event("agent.response", timeout=5.0)
```

---

## 3. Lifecycle

```python
from rosclaw.core.lifecycle import LifecycleMixin

class MyModule(LifecycleMixin):
    def _do_initialize(self):
        pass
    def _do_start(self):
        pass
    def _do_stop(self):
        pass

mod = MyModule()
mod.initialize()  # UNINITIALIZED -> READY
mod.start()       # READY -> RUNNING
mod.stop()        # RUNNING -> STOPPED
```

---

## 4. Runtime

```python
from rosclaw.core.runtime import Runtime, RuntimeConfig

config = RuntimeConfig(
    robot_id="ur5e_001",
    robot_model_path="./models/ur5e.urdf",
    safety_level="MODERATE",        # STRICT | MODERATE | LENIENT
    timeline_output_dir="./data",
    seekdb_backend="sqlite",        # "memory" | "sqlite"
    seekdb_path="./seekdb.sqlite",
)

runtime = Runtime(config)
runtime.initialize()
runtime.start()

# Status
status = runtime.get_status()
print(status["modules"])  # {'firewall': True, 'memory': True, ...}

runtime.stop()
```

---

## 5. MCP Hub

```python
from rosclaw.agent_runtime import MCPHub, AgentContext
from rosclaw.core.event_bus import EventBus

bus = EventBus()
hub = MCPHub(event_bus=bus, robot_id="ur5e_001")
hub.initialize()

# Async tool call
result = await hub.handle_tool_call("move_joints", {
    "joint_positions": [0.1, -0.2, 0.5, 0.0, 0.1, 0.0],
    "duration": 2.0,
})

# Context
hub.update_robot_description("UR5e 6-DOF arm")
```

---

## 6. e-URDF Parser

```python
from rosclaw.e_urdf import EURDFParser, RobotModel  # or EUrdfParser (alias)

parser = EURDFParser("./models/ur5e.urdf")
model = parser.get_model()

print(model.name)
print(model.get_joint_names())
print(model.get_joint_limits())
print(model.to_llm_context())  # Natural language description for LLM
```

---

## 7. Firewall

```python
from rosclaw.firewall import FirewallValidator
from rosclaw.core.event_bus import EventBus
from rosclaw.e_urdf import EURDFParser

bus = EventBus()
model = EURDFParser("./models/ur5e.urdf").get_model()

validator = FirewallValidator(
    robot_model=model,
    event_bus=bus,
    mujoco_model_path="./models/ur5e.xml",  # optional
    safety_level="MODERATE",
)
validator.initialize()

# Or validate directly
from rosclaw.firewall.validator import ValidationRequest
response = validator.validate(ValidationRequest(
    request_id="req1",
    robot_id="ur5e_001",
    trajectory=[[0.0]*6, [0.1]*6],
))
print(response.is_safe)
```

---

## 8. Practice (Timeline)

```python
from rosclaw.practice import UnifiedTimeline  # or TimelineChannel
from rosclaw.core.event_bus import EventBus

bus = EventBus()
timeline = UnifiedTimeline(
    robot_id="ur5e_001",
    event_bus=bus,              # REQUIRED
    output_dir="./practice_data",
    enable_mcap=False,          # MCAP recording (not yet implemented)
    buffer_size=100_000,        # Max timeline entries in memory
)
timeline.initialize()
timeline.start()

# Record LLM reasoning
timeline.record_llm_reasoning(
    instruction="pick up block",
    reasoning_steps=["identify", "reach", "grasp"],
    correlation_id="session_1",
)

# Record sensorimotor (1kHz, bypasses EventBus)
timeline.record_sensorimotor(
    joint_positions=[0.1]*6,
    joint_velocities=[0.0]*6,
    joint_torques=[0.5]*6,
    correlation_id="session_1",
)

# Query
print(timeline.get_summary())
entries = timeline.get_entries(correlation_id="session_1")
```

---

## 9. Memory (SeekDB)

```python
from rosclaw.memory import MemoryInterface, SeekDBSQLiteClient  # or SQLiteSeekDB (alias)
from rosclaw.core.event_bus import EventBus

# Option 1: In-memory (testing)
mem = MemoryInterface(robot_id="ur5e_001")

# Option 2: SQLite
client = SeekDBSQLiteClient("./seekdb.sqlite")
mem = MemoryInterface(robot_id="ur5e_001", seekdb_client=client)

mem.initialize()

# Store experience
mem.store_experience(
    event_id="exp1",
    event_type="praxis",
    instruction="pick up red block",
    outcome="success",
    duration_sec=3.2,
    tags=["grasp", "red"],
)

# Query
exp = mem.get_experience("exp1")
similar = mem.find_similar_experiences("pick up block", limit=5)
stats = mem.get_statistics()
```

---

## 10. Skill Manager

```python
from rosclaw.skill_manager import SkillRegistry, SkillExecutor, SkillEntry
from rosclaw.core.event_bus import EventBus

bus = EventBus()
registry = SkillRegistry(event_bus=bus)
executor = SkillExecutor(event_bus=bus, registry=registry)
registry.initialize()
executor.initialize()

# Register a skill
entry = SkillEntry(
    name="pick_and_place",
    description="Pick up object and place it",
    skill_type="programmed",
    parameters={"speed": 0.5},
    preconditions=["gripper:empty"],
)
registry.register(entry)

# Execute
result = executor.execute("pick_and_place", {"target": "red_block"})
print(result["status"])

# List
print(registry.list_skills())
```

---

## Class Name Aliases (Backward Compatible)

| Document Name | Actual Class | Import Path |
|---------------|-------------|-------------|
| `AgentRuntime` | `AgentContext` | `rosclaw.agent_runtime` |
| `EUrdfParser` | `EURDFParser` | `rosclaw.e_urdf` |
| `SQLiteSeekDB` | `SeekDBSQLiteClient` | `rosclaw.memory` |
| `MemorySeekDB` | `SeekDBMemoryClient` | `rosclaw.memory` |

---

## Migration Guide (E2E Issues Resolution)

Issues found during end-to-end testing (see `E2E_TEST_FINDINGS.md`):

### Issue 1: AgentRuntime class name
- **Problem**: Docs referenced `AgentRuntime`, actual class is `AgentContext`
- **Fix**: `AgentRuntime` alias added. Both work:
```python
from rosclaw.agent_runtime import AgentContext   # canonical
from rosclaw.agent_runtime import AgentRuntime   # alias (same class)
```

### Issue 2: EUrdfParser vs EURDFParser casing
- **Problem**: Docs used `EUrdfParser`, actual is `EURDFParser`
- **Fix**: `EUrdfParser` alias added. Both work:
```python
from rosclaw.e_urdf import EURDFParser   # canonical
from rosclaw.e_urdf import EUrdfParser   # alias (same class)
```

### Issue 3: SQLiteSeekDB class name
- **Problem**: Docs used `SQLiteSeekDB`, actual is `SeekDBSQLiteClient`
- **Fix**: `SQLiteSeekDB` alias added. Both work:
```python
from rosclaw.memory import SeekDBSQLiteClient  # canonical
from rosclaw.memory import SQLiteSeekDB         # alias (same class)
```

### Issue 4: PraxisEventType enum
- **Problem**: Tests expected `PraxisEventType.MOVE` but enum didn't exist
- **Fix**: `PraxisEventType` enum added to `core.types`:
```python
from rosclaw.core.types import PraxisEventType
PraxisEventType.SUCCESS   # "success"
PraxisEventType.FAILURE   # "failure"
PraxisEventType.EMERGENCY # "emergency"
PraxisEventType.MOVE      # "move"
PraxisEventType.GRASP     # "grasp"
PraxisEventType.VALIDATE  # "validate"
```
**Note**: `PraxisEvent.event_type` is still `str` — use `PraxisEventType.MOVE.value` to set.

### Issue 5: MCPHub requires event_bus
- **Problem**: `MCPHub()` fails — `event_bus` is required
- **Fix**: Pass `event_bus` (required positional arg):
```python
from rosclaw.core.event_bus import EventBus
from rosclaw.agent_runtime import MCPHub

bus = EventBus()
hub = MCPHub(event_bus=bus, robot_id="my_robot")  # event_bus is REQUIRED
```

### Issue 6: FirewallValidator constructor params
- **Problem**: Constructor needs `robot_model`, `event_bus`, `safety_level`
- **Fix**: See Section 7. Constructor signature:
```python
FirewallValidator(
    robot_model: RobotModel,       # REQUIRED - from EURDFParser.get_model()
    event_bus: EventBus,           # REQUIRED
    mujoco_model_path: str = None, # Optional - for MuJoCo collision check
    safety_level: str = "MODERATE" # STRICT | MODERATE | LENIENT
)
```

### Issue 7: UnifiedTimeline constructor params
- **Problem**: `UnifiedTimeline()` fails — `robot_id` and `event_bus` required
- **Fix**: See Section 8. Constructor signature:
```python
UnifiedTimeline(
    robot_id: str,                 # REQUIRED
    event_bus: EventBus,           # REQUIRED
    output_dir: str = "./practice_data",
    enable_mcap: bool = False,
    buffer_size: int = 100_000,
)
```

### Issue 8: SkillRegistry.register() takes SkillEntry
- **Problem**: `registry.register("name", lambda)` fails
- **Fix**: Pass a `SkillEntry` object:
```python
from rosclaw.skill_manager import SkillRegistry, SkillEntry

entry = SkillEntry(
    name="pick_and_place",
    description="Pick up object and place it",
    skill_type="programmed",
    parameters={"speed": 0.5},
    preconditions=["gripper:empty"],
    handler=my_handler_fn,          # Optional Callable
)
registry.register(entry)
```

### Issue 9: SeekDB API methods
- **Problem**: Docs referenced `store_experience`/`search_similar` on SeekDB
- **Fix**: These methods are on `MemoryInterface`, not `SeekDBClient`:
```python
# MemoryInterface (high-level)
mem.store_experience(event_id, event_type, instruction, outcome, ...)
mem.find_similar_experiences(query, limit=5)

# SeekDBClient (low-level)
client.insert(table, record)
client.query(table, filters)
```

---

## EventBus Topic Registry

| Topic | Publisher | Subscriber(s) | Priority |
|-------|-----------|---------------|----------|
| `agent.command` | MCPHub | FirewallValidator, UnifiedTimeline, Runtime | HIGH |
| `agent.response` | FirewallValidator, Drivers | MCPHub | HIGH |
| `safety.violation` | FirewallValidator | Runtime | CRITICAL |
| `firewall.status` | FirewallValidator | (monitoring) | LOW |
| `praxis.completed` | MCPDriver | UnifiedTimeline, SkillRegistry | NORMAL |
| `praxis.failed` | MCPDriver | UnifiedTimeline, SkillRegistry | HIGH |
| `praxis.recorded` | UnifiedTimeline | MemoryInterface | NORMAL |
| `timeline.status` | UnifiedTimeline | (monitoring) | LOW |
| `memory.status` | MemoryInterface | (monitoring) | LOW |
| `memory.experience.stored` | MemoryInterface | (monitoring) | LOW |
| `skill.execution.start` | SkillExecutor | UnifiedTimeline | NORMAL |
| `skill.execution.complete` | SkillExecutor | UnifiedTimeline, SkillRegistry | NORMAL |
| `swarm.message` | SwarmRuntimeManager | UnifiedTimeline | NORMAL |
| `robot.joint_states` | MCPDrivers | MCPHub | NORMAL |
| `robot.end_effector_pose` | MCPDrivers | MCPHub | NORMAL |
| `robot.emergency_stop` | Runtime | All drivers | CRITICAL |
| `runtime.status` | Runtime | (monitoring) | HIGH |

---

## Module Initialization Order

```
Runtime.__init__()
    +-- EventBus() created              <-- First, all modules depend on it

Runtime._do_initialize()
    +-- EURDFParser(model_path)         <-- Physical model loaded first
    +-- FirewallValidator(              <-- Needs e-URDF model + EventBus
    |       robot_model=model,
    |       event_bus=bus,
    |   )
    |   +-- .initialize()               <-- Subscribe only, NO publish
    +-- MemoryInterface(                <-- Needs EventBus + SeekDB
    |       robot_id=id,
    |       event_bus=bus,
    |       seekdb_client=client,
    |   )
    |   +-- .initialize()               <-- Subscribe only, NO publish
    +-- UnifiedTimeline(                <-- Needs EventBus
    |       robot_id=id,
    |       event_bus=bus,
    |   )
    |   +-- .initialize()               <-- Subscribe only, NO publish
    +-- SkillRegistry(event_bus=bus)
    |   +-- .initialize()
    +-- SkillExecutor(bus, registry)
        +-- .initialize()

Runtime._do_start()
    +-- module.start() for each         <-- MAY publish status events
    +-- bus.publish("runtime.status")   <-- Last, unified ready signal
```

**Rule**: Subscribe in `_do_initialize()`, publish in `_do_start()`. No publishing during init.
