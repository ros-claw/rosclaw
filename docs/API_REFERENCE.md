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
    event_bus=bus,
    output_dir="./practice_data",
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
