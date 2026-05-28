# ROSClaw v1.0 — Getting Started

> **Prerequisites**: Python 3.10+, `pip`, and a POSIX shell (Linux/macOS/WSL).
> **Time**: ~15 minutes

---

## 1. Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/rosclaw/rosclaw.git
cd rosclaw
pip install -e ".[dev]"
```

Verify the installation:

```bash
python -c "import rosclaw; print(rosclaw.__version__)"
# Expected: 1.0.0
```

Run the test suite to confirm everything is healthy:

```bash
pytest tests/ -q
# Expected: 270 passed
```

---

## 2. Your First Robot Program

The fastest way to understand ROSClaw is to run the included `hello_robot.py` example. It demonstrates the full lifecycle: EventBus → Runtime → Driver → Skills → Practice recording.

```bash
python examples/hello_robot.py
```

Expected output:

```
=== ROSClaw Hello Robot ===

1. EventBus created
2. Runtime started: hello_bot
3. Driver connected: positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
   Moved to: positions=[0.2, -0.3, 0.1, 0.0, 0.0, 0.0]
4. Skills registered: ['pick', 'place']
5. Skill executed: success
6. Practice event recorded
7. Event published to bus

=== Hello Robot complete ===
```

---

## 3. Core Concepts

ROSClaw is built around six **grounding engines** that bridge LLM reasoning and physical robots. All communication flows through the **EventBus** — no direct module-to-module calls.

### 3.1 EventBus (Central Nervous System)

```python
from rosclaw.core import EventBus, Event, EventPriority

bus = EventBus()

# Subscribe to a topic
bus.subscribe("robot.joint_states", lambda e: print(e.payload))

# Publish an event
bus.publish(Event(
    topic="robot.joint_states",
    payload={"positions": [0.1, 0.2, 0.0]},
    source="my_module",
    priority=EventPriority.NORMAL,
))
```

Key features:
- Sync and async subscribers
- Automatic event history (up to 10,000 events)
- Priority-based processing (CRITICAL → BACKGROUND)
- Wait for events with `await bus.await_event("topic")`

### 3.2 Runtime (Orchestrator)

```python
from rosclaw.core import Runtime, RuntimeConfig

config = RuntimeConfig(
    robot_id="my_robot",
    safety_level="MODERATE",
    timeline_output_dir="./practice_data",
    seekdb_backend="memory",
)
runtime = Runtime(config)
runtime.initialize()
runtime.start()
# ... do work ...
runtime.stop()
```

The Runtime manages:
- Lifecycle of all engines
- Safety envelope configuration
- Practice recording output
- Memory backend (memory or SQLite)

### 3.3 MCP Drivers (Robot Interface)

```python
from rosclaw.mcp_drivers import MuJoCoSimDriver

# Mock mode (no MuJoCo model required)
driver = MuJoCoSimDriver(robot_id="arm", model_path="", joint_dof=6)
driver.initialize()
driver.start()

# Read state
positions = driver.get_joint_positions()
velocities = driver.get_joint_velocities()

# Move joints
driver.move_joints([0.1, 0.2, 0.0, 0.0, 0.0, 0.0], duration=1.0)

driver.stop()
```

Drivers implement a common `BaseDriver` interface. ROSClaw provides:
- `MuJoCoSimDriver` — MuJoCo simulation with mock fallback
- `ROS2Driver` — ROS2 interface (requires ROS2 environment)
- `SerialDriver` — Serial device interface

### 3.4 Skills (Task Primitives)

```python
from rosclaw.skill_manager import SkillRegistry, SkillEntry, SkillExecutor

registry = SkillRegistry()
registry.register(SkillEntry(
    name="pick",
    description="Pick an object",
    skill_type="programmed",
    parameters={"object": "str"},
))

executor = SkillExecutor(bus, registry)
executor.initialize()
result = executor.execute("pick", {"object": "red_cube"})
```

Skills can be:
- **programmed** — Hard-coded behavior
- **learned** — Generated from demonstrations or LLM planning
- **composed** — Built from other skills

### 3.5 Practice Recording

```python
from rosclaw.practice import PracticeRecorder
from rosclaw.core.types import PraxisEvent, RobotState

recorder = PracticeRecorder(robot_id="my_robot", event_bus=bus)
recorder.initialize()
recorder.start()

evt = PraxisEvent(
    event_id="evt-001",
    event_type="success",
    robot_id="my_robot",
    agent_instruction="pick red_cube",
    duration_sec=1.2,
)
recorder.record_praxis_event(evt)
```

Practice events are automatically stored in SeekDB and can be retrieved for:
- Experience replay
- Failure analysis
- Skill learning

---

## 4. Running Benchmarks

Measure ROSClaw performance on your machine:

```bash
python benchmarks/run_benchmarks.py
```

This reports:
- EventBus throughput (target: >= 10,000 events/s)
- SeekDB insert/query latency
- SkillRegistry scale performance
- FirewallValidator trajectory validation speed

Results are saved to `benchmarks/results.md`.

---

## 5. Next Steps

| Topic | Where to Go |
|-------|-------------|
| Architecture deep dive | `docs/ARCHITECTURE_AUDIT.md` |
| API reference | `docs/API_REFERENCE.md` |
| Provider integration | `examples/demo_provider_mcp.py` |
| Security model | `docs/SECURITY_AUDIT.md` |
| Contributing | `CONTRIBUTING.md` |
| Roadmap | `docs/ROADMAP_v1.1.md` |

---

## 6. Common Issues

**ImportError with numpy**

If you see `cannot load module more than once per process`, this is a pytest-cov + numpy interaction. Run tests without coverage:

```bash
pytest tests/ -q
```

**MuJoCo not installed**

The `MuJoCoSimDriver` works in mock mode without MuJoCo. To enable full simulation:

```bash
pip install mujoco
```

Then provide a valid model path:

```python
driver = MuJoCoSimDriver(robot_id="arm", model_path="/path/to/robot.xml")
```

---

*Ready to build? See `02_writing_skills.md` for the next tutorial.*
