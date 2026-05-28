# ROSClaw v1.0 — Skills and Practice Recording

> **Prerequisites**: `01_getting_started.md`  
> **Time**: ~35 minutes  
> **Difficulty**: Beginner

---

## What You'll Learn

- Define and register custom skills
- Execute skills with parameters
- Record practice sessions (PraxisEvent)
- Query past experiences from SeekDB
- Compose skills into higher-level behaviors

---

## Overview

Skills are the atomic actions a robot can perform -- pick, place, move_home. Practice recording captures every skill execution so the robot (and its LLM planner) can learn from successes and failures.

In this tutorial you will build a small pick-and-place workflow, record each attempt, and query the results.

---

## Step 1: Define Your First Skill

A skill is described by a `SkillEntry`. The executor maps the name to a Python function.

```python
#!/usr/bin/env python3
"""Tutorial 02: Define and register a skill."""

from rosclaw.core import EventBus
from rosclaw.skill_manager import SkillRegistry, SkillEntry, SkillExecutor


# 1. Create infrastructure
bus = EventBus()
registry = SkillRegistry()
registry.initialize()

# 2. Define skill metadata
pick_skill = SkillEntry(
    name="pick",
    description="Pick an object from a known location",
    skill_type="programmed",
    parameters={"object": "str", "location": "str"},
)
registry.register(pick_skill)

print(f"Registered skills: {registry.list_skills()}")
print(f"Pick parameters: {registry.get('pick').parameters}")

registry.stop()
```

**Expected output**:

```
Registered skills: ['pick']
Pick parameters: {'object': 'str', 'location': 'str'}
```

---

## Step 2: Execute a Skill with a Handler

`SkillExecutor` connects skill names to actual Python callables.

```python
#!/usr/bin/env python3
"""Tutorial 02: Execute a skill with a handler."""

from rosclaw.core import EventBus
from rosclaw.skill_manager import SkillRegistry, SkillEntry, SkillExecutor


# 1. Handlers
bus = EventBus()
registry = SkillRegistry()
registry.initialize()

executor = SkillExecutor(bus, registry)
executor.initialize()

# 2. Register handler
@executor.handler("pick")
def handle_pick(params: dict) -> dict:
    obj = params.get("object", "unknown")
    loc = params.get("location", "unknown")
    print(f"  [Pick] Grabbing {obj} from {loc}")
    return {"status": "success", "object": obj}

# 3. Execute
result = executor.execute("pick", {"object": "red_cube", "location": "shelf_A"})
print(f"Result: {result}")

executor.stop()
registry.stop()
```

**Expected output**:

```
  [Pick] Grabbing red_cube from shelf_A
Result: {'status': 'success', 'object': 'red_cube'}
```

---

## Step 3: Record a Practice Session

Every skill execution should be recorded as a `PraxisEvent`. This creates training data for future learning.

```python
#!/usr/bin/env python3
"""Tutorial 02: Record a practice session."""

import numpy as np
from rosclaw.core import EventBus
from rosclaw.practice import PracticeRecorder
from rosclaw.core.types import PraxisEvent, RobotState


bus = EventBus()
recorder = PracticeRecorder(robot_id="tutorial_bot", event_bus=bus)
recorder.initialize()
recorder.start()

# Simulate a pick attempt
evt = PraxisEvent(
    event_id="pick_001",
    event_type="success",
    timestamp=0.0,
    robot_id="tutorial_bot",
    agent_instruction="pick red_cube from shelf_A",
    cot_trace=["plan approach", "grasp", "lift"],
    initial_state=RobotState(
        timestamp=0.0,
        joint_positions=np.zeros(6),
        joint_velocities=np.zeros(6),
        joint_torques=np.zeros(6),
    ),
    final_state=RobotState(
        timestamp=2.5,
        joint_positions=np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0]),
        joint_velocities=np.zeros(6),
        joint_torques=np.zeros(6),
    ),
    trajectory=[np.zeros(6), np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])],
    mcap_path=None,
    error_details=None,
    duration_sec=2.5,
)

recorder.record_praxis_event(evt)
print("Practice event recorded.")

recorder.stop()
```

**Expected output**:

```
Practice event recorded.
```

---

## Step 4: Query Past Experiences

Use SeekDB (via `MemoryInterface`) to retrieve and analyze recorded events.

```python
#!/usr/bin/env python3
"""Tutorial 02: Query past experiences from SeekDB."""

from rosclaw.core import EventBus
from rosclaw.memory import MemoryInterface


bus = EventBus()
memory = MemoryInterface("tutorial_bot", event_bus=bus)
memory.initialize()

# Store a few experiences
memory.store_experience("e1", "praxis", "pick red_cube", outcome="success", tags=["grasp"])
memory.store_experience("e2", "praxis", "pick blue_cube", outcome="failure", tags=["grasp", "slip"])
memory.store_experience("e3", "praxis", "place red_cube", outcome="success", tags=["place"])

# Query all experiences
all_exp = memory.list_experiences()
print(f"Total experiences: {len(all_exp)}")

# Find similar experiences
similar = memory.find_similar_experiences("pick cube")
print(f"Similar experiences: {len(similar)}")
for exp in similar:
    print(f"  - {exp['instruction']} ({exp['outcome']})")

# Statistics
stats = memory.get_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")

memory.stop()
```

**Expected output**:

```
Total experiences: 3
Similar experiences: 2
  - pick red_cube (success)
  - pick blue_cube (failure)
Success rate: 66.7%
```

---

## Step 5: Compose Skills into a Workflow

Higher-level behaviors combine multiple skills. Use the EventBus to coordinate between them.

```python
#!/usr/bin/env python3
"""Tutorial 02: Compose skills into a pick-and-place workflow."""

from rosclaw.core import EventBus
from rosclaw.skill_manager import SkillRegistry, SkillEntry, SkillExecutor


bus = EventBus()
registry = SkillRegistry()
registry.initialize()

# Register skills
registry.register(SkillEntry(name="pick",   description="Pick object",   skill_type="programmed", parameters={"object": "str"}))
registry.register(SkillEntry(name="place",  description="Place object",  skill_type="programmed", parameters={"location": "str"}))
registry.register(SkillEntry(name="verify", description="Verify placement", skill_type="programmed"))

executor = SkillExecutor(bus, registry)
executor.initialize()

@executor.handler("pick")
def do_pick(p):  print(f"  Picking {p['object']}");       return {"status": "ok"}

@executor.handler("place")
def do_place(p): print(f"  Placing at {p['location']}"); return {"status": "ok"}

@executor.handler("verify")
def do_verify(p): print("  Verifying...");               return {"status": "ok"}


# Workflow
def pick_and_place(object_name: str, location: str) -> dict:
    r1 = executor.execute("pick",   {"object": object_name})
    r2 = executor.execute("place",  {"location": location})
    r3 = executor.execute("verify", {})
    return {"steps": [r1, r2, r3], "final_status": r3["status"]}


result = pick_and_place("red_cube", "table_B")
print(f"Workflow complete: {result['final_status']}")

executor.stop()
registry.stop()
```

**Expected output**:

```
  Picking red_cube
  Placing at table_B
  Verifying...
Workflow complete: ok
```

---

## Complete Example

A full script that ties everything together: skills, execution, recording, and querying.

```python
#!/usr/bin/env python3
"""Tutorial 02: Complete skills and practice example."""

import numpy as np
from rosclaw.core import EventBus
from rosclaw.skill_manager import SkillRegistry, SkillEntry, SkillExecutor
from rosclaw.practice import PracticeRecorder
from rosclaw.memory import MemoryInterface
from rosclaw.core.types import PraxisEvent, RobotState


def main():
    bus = EventBus()

    # Skills
    registry = SkillRegistry()
    registry.initialize()
    registry.register(SkillEntry(name="pick", description="Pick object", skill_type="programmed"))

    executor = SkillExecutor(bus, registry)
    executor.initialize()

    @executor.handler("pick")
    def handle_pick(p):
        print(f"  Executing pick({p})")
        return {"status": "success"}

    # Recorder
    recorder = PracticeRecorder(robot_id="bot", event_bus=bus)
    recorder.initialize()
    recorder.start()

    # Execute and record
    result = executor.execute("pick", {"object": "cube"})

    recorder.record_praxis_event(PraxisEvent(
        event_id="run_001",
        event_type="success" if result["status"] == "success" else "failure",
        robot_id="bot",
        agent_instruction="pick cube",
        duration_sec=1.2,
        initial_state=RobotState(timestamp=0.0, joint_positions=np.zeros(6), joint_velocities=np.zeros(6), joint_torques=np.zeros(6)),
        final_state=RobotState(timestamp=1.2, joint_positions=np.zeros(6), joint_velocities=np.zeros(6), joint_torques=np.zeros(6)),
    ))

    # Memory
    memory = MemoryInterface("bot", event_bus=bus)
    memory.initialize()
    print(f"Experiences: {len(memory.list_experiences())}")

    # Cleanup
    recorder.stop()
    executor.stop()
    registry.stop()
    memory.stop()


if __name__ == "__main__":
    main()
```

---

## Try It Yourself

1. **Add a failure handler**: Create a `place` skill that returns `"failure"` 50% of the time. Record the outcomes and compute success rate.

2. **Skill parameters**: Extend the `pick` skill to accept `grip_force` (float). Validate that `grip_force` is between 0 and 100 before executing.

3. **Experience replay**: Query the last 3 failed experiences and print their instructions. Use this to generate a "retry list".

---

## Next Steps

- `03_runtime_lifecycle.md` -- Understand the 8-state lifecycle machine
- `docs/API_REFERENCE.md` -- Full SkillRegistry and PracticeRecorder API
- `examples/hello_robot.py` -- See skills in a full Runtime context

---

## Common Issues

**Skill not found during execution**

Ensure the skill is registered **before** calling `executor.execute()`:

```python
registry.register(SkillEntry(name="pick", ...))
executor.execute("pick", {...})   # OK
```

**PracticeRecorder not recording**

Call `recorder.start()` after `recorder.initialize()`:

```python
recorder.initialize()
recorder.start()   # Required!
```

**MemoryInterface returns empty list**

`MemoryInterface` stores experiences in SeekDB asynchronously. Add a small delay after storing before querying:

```python
memory.store_experience(...)
import time; time.sleep(0.1)
results = memory.list_experiences()
```

---

*Ready to orchestrate everything? See `03_runtime_lifecycle.md`.*
