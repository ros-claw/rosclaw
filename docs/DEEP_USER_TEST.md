# ROSClaw v1.0 深入用户场景测试报告

> **Date**: 2026-05-28
> **Tester**: rosclaw_qwen (Chief Architecture Reviewer)
> **Method**: 7 real-world user scenarios executed against actual API

---

## Summary

| # | Scenario | Result | Issues Found |
|---|----------|--------|--------------|
| 1 | Connect UR5 + Execute Pick | **PASS** | None |
| 2 | Create Custom Skill + Register | **FAIL** | `list_skills()` returns `list[str]`, not `list[SkillEntry]` |
| 3 | Configure LLM Providers | **PASS** | None |
| 4 | Record Practice Session + Export | **FAIL** | No public `record_agent_command()`, no manual `export_session()` |
| 5 | EventBus Custom Module Comm | **PASS** | Verified via source analysis |
| 6 | Query SeekDB History | **PASS** | Verified via source analysis |
| 7 | Firewall Safety Config | **FAIL** | `RobotModel.joints` needs `JointSpec` objects, not plain dicts |

**Result: 4/7 PASS, 3/7 FAIL** — 5 API-vs-documentation mismatches found

---

## Detailed Findings

### Scenario 1: Connect UR5 + Execute Pick Operation

**Result: PASS**

```python
from rosclaw.core.runtime import Runtime, RuntimeConfig

config = RuntimeConfig(
    robot_id="ur5e_001",
    enable_firewall=False,
    enable_memory=True,
    enable_practice=True,
    safety_level="MODERATE",
)
runtime = Runtime(config)
runtime.initialize()  # OK
runtime.start()       # OK

from rosclaw.agent_runtime import MCPHub
hub = MCPHub(event_bus=runtime.event_bus, robot_id="ur5e_001")
hub.initialize()      # OK - 5 tools registered

status = runtime.get_status()
# -> modules: {'firewall': False, 'memory': True, 'practice': True, 'swarm': False, 'skill_manager': True, 'e_urdf': False}
```

**What works:**
- Runtime lifecycle (init -> start -> stop)
- MCPHub creation with EventBus
- AgentContext attributes (session_id, robot_id)
- Tool registration (move_joints, grasp, get_robot_state, validate_trajectory, emergency_stop)
- Module status reporting

---

### Scenario 2: Create Custom Skill + Register to SkillRegistry

**Result: FAIL** — API mismatch in `list_skills()` return type

**What the user expects:**
```python
skills = registry.list_skills()
for s in skills:
    print(s.name, s.description)  # ERROR: 'str' object has no attribute 'name'
```

**What actually happens:**
```python
# registry.list_skills() returns list[str] (just skill names)
skills = registry.list_skills()
# -> ['custom_pick']  # list of STRINGS, not SkillEntry objects

# To get SkillEntry objects, use registry.get():
entry = registry.get("custom_pick")
print(entry.name, entry.description)  # OK
```

**Root cause**: `SkillRegistry.list_skills()` at `registry.py:116-120` returns `list[str]`:
```python
def list_skills(self, skill_type: Optional[str] = None) -> list[str]:
    """List all registered skill names."""
    if skill_type:
        return [s.name for s in self._skills.values() if s.skill_type == skill_type]
    return list(self._skills.keys())  # Returns strings!
```

**API_REFERENCE.md impact**: Section 11 shows `registry.list_skills()` without specifying return type.

**Fix needed**: Document that `list_skills()` returns `list[str]`, use `registry.get(name)` for `SkillEntry`.

---

### Scenario 3: Configure Different LLM Providers

**Result: PASS**

```python
from rosclaw.agent_runtime.llm_provider import get_provider, list_providers, LLMConfig

# List available providers
available = list_providers()  # -> ['deepseek', 'openai', 'qwen']

# Factory pattern
config = LLMConfig(api_key="sk-...", model="gpt-4o")
provider = get_provider("openai", config)

# Direct construction with kwargs
provider = DeepSeekProvider(api_key="sk-...", model="deepseek-chat", temperature=0.5)

# Backward-compat aliases
from rosclaw.agent_runtime import DeepSeekClient, DeepSeekConfig
# DeepSeekClient -> DeepSeekProvider
# DeepSeekConfig -> LLMConfig

# All providers have identical interface
plan = provider.plan_task("pick up block", {"joints": 6})
analysis = provider.analyze_failure("grasp failed", "timeout")
health = provider.health_check()
```

**What works:**
- Factory function `get_provider(name, config)`
- All 3 providers instantiate correctly
- Uniform interface (plan_task, analyze_failure, health_check)
- Keyword-argument construction
- Backward-compat aliases
- Environment variable support

---

### Scenario 4: Record Practice Session + Export MCAP

**Result: FAIL** — 2 missing public methods

**What the user expects:**
```python
timeline.record_agent_command(     # ERROR: no such method
    action="move_joints",
    joint_positions=[0.1]*6,
)
timeline.export_session("session_A")  # ERROR: no such method
```

**What actually works:**
```python
# Public recording methods:
timeline.record_llm_reasoning(instruction=..., reasoning_steps=..., correlation_id=...)
timeline.record_sensorimotor(joint_positions=..., joint_velocities=..., joint_torques=...)

# Agent commands are recorded AUTOMATICALLY via EventBus:
bus.publish(Event(
    topic="agent.command",
    payload={"action": "move_joints", ...},
    metadata={"request_id": "req_001"},
))
# -> UnifiedTimeline._on_agent_command() records it automatically

# Export is AUTOMATIC on praxis.completed:
bus.publish(Event(
    topic="praxis.completed",
    payload={"correlation_id": "session_A", "instruction": "...", ...},
))
# -> UnifiedTimeline._on_praxis_completed() auto-exports to:
#    output_dir/session_{id}/timeline.jsonl
#    output_dir/session_{id}/sensorimotor.npz
```

**Root causes:**
1. `record_agent_command` is private (`_on_agent_command`, line 124). Commands flow via EventBus.
2. Export is automatic on `praxis.completed` event, triggered by `_export_timeline()` (line 259). No manual export method.
3. MCAP export: `enable_mcap=True` flag exists but MCAP writer not fully implemented (sets `self._mcap_writer = True` on import success, line 103).

**API_REFERENCE.md impact**: Section 9 doesn't show that agent commands and export are EventBus-driven.

---

### Scenario 5: EventBus Custom Module Communication

**Result: PASS** (verified via source analysis)

```python
from rosclaw.core import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin

class MySensorModule(LifecycleMixin):
    def __init__(self, event_bus):
        super().__init__()
        self.event_bus = event_bus

    def _do_initialize(self):
        self.event_bus.subscribe("sensor.read", self._on_read)

    def _on_read(self, event):
        self.event_bus.publish(Event(
            topic="sensor.result",
            payload={"value": 42.0},
            source="my_sensor",
        ))

# Event attributes verified:
# event.topic, event.payload, event.source, event.priority,
# event.event_id, event.timestamp, event.metadata
```

**What works:**
- EventBus subscribe/publish pattern
- Event dataclass with all attributes
- Custom module lifecycle (initialize -> start -> stop)
- EventPriority enum (CRITICAL/HIGH/NORMAL/LOW/BACKGROUND)
- `bus.topics`, `bus.subscriber_count()`, `bus.get_history()`
- `bus.await_event()` for async request-response

---

### Scenario 6: Query SeekDB Historical Experiences

**Result: PASS** (verified via source analysis)

```python
from rosclaw.memory import MemoryInterface, SeekDBMemoryClient

mem = MemoryInterface(robot_id="ur5e_test", seekdb_client=SeekDBMemoryClient())
mem.initialize()

# Store
mem.store_experience(
    event_id="exp_001",
    event_type="praxis",
    instruction="pick up red block",
    outcome="success",
    duration_sec=2.1,
    tags=["pick", "red"],
)

# Query
exp = mem.get_experience("exp_001")       # Returns dict or None
similar = mem.find_similar_experiences("pick up block", limit=5)  # Returns list[dict]
stats = mem.get_statistics()              # Returns dict with total/success/failure counts
```

**What works:**
- `store_experience()` with all parameters
- `get_experience(id)` returns single experience dict
- `find_similar_experiences(query, limit)` uses keyword matching
- `get_statistics()` returns total/success/failure/emergency counts + success_rate
- Auto-ingestion from `praxis.recorded` EventBus events

---

### Scenario 7: Firewall Safety Configuration + Violation Detection

**Result: FAIL** — RobotModel construction not documented

**What the user expects:**
```python
# From API_REFERENCE.md Section 8:
model = EURDFParser("./models/ur5e.urdf").get_model()  # Correct way

# But if creating RobotModel manually:
model = RobotModel(name="test", joints={"j1": {"lower": -3.14, ...}})
# ERROR: joints must be dict[str, JointSpec], not dict[str, dict]
```

**What actually works:**
```python
from rosclaw.e_urdf.parser import RobotModel, JointSpec

# Correct manual construction:
model = RobotModel(
    name="test_robot",
    joints={
        "joint1": JointSpec(name="joint1", joint_type="revolute",
                           parent="base", child="link1",
                           limits={"lower": -3.14, "upper": 3.14, "velocity": 2.0, "effort": 10.0}),
    },
)

# Or use the parser (recommended):
parser = EURDFParser("./models/ur5e.urdf")
model = parser.get_model()

# SafetyEnvelope creation
from rosclaw.firewall.validator import SafetyEnvelope
envelope = SafetyEnvelope.from_robot_model(model, safety_level="MODERATE")

# Validation
from rosclaw.firewall.validator import ValidationRequest
req = ValidationRequest(
    request_id="req_001",
    robot_id="ur5e_001",
    trajectory=[[0.0]*6, [0.5]*6],
    duration_per_waypoint=[1.0],  # Optional: enables velocity check
)
response = validator.validate(req)
# response.is_safe, response.violations, response.warnings
# response.layers_checked, response.simulation_duration_ms
```

**Root cause**: `JointSpec` is a `@dataclass(init=False)` with custom `__init__` accepting `type` as alias for `joint_type` (URDF compat). API_REFERENCE.md doesn't show `JointSpec` construction.

---

## Issues Summary

| # | Module | Issue | Severity | Fix |
|---|--------|-------|----------|-----|
| D-1 | `skill_manager` | `list_skills()` returns `list[str]`, not `list[SkillEntry]` | Medium | Document return type; use `get(name)` for SkillEntry |
| D-2 | `practice` | No public `record_agent_command()` method | Medium | Document: agent commands flow via EventBus `agent.command` |
| D-3 | `practice` | No public `export_session()` method | Medium | Document: export is automatic on `praxis.completed` event |
| D-4 | `e_urdf` | `RobotModel` construction requires `JointSpec` objects | Low | Document `JointSpec` constructor in API_REFERENCE.md |
| D-5 | `practice` | MCAP writer not fully implemented | Low | Document `enable_mcap=True` is experimental |

---

## Recommended API_REFERENCE.md Updates

1. **Section 11 (Skill Manager)**: Add note that `list_skills()` returns `list[str]`
2. **Section 9 (Practice)**: Clarify that agent commands and export are EventBus-driven
3. **Section 7 (e-URDF)**: Add `JointSpec` constructor documentation
4. **Section 9 (Practice)**: Note that `enable_mcap=True` is experimental

---

## Architecture Observations

### Strengths
- Runtime lifecycle works flawlessly end-to-end
- LLM Provider abstraction is clean and extensible
- EventBus communication pattern is intuitive for custom modules
- Memory/SeekDB API is consistent and well-documented

### Weaknesses
- UnifiedTimeline's public API is too narrow (only 2 of 8 channels have public methods)
- SkillRegistry's `list_skills()` vs `get()` split is not obvious
- RobotModel manual construction is underdocumented
- MCAP export is incomplete despite the flag existing

### Recommendations for v1.1
1. Add `timeline.record_event(channel, data, correlation_id)` as generic public method
2. Consider `registry.list_skills(return_entries=False)` parameter
3. Add `timeline.export_session(correlation_id)` as public method
4. Complete MCAP writer implementation or remove the flag
