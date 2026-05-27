# ROSClaw v1.0 API Improvements

> **Date**: 2026-05-28
> **Trigger**: User feedback on 3 UX pain points + stress test results

---

## Summary

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | `driver.state` should return dict | Medium | Fixed (added `state_dict` property + `to_dict()`) |
| 2 | `SkillExecutor` should auto-create registry | Medium | Fixed (registry now optional) |
| 3 | `Runtime.status()` should return status | Low | Already supported (property + method) |
| 4 | `MuJoCoSimDriver.get_state()` bug | High | Fixed (returned `self._state` instead of `self._driver_state`) |
| 5 | Path traversal in timeline export | Medium | Documented (see SECURITY_AUDIT.md) |
| 6 | Missing input validation on joint values | High | Documented (stress test finding) |

---

## Improvement 1: Driver State as Dict

### Problem
Users expect `driver.state` to return a dict-like object they can inspect directly:
```python
# Before: driver.state returns DriverState dataclass
state = driver.state
print(state.connected)  # OK, but users want dict access
print(state["connected"])  # TypeError: 'DriverState' object is not subscriptable
```

### Solution
Added two features:

1. **`DriverState.to_dict()`** — convert any DriverState to a plain dict
2. **`BaseDriver.state_dict`** — property shortcut for `driver.state.to_dict()`

### New API
```python
from rosclaw.mcp_drivers import MuJoCoSimDriver

driver = MuJoCoSimDriver(robot_id="ur5e_001", model_path="./ur5e.xml")
driver.initialize()
driver.start()

# Option 1: DriverState dataclass (original)
state = driver.state
print(state.connected)          # True
print(state.joint_positions)    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Option 2: Plain dict (NEW)
state = driver.state_dict
print(state["connected"])       # True
print(state["joint_positions"]) # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Option 3: Manual conversion (NEW)
state = driver.state.to_dict()
print(state["error_code"])      # 0
```

### Files Changed
- `src/rosclaw/mcp_drivers/base.py`:
  - Added `DriverState.to_dict()` method
  - Added `BaseDriver.state_dict` property

---

## Improvement 2: SkillExecutor Auto-Creates Registry

### Problem
Users had to manually create and pass a `SkillRegistry` to `SkillExecutor`:
```python
# Before: registry is REQUIRED
registry = SkillRegistry(event_bus=bus)
executor = SkillExecutor(event_bus=bus, registry=registry)  # verbose
```

### Solution
Made `registry` parameter optional. If not provided, `SkillExecutor` creates one automatically.

### New API
```python
from rosclaw.skill_manager import SkillExecutor
from rosclaw.core.event_bus import EventBus

bus = EventBus()

# Option 1: Auto-create registry (NEW, simpler)
executor = SkillExecutor(event_bus=bus)
executor.initialize()
# executor.registry is automatically created

# Option 2: Share registry across components (original, still works)
from rosclaw.skill_manager import SkillRegistry
registry = SkillRegistry(event_bus=bus)
executor = SkillExecutor(event_bus=bus, registry=registry)
```

### Files Changed
- `src/rosclaw/skill_manager/executor.py`:
  - `SkillExecutor.__init__()` — `registry` parameter now `Optional`, defaults to `None`
  - When `registry=None`, auto-creates `SkillRegistry(event_bus=event_bus)`

---

## Improvement 3: Runtime.status() Already Works

### Problem
User reported `Runtime.status()` should return status.

### Solution
Already supported in two ways:

```python
from rosclaw.core.runtime import Runtime, RuntimeConfig

config = RuntimeConfig(robot_id="ur5e_001")
runtime = Runtime(config)
runtime.initialize()
runtime.start()

# Option 1: Property (recommended)
status = runtime.status
print(status["robot_id"])       # "ur5e_001"
print(status["runtime_state"])  # "RUNNING"
print(status["modules"])        # {"firewall": True, "memory": True, ...}

# Option 2: Method (also works)
status = runtime.get_status()
print(status["drivers"])        # []
```

### Status Dict Structure
```python
{
    "robot_id": "ur5e_001",
    "runtime_state": "RUNNING",
    "event_bus": {
        "topics": ["agent.command", "praxis.completed", ...],
        "history_size": 42,
    },
    "modules": {
        "firewall": True,
        "memory": True,
        "practice": True,
        "swarm": False,
        "skill_manager": True,
        "e_urdf": True,
    },
    "drivers": ["mujoco_arm"],
}
```

### No Files Changed
Already implemented in `src/rosclaw/core/runtime.py` lines 265-287.

---

## Bug Fix: MuJoCoSimDriver.get_state()

### Problem
`MuJoCoSimDriver.get_state()` returned `self._state` which does not exist — should be `self._driver_state`.

```python
# Before: crash
driver.get_state()
# AttributeError: 'MuJoCoSimDriver' object has no attribute '_state'
```

### Fix
Changed `return self._state` to `return self._driver_state`.

### Files Changed
- `src/rosclaw/mcp_drivers/mujoco_sim_driver.py` line 118

---

## Stress Test Results

```
=== ROSClaw v1.0 压力测试 ===

[压力1] 不传参数实例化...
✗ 必须传参数: MuJoCoSimDriver.__init__() missing 2 required positional arguments

[压力2] 错误参数类型...
✓ 正确拒绝非法handler: TypeError

[压力3] 注册空技能名...
✓ 正确拒绝空技能名: ValueError

[压力4] 重复初始化...
✓ 正确拒绝重复初始化: Cannot initialize: already in state READY

[压力5] 未初始化就使用...
✓ 正确拒绝: Cannot perform move_joints: driver lifecycle state is UNINITIALIZED

[压力6] 超大关节值...
✗ 接受了危险值 (SECURITY ISSUE - see SECURITY_AUDIT.md S-2)

[压力7] 并发EventBus操作...
✓ 并发安全: 500 events

[压力8] 大量对象创建...
✓ 可管理1000个skills

=== 压力测试完成 ===
通过: 6/8 (75%)
失败: 2/8
```

### Stress Test Failures

**Test 1 — 不传参数实例化**: Expected behavior. `MuJoCoSimDriver` requires `robot_id` and `model_path`. This is by design — drivers need identity and model configuration.

**Test 6 — 超大关节值**: Security issue documented in `SECURITY_AUDIT.md` as S-2. The `_validate_joint_positions()` method checks for `abs(p) > 1e6` but the stress test used values that pass this check while still being dangerous. Recommendation: tighten bounds to realistic robot limits.

---

## Migration Guide

### For Existing Code

No breaking changes. All existing code continues to work:

```python
# Before (still works)
registry = SkillRegistry(event_bus=bus)
executor = SkillExecutor(event_bus=bus, registry=registry)

state = driver.state
connected = state.connected

# After (new simpler options)
executor = SkillExecutor(event_bus=bus)  # registry auto-created

state = driver.state_dict  # dict access
connected = state["connected"]
```

### New Patterns Recommended

```python
# Pattern 1: Quick status check
if driver.state_dict["connected"]:
    driver.move_joints([0.0] * 6)

# Pattern 2: Minimal setup
from rosclaw.skill_manager import SkillExecutor
executor = SkillExecutor(event_bus=bus)
executor.initialize()
executor.register(SkillEntry(name="pick", description="Pick object", skill_type="programmed"))
result = executor.execute("pick", {"target": "block"})

# Pattern 3: Runtime orchestration
from rosclaw.core.runtime import Runtime, RuntimeConfig
runtime = Runtime(RuntimeConfig(robot_id="ur5e_001"))
runtime.initialize()
runtime.start()
print(runtime.status["modules"])  # dict of all module states
```

---

## v1.1 Recommendations

Based on stress testing and user feedback:

### Priority 1: Safety
- Tighten joint position validation bounds (currently `1e6`, should match robot limits from e-URDF)
- Add `correlation_id` validation in timeline export (path traversal prevention)
- Add EventBus rate limiting

### Priority 2: API Ergonomics
- Add `Runtime.register_driver()` convenience method that auto-initializes the driver
- Add `SkillRegistry.register_from_handler()` factory for quick skill registration
- Add `UnifiedTimeline.get_session(correlation_id)` to retrieve session data

### Priority 3: Developer Experience
- Add `Runtime.quick_start()` one-liner for common configurations
- Add `DriverState.is_safe()` method checking error_code + joint limits
- Add rich `__repr__` to all major classes for better debugging

---

## Files Changed Summary

| File | Change | Lines |
|------|--------|-------|
| `src/rosclaw/mcp_drivers/base.py` | Added `DriverState.to_dict()`, `BaseDriver.state_dict` | +15 |
| `src/rosclaw/mcp_drivers/mujoco_sim_driver.py` | Fixed `get_state()` returning `self._state` → `self._driver_state` | 1 |
| `src/rosclaw/skill_manager/executor.py` | Made `registry` parameter optional | 1 |

**Total**: 3 files, ~17 lines changed. All changes are backward-compatible.
