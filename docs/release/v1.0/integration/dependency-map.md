# Dependency Map

> **Author**: Release Integrator  
> **Date**: 2026-05-28  
> **Status**: DRAFT  
> **RFC Reference**: RFC-0001 Architecture Freeze §4 Anti-Patterns

---

## Executive Summary

Scanned all cross-module imports in `src/rosclaw/` to identify EventBus boundary violations per RFC-0001.

**Findings:**
- **15 cross-module imports** detected
- **1 P0 violation**: MCP server bypasses EventBus for firewall integration
- **2 P1 violations**: Type imports that should use core/types.py
- **12 acceptable**: Runtime orchestration and type dependencies

---

## Critical Startup Chain

```
CLI (src/rosclaw/cli.py:main)
  └─> Runtime (src/rosclaw/core/runtime.py:Runtime)
        ├─> EventBus (src/rosclaw/core/event_bus.py)
        ├─> EURDFParser (src/rosclaw/e_urdf/parser.py)
        ├─> FirewallValidator (src/rosclaw/firewall/validator.py)
        │     └─> RobotModel (src/rosclaw/e_urdf/parser.py)
        ├─> MemoryInterface (src/rosclaw/memory/interface.py)
        │     └─> SeekDBSQLiteClient (src/rosclaw/memory/seekdb_client.py)
        ├─> UnifiedTimeline (src/rosclaw/practice/timeline.py)
        ├─> SwarmRuntimeManager (src/rosclaw/swarm/manager.py)
        ├─> SkillRegistry (src/rosclaw/skill_manager/registry.py)
        ├─> SkillExecutor (src/rosclaw/skill_manager/executor.py)
        └─> ProviderLoader (src/rosclaw/provider/loader.py)
```

**Startup sequence:**
1. `cli.py:main()` parses arguments
2. `cli.py:cmd_run()` instantiates Runtime with config
3. `Runtime.__init__()` creates EventBus + all module instances
4. `Runtime.initialize()` calls each module's `initialize()`
5. `Runtime.start()` calls each module's `start()`
6. Runtime enters RUNNING state

---

## Cross-Module Import Analysis

### Acceptable Imports (12)

#### core/runtime.py → all modules
```python
from rosclaw.e_urdf.parser import EURDFParser
from rosclaw.firewall.validator import FirewallValidator
from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import SeekDBSQLiteClient, SeekDBMemoryClient
from rosclaw.practice.timeline import UnifiedTimeline
from rosclaw.swarm.manager import SwarmRuntimeManager
from rosclaw.skill_manager.registry import SkillRegistry
from rosclaw.skill_manager.executor import SkillExecutor
from rosclaw.provider.loader import ProviderLoader
```
**Justification**: Runtime is the orchestrator per RFC-0001 §2.1. It MUST instantiate all modules to manage their lifecycle.

#### firewall/validator.py → e_urdf/parser.py
```python
from rosclaw.e_urdf.parser import RobotModel
```
**Justification**: Firewall needs robot model to validate joint limits and safety envelopes. This is a type dependency, not runtime communication.

#### provider/* internal imports
```python
from rosclaw.provider.core.errors import ...
from rosclaw.provider.core.manifest import ...
```
**Justification**: Internal provider module imports are acceptable.

---

### P0 Violations (1)

#### mcp/ur5_server.py → firewall/decorator.py
```python
# src/rosclaw/mcp/ur5_server.py:8
from rosclaw.firewall.decorator import (
    firewall_protected,
    SafetyViolation,
)
```

**Problem**: MCP server directly imports firewall decorator, bypassing EventBus.

**Why it matters**:
- Violates RFC-0001 §2.2: "Modules MUST NOT import each other directly"
- MCP server should request safety validation via EventBus, not direct function call
- Makes firewall changes breaking changes for MCP server

**Suggested fix**:
```python
# Instead of:
@firewall_protected(safety_level="STRICT")
def execute_trajectory(self, waypoints):
    ...

# Use EventBus:
async def execute_trajectory(self, waypoints):
    validation_request = EventBus.publish(
        "firewall.validate",
        {"waypoints": waypoints, "safety_level": "STRICT"}
    )
    if not validation_request["approved"]:
        raise SafetyViolation(validation_request["reason"])
    ...
```

**Verification**:
```bash
grep -n "from rosclaw.firewall" src/rosclaw/mcp/ur5_server.py
# Should return empty after fix
```

---

### P1 Violations (2)

#### agent_runtime/mcp_hub.py → e_urdf/parser.py
```python
# src/rosclaw/agent_runtime/mcp_hub.py:12
from rosclaw.e_urdf.parser import Vec3
```

**Problem**: Type-only import, but Vec3 should be in core/types.py for consistency.

**Suggested fix**: Move `Vec3` to `src/rosclaw/core/types.py` and update imports.

**Verification**:
```bash
grep -r "class Vec3" src/rosclaw/
# Should find in core/types.py, not e_urdf/parser.py
```

#### practice/recorder.py → data/flywheel.py
```python
# src/rosclaw/practice/recorder.py:7
from rosclaw.data.flywheel import DataFlywheel, EventType
from rosclaw.data.flywheel import RobotState as FlywheelRobotState
```

**Problem**: Type imports from data layer. RobotState and EventType should be in core/types.py.

**Suggested fix**: Move `RobotState` and `EventType` to `src/rosclaw/core/types.py` and update imports.

**Verification**:
```bash
grep -r "class RobotState" src/rosclaw/
# Should find in core/types.py
```

---

## Module Dependency Graph

```
                    ┌─────────────────┐
                    │   cli.py        │
                    │   (entry)       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ core/runtime.py │
                    │ (orchestrator)  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        │         ┌──────────▼──────────┐         │
        │         │ core/event_bus.py   │         │
        │         │ (communication)     │         │
        │         └──────────┬──────────┘         │
        │                    │                    │
        │    ┌───────────────┼───────────────┐    │
        │    │               │               │    │
┌───────▼────┴──┐    ┌──────▼──────┐    ┌──▼────▼──┐
│ e_urdf/       │    │ firewall/   │    │ memory/  │
│ parser.py     │◄───│ validator   │    │ interface│
│ (robot model) │    │ (safety)    │    │ (storage)│
└───────┬───────┘    └─────────────┘    └──────────┘
        │
┌───────▼───────────┐    ┌──────────────┐    ┌──────────────┐
│ practice/         │    │ skill_mgr/   │    │ swarm/       │
│ timeline.py       │    │ registry     │    │ manager      │
│ (event recording) │    │ (skills)     │    │ (multi-robot)│
└───────┬───────────┘    └──────────────┘    └──────────────┘
        │
┌───────▼───────────┐
│ data/             │
│ flywheel.py       │
│ (type defs)       │
└───────────────────┘
```

**Legend:**
- `→` : Instantiates (acceptable for Runtime)
- `◄──` : Type dependency (acceptable)
- `⚠️` : Direct import (violation)

---

## Violations by Module

| Module | Imports From | Severity | Status |
|--------|-------------|----------|--------|
| mcp/ur5_server | firewall/decorator | P0 | Open |
| agent_runtime/mcp_hub | e_urdf/parser (Vec3) | P1 | Open |
| practice/recorder | data/flywheel (types) | P1 | Open |

---

## Recommendations

### Immediate Actions (P0)

1. **Refactor mcp/ur5_server.py** to use EventBus for firewall validation
   - Remove direct import of `firewall.decorator`
   - Implement async validation request/response via EventBus
   - Update tests to verify EventBus-based flow

### Short-Term (P1)

2. **Move shared types to core/types.py**:
   - `Vec3` from e_urdf/parser.py
   - `RobotState` from data/flywheel.py
   - `EventType` from data/flywheel.py
   - Update all imports across codebase

### Long-Term

3. **Add import linting** to CI pipeline:
   ```bash
   # Check for cross-module imports (excluding core)
   grep -r "^from rosclaw\." src/ | \
     grep -v "from rosclaw.core" | \
     grep -v "^src/rosclaw/core/"
   ```

---

## Verification Commands

```bash
# Scan all cross-module imports
find src/rosclaw -name "*.py" -exec grep -l "^from rosclaw\." {} \; | \
  xargs grep "^from rosclaw\." | \
  grep -v "from rosclaw.core" | \
  grep -v "^src/rosclaw/core/"

# Verify startup chain
python3 -c "from rosclaw.cli import main; print('CLI import OK')"
python3 -c "from rosclaw.core import Runtime; print('Runtime import OK')"

# Verify EventBus singleton
grep -r "EventBus()" src/rosclaw/ | grep -v test
# Should only find in core/runtime.py
```

---

## Related Documents

- [RFC-0001: Architecture Freeze](../RFC-0001-architecture-freeze.md)
- [RFC-0005: Acceptance Gates](../RFC-0005-acceptance-gates.md)
- [Event Flow Map](./event-flow-map.md)
- [Audit Report: Runtime](../audits/audit-runtime.md)
