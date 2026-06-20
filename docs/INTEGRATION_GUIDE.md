# ROSClaw v1.0 Integration Guide

**Author**: ROSClaw Contributors  
**Date**: 2026-05-29  
**Status**: ✅ COMPLETE

---

## Overview

This guide covers the integration of KNOW (Knowledge Graph) and HOW (Heuristic Recovery) modules into ROSClaw v1.0.

---

## Runtime Configuration

### Enabling KNOW and HOW

```python
from rosclaw.core import Runtime, RuntimeConfig

config = RuntimeConfig(
    robot_id="my_robot",
    enable_knowledge=True,   # Enable KnowledgeInterface
    enable_how=True,         # Enable HeuristicEngine
    enable_firewall=True,
    enable_memory=True,
    enable_practice=True,
)

runtime = Runtime(config)
runtime.initialize()
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_knowledge` | bool | False | Enable Knowledge Graph module |
| `enable_how` | bool | False | Enable Heuristic Recovery module |
| `enable_firewall` | bool | True | Enable safety validation |
| `enable_memory` | bool | True | Enable SeekDB storage |
| `enable_practice` | bool | True | Enable practice recording |
| `seekdb_backend` | str | "memory" | "memory" or "sqlite" |
| `seekdb_path` | str | "./seekdb.sqlite" | SQLite file path |
| `seekdb_url` | str \| None | `ROSCLAW_SEEKDB_URL` env var | Optional SeekDB endpoint for practice event persistence |
| `seekdb_fallback_dir` | str | `ROSCLAW_SEEKDB_FALLBACK_DIR` or `/data/rosclaw/fallback` | Local JSON fallback when SeekDB is unreachable |

---

## Accessing KNOW and HOW

After initialization:

```python
# KNOW interface
knowledge = runtime.knowledge
caps = knowledge.query_robot_capabilities("my_robot")

# HOW engine
how = runtime.how
# Note: suggest_recovery is async
import asyncio
suggestion = asyncio.run(how.suggest_recovery("joint_limit_exceeded"))
```

---

## Enabling Practice / SeekDB Persistence

When `enable_practice=True` and `seekdb_url` is configured, `Runtime` automatically
assembles a `SeekDBBridge` and passes it to `EpisodeRecorder`. Every finalized
practice episode is then forwarded to SeekDB in addition to the local artifact
directory.

### Minimal configuration

```python
config = RuntimeConfig(
    robot_id="my_robot",
    enable_practice=True,
    seekdb_url="http://localhost:2881",
    seekdb_fallback_dir="/data/rosclaw/fallback",
)
runtime = Runtime(config)
runtime.initialize()
```

Or via environment variables:

```bash
export ROSCLAW_SEEKDB_URL=http://localhost:2881
export ROSCLAW_SEEKDB_FALLBACK_DIR=/data/rosclaw/fallback
```

### Requirements

Install the optional `rosclaw[practice]` extra so that `SeekDBBridge` can import
`rosclaw_practice`:

```bash
pip install -e ".[practice]"
```

### Behavior notes

- If `seekdb_url` is unset, no bridge is created and `EpisodeRecorder` behavior is unchanged.
- If `rosclaw_practice` is not installed, `Runtime` logs an info message and disables SeekDB forwarding; initialization continues normally.
- SeekDB submission failures are non-fatal: local artifact writes and `praxis.recorded` publication always complete.
- Failed submissions are written as JSON files to `seekdb_fallback_dir`.

See `docs/practice/SEEKDB_INTEGRATION.md` for the full data-mapping table and
offline verification steps.

---

## Adding Knowledge Graph Entries

### Method 1: Direct SeekDB Insert

```python
from rosclaw.memory.seekdb_client import SeekDBMemoryClient

seekdb = SeekDBMemoryClient()
seekdb.connect()

seekdb.insert(
    "knowledge_graph",
    {
        "id": "ur5e_pick",
        "robot_id": "universal_robots_ur5e",
        "capability": "pick_and_place",
        "skill_type": "programmed",
        "parameters": '{"object": "str", "location": "str"}',
        "description": "Pick an object from a location",
    },
)
```

### Method 2: Through KnowledgeInterface (Recommended)

```python
from rosclaw.know import KnowledgeInterface

know = KnowledgeInterface(seekdb_client=seekdb, robot_id="ur5e")
know.initialize()

# Capabilities are auto-queried from SeekDB
```

### Required Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | str | ✅ | Unique entry ID |
| `robot_id` | str | ✅ | Robot identifier |
| `capability` | str | ✅ | Capability name |
| `skill_type` | str | ✅ | "programmed", "learned", or "composed" |
| `parameters` | str | ❌ | JSON parameter schema |
| `description` | str | ❌ | Human-readable description |

---

## Adding Heuristic Rules

### Automatic Seeding

HeuristicEngine automatically seeds default rules on initialization:

```python
from rosclaw.how import HeuristicEngine

how = HeuristicEngine(seekdb_client=seekdb)
await how.initialize()  # Seeds 10 default rules
```

### Custom Rules

```python
# Insert directly into SeekDB
seekdb.insert(
    "heuristic_rules",
    {
        "id": "custom_rule_1",
        "condition": "collision_detected",
        "action": "Stop immediately and re-plan path",
        "priority": 1,
        "source": "custom",
        "success_count": 0,
        "failure_count": 0,
    },
)
```

### Default Rules (Auto-Seeded)

| Condition | Action | Priority |
|-----------|--------|----------|
| `joint_limit_exceeded` | Reduce velocity by 50% and re-plan | 2 |
| `collision_detected` | Stop immediately and re-plan | 1 |
| `sensor_timeout` | Retry sensor read, switch to backup | 3 |
| `trajectory_deviation` | Re-calculate trajectory | 2 |
| `power_fluctuation` | Reduce speed, monitor | 3 |

---

## Common Errors and Solutions

### Error: `RuntimeError: event loop is already running`

**Cause**: Initializing Runtime with `enable_how=True` from an async context.

**Workaround**:
```python
# Option 1: Initialize from sync context
def main():
    runtime = Runtime(config)
    runtime.initialize()

# Option 2: Use subprocess or thread
import threading
def init_runtime():
    runtime = Runtime(config)
    runtime.initialize()

t = threading.Thread(target=init_runtime)
t.start()
t.join()
```

**Fix Status**: Partially fixed in c4ea448. Full fix planned for v1.1.

---

### Error: `ImportError: cannot import name 'KnowledgeInterface'`

**Cause**: KNOW module not installed or corrupted.

**Solution**:
```bash
pip install -e ".[dev]"
python -c "from rosclaw.know import KnowledgeInterface; print('OK')"
```

---

### Error: `heuristic_rules` table empty after init

**Cause**: HeuristicEngine seeding failed or was skipped.

**Solution**:
```python
# Manual seeding
import asyncio
asyncio.run(how.seed_defaults())

# Verify
results = seekdb.query("heuristic_rules", filters={}, limit=10)
print(f"Rules: {len(results)}")
```

---

### Error: Knowledge query returns empty list

**Cause**: No data in `knowledge_graph` for the robot_id.

**Solution**:
```python
# Check if data exists
results = seekdb.query("knowledge_graph", filters={"robot_id": "my_robot"})
if not results:
    # Seed data
    seekdb.insert("knowledge_graph", {...})
```

---

## Performance Tuning

### KNOW Query Optimization

- **In-memory backend**: Use for development (fastest)
- **SQLite backend**: Use for production with large datasets
- **Batch inserts**: Insert multiple records in a single transaction

### HOW Recovery Optimization

- **Rule caching**: HeuristicEngine caches rules in memory
- **Pre-seeding**: Call `seed_defaults()` during setup, not during hot path
- **Async access**: Always use `await` for `suggest_recovery()`

### EventBus Optimization

- **Topic namespacing**: Use hierarchical topics (e.g., `robot.joint_states`)
- **Subscriber limits**: Keep subscriber count per topic < 10 for optimal throughput
- **History size**: Default is 10,000 events; increase if needed

### Benchmark Targets

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| KNOW Query | < 100 ms | 0.0002 ms | ✅ |
| HOW Recovery | < 10 ms | 0.0013 ms | ✅ |
| EventBus | > 10,000/s | 114,214/s | ✅ |
| SeekDB Query | < 50 ms | 6.46 ms | ✅ |
| E2E Pipeline | < 500 ms | 0.0049 ms | ✅ |

---

## Testing Integration

### Smoke Test

```bash
pytest tests/integration/test_know_how_smoke.py -v
```

### Error Path Test

```bash
pytest tests/integration/test_error_paths.py -v
```

### Performance Benchmark

```bash
python benchmarks/integration_performance.py
```

---

## Troubleshooting Checklist

- [ ] `enable_knowledge=True` in RuntimeConfig
- [ ] `enable_how=True` in RuntimeConfig
- [ ] SeekDB connected (`seekdb.connect()`)
- [ ] `knowledge_graph` table has data
- [ ] `heuristic_rules` table has data
- [ ] Runtime initialized before queries
- [ ] Async functions use `await` or `asyncio.run()`
- [ ] EventBus topics use correct namespacing

---

## Migration from v0.x

### KNOW Module (New in v1.0)

Previously, robot capabilities were hardcoded. Now they are stored in SeekDB:

```python
# v0.x (hardcoded)
# capabilities = ["pick", "place", "move"]

# v1.0 (from Knowledge Graph)
know = runtime.knowledge
caps = know.query_robot_capabilities("my_robot")
```

### HOW Module (New in v1.0)

Previously, failure recovery required LLM calls. Now heuristic rules provide fast recovery:

```python
# v0.x (slow LLM call)
# recovery = llm.query("How to fix joint limit error?")

# v1.0 (fast heuristic)
how = runtime.how
suggestion = asyncio.run(how.suggest_recovery("joint_limit_exceeded"))
```

---

## Next Steps

1. **Seed your robot's capabilities** into `knowledge_graph`
2. **Add custom heuristic rules** for your use case
3. **Run benchmarks** to establish your performance baseline
4. **Monitor SeekDB growth** and implement cleanup policies

---

*For questions, see `docs/INTEGRATION_QUALITY_REPORT.md` or check the status board.*
