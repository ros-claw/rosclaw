# Tutorial Review: 01_getting_started.md

**Reviewer**: Qwen AI Assistant
**Date**: 2026-05-28
**Tutorial**: `tutorials/01_getting_started.md` (247 lines)
**Score**: 8.5/10

---

## Strengths

1. **Clear prerequisites** — Python 3.10+, POSIX shell, ~15 min estimate
2. **Installation verification** — version check + pytest in one step
3. **Complete workflow demo** — `hello_robot.py` shows full lifecycle
4. **Expected output** for every code block
5. **All 6 grounding engines** introduced concisely
6. **Benchmark section** — users validate their environment
7. **Common Issues** addresses real problems (numpy, MuJoCo)

---

## Issues and Improvements

### P0: Missing Error Handling in Runtime Example (line 94-106)

```python
# CURRENT — no cleanup on failure
runtime = Runtime(config)
runtime.initialize()
runtime.start()
# ... do work ...
runtime.stop()

# RECOMMENDED
runtime = Runtime(config)
try:
    runtime.initialize()
    runtime.start()
    # ... do work ...
finally:
    runtime.stop()
```

### P0: Missing "What's Next" Narrative (line 207-216)

Current section is just a link table. Should include learning paths:
- Application Developers: 01 → 02 → 04 → 05
- System Integrators: 01 → 03 → 10 → 09
- Researchers: 01 → 07 → 08 → 06

### P1: No Async EventBus Example (line 68-82)

Only sync `subscribe()` shown. Should demonstrate:
```python
bus.subscribe_async("topic", async_handler)
event = await bus.await_event("robot.ready", timeout=5.0)
```

### P1: RuntimeConfig Options Not Explained (line 94-100)

Missing: what safety levels exist (STRICT/MODERATE/PERMISSIVE), when to use "sqlite" vs "memory" backend.

### P1: Driver Missing State Checks (line 120-131)

Should verify `driver.state == "READY"` after `initialize()`.

### P2: Practice Event Example Incomplete (line 164-179)

Missing sensorimotor data (the key differentiator), no `recorder.stop()` cleanup.

### P2: Common Issues Too Brief (line 220-243)

Should add: EventBus subscriber not receiving, Runtime state errors, stale driver state, practice events not saving.

---

## Template Compliance: 75%

| Section | Status |
|---------|--------|
| Prerequisites | ✅ |
| Time estimate | ✅ |
| Learning Objectives | ❌ Implicit only |
| Difficulty level | ❌ Missing |
| Step-by-step | ✅ |
| Complete example | ✅ |
| Try It Yourself | ❌ Missing |
| Next Steps | ✅ |
| Common Issues | ✅ |

---

## Action Items

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Add try/finally to Runtime example | 5 min |
| P0 | Write "What's Next" narrative with learning paths | 30 min |
| P1 | Add async subscriber example | 15 min |
| P1 | Add RuntimeConfig options table | 20 min |
| P1 | Add driver state checks | 10 min |
| P2 | Expand Common Issues (+4 entries) | 30 min |
| P2 | Add sensorimotor data to PraxisEvent | 15 min |

**Total estimated effort**: ~2 hours

---

**Verdict**: Solid foundation. Fix P0 items before publishing.
