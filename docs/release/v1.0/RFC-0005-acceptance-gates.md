# RFC-0005: ROSClaw v1.0 Acceptance Gates

> **Status**: DRAFT
> **Author**: Release Commander
> **Date**: 2026-05-28

---

## P0: MUST PASS before release (Release Blockers)

| # | Gate | Verification Method | Status |
|---|------|---------------------|--------|
| 1 | Repo clean installs (`pip install -e .`) | Manual + CI | _ |
| 2 | CLI starts (`rosclaw --version`, `rosclaw status`) | `tests/test_cli.py` | _ |
| 3 | Runtime starts and reaches RUNNING state | `tests/test_core.py` | _ |
| 4 | At least one Agent Runtime demo runs end-to-end | `examples/hello_robot.py` | _ |
| 5 | At least one Provider registers and is callable | `tests/test_provider.py` | _ |
| 6 | EventBus publishes/subscribes core events | `tests/test_core.py` | _ |
| 7 | Practice records a PraxisEvent | `tests/test_provider.py` (integration) | _ |
| 8 | Memory can query a stored event | `tests/test_data_layer.py` | _ |
| 9 | No module bypasses Runtime for core loop | `grep` cross-module imports | _ |
| 10 | README-claimed v1.0 capabilities have demo or test | Manual review | _ |
| 11 | No `_state` attribute collision (ROLE_SWAP fix verified) | `tests/test_core.py` | _ |
| 12 | All existing tests pass (157+) | `pytest` | _ |

## P1: SHOULD FIX before release (Strong Recommendations)

| # | Gate | Verification Method | Status |
|---|------|---------------------|--------|
| 1 | Configuration system unified (single entry point) | Code review | _ |
| 2 | Log format consistent across modules | Code review | _ |
| 3 | Contract tests exist for EventBus, Provider, PraxisEvent | `tests/contracts/` | _ |
| 4 | Schema versioning on PraxisEvent and MemorySchema | Code review | _ |
| 5 | Error handling consistent (structured error types) | Code review | _ |
| 6 | Examples stable and documented | Run examples | _ |
| 7 | Thread safety limitations documented | Docs review | _ |
| 8 | Memory buffer growth has size limits | Code review + test | _ |

## P2: DEFER to v1.1 (Not Blocking)

| # | Item | Notes |
|---|------|-------|
| 1 | Advanced Swarm scheduling | Interface-ready is sufficient |
| 2 | Complete Darwin Arena | v1.1 |
| 3 | Advanced How/Know auto-recovery | v1.1 |
| 4 | Complete website integration | v1.1 |
| 5 | Multi-robot real hardware demo | v1.1 |
| 6 | Large-scale benchmark | v1.1 |
| 7 | Sandbox digital twin integration | In development, v1.1 |
| 8 | MCAP format write support | v1.1 |
| 9 | Prometheus metrics / OpenTelemetry | v1.1 |

## Release Candidate Process

```text
RC0: Current state freeze (no new features)
RC1: Fix all P0 issues
RC2: Fix P1 issues where feasible
RC3: Documentation, demos, README alignment
v1.0: Tag release
```

## v1.0 Minimum Viable Loop

The release proves THIS loop works, nothing more:

```text
Agent gives goal
    |
Agent Runtime normalizes context
    |
Runtime selects Provider / Skill
    |
Firewall checks safety (basic joint limits)
    |
Practice records execution event
    |
Event Bus publishes PraxisEvent
    |
SeekDB indexes event
    |
Memory can retrieve it
```

v1.0 narrative: "ROSClaw v1.0 is the first unified runtime foundation for Physical AI systems."

NOT: "We have completely implemented all physical AI."
