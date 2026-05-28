# ROSClaw v1.0 Unified Integration Vision

**Date**: 2026-05-28  
**Status**: Integration Phase Active  
**Target**: Full v1.0 Integration (No v1.1 deferral for core modules)

---

## Executive Summary

We have completed comprehensive audits of all 9 modules. The findings reveal a **strong foundation with clear integration gaps**. This document unifies our understanding and defines the path forward.

**Key Insight**: ROSClaw v1.0 is NOT a collection of independent modules. It is a **unified Physical Intelligence Runtime** where every module plays a critical role in the grounding pipeline.

---

## The ROSClaw v1.0 Vision: Unified Grounding Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ROSClaw v1.0 Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Agent Runtime (LLM Interface)                                       │
│       ↓                                                               │
│  Provider Layer (Capability Routing)                                 │
│       ↓                                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │  KNOW    │───→│   HOW    │───→│ FIREWALL │───→│  SANDBOX │       │
│  │(Knowledge│    │(Heuristic│    │(Safety   │    │(Digital  │       │
│  │  Graph)  │    │ Recovery)│    │  Rules)  │    │  Twin)   │       │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘       │
│       ↓               ↓               ↓               ↓              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    PRACTICE (Timeline Grounding)              │   │
│  │              Records all physical interactions                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    MEMORY (SeekDB Knowledge Plane)            │   │
│  │         Unified storage for all experience data               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    DASHBOARD (Observability)                  │   │
│  │         Real-time monitoring and visualization                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  SWARM (Multi-Agent Coordination) - Architecturally Ready           │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Status Overview

### ✅ STRONG FOUNDATION (Ready for Integration)

| Module | Status | Audit Score | Role in v1.0 |
|--------|--------|-------------|--------------|
| **Practice** | ✅ Production Ready | 22/22 tests | Records all physical interactions to MCAP |
| **Memory** | ✅ Integrated (1%) | 0 P0 | SeekDB Knowledge Plane foundation |
| **Swarm** | ✅ Architecturally Ready | PASS | Multi-agent coordination interfaces |
| **Sandbox** | ✅ S0 Complete | 22 tests | Digital twin with MuJoCo engine |

### ⚠️ INTEGRATION GAPS (Need Work)

| Module | Status | Critical Issues | Impact |
|--------|--------|----------------|--------|
| **Provider** | ⚠️ 1 P0 (mitigated) | Sync/async boundary violation | Blocks third-party providers |
| **Dashboard** | ⚠️ NOT READY | No runtime observability | Cannot show live robot state |
| **Know** | ❌ NOT INTEGRATED | Zero code in v1.0 | No structured knowledge for LLM |
| **How** | ❌ NOT INTEGRATED | Zero code in v1.0 | No heuristic failure recovery |

### 🔴 RED TEAM FINDINGS

- **270/270 tests pass** (113% of P0 requirement)
- **All 12 P0 gates PASS**
- **2 High severity** (documentation issues)
- **5 Medium severity** (tutorial improvements)

---

## Why Every Module Matters

### 1. KNOW (Knowledge Graph) - The Brain
**Current State**: NOT INTEGRATED (0 lines in v1.0)  
**Why Critical**: 
- Provides structured robot capabilities to LLM
- Enables symptom-based failure recovery without LLM calls
- Reduces hallucination by grounding LLM in facts

**Integration Effort**: ~80 hours over 3-4 weeks  
**Value**: Transforms ROSClaw from "API wrapper" to "intelligent runtime"

### 2. HOW (Heuristic Recovery) - The Reflexes
**Current State**: NOT INTEGRATED (0 lines in v1.0)  
**Why Critical**:
- Provides fast failure recovery (<10ms vs 500ms LLM call)
- Learns from past failures (feedback loop)
- Reduces API costs (cached heuristics vs LLM calls)

**Integration Effort**: ~8.5 hours  
**Value**: Makes ROSClaw self-improving

### 3. DASHBOARD - The Eyes
**Current State**: NOT READY (static UI only)  
**Why Critical**:
- Users cannot see live robot state
- Cannot monitor module health
- Cannot visualize PraxisEvents or timelines

**Integration Effort**: ~16 hours for minimum viable runtime observability  
**Value**: Makes ROSClaw observable and debuggable

### 4. PROVIDER - The Nervous System
**Current State**: Well-architected, 1 P0 (mitigated)  
**Why Critical**:
- Routes LLM requests to appropriate models
- Supports diverse runtimes (Python, HTTP, ROS2)
- Enables capability-based agent planning

**Integration Effort**: ~4 hours to fix P0  
**Value**: Unified capability abstraction

### 5. PRACTICE - The Black Box
**Current State**: Production ready (22/22 tests)  
**Why Critical**:
- Records all physical interactions to MCAP
- Enables replay and debugging
- Foundation for learning from experience

**Integration Effort**: Already integrated  
**Value**: Complete physical interaction history

### 6. MEMORY - The Hippocampus
**Current State**: Integrated (1% of independent module)  
**Why Critical**:
- SeekDB Knowledge Plane foundation
- Stores PraxisEvents
- Enables experience-based learning

**Integration Effort**: Already integrated, needs expansion  
**Value**: Unified experience storage

### 7. FIREWALL/SANDBOX - The Immune System
**Current State**: v1.0 basic + sandbox S0 complete  
**Why Critical**:
- Validates trajectories before execution
- Prevents physical damage
- Enables safe exploration

**Integration Effort**: ~8 hours to integrate sandbox  
**Value**: Physical safety guarantees

### 8. SWARM - The Colony
**Current State**: Architecturally ready  
**Why Critical**:
- Multi-agent coordination interfaces
- Capability-based task assignment
- Foundation for distributed robotics

**Integration Effort**: Already integrated (interfaces only)  
**Value**: Multi-robot coordination ready

---

## Unified Integration Plan

### Phase 1: Critical Path (Week 1-2, ~40 hours)

**Goal**: Integrate KNOW and HOW to make v1.0 intelligent

| Task | Owner | Hours | Deliverable |
|------|-------|-------|-------------|
| Create `src/rosclaw/know/` module | KNOW | 20 | KnowledgeInterface online |
| Add `enable_knowledge` to RuntimeConfig | ROSCLAW | 1 | Config flag |
| Add `knowledge.*` EventBus topics | ROSCLAW | 2 | Event topics |
| Fill knowledge_graph from e-URDF | ROSCLAW + KNOW | 4 | Capability triples |
| Add MCP `query_knowledge` tool | ROSCLAW | 4 | LLM can query knowledge |
| Integrate HOW HeuristicEngine | HOW | 8.5 | Fast failure recovery |

**Success Criteria**:
- LLM can query robot capabilities via MCP
- Firewall blocks come with recovery suggestions
- Agent gets fast heuristic recovery (<10ms)

### Phase 2: Observability (Week 3, ~16 hours)

**Goal**: Make v1.0 observable and debuggable

| Task | Owner | Hours | Deliverable |
|------|-------|-------|-------------|
| Implement `rosclaw-agent-daemon` | DASHBOARD | 4 | Heartbeat + events |
| Add WebSocket `/api/events/stream` | DASHBOARD | 4 | Real-time push |
| Parse actual MCAP files | DASHBOARD | 4 | Real topic lists |
| Add runtime status API | DASHBOARD | 4 | Live connection status |

**Success Criteria**:
- Dashboard shows live robot state
- Users can see PraxisEvents in real-time
- MCAP files are properly parsed

### Phase 3: Provider Polish (Week 3, ~4 hours)

**Goal**: Fix P0 and add integration tests

| Task | Owner | Hours | Deliverable |
|------|-------|-------|-------------|
| Fix sync/async boundary in ProviderRegistry | PROVIDER | 2 | Third-party providers work |
| Fix Runtime._health poking | ROSCLAW | 1 | Proper lifecycle |
| Add Provider + Runtime integration test | PROVIDER | 1 | Verified integration |

**Success Criteria**:
- Third-party providers can register from async context
- No encapsulation violations

### Phase 4: Sandbox Integration (Week 4, ~8 hours)

**Goal**: Replace basic firewall with full digital twin

| Task | Owner | Hours | Deliverable |
|------|-------|-------|-------------|
| Create SandboxRuntimeAdapter | SANDBOX | 2 | Lifecycle integration |
| Replace static collision check with dynamic | SANDBOX | 3 | ISSUE-SB-001 fixed |
| Migrate MCP server to EventBus validation | SANDBOX + ROSCLAW | 3 | ISSUE-SB-003 fixed |

**Success Criteria**:
- Dynamic collision detection works
- Firewall violations visible to Practice/Memory
- Single MuJoCo model loaded

---

## Cross-Module Dependencies

```
KNOW ──provides──→ Agent Runtime (capabilities, symptoms, analogies)
  ↓
  └──provides──→ HOW (bridge_index.json, code_patterns)

HOW ──provides──→ Firewall (recovery suggestions)
  ↓
  └──provides──→ Agent Runtime (fast heuristic recovery)

PRACTICE ──records──→ MEMORY (PraxisEvents)
  ↓
  └──triggers──→ KNOW (incremental ingest)

MEMORY ──stores──→ SeekDB (unified Knowledge Plane)
  ↓
  └──serves──→ KNOW, HOW, DASHBOARD

DASHBOARD ──observes──→ Runtime (live state)
  ↓
  └──visualizes──→ PRACTICE (timelines), MEMORY (experiences)

SANDBOX ──validates──→ Firewall (dynamic collision)
  ↓
  └──publishes──→ MEMORY (episodes), DASHBOARD (status)

SWARM ──coordinates──→ Multiple agents (task assignment)
  ↓
  └──reads──→ KNOW (capabilities), MEMORY (past performance)
```

---

## Shared Dependencies

### Python Dependencies (pyproject.toml)

**Core Runtime**:
- `mujoco>=3.1.0` - Physics simulation (used by SANDBOX, FIREWALL)
- `numpy>=1.24.0` - Numerical operations (used by all modules)
- `pydantic>=2.0.0` - Data validation (used by all modules)
- `pyyaml>=6.0.0` - Config parsing (used by KNOW, SANDBOX)

**Knowledge & Learning**:
- `sentence-transformers>=2.2.0` - Embeddings (used by KNOW, HOW)
- `networkx>=3.1.0` - Graph algorithms (used by KNOW)
- `scipy>=1.11.0` - DTW distance (used by MEMORY)

**ROS 2 Integration**:
- `rclpy>=3.0.0` - ROS 2 Python client (used by PRACTICE, SANDBOX)
- `mcap>=0.0.10` - MCAP file parsing (used by PRACTICE, DASHBOARD)

**LLM Integration**:
- `openai>=1.0.0` - OpenAI API (used by PROVIDER)
- `anthropic>=0.18.0` - Anthropic API (used by PROVIDER)
- `requests>=2.31.0` - HTTP client (used by PROVIDER)

**Web & Dashboard**:
- `fastapi>=0.104.0` - Web framework (used by DASHBOARD)
- `uvicorn>=0.24.0` - ASGI server (used by DASHBOARD)
- `websockets>=12.0` - Real-time push (used by DASHBOARD)

### Shared Infrastructure

**SeekDB** (Knowledge Plane):
- SQLite backend for v1.0
- Future: PostgreSQL with pgvector for v1.1
- Used by: MEMORY, KNOW, HOW, PRACTICE

**EventBus** (Central Nervous System):
- In-process pub/sub for v1.0
- Future: Redis Streams or NATS for v1.1
- Used by: All modules

**e-URDF-Zoo** (Physical DNA):
- YAML configs for robot models
- Used by: SANDBOX, FIREWALL, SWARM, KNOW

---

## Testing Strategy

### Unit Tests (Per Module)
- Each module maintains its own test suite
- Target: 80% coverage minimum
- Current: 72% overall (core modules at 90%+)

### Integration Tests (Cross-Module)
- Test EventBus message flow
- Test SeekDB read/write patterns
- Test MCP tool invocation
- Target: All critical paths covered

### End-to-End Tests (Full Pipeline)
- `hello_robot.py` demo (already exists)
- Knowledge query → Agent planning → Firewall validation → Practice recording → Memory storage → Dashboard visualization
- Target: 1 complete E2E test per sprint

### Benchmark Tests (Performance)
- EventBus throughput: 216,582 events/s ✅
- SeekDB insert latency: 0.0023 ms ✅
- Firewall validation speed: 0.0796 ms ✅
- Target: All benchmarks pass

---

## Success Criteria for v1.0 Release

### P0 Gates (Must Pass)

1. ✅ Repo clean installs
2. ✅ CLI starts
3. ✅ Runtime reaches RUNNING state
4. ✅ Agent Runtime demo runs end-to-end
5. ⚠️ Provider registers and is callable (fix P0)
6. ✅ EventBus publishes/subscribes
7. ✅ Practice records PraxisEvent
8. ✅ Memory queries stored event
9. ✅ No module bypasses Runtime
10. ✅ README capabilities have tests
11. ✅ No `_state` attribute collision
12. ✅ All tests pass (270/270)

### P1 Gates (Should Pass)

1. ⚠️ KNOW module integrated (Week 1-2)
2. ⚠️ HOW module integrated (Week 2)
3. ⚠️ Dashboard shows live runtime state (Week 3)
4. ⚠️ Provider sync/async fixed (Week 3)
5. ⚠️ Sandbox dynamic collision (Week 4)

### Release Definition

**ROSClaw v1.0 = Unified Physical Intelligence Runtime**

A complete system where:
- LLM queries structured knowledge (KNOW)
- Agent gets fast heuristic recovery (HOW)
- Actions are validated by digital twin (SANDBOX)
- All interactions are recorded (PRACTICE)
- Experiences are stored and queryable (MEMORY)
- System state is observable (DASHBOARD)
- Multiple agents can coordinate (SWARM)

**NOT** a collection of independent modules.

---

## Next Steps

### For Module Owners

1. **Read this document** - Understand your role in the unified system
2. **Read your audit report** - Understand specific issues
3. **Read related audits** - Understand dependencies
4. **Start integration work** - Follow the phased plan above

### For Release Commander

1. **Prioritize Phase 1** - KNOW + HOW integration is critical path
2. **Monitor progress** - Weekly status updates
3. **Resolve blockers** - Cross-module dependencies
4. **Verify gates** - Run acceptance tests before release

### For All Contributors

1. **No new features** - Focus on integration only
2. **No breaking changes** - Respect frozen architecture
3. **Document everything** - Update README, tutorials, API docs
4. **Test everything** - Unit + integration + E2E

---

## Conclusion

ROSClaw v1.0 has a **strong foundation** with production-ready modules. The integration gaps are **clear and addressable**. By following the unified plan, we can deliver a **cohesive Physical Intelligence Runtime** that transforms how robots learn from experience.

**The vision is clear. The path is defined. Let's build it together.**

---

**Questions?** Ask in the relevant tmux session or consult the audit reports in `docs/release/v1.0/audits/`.
