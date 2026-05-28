# RFC-0001: ROSClaw v1.0 Architecture Freeze

> **Status**: DRAFT — Pending review by all auditors
> **Author**: Release Commander
> **Date**: 2026-05-28
> **Scope**: Freeze the architectural boundaries for v1.0 release

---

## 1. Purpose

This RFC freezes the core architecture decisions for ROSClaw v1.0. All Claude Code instances (integrators, auditors, domain owners) MUST align their work to these boundaries. Any violation found during audit rounds must be flagged as P0 or P1.

## 2. Four Public Infrastructure Pillars

### 2.1 rosclaw-runtime (The Orchestrator)

- **Location**: `src/rosclaw/core/runtime.py`
- **Responsibility**: Unified lifecycle management, configuration, plugin loading, event orchestration
- **Rule**: ALL modules MUST be instantiated and managed by Runtime. No module may start itself.
- **Lifecycle states**: UNINITIALIZED -> INITIALIZING -> READY -> STARTING -> RUNNING -> STOPPING -> STOPPED -> ERROR

### 2.2 rosclaw-event-bus (The Decoupling Layer)

- **Location**: `src/rosclaw/core/event_bus.py`
- **Responsibility**: Publish/subscribe event communication between modules
- **Rule**: Modules MUST NOT import each other directly. Communication MUST flow through EventBus.
- **Core events for v1.0**:
  - `PraxisEvent` — practice publishes, memory/how subscribes
  - `PraxisFailedEvent` — practice publishes, memory/how/darwin subscribe
  - `RuntimeLifecycleEvent` — runtime publishes, all modules subscribe
  - `AgentCommand` — agent_runtime publishes, provider/firewall subscribe
  - `SkillLoaded` — skill_manager publishes, dashboard subscribes

### 2.3 SeekDB as Knowledge Plane (NOT just Memory DB)

- **Location**: `src/rosclaw/memory/seekdb_client.py` + independent `part/rosclaw-memory/`
- **Responsibility**: Shared storage for ALL modules
  - Memory: experience graph
  - Practice: event index
  - Know: knowledge graph
  - How: heuristic rules
  - Darwin: evaluation results
  - Skill: skill metadata
- **Rule**: SeekDB is Infrastructure Layer, not Memory Layer. Any module may read/write via its contract.
- **v1.0 minimum**: Support Memory + Practice event storage with SQLite backend.

### 2.4 e-URDF as Physical DNA Registry

- **Location**: `src/rosclaw/e_urdf/parser.py` + `part/e-urdf-zoo/`
- **Responsibility**: Robot physical identity definition
  - `safety.yaml` -> consumed by Firewall/Sandbox
  - `capabilities.yaml` -> consumed by Swarm
  - `semantic.yaml` -> consumed by Dashboard
  - `benchmark.yaml` -> consumed by Darwin
- **Rule**: e-URDF is NOT just a model repository. It is the Physical DNA Registry that all grounding engines reference.

## 3. Module Boundary Contract

Each module MUST answer these questions:

```text
1. What Events do you PUBLISH?
2. What Events do you SUBSCRIBE to?
3. What Schema do you DEPEND on?
4. What Contract do you EXPOSE?
5. Can you be managed by Runtime lifecycle?
```

### Module Inventory for v1.0

| Module | v1.0 Status | v1.0 Scope |
|--------|-------------|------------|
| core (Runtime + EventBus + Lifecycle) | Integrated | P0 - Must be stable |
| agent_runtime (MCP Hub + AI Collaboration) | Integrated | P0 - Must expose MCP |
| e_urdf (Parser) | Integrated | P0 - Must parse robot configs |
| firewall (Decorator + Validator) | Integrated | P0 - Basic joint validation |
| sandbox (Digital Twin) | In Development (tmux: sandbox) | P2 - Integration in v1.1 |
| memory (Interface + SeekDB) | Integrated (simplified) | P0 - Basic event storage |
| practice (Recorder + Timeline) | Integrated | P0 - PraxisEvent capture |
| skill_manager (Registry/Loader/Executor) | Integrated | P1 - Basic skill execution |
| swarm (Manager) | Integrated (minimal) | P1 - Multi-agent-ready interfaces |
| mcp_drivers (ROS2/MuJoCo/Serial) | Integrated | P1 - Mock mode sufficient |
| data (RingBuffer/Flywheel) | Integrated | P2 - Basic buffering |
| provider (Loader/Registry/Runtime) | Integrated | P0 - ProviderContract |
| dashboard (UI) | Independent (tmux: dashboard) | P1 - Runtime observability |

### Independent Modules (part/) - Not Yet Integrated

| Module | Code Size | Integration Priority |
|--------|-----------|---------------------|
| rosclaw-memory (independent) | 95,620 lines | P1 - Schema alignment |
| rosclaw-practice (independent) | 1,336 lines | P1 - PraxisEvent alignment |
| rosclaw-swarm (independent) | 2,045 lines | P2 - Interface readiness |
| rosclaw-how | 7,840 lines | P2 - v1.1 |
| rosclaw-know | 8,840 lines | P2 - v1.1 |
| rosclaw-darwin | 2,176 lines | P2 - v1.1 |
| rosclaw-sandbox | In Development | P2 - v1.1 |

## 4. Anti-Patterns (P0 Violations)

These patterns MUST NOT exist in v1.0 release:

1. **Runtime Bypass**: Module A directly imports Module B's internals
2. **Event Bus Decoration**: EventBus exists but modules still call each other directly
3. **SeekDB as Memory-Only**: SeekDB used only by memory module, not as Knowledge Plane
4. **e-URDF as Model Repo**: e-URDF treated as just a file storage, not a DNA Registry
5. **Hardcoded Agent**: Any module hardcodes Claude/GPT/Qwen instead of using AgentRuntime abstraction
6. **Self-Starting Module**: Any module that starts itself outside Runtime lifecycle

## 5. Known Gaps (from GAP_ANALYSIS.md + ROLE_SWAP_REVIEW.md)

| Gap | Source | Severity | v1.0 Action |
|-----|--------|----------|-------------|
| Firewall only basic joint limits | GAP_ANALYSIS | P1 | Document as v1.0 scope |
| Memory only in-memory backend | GAP_ANALYSIS | P1 | SQLite backend minimum |
| MCAP recording not implemented | GAP_ANALYSIS | P2 | Defer to v1.1 |
| Real hardware drivers missing | GAP_ANALYSIS | P2 | Mock mode acceptable |
| _state attribute collision | ROLE_SWAP_REVIEW | P0 | FIXED - verify |
| API usability 6/10 | ROLE_SWAP_REVIEW | P1 | Improve examples |
| Thread safety only in EventBus | ARCH_AUDIT | P1 | Document limitations |
| Memory buffer growth in Timeline | ARCH_AUDIT | P1 | Add size limits |

## 6. Freeze Date

Architecture is frozen as of this RFC approval. No new modules, no new inter-module dependencies, no structural changes without a new RFC.

All work from this point forward is: **audit -> fix -> test -> document -> release**.
