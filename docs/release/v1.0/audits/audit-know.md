# Knowledge Graph (rosclaw-know) Audit Report — v1.0 Full Integration

> **Auditor**: Knowledge Graph domain owner session
> **Date**: 2026-05-28
> **Constraint**: ALL integration must land in v1.0. No v1.1 deferral.

---

## Executive Summary

**rosclaw-know is NOT integrated into ROSClaw v1.0.** There is no `src/rosclaw/know/` directory, no Knowledge module in the Runtime lifecycle, no MCP tool for knowledge queries, and no code path that writes to the `knowledge_graph` SeekDB table.

**However, the groundwork is ready:**
- SeekDB schema defines `knowledge_graph` (6 columns) and `heuristic_rules` (7 columns)
- e-URDF parses `semantic_tags` but they are dead-end data
- `part/rosclaw-know/` (8,840 lines, 349 clusters, 7 domains) is mature and ready for adapter integration
- Runtime's `LifecycleMixin` + `_modules` registry pattern is the exact integration point

**Key architectural decision**: Know is **NOT a service** (no FastAPI port, no resident process). It is a **Runtime-managed batch+query hybrid module**:
- **Query side** (online): `KnowledgeInterface` serves symptom matching, capability lookup, analogy retrieval — zero LLM calls in hot path
- **Batch side** (offline/event-triggered): `KnowledgeBatchEngine` wraps the harvester/weaver/muse pipeline, triggered by EventBus events or cron
- **Asset pipeline**: `bridge_index.json` + `code_patterns/*.md` are shared artifacts consumed by Agent Runtime and How

**This audit proposes a 3-phase v1.0 integration**: Foundation (Week 1) → Query Online (Week 2) → Batch Pipeline (Week 3-4). Total effort: **~80 hours**.

---

## 1. Current State: NOT INTEGRATED

### 1.1 Module Presence

| Check | Result | Evidence |
|-------|--------|----------|
| `src/rosclaw/know/` exists | **NO** | No `know/` in `src/rosclaw/` |
| Runtime manages know lifecycle | **NO** | `runtime.py` lines 91-106: `_firewall`, `_memory`, `_practice`, `_swarm`, `_skill_manager` — no `_knowledge` |
| RuntimeConfig has `enable_knowledge` | **NO** | `RuntimeConfig` (lines 52-71): no knowledge flag |
| EventBus has knowledge topics | **NO** | No `knowledge.*` topics in event_bus.py or runtime.py |
| MCP Hub exposes knowledge tools | **NO** | 11 tools registered, none for knowledge query |
| SeekDB `knowledge_graph` has data | **NO** | Table defined but empty (audit-memory.md confirms) |
| e-URDF `semantic_tags` consumed | **NO** | Parsed (parser.py:256-260) but never read post-parse |

### 1.2 Independent Module Inventory (`part/rosclaw-know/`)

| Component | Lines | Phase | Purpose | v1.0 Relevance |
|-----------|-------|-------|---------|----------------|
| `pipeline.py` | ~106 | 1 | Orchestrator: harvest → weave → muse → publish | Batch engine wrapper |
| `harvester.py` | ~196 | 1 | Async LLM extraction of (symptom, fix) from wiki | Batch side |
| `weaver.py` | ~83 | 1 | NetworkX graph with cross-domain edges | Batch side |
| `muse.py` | ~306 | 1 | LLM cross-domain analogy compiler | Batch side |
| `feedback_distill.py` | ~178 | 4 | Outcomes → per-pattern uplift metrics | Batch side |
| `bridge_reweighter.py` | ~158 | 4 | Merge metrics back into bridge_index | Batch side |
| `source_manifest.py` | ~188 | 5 | Content-hash dirty detection | Batch side |
| `incremental_pipeline.py` | ~291 | 5 | Selective Muse on new nodes only | Batch side |
| `active_learning.py` | ~236 | 7 | Blind-spot → DeepSeek draft → auto-ingest | Batch side |
| `awesome_fetcher.py` | ~388 | 8 | Curated GitHub awesome list ingest | Batch side |
| `prompts.py` | ~220 | — | Extractor / Muse prompt templates | Shared (both sides) |
| `curated_patterns.py` | ~332 | — | 7 hand-written safety patterns | **Query side** — must be online |
| `seekdb_align.py` | ~125 | — | Dedup via sentence-transformer + SeekDB | Batch side |
| **Total** | **~8,840** | | | |

**349 clusters** at v0.8.1, 7 embodied-AI domains, 23/23 staging clusters at perfect markdown completeness.

---

## 2. Architecture: Know is NOT a Service

### 2.1 Why Not a Service?

| Concern | If Service (FastAPI) | If Runtime Module (Hybrid) |
|---------|---------------------|---------------------------|
| Process model | Resident daemon + port | Managed by Runtime lifecycle |
| LLM calls | Would need async queue | Batch side only, EventBus-triggered |
| Startup time | ~10s (model load + asset sync) | Query side: <100ms; Batch side: on-demand |
| Resource footprint | Always-on ~2GB RAM | Query side: ~250KB; Batch side: released after run |
| Coupling to How | Two services talking | Both managed by Runtime, share assets via filesystem |
| Failure mode | Service down = no knowledge | Query side is local code; batch failure is non-blocking |

**Conclusion**: A resident Know service would be an architectural mistake. It would duplicate How's API surface, add a failure point, and waste RAM on a batch-oriented workload.

### 2.2 The Correct Model: Hybrid Batch + Query Module

```
┌─────────────────────────────────────────────────────────────┐
│                    ROSClaw Runtime                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Firewall   │  │    Memory    │  │  Knowledge       │  │
│  │  (resident)  │  │  (resident)  │  │  (hybrid)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                       │ Query │ Batch │    │
│                                       │ side  │ side  │    │
└─────────────────────────────────────────────────────────────┘
              │                              │
              │ EventBus                     │ EventBus (trigger)
              │                              │
┌─────────────▼──────────────┐    ┌──────────▼────────────────┐
│  Agent Runtime / MCP Hub   │    │  part/rosclaw-know/       │
│  • query_knowledge MCP tool│    │  • harvester/weaver/muse  │
│  • symptom match in prompt │    │  • feedback_distill       │
│  • capability injection    │    │  • active_learning        │
│                            │    │  (triggered, not resident)│
└────────────────────────────┘    └───────────────────────────┘
```

### 2.3 Module Boundary Contract (RFC-0001 Section 3)

```text
Module: rosclaw.know

1. What Events do you PUBLISH?
   - KnowledgeRefreshedEvent — "knowledge.assets_refreshed" (bridge_index updated)
   - KnowledgeIngestProgressEvent — "knowledge.ingest_progress" (batch status)
   - PatternMatchedEvent — "knowledge.pattern_matched" (CATALYST hit logged)

2. What Events do you SUBSCRIBE to?
   - PraxisCompletedEvent — "praxis.completed" → trigger incremental ingest
   - KnowledgeIngestRequestEvent — "knowledge.ingest_request" → manual batch trigger
   - RuntimeLifecycleEvent — "runtime.status" → load/unload assets on state change

3. What Schema do you DEPEND on?
   - bridge_index.json schema (symptom_clusters, cross_domain_analogies, priority)
   - code_patterns/*.md schema (Fix, Anti-pattern, Cross-domain, Patch)
   - SeekDB knowledge_graph table (subject/predicate/object triples)
   - SeekDB heuristic_rules table (condition/action/priority)

4. What Contract do you EXPOSE?
   - KnowledgeInterface.query_symptom(error_log) → list[matched_clusters]
   - KnowledgeInterface.get_capabilities(robot_id) → list[str]
   - KnowledgeInterface.get_analogies(symptom) → list[cross_domain_analogies]
   - KnowledgeInterface.get_safety_rule(safety_label) → str (hard-coded constraint)
   - KnowledgeBatchEngine.run_incremental_ingest(paths) → summary_dict
   - KnowledgeBatchEngine.run_full_pipeline() → summary_dict

5. Can you be managed by Runtime lifecycle?
   - YES. KnowledgeInterface is resident (READY state). KnowledgeBatchEngine is
transient (STARTED → runs → STOPPED per invocation).
```

---

## 3. v1.0 Integration Design

### 3.1 Source Layout

```
src/rosclaw/know/
  __init__.py              # exports KnowledgeInterface, KnowledgeBatchEngine
  interface.py             # Query side: symptom match, capability lookup (~200 lines)
  batch_engine.py          # Batch side: wraps rosclaw-know pipeline (~150 lines)
  asset_loader.py          # bridge_index.json + code_patterns/*.md loader (~100 lines)
  curated_patterns.py      # 7 hand-written safety patterns (copied from rosclaw-know)
  config.py                # Know-specific settings (model paths, similarity_floor)
  prompts.py               # Extractor / Muse prompts (copied from rosclaw-know)
```

### 3.2 Runtime Integration

**`RuntimeConfig`** additions:
```python
enable_knowledge: bool = True                    # NEW
knowledge_assets_path: str = "data/knowledge_assets"  # NEW
knowledge_similarity_floor: float = 0.5          # NEW
```

**`Runtime._do_initialize()`** additions (~20 lines):
```python
if self.config.enable_knowledge:
    from rosclaw.know.interface import KnowledgeInterface
    from rosclaw.know.batch_engine import KnowledgeBatchEngine
    self._knowledge = KnowledgeInterface(
        robot_id=self.config.robot_id,
        event_bus=self.event_bus,
        assets_path=self.config.knowledge_assets_path,
        similarity_floor=self.config.knowledge_similarity_floor,
    )
    self._knowledge_batch = KnowledgeBatchEngine(
        event_bus=self.event_bus,
        assets_path=self.config.knowledge_assets_path,
    )
    self._modules.append(self._knowledge)
    # Batch engine is NOT in _modules — it's transient, started per-event
```

**`Runtime` properties** addition:
```python
@property
def knowledge(self) -> Optional[Any]:
    return self._knowledge
```

### 3.3 KnowledgeInterface — Query Side (Resident)

This is the **hot path**. Zero LLM calls. Pure rules + vector cosine.

```python
class KnowledgeInterface(LifecycleMixin):
    """
    Online knowledge query engine for Agent Runtime.

    Loads bridge_index.json + code_patterns/*.md at startup.
    Answers queries via in-memory numpy cosine over symptom embeddings.
    """

    def query_symptom(self, error_log: str, top_k: int = 3) -> list[dict]:
        """
        Match an error log against symptom clusters.
        Returns ranked matches with analogies + diffs.
        Skips soft-deprecated clusters (priority < 0).
        """

    def get_capabilities(self, robot_id: str) -> list[str]:
        """
        Query knowledge_graph for robot capabilities.
        Falls back to e-URDF semantic_tags if knowledge_graph empty.
        """

    def get_safety_rule(self, safety_label: str) -> str:
        """
        Return hard-coded safety constraint for known symptoms.
        Same 5 labels as rosclaw-how: Torque_Overflow, Velocity_Divergence,
        Memory_Exhaustion, Numerical_Instability, Compile_Error.
        """

    def get_cross_domain_analogies(self, symptom: str, max_analogies: int = 3) -> list[dict]:
        """
        Return cross-domain analogies for a matched symptom.
        Used by Agent Runtime to enrich LLM prompts.
        """
```

**Why numpy cosine, not SeekDB query?**
- v1.0 SeekDB is SQLite-backed, no vector extension
- 349 clusters × 384 dims = ~500KB embedding matrix — trivial in RAM
- numpy cosine is <5ms per query vs. SQL round-trip + deserialization
- When v1.1 upgrades SeekDB to vector-capable, swap to SeekDB query with zero API change

### 3.4 KnowledgeBatchEngine — Batch Side (Transient)

Triggered by EventBus events, NOT resident.

```python
class KnowledgeBatchEngine:
    """
    Offline knowledge refinery wrapper.

    Wraps part/rosclaw-know/ pipeline. Runs as transient task,
    not a LifecycleMixin (it doesn't stay in Runtime._modules).
    """

    def run_full_pipeline(self, *, max_pages: int | None = None) -> dict:
        """Run complete harvest → weave → muse → publish."""

    def run_incremental_ingest(self, new_paths: list[Path]) -> dict:
        """Ingest new documents without re-running full pipeline."""

    def run_feedback_reweight(self) -> dict:
        """Distill feedback outcomes and reweight bridge_index."""
```

**Event triggers**:
| Event Topic | Handler | When |
|-------------|---------|------|
| `praxis.completed` | `_on_praxis_completed` | After each practice execution, extract new wiki pages from COT trace |
| `knowledge.ingest_request` | `_on_ingest_request` | Manual/admin trigger |
| `knowledge.full_rebuild` | `_on_full_rebuild` | Nightly cron or operator command |

### 3.5 MCP Hub Integration

Add 2 tools to `mcp_hub.py`:

```python
# 1. Query knowledge for agent reasoning
def _register_query_knowledge_tool(self) -> None:
    self._tools["query_knowledge"] = {
        "name": "query_knowledge",
        "description": "Query the Knowledge Graph for robot capabilities, known failure patterns, or cross-domain engineering analogies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["symptom", "capability", "analogy"],
                    "description": "Type of knowledge to query",
                },
                "query": {"type": "string", "description": "The query text"},
            },
            "required": ["query_type", "query"],
        },
    }

# 2. Get safety heuristic for known dangerous conditions
def _register_get_safety_heuristic_tool(self) -> None:
    self._tools["get_safety_heuristic"] = {
        "name": "get_safety_heuristic",
        "description": "Get a safety heuristic rule for a known dangerous condition (torque overflow, velocity divergence, memory exhaustion, numerical instability).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "condition": {
                    "type": "string",
                    "enum": ["torque_overflow", "velocity_divergence", "memory_exhaustion", "numerical_instability"],
                },
            },
            "required": ["condition"],
        },
    }
```

### 3.6 e-URDF Integration

When `RobotModelLoader` parses a URDF:
1. Write `semantic_tags` to `knowledge_graph` as triples
2. Write `capabilities.yaml` entries to `knowledge_graph`
3. This happens at Runtime initialization, before KnowledgeInterface loads assets

```python
# In e_urdf/parser.py or knowledge_bridge.py
for tag in link.semantic_tags:
    seekdb.insert("knowledge_graph", {
        "subject": robot_name,
        "predicate": "has_capability",
        "object": tag,
        "confidence": 1.0,
        "source": "e_urdf",
        "timestamp": time.time(),
    })
```

### 3.7 Asset Pipeline

```
part/rosclaw-know/
  data/assets/bridge_index.json      ──┐
  data/assets/code_patterns/*.md       │  symlink or copy at build time
                                       ▼
src/rosclaw/know/data/
  bridge_index.json                  ──┐
  code_patterns/                       │  loaded by KnowledgeInterface at startup
                                       ▼
  v1.0 Runtime / Agent Runtime / How
```

**Build-time asset sync**: `make sync-know-assets` copies/symlinks from `part/rosclaw-know/data/assets/` to `src/rosclaw/know/data/`. This is a build step, not runtime dependency.

---

## 4. Cross-Module Value

### 4.1 Know ↔ Agent Runtime

| Agent Runtime Need | KnowledgeInterface Provides | Value |
|-------------------|----------------------------|-------|
| "Why did my code fail?" | `query_symptom(error_log)` → matched cluster + fix | Reduces debugging time |
| "How do I fix torque overflow?" | `get_safety_rule("Torque_Overflow")` → hard constraint | Prevents physical damage |
| "What can this robot do?" | `get_capabilities(robot_id)` → [grasp, lift, navigate] | Better task planning |
| "Any similar failures?" | `get_cross_domain_analogies(symptom)` → cross-domain insights | Transfers solutions across domains |

### 4.2 Know ↔ Practice

| Practice Event | Know Action | Value |
|---------------|-------------|-------|
| `praxis.completed` with `outcome=failed` | Extract error from COT trace → run incremental ingest | Failures become new knowledge |
| `praxis.completed` with `outcome=success` | Extract technique from COT trace → run incremental ingest | Successes become reusable patterns |

### 4.3 Know ↔ Memory

| Memory Query | Know Enhancement | Value |
|-------------|-----------------|-------|
| "Find similar experiences" | + knowledge_graph subject matching | Semantic + episodic memory unified |
| "What do we know about gripper failures?" | `knowledge_graph` triple query + `experience_graph` event query | Structured facts + concrete experiences |

### 4.4 Know ↔ How (Future-Ready)

v1.0 does NOT integrate the full rosclaw-how CATALYST service (that's a separate service with pyseekdb embedded). But v1.0 Know lays the foundation:
- `bridge_index.json` format is identical — How can read v1.0 Know assets directly
- `heuristic_rules` table schema is ready — How can write to it in v1.1
- `outcomes` collection schema is ready — How can export to rosclaw-know in v1.1

### 4.5 Know ↔ Firewall

| Firewall Check | Know Enhancement | Value |
|---------------|-----------------|-------|
| Joint torque validation | `get_safety_rule("Torque_Overflow")` provides threshold + fix | Not just "too high" but "clamp with torch.clamp" |
| Velocity validation | `get_safety_rule("Velocity_Divergence")` provides anti-windup hint | Safety + education combined |

---

## 5. Implementation Plan

### Phase 1: Foundation (Week 1, ~20h)

| Task | File(s) | Hours | Owner |
|------|---------|-------|-------|
| Create `src/rosclaw/know/` package skeleton | `__init__.py`, `config.py` | 2 | core |
| Port 7 curated safety patterns | `curated_patterns.py` | 3 | know |
| Port prompts.py (extractor + muse) | `prompts.py` | 2 | know |
| Implement AssetLoader (bridge_index + code_patterns) | `asset_loader.py` | 4 | know |
| Implement KnowledgeInterface.query_symptom() | `interface.py` | 6 | know |
| Implement KnowledgeInterface.get_safety_rule() | `interface.py` | 2 | know |
| Unit tests for interface | `tests/test_knowledge_interface.py` | 4 | know |

**Deliverable**: `KnowledgeInterface` can load assets and answer `query_symptom()` + `get_safety_rule()`.

### Phase 2: Query Online + Runtime Integration (Week 2, ~25h)

| Task | File(s) | Hours | Owner |
|------|---------|-------|-------|
| Add `enable_knowledge` to RuntimeConfig | `core/runtime.py` | 1 | core |
| Register KnowledgeInterface in Runtime lifecycle | `core/runtime.py` | 3 | core |
| Add `Runtime.knowledge` property | `core/runtime.py` | 1 | core |
| Implement KnowledgeInterface.get_capabilities() | `know/interface.py` | 3 | know |
| Implement KnowledgeInterface.get_cross_domain_analogies() | `know/interface.py` | 3 | know |
| e-URDF → knowledge_graph bridge | `e_urdf/knowledge_bridge.py` or `know/bridge.py` | 4 | e_urdf + know |
| Add MCP `query_knowledge` tool | `agent_runtime/mcp_hub.py` | 4 | agent_runtime |
| Add MCP `get_safety_heuristic` tool | `agent_runtime/mcp_hub.py` | 2 | agent_runtime |
| Inject capabilities into AgentContext | `agent_runtime/mcp_hub.py` | 2 | agent_runtime |
| Integration tests | `tests/test_know_integration.py` | 5 | know + agent_runtime |

**Deliverable**: Agent Runtime can query knowledge via MCP. Runtime manages KnowledgeInterface lifecycle.

### Phase 3: Batch Pipeline + EventBus (Week 3-4, ~35h)

| Task | File(s) | Hours | Owner |
|------|---------|-------|-------|
| Implement KnowledgeBatchEngine wrapper | `know/batch_engine.py` | 6 | know |
| Wire EventBus subscriptions (praxis.completed → ingest) | `know/batch_engine.py` | 4 | know |
| Add `knowledge.*` Event topics to EventBus docs | `core/event_bus.py` + docs | 2 | core |
| Implement seekdb_align.py adapter for v1.0 SeekDB | `know/seekdb_align.py` | 4 | know |
| Implement source_manifest.py adapter | `know/source_manifest.py` | 3 | know |
| Build-time asset sync script | `scripts/sync_know_assets.py` + Makefile | 3 | devops |
| End-to-end test: load URDF → query knowledge → MCP tool | `tests/test_know_e2e.py` | 6 | know |
| Performance benchmark: query latency p95 | `benchmarks/` | 4 | know |
| Documentation: know module contract + API | `docs/know/` | 4 | docs |

**Deliverable**: Full v1.0 Know module with batch+query hybrid, EventBus integration, and E2E tests.

### Total Effort

| Phase | Hours | Risk |
|-------|-------|------|
| Foundation (Query core) | 20 | Low — mostly porting + wiring |
| Query Online + Runtime | 25 | Medium — touches 4 modules (core, know, e_urdf, agent_runtime) |
| Batch Pipeline + Events | 35 | Medium — dependency on part/rosclaw-know/ imports |
| **Total** | **~80** | **Manageable** |

---

## 6. Issue Inventory

### ISSUE-K001 [P0] — No `src/rosclaw/know/` module exists

**Problem**: Knowledge Graph has zero code in v1.0. The `part/rosclaw-know/` module (8,840 lines) is not referenced by any v1.0 code.

**Evidence**: `find src/rosclaw -type d | grep know` returns nothing.

**Why it matters**: This is a P0 because the constraint is "everything in v1.0". Without a Know module, Agent Runtime has no structured knowledge source. It cannot answer "what can this robot do?" or "how do I fix this error?". The LLM must infer everything from free-form text, increasing hallucination risk.

**Suggested fix**: Create `src/rosclaw/know/` per Phase 1 above. Start with `KnowledgeInterface` (query side) which has no external dependencies beyond numpy + sentence-transformers (already in pyproject.toml).

**Verification**: `import rosclaw.know` succeeds; `KnowledgeInterface` instantiates under Runtime.

---

### ISSUE-K002 [P0] — RuntimeConfig lacks knowledge enablement

**Problem**: `RuntimeConfig` has flags for firewall, memory, practice, swarm, skill_manager, provider — but no `enable_knowledge`.

**Evidence**: `src/rosclaw/core/runtime.py` lines 52-71.

**Why it matters**: Without a config flag, operators cannot enable/disable Know per-deployment. It also signals that Know is not a first-class module.

**Suggested fix**: Add `enable_knowledge: bool = True` and `knowledge_assets_path: str = "data/knowledge_assets"` to `RuntimeConfig`.

**Verification**: `RuntimeConfig(enable_knowledge=True)` works; Runtime initializes Know module.

---

### ISSUE-K003 [P0] — No knowledge topics in EventBus

**Problem**: EventBus has `robot.joint_states`, `agent.command`, `robot.emergency_stop`, `safety.violation`, `runtime.status` — but no `knowledge.*` topics.

**Evidence**: `src/rosclaw/core/event_bus.py` + `runtime.py:_setup_internal_subscriptions()`.

**Why it matters**: Without knowledge topics, Practice cannot trigger Know ingest, and Know cannot notify Agent Runtime when assets refresh. The modules are siloed.

**Suggested fix**: Add topic constants to event_bus.py:
```python
# Knowledge topics
KNOWLEDGE_INGEST_REQUEST = "knowledge.ingest_request"
KNOWLEDGE_ASSETS_REFRESHED = "knowledge.assets_refreshed"
KNOWLEDGE_PATTERN_MATCHED = "knowledge.pattern_matched"
```

**Verification**: Events with these topics flow through EventBus without errors.

---

### ISSUE-K004 [P1] — knowledge_graph table is empty (Anti-Pattern #3)

**Problem**: SeekDB `knowledge_graph` table is defined but has no writer.

**Evidence**: `seekdb_client.py:52-62` defines schema; audit-memory.md confirms "no v1.0 writer".

**Why it matters**: RFC-0001 Anti-Pattern #3: "SeekDB as Memory-Only". An empty knowledge_graph makes SeekDB a de facto Memory-only store. This is a structural lie — the schema promises a Knowledge Plane that doesn't exist.

**Suggested fix**: Implement e-URDF → knowledge_graph bridge (Phase 2, ~4h). Every URDF parse writes capability triples.

**Verification**: After loading a URDF with semantic tags, `seekdb.count("knowledge_graph") > 0`.

---

### ISSUE-K005 [P1] — MCP Hub has no knowledge query tools

**Problem**: 11 MCP tools exist, none for knowledge queries.

**Evidence**: `mcp_hub.py` tools list — no `query_knowledge`, `get_safety_heuristic`.

**Why it matters**: The LLM cannot ask "What is this robot's max torque?" or "Has this skill succeeded before?" It must infer from free-form `robot_model_description`, which is unreliable.

**Suggested fix**: Add `query_knowledge` and `get_safety_heuristic` MCP tools (Phase 2, ~6h).

**Verification**: `tests/test_mcp_hub.py` — tool exists and returns non-empty for seeded knowledge.

---

### ISSUE-K006 [P1] — e-URDF semantic_tags are dead-end data

**Problem**: `e_urdf.parser.py` parses `semantic_tags` but no module consumes them.

**Evidence**: `parser.py:256-260` parses tags; no downstream reader.

**Why it matters**: Wasted parsing effort + missed grounding opportunity. Tags like "grasp", "lift", "precision_grip" represent the robot's capabilities — exactly what the LLM needs for task planning.

**Suggested fix**: Write tags to knowledge_graph at parse time; expose via `KnowledgeInterface.get_capabilities()`.

**Verification**: Load URDF → query capabilities → assert tags present.

---

### ISSUE-K007 [P1] — AgentContext lacks knowledge grounding

**Problem**: `AgentContext.to_mcp_context()` sends free-form `robot_model_description` but no structured capabilities.

**Evidence**: `mcp_hub.py:45-60` — no `capabilities`, `known_facts`, or `heuristics` fields.

**Why it matters**: The LLM reasons from a blob of text instead of structured triples. This increases hallucination and reduces precision.

**Suggested fix**: Add `robot_capabilities: list[str]` to AgentContext, populated from knowledge_graph at initialization.

**Verification**: MCP context JSON includes structured capability list.

---

### ISSUE-K008 [P2] — heuristic_rules table empty but undocumented

**Problem**: `heuristic_rules` table is defined but empty and undocumented.

**Evidence**: `seekdb_client.py:64-75` — no docstring explaining v1.1 purpose.

**Why it matters**: Future contributors may repurpose this table. It's reserved for How module integration.

**Suggested fix**: Add comment: "Reserved for How module v1.1 — do not use for other purposes."

**Verification**: Code review.

---

### ISSUE-K009 [P2] — No build-time asset sync from part/rosclaw-know/

**Problem**: v1.0 `src/rosclaw/know/` has no mechanism to ingest assets from `part/rosclaw-know/data/assets/`.

**Why it matters**: The 349 clusters in rosclaw-know are valuable. Without asset sync, v1.0 Know starts from zero.

**Suggested fix**: Add `scripts/sync_know_assets.py` that copies/symlinks `bridge_index.json` + `code_patterns/` at build time.

**Verification**: `make sync-know-assets` populates `src/rosclaw/know/data/`.

---

## 7. Acceptance Gates for Know v1.0

| Gate | Verification | Phase |
|------|-------------|-------|
| `rosclaw.know` imports without error | `python -c "import rosclaw.know"` | 1 |
| Runtime initializes KnowledgeInterface | `tests/test_core.py` extension | 2 |
| Query symptom returns ranked matches | `tests/test_knowledge_interface.py` | 1 |
| Safety rule returns hard-coded constraint | `tests/test_knowledge_interface.py` | 1 |
| MCP `query_knowledge` tool exists | `tests/test_mcp_hub.py` | 2 |
| MCP `get_safety_heuristic` tool exists | `tests/test_mcp_hub.py` | 2 |
| e-URDF semantic tags write to knowledge_graph | `tests/test_know_e2e.py` | 2 |
| EventBus `knowledge.*` topics work | `tests/test_event_bus.py` | 3 |
| Praxis event triggers incremental ingest | `tests/test_know_e2e.py` | 3 |
| Build-time asset sync works | `make sync-know-assets && test -f src/rosclaw/know/data/bridge_index.json` | 3 |
| No regression in 157+ existing tests | `pytest` | All |

---

## 8. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| sentence-transformers model download (~120MB) fails in CI | Medium | Blocks tests | Pin model in `tests/conftest.py` with mock embeddings fallback |
| part/rosclaw-know/ API changes break adapter | Low | Build failure | Lock to git tag in Makefile; adapter tests catch drift |
| Embedding dimension mismatch (384 vs future change) | Low | Silent failures | Store `embedding_dim` in bridge_index.json; assert at load time |
| Batch engine LLM costs (DeepSeek API) | Medium | Budget overrun | `ROSCLAW_KNOW_MOCK_LLM=1` for tests; batch only runs on explicit trigger |
| Asset file size (bridge_index.json + 349 patterns) | Low | Slow startup | Lazy-load patterns; keep bridge_index in memory (~2MB) |

---

## 9. Summary

**Status**: NOT INTEGRATED. Must be integrated in v1.0 per release constraint.

**Architecture**: Know is a **hybrid batch+query module** managed by Runtime lifecycle. Not a service.

**Effort**: ~80 hours over 3-4 weeks (Foundation → Query Online → Batch Pipeline).

**P0 Blockers** (must fix before RC1):
1. Create `src/rosclaw/know/` module
2. Add `enable_knowledge` to RuntimeConfig
3. Add `knowledge.*` EventBus topics

**P1 Issues** (must fix before RC2):
4. Fill knowledge_graph from e-URDF
5. Add MCP knowledge query tools
6. Consume semantic_tags
7. Enrich AgentContext with capabilities

**P2 Items** (RC3 or post-release):
8. Document heuristic_rules purpose
9. Build-time asset sync from part/rosclaw-know/

**The v1.0 Know module will give Agent Runtime**: structured robot capabilities, symptom-based failure recovery, cross-domain engineering analogies, and safety heuristics — all with zero LLM calls in the hot path.

---

*End of Audit Report*
