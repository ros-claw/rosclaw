# ROSClaw-How (Heuristic Rules & Recovery) v1.0 Integration Audit

> **Auditor**: Heuristic Rules & Recovery Auditor  
> **Date**: 2026-05-28  
> **Scope**: `rosclaw-how` integration status in v1.0 vs. RFC-0001 / RFC-0005 / independent module  
> **Status**: **NOT INTEGRATED** — v1.0 has zero how module code; SeekDB `heuristic_rules` table is a ghost schema

---

## Executive Summary

**rosclaw-how is completely absent from v1.0.**

| Dimension | Finding |
|-----------|---------|
| Source module | `src/rosclaw/how/` — **does not exist** |
| Runtime registration | `RuntimeConfig.enable_how` — **does not exist** |
| SeekDB table | `heuristic_rules` schema defined — **zero reads, zero writes** |
| EventBus subscribers | `praxis.failed` — subscribed by Timeline only, **no how handler** |
| Agent Runtime recovery | `analyze_failure()` calls DeepSeek API directly — **no heuristic lookup** |
| Firewall safety | 3-layer validation (e-URDF / MuJoCo / semantic) — **no heuristic rule injection** |
| MCP tools | No heuristic query tool exposed |

**Verdict**: v1.0 has **0% how integration**. The `heuristic_rules` table in SeekDB is a "ghost schema" — defined but never populated or queried. All failure recovery is delegated to the LLM (`DeepSeekClient.analyze_failure`), which is slow, non-deterministic, and learns nothing from past failures.

**However**, rosclaw-how (the independent module at `part/rosclaw-how/`, 7,840 lines) is a **mature, production-ready service** with 117 passing tests, a complete feedback loop, and proven value in Frontier-Engineering benchmarks. A **minimal v1.0 integration** is feasible and would materially improve the release narrative.

---

## 1. Current Integration Status

### 1.1 Module Existence Check

```bash
$ ls /home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0/src/rosclaw/
agent_runtime  cli.py  core  data  e_urdf  firewall  __init__.py
mcp  mcp_drivers  memory  practice  provider  skill_manager  specs  swarm
# NO "how" directory
```

**Finding**: No `src/rosclaw/how/` directory. No `HowInterface`, `HeuristicEngine`, or `RecoveryPlanner` class.

### 1.2 RuntimeConfig Check

```python
# src/rosclaw/core/runtime.py:52-71
@dataclass
class RuntimeConfig:
    enable_firewall: bool = True
    enable_memory: bool = True
    enable_practice: bool = True
    enable_swarm: bool = False
    enable_skill_manager: bool = True
    enable_provider: bool = True
    # enable_how: bool = False  <- MISSING
```

**Finding**: No `enable_how` flag. Runtime lifecycle does not initialize, start, or stop any how module.

### 1.3 SeekDB `heuristic_rules` Table

```python
# src/rosclaw/memory/seekdb_client.py:64-75
"heuristic_rules": {
    "columns": {
        "id": "TEXT PRIMARY KEY",
        "condition": "TEXT NOT NULL",
        "action": "TEXT NOT NULL",
        "priority": "INTEGER DEFAULT 0",
        "success_count": "INTEGER DEFAULT 0",
        "failure_count": "INTEGER DEFAULT 0",
        "last_triggered": "REAL",
    },
    "indices": ["priority"],
},
```

**Finding**: Schema exists with 7 columns. **Zero code paths read or write this table.**

Evidence:
```bash
$ grep -rn "heuristic_rules" /home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0/src/rosclaw/ --include="*.py"
src/rosclaw/memory/seekdb_client.py:64    "heuristic_rules": {
# Only the schema definition. No insert(), no query(), no update().
```

### 1.4 EventBus — `praxis.failed` Flow

```python
# src/rosclaw/practice/timeline.py:95
self._event_bus.subscribe("praxis.failed", self._on_praxis_failed)

# src/rosclaw/practice/timeline.py:189-194
def _on_praxis_failed(self, event: Event) -> None:
    correlation_id = event.payload.get("correlation_id", "unknown")
    self._record(TimelineChannel.PRAXIS_EVENT, {
        "event_type": "failure",
        "error": event.payload.get("error", "unknown"),
    }, correlation_id=correlation_id)
```

**Finding**: `praxis.failed` is recorded to the timeline as a JSON entry. **No subscriber queries heuristic rules for recovery suggestions.** The failure dies on the timeline; no recovery action is triggered.

### 1.5 Agent Runtime — Failure Recovery

```python
# src/rosclaw/agent_runtime/ai_collaboration.py:117-161
def analyze_failure(self, task_description: str, error_log: str) -> dict:
    """Analyze a task failure and suggest recovery."""
    system_prompt = """You are a robot failure analyst for ROSClaw.
    Analyze task failures and suggest recovery strategies.
    ..."""
    # Calls DeepSeek API directly — NO heuristic rule lookup
    response = client.chat.completions.create(...)
```

**Finding**: `DeepSeekClient.analyze_failure()` sends the error log to the LLM and waits for a JSON response. This is:
- **Slow**: ~500-2000ms per call (network round-trip + LLM inference)
- **Non-deterministic**: Same error may yield different recovery strategies
- **Stateless**: Does not learn from past failures on this robot
- **Expensive**: Costs API tokens for every failure

**What it should do**: First query local `heuristic_rules` table for known conditions (O(1) with index), fall back to LLM only for unknown errors.

### 1.6 Firewall — Safety Validation

```python
# src/rosclaw/firewall/validator.py:222-256
def validate(self, request: ValidationRequest) -> ValidationResponse:
    """Run 3-layer validation pipeline."""
    layer1 = self._check_eurdf_limits(request)      # Static joint limits
    layer2 = self._check_mujoco_collision(request)  # Physics simulation
    layer3 = self._check_semantic_safety(request)   # Velocity limits
```

**Finding**: Firewall has **3 hardcoded validation layers**. No heuristic rule injection:
- No "if torque > 200 Nm, suggest anti-windup clamp" rule
- No "if velocity divergence detected, add output saturation" rule
- No priority-based rule ordering
- No rule success/failure tracking

When Firewall blocks an action, it publishes `safety.violation` with a description string. **No recovery strategy is suggested.**

### 1.7 Runtime — `safety.violation` Handler (Only Emergency Stop)

```python
# src/rosclaw/core/runtime.py:245-259
def _setup_internal_subscriptions(self) -> None:
    self.event_bus.subscribe("safety.violation", self._on_safety_violation)

def _on_safety_violation(self, event: Event) -> None:
    """Handle safety violation events."""
    print(f"[Runtime] SAFETY VIOLATION: {event.payload}")
    self.event_bus.publish(Event(
        topic="robot.emergency_stop",
        payload={"reason": event.payload},
        source="runtime",
        priority=EventPriority.CRITICAL,
    ))
```

**Finding**: Runtime **already subscribes** to `safety.violation` — this is the closest existing integration point. But the handler does nothing except print and trigger `robot.emergency_stop`. **No heuristic recovery is attempted before the hard stop.**

**What it should do**: Before publishing `robot.emergency_stop`, query `heuristic_rules` for a recovery strategy. If a recovery is found, publish `recovery.suggestion` and give the agent a chance to apply it before escalating to emergency stop.

### 1.8 Agent Runtime — `llm_provider.py` Also Has `analyze_failure`

```python
# src/rosclaw/agent_runtime/llm_provider.py:171-200
def analyze_failure(self, task_description: str, error_log: str) -> dict:
    """Analyze a task failure and suggest recovery."""
    # Identical DeepSeek API call — NO heuristic lookup
    response = client.chat.completions.create(...)
```

**Finding**: **`llm_provider.py:171` has a second `analyze_failure()`**, identical to `ai_collaboration.py:117`. Both call DeepSeek API directly. **If how is integrated, BOTH files need modification.** Report previously only audited `ai_collaboration.py`.

### 1.9 Provider Layer — `SafetySpec` Is Isolated from `heuristic_rules`

```python
# src/rosclaw/provider/core/manifest.py:141
safety: SafetySpec = field(default_factory=SafetySpec)

# src/rosclaw/provider/client.py:63, 164, 206, 310
safety_level: str = "MODERATE"  # STRICT | MODERATE | LENIENT
```

**Finding**: Provider manifest defines `SafetySpec` (`executable`, `requires_guard`, `fallback_provider`) and client accepts `safety_level`. **These concepts are completely isolated from `heuristic_rules`.** There is no "load stricter heuristic rules when safety_level=STRICT" logic.

### 1.10 `praxis.failed` Event Payload Inspection

```python
# src/rosclaw/practice/timeline.py:189-194
def _on_praxis_failed(self, event: Event) -> None:
    correlation_id = event.payload.get("correlation_id", "unknown")
    self._record(TimelineChannel.PRAXIS_EVENT, {
        "event_type": "failure",
        "error": event.payload.get("error", "unknown"),
    }, ...)
```

**Finding**: The `praxis.failed` payload only contains `correlation_id` + `error` string. **Missing fields rosclaw-how needs**: `error_log` (full traceback), `previous_scores` (verifier history), `current_iteration` (to decide FREE_EXPLORATION vs CATALYST). Even if how existed, it could not make an informed decision with this minimal payload.

### 1.11 Sandbox Module — Does Not Exist in v1.0

```bash
$ ls /home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0/src/rosclaw/sandbox*
No sandbox in src
```

**Finding**: `src/rosclaw/sandbox/` **does not exist** (RFC-0001 marks it as "In Development, P2, v1.1"). The Cross-Module Value question "How does it enhance Sandbox?" is moot for v1.0.

### 1.12 `heuristic_rules` Schema vs. rosclaw-how Actual Architecture — **Incompatible**

v1.0 SeekDB `heuristic_rules` schema (simple key-value):
```
id | condition | action | priority | success_count | failure_count | last_triggered
```

rosclaw-how actual architecture (vector semantic search + feedback loop):
| Collection | Purpose | Schema |
|---|---|---|
| `symptom_index` | Vector-indexed symptom clusters | `symptom`, `keywords`, `cross_domain_analogies`, `associated_patterns`, `priority`, `content_hash` |
| `code_pattern_library` | Markdown code patterns | `source_file`, `content_hash`, `document` (full markdown body) |
| `injection_outcomes` | Feedback loop data | `symptom`, `pattern_id`, `similarity`, `pre_score`, `post_score`, `delta_score`, `iterations_to_resolve` |

**Finding**: The two schemas are **completely incompatible**. v1.0's `heuristic_rules` is a condition-action lookup table. rosclaw-how is a vector-semantic search system with cross-domain analogies, unified diff snippets, and per-pattern uplift tracking. **The "re-use existing heuristic_rules table" proposal in Section 3 is a v1.0 shim, NOT a rosclaw-how integration.**

---

## 2. What We Have (Independent Module)

The **independent** `rosclaw-how` module (`/home/ubuntu/rosclaw/rosclaw_wiki/rosclaw-how/`) is a **complete, tested, benchmarked service**:

| Capability | Status | Evidence |
|------------|--------|----------|
| 3-strategy state router (SAFETY/FREE_EXP/CATALYST) | ✅ | `state_router.py` — pure rules, <1ms |
| Error normalizer (10 symptom labels) | ✅ | `error_normalizer.py` — regex, zero LLM |
| Vector search (SeekDB + sentence-transformers) | ✅ | `semantic_router.py` — p95 ~400ms |
| In-memory fallback router | ✅ | `inmemory_router.py` — numpy cosine |
| Safety rule injection (hardcoded constraints) | ✅ | `SAFETY_RULES` — Torque/Velocity/Memory/Instability/Compile |
| Cross-domain analogy + diff snippet assembly | ✅ | `assemble_inspiration()` — ≤400 tokens |
| Feedback loop (injection_id → post_score → delta) | ✅ | `outcomes.py` — per-pattern uplift/win_rate |
| Blind-spot detector (Unknown_Error prefix tracking) | ✅ | `blind_spots.py` — 1h sliding window |
| Hot reload (delta-sync with content-hash) | ✅ | `asset_loader.py` — SHA-256 fingerprint |
| Staging/production/demoted priority gate | ✅ | `priority` field: -1/0/+1 |
| Admin promote endpoint | ✅ | `/admin/promote` — maturity mutation |
| Operator dashboard (HTML+JS, no CDN) | ✅ | `/ui` — polls health/stats/blind_spots |
| Tests | ✅ | 117/117 pass |
| Frontier-Eng A/B benchmark | ✅ | `verify_how.py` — PIDTuning task |

**This is not a prototype. It is a production service.** Deferring it to v1.1 is leaving real value on the table.

---

## 3. v1.0 Minimum Viable Integration Proposal

### 3.1 Proposal: "Heuristic Safety Overlay"

**Goal**: Add the smallest possible how integration that provides real value to v1.0's core loop.

**Scope**:
1. **Firewall enrichment**: When Firewall blocks an action, query `heuristic_rules` for a recovery suggestion
2. **Agent Runtime fallback**: When `analyze_failure()` is called, check local rules first before hitting DeepSeek
3. **PraxisFailedEvent handler**: Auto-populate `heuristic_rules` from `error_details` + `recovery_strategy`

**Why this is minimal**: Reuses existing SeekDB table, existing EventBus topics, existing Agent Runtime code paths. No new services, no new dependencies.

### 3.2 Concrete Changes

#### Change A: `src/rosclaw/how/__init__.py` + `heuristic_engine.py` (~150 lines)

```python
# src/rosclaw/how/heuristic_engine.py
"""Minimal heuristic rule engine for v1.0.

Queries SeekDB heuristic_rules table for known failure conditions.
Falls back to LLM for unknown errors.
"""
from rosclaw.memory.seekdb_client import SeekDBClient

class HeuristicEngine:
    def __init__(self, seekdb: SeekDBClient):
        self._seekdb = seekdb

    def find_recovery(self, error_log: str, error_type: str = "") -> dict | None:
        """Query heuristic_rules for matching recovery strategy."""
        # 1. Exact match on error_type
        rules = self._seekdb.query(
            "heuristic_rules",
            filters={"condition": error_type},
            order_by="-priority",
            limit=1,
        )
        if rules:
            return {"strategy": rules[0]["action"], "source": "heuristic"}

        # 2. Keyword match on error_log (fallback)
        all_rules = self._seekdb.query("heuristic_rules", limit=100)
        for rule in all_rules:
            if rule["condition"] in error_log:
                return {"strategy": rule["action"], "source": "heuristic"}
        return None

    def record_rule(self, condition: str, action: str, priority: int = 0) -> str:
        """Record a new heuristic rule to SeekDB."""
        return self._seekdb.insert("heuristic_rules", {
            "id": f"rule_{condition[:20]}",
            "condition": condition,
            "action": action,
            "priority": priority,
        })
```

#### Change B: `src/rosclaw/core/runtime.py` (~10 lines)

```python
# In RuntimeConfig:
enable_how: bool = True          # NEW

# In Runtime._do_initialize():
if self.config.enable_how:
    from rosclaw.how.heuristic_engine import HeuristicEngine
    self._how = HeuristicEngine(self._memory._seekdb)
    # Subscribe to praxis.failed for auto-learning
    self.event_bus.subscribe("praxis.failed", self._on_praxis_failed)
```

#### Change C: `src/rosclaw/agent_runtime/ai_collaboration.py` (~15 lines)

```python
# In DeepSeekClient.analyze_failure():
def analyze_failure(self, task_description: str, error_log: str,
                    heuristic_engine=None) -> dict:
    # 1. Try heuristic first (fast, deterministic, free)
    if heuristic_engine:
        heuristic = heuristic_engine.find_recovery(error_log)
        if heuristic:
            return {
                "root_cause": "matched_heuristic",
                "severity": "medium",
                "recovery_strategy": heuristic["strategy"],
                "preventive_measures": ["from heuristic_rules table"],
            }
    # 2. Fall back to LLM (slow, expensive, but handles novel errors)
    ...existing LLM code...
```

#### Change D: `src/rosclaw/firewall/validator.py` (~20 lines)

```python
# In _on_agent_command(), after publishing safety.violation:
if not response.is_safe:
    self._event_bus.publish(Event(
        topic="safety.violation",
        payload={...},
        ...
    ))
    # NEW: Query heuristic for recovery suggestion
    if hasattr(self, '_how') and self._how:
        recovery = self._how.find_recovery(
            error_log="; ".join(v.description for v in response.violations)
        )
        if recovery:
            self._event_bus.publish(Event(
                topic="safety.recovery_suggestion",
                payload={
                    "request_id": request.request_id,
                    "suggestion": recovery["strategy"],
                    "source": "heuristic",
                },
                source="firewall_validator",
            ))
```

#### Change E: Seed `heuristic_rules` with 5 safety rules (~30 lines)

```python
# In Runtime._do_initialize() or a migration script:
if self.config.enable_how and self._how:
    default_rules = [
        ("joint limit exceeded", "Reduce Kp gain by 30% and re-validate", 1),
        ("collision detected", "Replan trajectory with larger clearance", 1),
        ("velocity exceeds limit", "Add output saturation clamp", 1),
        ("torque overflow", "Check PID anti-windup; clamp to rated limit", 2),
        ("timeout", "Reduce waypoint count; check network latency", 0),
    ]
    for condition, action, priority in default_rules:
        self._how.record_rule(condition, action, priority)
```

### 3.3 Integration Effort Estimate

| Task | Lines | Hours | Risk |
|------|-------|-------|------|
| Create `src/rosclaw/how/` module (engine + __init__) | ~150 | 2 | Low |
| Wire into Runtime lifecycle | ~20 | 1 | Low |
| Modify Agent Runtime to query heuristics first | ~15 | 1 | Low |
| Modify Firewall to publish recovery suggestions | ~20 | 1 | Low |
| Seed default safety rules | ~30 | 1 | Low |
| Add tests (heuristic lookup, fallback, seeding) | ~80 | 2 | Low |
| Update RuntimeConfig docs / examples | ~10 | 0.5 | Low |
| **Total** | **~325** | **~8.5** | **Low** |

**This is a single-day integration.** The independent module already has all the logic; we just need a thin v1.0 adapter.

---

## 4. Cross-Module Value Analysis

### 4.1 How -> Practice (Failure Recovery)

| Current State | With How Integration |
|---------------|----------------------|
| Practice records failures to timeline | Practice records failures + **receives recovery suggestion** |
| Agent must wait for LLM to analyze | Agent gets **instant heuristic recovery** (<10ms) |
| Same failure repeated = same LLM cost | Same failure repeated = **free cached rule** |

### 4.2 How -> Memory (Learning from Failures)

| Current State | With How Integration |
|---------------|----------------------|
| Memory stores `error_details` as text | Memory stores `error_details` + **links to heuristic_rule** |
| No correlation between errors and fixes | `heuristic_rules.success_count/failure_count` track efficacy |
| `find_similar_experiences` does keyword match | Can query "what heuristic worked for similar errors?" |

### 4.3 How -> Firewall (Safety Heuristics)

| Current State | With How Integration |
|---------------|----------------------|
| Firewall blocks with description string | Firewall blocks with **recovery suggestion** |
| Agent must infer fix from violation text | Agent receives **concrete action** ("Reduce Kp by 30%") |
| No learning from repeated violations | Violation -> heuristic -> success -> **rule priority boosted** |

---

## 5. Issue List

### P0 Issues (Release Blockers — if integration is accepted)

**None.** How is not integrated, so it cannot block release. If the integration proposal is accepted, all changes are additive and low-risk.

### P1 Issues (Should Fix Before Release — if integrated)

---

#### ISSUE-001: `heuristic_rules` table has no query performance for substring match

**Severity**: P1  
**Module**: memory + how  
**Detected by**: how-auditor  
**Status**: open (only relevant if integrated)

**Problem**: The proposed `HeuristicEngine.find_recovery()` does a linear scan of all rules for substring matching. With 100+ rules, this is O(N) and will degrade.

**Evidence**:
- `seekdb_client.py:133-148` — `SeekDBMemoryClient.query()` does `for r in records: match = all(...)`
- `seekdb_client.py:221-235` — `SeekDBSQLiteClient.query()` does `SELECT * FROM {table}` with `WHERE` only for exact equality
- No `LIKE` or `FULLTEXT` index on `condition` column

**Why it matters**: If v1.0 seeds 50+ heuristic rules, substring matching on every failure becomes a bottleneck.

**Suggested fix**:
```sql
-- Add FULLTEXT index on heuristic_rules.condition
CREATE VIRTUAL TABLE heuristic_rules_fts USING fts5(condition, action);
-- Or simpler: add LIKE index
CREATE INDEX idx_heuristic_condition ON heuristic_rules(condition);
```

**Verification**:
```python
# Benchmark 100 rules
for i in range(100):
    engine.record_rule(f"error_type_{i}", f"action_{i}")
import time; t0 = time.time()
engine.find_recovery("some long error log with error_type_50 in it")
print(f"Lookup: {(time.time()-t0)*1000:.1f}ms")  # Should be <10ms
```

---

#### ISSUE-002: No feedback loop for heuristic rule efficacy

**Severity**: P1  
**Module**: how  
**Detected by**: how-auditor  
**Status**: open

**Problem**: The v1.0 proposal records rules but does not track whether they actually helped. The independent `rosclaw-how` module has a complete `outcomes.py` feedback loop; v1.0 integration would not.

**Evidence**:
- `heuristic_rules` schema has `success_count` and `failure_count` columns
- No code updates these counters after a recovery is attempted
- No `delta_score` equivalent to measure rule efficacy

**Why it matters**: Without efficacy tracking, rules cannot be auto-demoted (priority=-1) when they consistently fail. The system cannot self-improve.

**Suggested fix**:
```python
# After Agent Runtime applies recovery and reports outcome:
def _on_praxis_completed(self, event: Event) -> None:
    if event.payload.get("heuristic_id"):
        outcome = "success" if event.payload.get("success") else "failure"
        self._how.update_rule_stats(event.payload["heuristic_id"], outcome)
```

**Verification**: N/A (requires v1.1-level feedback infrastructure)

---

#### ISSUE-003: Agent Runtime `analyze_failure()` has no heuristic_engine parameter

**Severity**: P1  
**Module**: agent_runtime  
**Detected by**: how-auditor  
**Status**: open

**Problem**: `DeepSeekClient.analyze_failure()` signature is `(task_description, error_log)`. No injection point for `HeuristicEngine`.

**Evidence**:
- `ai_collaboration.py:117` — `def analyze_failure(self, task_description: str, error_log: str)`
- No optional `heuristic_engine` parameter
- No way for the caller (e.g., `mcp_hub.py`) to pass a heuristic engine

**Why it matters**: Without this parameter, the integration proposal's Change C cannot be implemented cleanly.

**Suggested fix**:
```python
def analyze_failure(self, task_description: str, error_log: str,
                    heuristic_engine: Optional[Any] = None) -> dict:
```

**Verification**:
```bash
grep -n "analyze_failure" src/rosclaw/agent_runtime/*.py
# Confirm all call sites pass heuristic_engine
```

---

### P2 Issues (Defer to v1.1)

---

#### ISSUE-004: No vector search for semantic error matching

**Severity**: P2  
**Module**: how  
**Detected by**: how-auditor  
**Status**: deferred

**Problem**: The v1.0 proposal uses exact/substring matching on `error_log`. The independent `rosclaw-how` module uses sentence-transformers (384-d embeddings) for semantic matching — "torque overflow" and "joint effort exceeded" match even though words differ.

**Why it matters**: Substring matching fails when error messages use different vocabulary for the same root cause.

**Suggested fix**: v1.1 adds `sentence-transformers` dependency and a `heuristic_embeddings` table (or vector column on `heuristic_rules`).

**Verification**: N/A (v1.1)

---

#### ISSUE-005: No bridge to independent `rosclaw-how` service

**Severity**: P2  
**Module**: how  
**Detected by**: how-auditor  
**Status**: deferred

**Problem**: The v1.0 proposal builds a minimal in-process heuristic engine. The independent `rosclaw-how` service (port 47820) has a full feedback loop, blind-spot detection, and admin dashboard. These capabilities are not exposed to v1.0.

**Why it matters**: v1.0's minimal engine is a one-way street (rules help agents, but agents don't feed back to improve rules). The independent service has closed-loop learning.

**Suggested fix**: v1.1 adds an HTTP client to `rosclaw-how` service, or embeds the full module.

**Verification**: N/A (v1.1)

---

#### ISSUE-006: No MCP tool for heuristic queries

**Severity**: P2  
**Module**: agent_runtime / mcp  
**Detected by**: how-auditor  
**Status**: deferred

**Problem**: The independent `rosclaw-how` module exposes MCP tools (via `rosclaw-know-how-mcp`). v1.0 has no equivalent.

**Why it matters**: Agents using MCP cannot query heuristic rules programmatically.

**Suggested fix**: v1.1 adds `@mcp.tool()` wrappers around `HeuristicEngine.find_recovery()` and `record_rule()`.

**Verification**: N/A (v1.1)

---

## 6. Comparison with Reference Audits

### vs. audit-memory.md

| Aspect | Memory | How |
|--------|--------|-----|
| v1.0 module exists | Yes (`src/rosclaw/memory/`, 621 lines) | No (zero lines) |
| SeekDB table used | `experience_graph` actively written | `heuristic_rules` ghost schema |
| EventBus integration | Subscribes `praxis.recorded` | No subscribers |
| Runtime lifecycle | Managed by Runtime | Not registered |
| Independent module | 95,620 lines (P1 for v1.0) | 7,840 lines (P2 for v1.1) |
| Test coverage | 10 tests | 0 tests |

### vs. audit-practice.md

| Aspect | Practice | How |
|--------|----------|-----|
| v1.0 module exists | Yes (`src/rosclaw/practice/`, ~800 lines) | No |
| RFC compliance | 100% | N/A (not in RFC) |
| Post-RFC extensions | Pluggable backends, safety guards, MCP tools | N/A |
| Test coverage | 22/22 pass | 0 |
| Production readiness | Yes | Independent module: Yes; v1.0: No |

---

## 7. Recommendation

### Option A: Minimal Integration (Recommended)

**Accept the "Heuristic Safety Overlay" proposal** (~8.5 hours, ~325 lines). This adds real value to v1.0:

1. Firewall blocks come with **recovery suggestions** (not just "BLOCKED")
2. Agent Runtime gets **fast, deterministic failure recovery** (not just slow LLM calls)
3. SeekDB `heuristic_rules` table becomes **active** (not a ghost schema)
4. v1.0 narrative strengthens: "Unified runtime foundation for Physical AI **with learned safety heuristics**"

**Risk**: Low. All changes are additive. No existing behavior is modified.

### Option B: Document as v1.1

Keep RFC-0001's current P2 classification. Add a note to `RFC-0001`:

> "rosclaw-how is deferred to v1.1. The `heuristic_rules` SeekDB table is reserved but unused."

**Risk**: v1.0 ships without failure recovery learning. Every failure costs an LLM call. No improvement over time.

### Option C: Hybrid (Pragmatic Compromise)

1. **v1.0 RC3**: Add `HeuristicEngine` + 5 seed rules + Firewall integration (Option A, but smaller)
2. **v1.1**: Full `rosclaw-how` service integration with feedback loop

This gives v1.0 a "heuristic safety net" without committing to the full feedback infrastructure.

---

## 8. Summary Table

| Category | Count | Detail |
|----------|-------|--------|
| P0 Issues | **0** | No release blockers (module not integrated) |
| P1 Issues | **3** | ISSUE-001 to ISSUE-003 (if integration accepted) |
| P2 Issues | **3** | ISSUE-004 to ISSUE-006 (defer to v1.1) |
| v1.0 how code lines | **0** | No `src/rosclaw/how/` directory |
| Independent how code lines | **7,840** | `part/rosclaw-how/`, production-ready |
| SeekDB `heuristic_rules` reads | **0** | Ghost schema |
| SeekDB `heuristic_rules` writes | **0** | Ghost schema |
| EventBus how subscribers | **0** | None |
| MCP heuristic tools | **0** | None |
| Integration effort (Option A) | **~8.5h** | ~325 lines, low risk |

---

*Audit completed by Heuristic Rules & Recovery Auditor. No v1.0 source code was modified.*
