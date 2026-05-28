# ROSClaw Dashboard v1.0 — Observability Audit Report

> **Auditor:** rosclaw-dashboard domain owner
> **Date:** 2026-05-28
> **Scope:** v1.0 dashboard integration readiness for Physical AI Runtime control panel
> **Reference:** [Implementation Plan](../../../../实施方案.md)

---

## Executive Summary

The current dashboard (`v0.1.0 MVP`) provides a **static control-plane skeleton** with CRUD APIs for robots, missions, skills, MCAP files, memory entries, and safety rules. However, it **lacks runtime observability** — the critical path from "Runtime starts → Events generated → Dashboard displays" is **not yet closed**. Dashboard cannot read live robot state, cannot show module health (Agent, Provider, Memory, Practice, Swarm), and cannot visualize timelines or PraxisEvents.

**Verdict:** Dashboard is **not ready** for v1.0 "Physical AI Runtime control panel" without closing the runtime observability gap.

---

## Audit Targets & Findings

### 1. Runtime State Access

| Question | Status | Finding |
|----------|--------|---------|
| Can Dashboard read Runtime state? | **MISSING** | No runtime status API exists. `/api/robots/{id}/health` returns only `{"status": "offline", "robot_id": "..."}` — a static DB field, not live runtime telemetry. |
| Is there an API for Runtime status? | **MISSING** | No `/api/runtime/*` endpoints. No heartbeat, no topic discovery, no diagnostics, no TF status. |

**Gap Detail:**
The implementation plan (Section 4.3 "Live Robot Console") specifies real-time data from:
- ROS 2 topics
- diagnostics
- foxglove_bridge websocket
- rosclaw-agent-daemon
- system metrics exporter

**Current state:** None of these data sources are connected. The backend is a pure SQLite CRUD service with no ROS 2 bridge, no daemon connection, and no WebSocket/SSE streaming.

**Required for v1.0:**
- `GET /api/runtime/{robot_id}/status` — online/offline, bridge connection, daemon heartbeat
- `GET /api/runtime/{robot_id}/topics` — live ROS topic list with hz
- `GET /api/runtime/{robot_id}/diagnostics` — diagnostic messages
- WebSocket or SSE endpoint for real-time push

---

### 2. Module Status Visibility

| Module | Can Dashboard Show Status? | Data Available? | What's Missing |
|--------|---------------------------|-----------------|----------------|
| **Agent** | No | Agent ID stored in `missions.agent_id` (string) | No agent registry. No agent health/connection status. No agent event stream. |
| **Provider** | No | None | No provider concept in schema. No LLM provider status, cost, latency. |
| **Memory** | Partial | `MemoryEntry` table with 4 seeded entries | No live memory service connection. No `rosclaw-memory` API integration. Static seeded data only. |
| **Practice** | No | None | No practice module API. No training run status, no benchmark results. |
| **Swarm** | No | None | No multi-robot fleet concept. No swarm coordinator status. |

**Gap Detail:**
The dashboard's Memory page (`/memory`) renders seeded SQLite rows. It does **not** connect to the actual `rosclaw-memory` service. The Safety page shows seeded audits/rules but has no live safety event stream.

**Required for v1.0:**
- Agent registry API (`/api/agents`) with connection status
- Provider status API (`/api/providers`) with cost/latency
- Memory service bridge (`/api/memory/retrieve`, `/api/memory/search`)
- Practice run status (`/api/practice/runs`)
- Swarm fleet view (`/api/swarm/fleets/{id}/robots`)

---

### 3. e-URDF Visualization

| Question | Status | Finding |
|----------|--------|---------|
| Can Dashboard read semantic.yaml / capabilities.yaml? | **PARTIAL** | `robot.eurdf.yaml` is parsed during import (`robot_service.py:import_robot_from_directory`). Sensors, actuators, skills are extracted into SQLite. |
| Can Dashboard visualize e-URDF? | **NO** | No 3D viewer. No WebGL/Three.js/R3F integration. Embodiment page shows static text only. |

**Gap Detail:**
The e-URDF parser exists and populates DB tables (`sensors`, `actuators`, `skills`). However:
- No `semantic.yaml` or `capabilities.yaml` reader (plan specifies `robot.eurdf.yaml` + `robot.skills.yaml` + `robot.safety.yaml`)
- No 3D robot viewer (React Three Fiber placeholder not implemented)
- No link/joint/frame tree visualization
- No sensor/actuator overlay on 3D model
- No workspace / keep-out zone visualization

**Required for v1.0:**
- e-URDF schema validator CLI (`rosclaw-dashboard validate`)
- 3D embodiment viewer (minimum: URDF → Three.js loader)
- Body-part tree with click-to-inspect
- Topic binding table per sensor/actuator

---

### 4. Timeline / PraxisEvent Visualization

| Question | Status | Finding |
|----------|--------|---------|
| Can Dashboard visualize MCAP data? | **PARTIAL** | MCAP files can be registered and listed. Topic list is **hardcoded placeholder** (`mcap.py:37-51`). No actual MCAP parsing. |
| Can Dashboard show timeline / PraxisEvent? | **NO** | No unified timeline component. No PraxisEvent schema or API. |

**Gap Detail:**
The MCAP router (`mcap.py`) has:
- `GET /api/mcap` — list registered files
- `GET /api/mcap/{id}/topics` — **returns static array**, not parsed from file
- `GET /api/mcap/{id}/foxglove-layout` — **returns hardcoded layout**, not generated from e-URDF sensors

The implementation plan specifies:
- MCAP indexer (topic stats, message count, frequency, dropped frames)
- Failure segment extraction
- Task/skill segment binding
- Unified timeline with swimlanes (sensor stream, agent intent, skill execution, safety event, memory write)
- Click event → Agent trace + Foxglove segment + memory summary

**Required for v1.0:**
- Real MCAP parser integration (`mcap` Python package already in repo)
- `mcap_topics` table with message_count, avg_hz, first/last timestamp
- `mcap_segments` table for failure/skill/task segments
- Unified timeline component (frontend)
- PraxisEvent schema and ingestion API

---

### 5. Minimum Demo Path

**Is there a path: Runtime starts → Events generated → Dashboard displays?**

**Answer: NO.**

The current demo path is:
```
Seed script → SQLite rows → Dashboard renders static tables
```

The required v1.0 demo path is:
```
Runtime (ROS 2 / Agent) starts
    → rosclaw-agent-daemon publishes events
    → foxglove_bridge streams topics
    → Dashboard API receives events
    → Dashboard displays live status + timeline
```

**What's missing for this path:**

| Component | Status |
|-----------|--------|
| `rosclaw-agent-daemon` | **Not implemented** |
| Robot heartbeat publisher | **Not implemented** |
| Topic discovery collector | **Not implemented** |
| Agent event publisher (`/rosclaw/agent_events`) | **Not implemented** |
| Skill event publisher (`/rosclaw/skill_events`) | **Not implemented** |
| Safety event publisher (`/rosclaw/safety_events`) | **Not implemented** |
| WebSocket/SSE event stream API | **Not implemented** |
| Live telemetry ingestion | **Not implemented** |

---

## Missing APIs (v1.0 Blockers)

### Critical (Must Have)

| API | Purpose | Priority |
|-----|---------|----------|
| `GET /api/runtime/{robot_id}/status` | Live runtime connection status | P0 |
| `GET /api/runtime/{robot_id}/topics` | Live ROS topic discovery | P0 |
| `WS /api/events/stream` | Real-time event push (agent/skill/safety) | P0 |
| `GET /api/agents` | Agent registry and health | P0 |
| `GET /api/mcap/{id}/topics` (real) | Parse actual MCAP file for topics | P0 |
| `POST /api/events/ingest` | PraxisEvent ingestion endpoint | P0 |

### Important (Should Have)

| API | Purpose | Priority |
|-----|---------|----------|
| `GET /api/memory/search` | SeekDB semantic search bridge | P1 |
| `GET /api/practice/runs` | Practice/training run status | P1 |
| `GET /api/swarm/fleets` | Multi-robot fleet management | P1 |
| `GET /api/providers` | LLM provider status and cost | P1 |
| `GET /api/mcap/{id}/segments` | Failure/skill segment extraction | P1 |

### Deferred (v1.1)

| API | Purpose | Priority |
|-----|---------|----------|
| `GET /api/simulation/scenarios` | Arena/simulation sync | P2 |
| `POST /api/approvals/{id}/approve` | Human approval gate (UI exists, API partial) | P2 |
| `GET /api/robots/{id}/telemetry` | Historical telemetry queries (DuckDB) | P2 |

---

## Missing Event Schemas

### PraxisEvent (Unified Event Schema)

The implementation plan defines a unified event schema (Section 16). This schema is **not implemented** in code:

```json
{
  "event_id": "evt_001",
  "robot_id": "go2_lab_001",
  "mission_id": "mission_001",
  "timestamp": 1716888000.123,
  "type": "skill_started",
  "source": "rosclaw-agent",
  "payload": { ... }
}
```

**Required event types for v1.0:**
- `mission_created`, `mission_started`, `mission_paused`, `mission_completed`, `mission_failed`
- `skill_selected`, `skill_started`, `skill_completed`, `skill_failed`
- `safety_check_passed`, `safety_check_failed`, `approval_required`
- `memory_written`, `failure_detected`, `mcap_segment_created`

**Current state:** `AgentEvent` table exists with `event_type` (string) and `payload_json` (string), but no schema validation, no typed event ingestion, no event stream.

---

## Missing Dashboard Adapters

| Adapter | Purpose | Status |
|---------|---------|--------|
| **ROS 2 Bridge Adapter** | Connect to live ROS 2 topics | **Missing** |
| **Foxglove Bridge Adapter** | Stream to/from Foxglove | **Missing** |
| **Agent Daemon Adapter** | Receive agent events | **Missing** |
| **MCAP Parser Adapter** | Read actual MCAP files | **Missing** (placeholder only) |
| **Memory Service Adapter** | Connect to rosclaw-memory | **Missing** |
| **SeekDB Adapter** | Semantic search bridge | **Missing** |
| **WebSocket/SSE Adapter** | Real-time push to frontend | **Missing** |

---

## Features to Defer to v1.1

Based on the implementation plan's sprint structure and current MVP state, these features should be explicitly deferred:

| Feature | Original Sprint | Defer Reason |
|---------|-----------------|--------------|
| Arena / Simulation Sync | Sprint 8 | No arena/practice module integration exists |
| Complex 3D dynamics viewer | Sprint 1+ | 3D viewer is placeholder; full URDF loader is large |
| Generative UI (Agentic UI) | Sprint 4+ | No structured UI intent protocol implemented |
| Multi-robot swarm scheduling | Sprint 4+ | No swarm module API |
| Complex permission system | — | Authentication stubs exist but no auth routes |
| Self-hosted MCAP player | — | Explicitly out of scope per plan (use Foxglove) |
| Complete safety approval workflow | Sprint 7 | Toggle exists but no approval gate logic |

---

## Recommendations

### Immediate Actions (Before v1.0)

1. **Implement `rosclaw-agent-daemon`** (minimal): A Python daemon that publishes heartbeat + agent/skill/safety events to a WebSocket or HTTP endpoint. This is the **minimum viable runtime connector**.

2. **Add WebSocket/SSE endpoint** (`/api/events/stream`): Push real-time events to the dashboard frontend. Can reuse existing `AgentEvent` table as event log.

3. **Replace MCAP topic placeholder** with actual MCAP parsing: The `mcap` Python package is already a submodule. Use it to parse topic lists, message counts, and time ranges.

4. **Add runtime status API**: Even if synthetic, provide `GET /api/runtime/{robot_id}/status` that returns bridge connection, daemon heartbeat, and topic list.

### v1.0 Definition Adjustment

Consider redefining v1.0 as:

> **v1.0 = Static Control Plane + One Live Robot Demo**
>
> Dashboard can register a robot, import e-URDF, show static embodiment, receive events from a single running agent-daemon, display basic mission trace, and open MCAP in Foxglove.

This is achievable with the 4 immediate actions above. The full "Physical AI Runtime control panel" with multi-robot, swarm, practice, and arena integration is a **v1.1 or v2.0** target.

---

## Appendix: Current API Coverage vs Plan

| API (from plan) | Status | Notes |
|-----------------|--------|-------|
| `GET /api/robots` | **Done** | |
| `POST /api/robots/import` | **Done** | e-URDF import works |
| `GET /api/robots/{id}/embodiment` | **Done** | |
| `GET /api/robots/{id}/skills` | **Done** | |
| `GET /api/missions` | **Done** | |
| `GET /api/missions/{id}/trace` | **Done** | |
| `POST /api/missions/{id}/pause` | **Done** | |
| `POST /api/missions/{id}/abort` | **Done** | |
| `GET /api/mcap` | **Done** | |
| `GET /api/mcap/{id}/foxglove-layout` | **Placeholder** | Hardcoded layout |
| `GET /api/mcap/{id}/topics` | **Placeholder** | Hardcoded topics |
| `GET /api/mcap/{id}/segments` | **Missing** | |
| `GET /api/skills` | **Done** | Added during audit |
| `POST /api/skills/{id}/run` | **Done** | Minimal (DB only) |
| `GET /api/memory` | **Done** | Added during audit |
| `GET /api/safety/audits` | **Done** | Added during audit |
| `GET /api/safety/rules` | **Done** | Added during audit |
| `POST /api/approvals/{id}/approve` | **Missing** | |
| `WS /api/events/stream` | **Missing** | |
| `GET /api/runtime/*` | **Missing** | |
| `GET /api/agents` | **Missing** | |

---

*End of Audit Report*
