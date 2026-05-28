# ROSClaw v1.0 Audit: Sandbox / Digital Twin Firewall

> **Auditor**: Sandbox Domain Owner (rosclaw-sandbox session)
> **Date**: 2026-05-28
> **Scope**: v1.0 firewall completeness, sandbox integration readiness, architecture alignment, MuJoCo overlap

---

## Executive Summary

The v1.0 firewall module (809 lines across `decorator.py` + `validator.py`) provides **basic but functional** trajectory validation through MuJoCo simulation and e-URDF soft limits. It is integrated into the Runtime lifecycle and EventBus, satisfying the v1.0 Minimum Viable Loop requirement for basic safety checking.

However, significant gaps exist between the v1.0 firewall and the design vision in `safety_firewall_design.md` (4-layer architecture). The rosclaw-sandbox module (in development at S0 stage) is designed to close these gaps in v1.1.

**Overall Assessment**: v1.0 firewall is **adequate for v1.0 release** but **not production-grade**. Sandbox integration path is **architecturally sound** and **aligned with RFC-0001**.

---

## 1. v1.0 Firewall Completeness Assessment

### 1.1 What Exists (809 lines)

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| DigitalTwinFirewall | `firewall/decorator.py` | 438 | Integrated (UR5 MCP) |
| FirewallValidator | `firewall/validator.py` | 342 | Integrated (Runtime) |
| SafetyEnvelope | `firewall/validator.py` | ~40 | Working |
| ValidationRequest/Response | `firewall/validator.py` | ~30 | Working |

### 1.2 Three-Layer Validation — Claimed vs Implemented

| Layer | Claimed | Actually Implemented | Gap |
|-------|---------|---------------------|-----|
| Layer 1: eURDF Soft Limits | ✅ Joint limit checking | ✅ `_check_eurdf_limits()` with configurable soft factor (0.90/0.95/0.99) | Minor: no velocity/acceleration limits at this layer |
| Layer 2: MuJoCo Collision | ✅ Physics-based collision detection | ✅ `_check_mujoco_collision()` — iterates waypoints, checks `ncon` | **Major**: only checks static poses, no dynamic simulation. Sets `qpos` and calls `mj_forward`, never `mj_step` |
| Layer 3: Semantic Safety | ✅ Keepout zones + velocity checks | ⚠️ Partial — keepout zones are **skipped** (FK not computed), velocity checks only work if `duration_per_waypoint` is provided | **Major**: keepout zones always produce warnings, never violations |

### 1.3 What the Firewall CAN Do

- ✅ Load MJCF models and validate trajectories against joint limits
- ✅ Detect geometric collisions at static waypoints
- ✅ Integrate with EventBus (subscribes to `agent.command`, publishes `agent.response` + `safety.violation`)
- ✅ Lifecycle management via `LifecycleMixin`
- ✅ Three safety levels: STRICT, MODERATE, LENIENT
- ✅ Used by UR5 MCP server for real trajectory validation

### 1.4 What the Firewall CANNOT Do

- ❌ **Dynamic simulation**: never calls `mj_step()`, only `mj_forward()` — cannot detect dynamic collisions
- ❌ **Risk scoring**: no `risk_score` output, only boolean `is_safe`
- ❌ **MODIFY_AND_ALLOW**: no ability to suggest modified safe actions
- ❌ **REQUIRE_HUMAN_CONFIRMATION**: no human-in-the-loop support
- ❌ **Workspace boundary checking**: no Cartesian workspace limits
- ❌ **Episode recording**: no trace/replay of firewall decisions
- ❌ **Forward prediction**: no trajectory rollout over time horizon
- ❌ **Self-collision filtering**: `allowed_contacts` list exists but geom names may not match
- ❌ **Multi-robot**: single robot only

### 1.5 Design Doc vs Implementation Gap

The `safety_firewall_design.md` envisions a **4-layer architecture**:

| Layer | Design Doc | v1.0 Implementation |
|-------|-----------|-------------------|
| Layer 1: Hard Limits | Joint/velocity/acceleration/torque limits, < 0.1ms | ⚠️ Joint position limits only (in validator) |
| Layer 2: Analytical Check | Pinocchio + Ruckig, FK/IK, workspace, < 1ms | ❌ Not implemented |
| Layer 3: Digital Twin | MJX GPU parallel simulation, domain randomization, < 10ms | ⚠️ Basic MuJoCo static collision only |
| Layer 4: Predictive Safety | Neural Twin (Cosmos/V-JEPA 2), 5-10s prediction | ❌ Not implemented |

**Verdict**: v1.0 implements ~25% of the design vision. This is **acceptable for v1.0 scope** per RFC-0005 (P2 deferral for sandbox digital twin).

---

## 2. Issues Found

### ISSUE-SB-001: FirewallValidator Never Calls mj_step()

**Severity**: P1
**Module**: firewall
**Owner**: sandbox
**Status**: open

#### Problem
The MuJoCo collision layer only checks static poses by setting `qpos` and calling `mj_forward()`. It never simulates physics dynamics with `mj_step()`, meaning dynamic collisions (e.g., swinging arm hitting an obstacle mid-trajectory) are undetectable.

#### Evidence
- file: `src/rosclaw/firewall/validator.py`
- line: 289-308 (`_check_mujoco_collision`)
- direct observation: `mj_forward` called, `mj_step` never called

#### Why it matters
A trajectory that passes through a collision-free set of waypoints may still collide during the dynamic transition between waypoints. This is the primary use case for a physics-based firewall.

#### Expected behavior
The firewall should simulate the trajectory forward in time using `mj_step()`, applying controls and stepping physics, then check contacts at each physics substep.

#### Suggested fix
Replace static pose checking with trajectory rollout in a cloned `MjData` instance. The sandbox module's `FirewallGate.check()` already implements this pattern.

#### Verification
```bash
# A trajectory where waypoints are collision-free but mid-motion collides
rosclaw firewall check --action mid_swing_collision.json
# Expected: BLOCK  (currently: ALLOW)
```

---

### ISSUE-SB-002: Semantic Keepout Zones Always Skipped

**Severity**: P1
**Module**: firewall
**Owner**: sandbox
**Status**: open

#### Problem
`_check_semantic_safety()` always produces warnings for keepout zones because forward kinematics (FK) is never computed. The method appends `"Keepout zone defined but FK not computed — skipped"` for every zone.

#### Evidence
- file: `src/rosclaw/firewall/validator.py`
- line: 313-322 (`_check_semantic_safety`)

#### Why it matters
Keepout zones are a critical safety feature (e.g., "never enter the human workspace"). Without FK, they provide zero protection.

#### Suggested fix
Compute FK via `mj_forward()` (already called) and read `body_pos` / `site_pos` from `MjData` to check if any link enters a keepout zone.

---

### ISSUE-SB-003: DigitalTwinFirewall Not EventBus-Integrated

**Severity**: P1
**Module**: firewall
**Owner**: sandbox
**Status**: open (documented as v1.0 scope)

#### Problem
`DigitalTwinFirewall` (decorator.py) is a standalone class used directly by `ur5_server.py`. It does not participate in the EventBus, violating RFC-0001 §2.2 ("Modules MUST NOT import each other directly").

#### Evidence
- file: `src/rosclaw/mcp/ur5_server.py`
- line: 41-42 (direct import of `DigitalTwinFirewall`)
- file: `src/rosclaw/firewall/decorator.py`
- No EventBus import or subscription

#### Why it matters
Firewall results from the MCP server are invisible to Practice, Memory, and Dashboard. No `safety.violation` event is published when the MCP server blocks a trajectory.

#### Expected behavior
The MCP server should publish `agent.command` events and subscribe to `agent.response`, using the FirewallValidator (which is EventBus-integrated) instead of the standalone DigitalTwinFirewall.

#### Suggested fix
Replace `DigitalTwinFirewall` usage in `ur5_server.py` with event-based validation through the Runtime's FirewallValidator. Add a compatibility shim for backward compatibility.

---

### ISSUE-SB-004: No FirewallDecision Standard Type

**Severity**: P2
**Module**: firewall
**Owner**: sandbox
**Status**: deferred to v1.1

#### Problem
The v1.0 firewall uses two different result types: `ValidationResult` (decorator.py) and `ValidationResponse` (validator.py). Neither matches the sandbox's `FirewallDecision` schema which includes `risk_score`, `replay_id`, and 5-level decision types.

#### Why it matters
When sandbox replaces the v1.0 firewall, all consumers (Runtime, MCP server, Practice) will need to adapt to a new result type.

#### Suggested fix
Define `FirewallDecision` as a shared type in `core/types.py` during v1.1 integration. The v1.0 firewall can add a `to_firewall_decision()` adapter method.

---

### ISSUE-SB-005: MuJoCo Usage Overlap with Sandbox

**Severity**: P2
**Module**: firewall, mcp_drivers, sandbox
**Owner**: sandbox
**Status**: deferred to v1.1

#### Problem
Three separate MuJoCo usage patterns exist:

| Component | MuJoCo Usage | Model Source |
|-----------|-------------|-------------|
| `firewall/decorator.py` | `MjModel.from_xml_path()` + `mj_step()` | `model_path` param |
| `firewall/validator.py` | `MjModel.from_xml_path()` + `mj_forward()` | `mujoco_model_path` param |
| `mcp_drivers/mujoco_sim_driver.py` | `MjModel.from_xml_path()` + `mj_step()` | `model_path` param |
| `sandbox/engines/mujoco/engine.py` | `MjModel.from_xml_path()` + `mj_step()` | `profile.mjcf_path` |

#### Why it matters
- Model may be loaded 3 times for the same robot (firewall decorator + firewall validator + sim driver)
- Inconsistent timestep, gravity, and solver settings across instances
- Memory waste from duplicate model compilation

#### Suggested fix
In v1.1, sandbox should own all MuJoCo model loading. The firewall validator should delegate to sandbox's engine. The sim driver should use sandbox as its simulation backend instead of managing its own MjModel.

---

## 3. Sandbox Integration Readiness

### 3.1 Can Sandbox Plug into v1.0 Runtime Lifecycle?

**YES** — with minor adaptation.

The sandbox already uses a session-based lifecycle (`SandboxSession`) with states: `created → running → paused → closed → failed`. This maps directly to v1.0's `LifecycleMixin` states.

Integration adapter (already designed in `rosclaw_v1.0_sandbox_集成方案.md`):

```python
class SandboxRuntimeAdapter:
    name = "sandbox"
    async def start(self): ...
    async def stop(self): ...
    def health(self): ...
```

**Status**: Adapter not yet implemented in code. Planned for v1.1.

### 3.2 Can Sandbox Use e-URDF safety.yaml?

**YES** — the `eurdf/loader.py` already reads `safety.yaml`, `capabilities.yaml`, `semantic.yaml`, and `benchmark.yaml` if present. The `RobotEmbodimentProfile.safety` field is passed to the `FirewallGate` for constraint checking.

**Current limitation**: The e-URDF-Zoo skeleton configs don't include these YAML files yet. The loader gracefully falls back to defaults.

### 3.3 Can Sandbox Subscribe to AgentCommand Events?

**YES** — the `events/schemas.py` already defines `FirewallActionAllowed` and `FirewallActionBlocked` events. The `events/publisher.py` includes a `RuntimePublisher` that adapts to v1.0's EventBus.

**Current limitation**: The `RuntimePublisher` is not yet wired into the firewall gate's check flow. This is a v1.1 integration task.

### 3.4 Integration Checklist for v1.1

- [ ] Create `SandboxRuntimeAdapter` in `src/rosclaw/sandbox/runtime_adapter.py`
- [ ] Register sandbox module in `Runtime._do_initialize()`
- [ ] Add `configs/sandbox.yaml` to v1.0 config directory
- [ ] Wire `FirewallGate` to publish events via `RuntimePublisher`
- [ ] Replace `DigitalTwinFirewall` in `ur5_server.py` with EventBus-based validation
- [ ] Unify MuJoCo model loading (single model, shared by sandbox + driver)
- [ ] Add `FirewallDecision` to `core/types.py` as shared schema
- [ ] Add `ActionGroundingInterceptor` to `agent_runtime/interceptors.py`
- [ ] Practice subscribes to `SandboxTaskFailed` and `FirewallActionBlocked`
- [ ] Add sandbox status to `rosclaw status` output

---

## 4. Architecture Alignment Verification

### 4.1 RFC-0001 Compliance

| RFC-0001 Rule | Sandbox Compliance | Notes |
|---------------|-------------------|-------|
| §2.1 All modules managed by Runtime | ✅ Designed, not yet implemented | `SandboxRuntimeAdapter` planned |
| §2.2 No direct module imports | ✅ Sandbox only imports `core.types`, `core.session` | EventBus via `RuntimePublisher` |
| §2.3 SeekDB as Knowledge Plane | ✅ Sandbox does NOT write to SeekDB directly | Events → Practice → Memory → SeekDB |
| §2.4 e-URDF as Physical DNA | ✅ Sandbox reads from e-URDF-Zoo via `eurdf/loader.py` | Uses `e_urdf.json` + optional YAMLs |
| Anti-Pattern 1: Runtime Bypass | ✅ No direct imports of memory/practice/dashboard |  |
| Anti-Pattern 2: Event Bus Decoration | ✅ Events defined, publisher interface pluggable |  |
| Anti-Pattern 6: Self-Starting Module | ✅ Sandbox only starts when explicitly created |  |

### 4.2 Integration Path: Standalone → rosclaw.sandbox Namespace

**Verified correct**. The sandbox currently installs as `rosclaw-sandbox` package with `rosclaw.sandbox` Python namespace. The `pkgutil.extend_path` bridge added to v1.0's `__init__.py` enables namespace sharing.

For v1.1, the code can be moved into `src/rosclaw/sandbox/` within the v1.0 repo without breaking imports.

### 4.3 RFC-0005 Acceptance Gate Alignment

| Gate | Sandbox Relevance | Status |
|------|------------------|--------|
| P0 #9: No module bypasses Runtime | Sandbox will register via `SandboxRuntimeAdapter` | v1.1 |
| P2 #7: Sandbox digital twin integration | In development (S0 complete) | In Progress |

---

## 5. MuJoCo Usage Overlap Analysis

### Current State (v1.0 + sandbox)

```
┌─────────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
│ DigitalTwinFirewall │   │ FirewallValidator     │   │ MuJoCoSimDriver     │
│ (decorator.py)      │   │ (validator.py)        │   │ (mujoco_sim_driver) │
│                     │   │                       │   │                     │
│ Own MjModel ✓       │   │ Own MjModel ✓         │   │ Own MjModel ✓       │
│ mj_step() ✓         │   │ mj_forward() only     │   │ mj_step() ✓         │
│ Trajectory rollout  │   │ Static pose check     │   │ Driver simulation   │
│ Collision check ✓   │   │ Collision check ✓     │   │ No collision check  │
│ Joint limits ✓      │   │ Joint limits ✓        │   │ No limit check      │
│ Torque limits ✓     │   │ Torque via safety     │   │ No torque check     │
│ Standalone ✗        │   │ EventBus ✓            │   │ Driver lifecycle ✓  │
└─────────────────────┘   └──────────────────────┘   └─────────────────────┘

┌─────────────────────────────────┐
│ Sandbox MujocoEngine            │
│ (engines/mujoco/engine.py)      │
│                                 │
│ Own MjModel ✓                   │
│ mj_step() ✓                     │
│ Full simulation loop ✓          │
│ Collision + joint + workspace ✓ │
│ Risk scoring ✓                  │
│ Episode recording ✓             │
│ EventBus events ✓               │
│ Standalone + Runtime adapter ✓  │
└─────────────────────────────────┘
```

### Recommended v1.1 Consolidation

```
                    ┌─────────────────────┐
                    │  Sandbox MujocoEngine │
                    │  (Single MuJoCo owner)│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼──────┐ ┌───────▼───────┐
    │ Firewall Gate  │ │ Task Runtime │ │ Sim Driver    │
    │ (safety check) │ │ (episodes)   │ │ (hardware     │
    │                │ │              │ │  abstraction) │
    └────────────────┘ └─────────────┘ └───────────────┘
```

---

## 6. Recommendations

### For v1.0 Release (No Action Required)

1. **Document firewall limitations** in README and API docs — current basic joint validation is v1.0 scope per RFC-0001 and RFC-0005
2. **Keep DigitalTwinFirewall** as-is for UR5 MCP backward compatibility
3. **No new features** — architecture is frozen per RFC-0001 §6

### For v1.1 (Sandbox Integration)

1. **P0**: Implement `SandboxRuntimeAdapter` and register in Runtime
2. **P0**: Replace static `mj_forward` collision check with dynamic `mj_step` rollout (fixes ISSUE-SB-001)
3. **P1**: Implement FK-based keepout zone checking (fixes ISSUE-SB-002)
4. **P1**: Migrate MCP server from `DigitalTwinFirewall` to EventBus-based validation (fixes ISSUE-SB-003)
5. **P1**: Consolidate MuJoCo model loading (fixes ISSUE-SB-005)
6. **P2**: Add `FirewallDecision` to shared types (fixes ISSUE-SB-004)

---

### ISSUE-SB-006: Firewall Pre-Check for Joint Limits ✅ FIXED

**Severity**: P1
**Module**: firewall
**Owner**: sandbox
**Status**: **fixed** (2026-05-28)

#### Problem
The `FirewallGate.check()` method only validated joint limits during simulation rollout, not in the initial action request. This meant dangerous actions with out-of-bounds target values could pass initial validation.

#### Evidence
- file: `src/rosclaw/sandbox/firewall/gate.py`
- line: 52-53 (original code)
- test: `examples/actions/bad_table_collision.json` with values `[3.5, -3.5, 3.5, -3.5, 3.5, -3.5]` returned `ALLOW`

#### Why it matters
Joint limit violations should be caught immediately, before any simulation. This is a Layer 1 (Hard Limits) check that should be < 0.1ms, not deferred to Layer 3 (Digital Twin).

#### Fix Applied
Added pre-check logic in `FirewallGate.check()` that validates requested action values against joint limits before simulation:

```python
# Pre-check: validate requested action values against joint limits
if self._profile:
    limits = self._profile.get_joint_limits()
    target_values = action_request.get("values", action_request.get("target_pose", []))
    for i, (jname, (lo, hi)) in enumerate(limits.items()):
        if i < len(target_values):
            val = target_values[i]
            if val < lo or val > hi:
                violations.append(
                    f"joint_limit: {jname} target={val:.3f} outside [{lo:.3f}, {hi:.3f}]"
                )
```

#### Verification
```bash
$ rosclaw-sandbox firewall check --robot universal_robots_ur5e --world tabletop \
    --action examples/actions/bad_table_collision.json
[FIREWALL BLOCKED] joint_limit: joint_0 target=3.500 outside [-3.140, 3.140] (risk=0.90)
  Violated: joint_limit
  Replay:   firewall_ep_f0d410f3
```

---

### ISSUE-SB-007: Top-Level Sandbox Import Missing ✅ FIXED

**Severity**: P1
**Module**: core
**Owner**: sandbox
**Status**: **fixed** (2026-05-28)

#### Problem
The `rosclaw.sandbox` package did not export the `Sandbox` class at the top level, requiring users to import from `rosclaw.sandbox.sandbox_api` instead of `rosclaw.sandbox`.

#### Evidence
- file: `src/rosclaw/sandbox/__init__.py`
- test: `python3 -c "from rosclaw.sandbox import Sandbox"` raised `ImportError`

#### Why it matters
API usability. The `Sandbox` class is the primary user-facing interface and should be importable from the package root, consistent with Python packaging conventions.

#### Fix Applied
Added `Sandbox` to `__all__` in `src/rosclaw/sandbox/__init__.py`:

```python
from rosclaw.sandbox.sandbox_api import Sandbox

__all__ = [
    "__version__",
    "FirewallDecision",
    "RobotEmbodimentProfile",
    "Sandbox",  # ← added
    "SandboxEnv",
    "SandboxSession",
    "StepResult",
]
```

#### Verification
```bash
$ python3 -c "from rosclaw.sandbox import Sandbox; print(Sandbox)"
<class 'rosclaw.sandbox.sandbox_api.Sandbox'>
```

---

### ISSUE-SB-008: EventBus Publishing Not Wired into Core Modules ✅ FIXED

**Severity**: P1
**Module**: sandbox, firewall, tasks
**Owner**: sandbox
**Status**: **fixed** (2026-05-28)

#### Problem
The sandbox, firewall, and task runtime modules did not publish events to the EventBus. This meant:
- Practice could not subscribe to sandbox episodes
- Memory could not record firewall blocks
- Dashboard could not display real-time sandbox state

#### Evidence
- `sandbox_api.py`: No event publishing in `reset()`, `step()`, `close()`
- `firewall/gate.py`: No event publishing in `check()`
- `tasks/runtime.py`: No event publishing in `run_episode()`

#### Fix Applied
Added optional `publisher` parameter to all three modules (defaults to `NullPublisher`):

1. **Sandbox** publishes:
   - `SandboxSessionStarted` on `reset()`
   - `SandboxStepExecuted` on each `step()`
   - `SandboxSessionStopped` on `close()`

2. **FirewallGate** publishes:
   - `FirewallActionAllowed` when action passes
   - `FirewallActionBlocked` when action is blocked

3. **TaskRuntime** publishes:
   - `SandboxTaskStarted` at episode start
   - `SandboxTaskSucceeded` on success
   - `SandboxTaskFailed` on failure/timeout

All events include `robot_id`, `task_id`, `session_id`, and a `payload` dict for extensibility.

#### Verification
```bash
$ python3 -m pytest tests/test_event_publishing.py -v
tests/test_event_publishing.py::TestSandboxEventPublishing::test_reset_publishes_session_started PASSED
tests/test_event_publishing.py::TestSandboxEventPublishing::test_step_publishes_step_executed PASSED
tests/test_event_publishing.py::TestSandboxEventPublishing::test_close_publishes_session_stopped PASSED
tests/test_event_publishing.py::TestFirewallEventPublishing::test_allow_publishes_firewall_action_allowed PASSED
tests/test_event_publishing.py::TestFirewallEventPublishing::test_block_publishes_firewall_action_blocked PASSED
tests/test_event_publishing.py::TestNullPublisherDefault::test_sandbox_default_no_events PASSED
```

---

## Appendix: Sandbox S0-S1 Development Status

The rosclaw-sandbox module has completed Sprint S0 with:

| Component | Status | Files |
|-----------|--------|-------|
| Core types (RobotEmbodimentProfile, StepResult, FirewallDecision) | ✅ Complete | `core/types.py` |
| Session lifecycle | ✅ Complete | `core/session.py` |
| Environment protocol | ✅ Complete | `core/env.py` |
| Engine registry | ✅ Complete | `core/registry.py` |
| MuJoCo engine | ✅ MVP | `engines/mujoco/engine.py` |
| e-URDF-Zoo bridge | ✅ MVP | `eurdf/loader.py` |
| Model validator | ✅ MVP | `validator/model_validator.py` |
| Task runtime | ✅ Skeleton | `tasks/runtime.py` |
| Firewall gate | ✅ MVP | `firewall/gate.py` |
| Episode recorder | ✅ MVP | `traces/recorder.py` |
| Replay engine | ✅ MVP | `traces/replay.py` |
| Event schemas | ✅ Complete | `events/schemas.py` |
| Event publishers | ✅ Complete | `events/publisher.py` |
| CLI (doctor/robots/validate/run/firewall/replay) | ✅ Working | `cli/main.py` |
| Tests (22 tests, all passing) | ✅ Complete | `tests/test_*.py` |
| Config YAML (3 worlds, 4 tasks) | ✅ Complete | `configs/` |
| README | ✅ Complete | `README.md` |

**Verified Commands**:
```
✅ rosclaw-sandbox --help
✅ rosclaw-sandbox doctor (Python 3.12.3, MuJoCo 3.8.1, 63 robots)
✅ rosclaw-sandbox robots list (63 robots from e-URDF-Zoo)
✅ rosclaw-sandbox validate unitree_go2 (correctly reports FAIL for skeleton)
✅ rosclaw-sandbox run --robot unitree_go2 --world empty --steps 50 --record
✅ rosclaw-sandbox replay runs/ep_*
✅ rosclaw-sandbox firewall check --robot unitree_go2 --action safe_reach.json
✅ python -c "import rosclaw.sandbox; print(rosclaw.sandbox.__version__)" → 0.1.0
✅ pytest tests/ → 22 passed
```
