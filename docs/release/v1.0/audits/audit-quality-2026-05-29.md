# ROSClaw v1.0 Quality Audit Report

**Audit Date:** 2026-05-29
**Auditor:** KNOW Domain (Claude)
**Modules:** swarm/, dashboard/, sdk_to_mcp/, cli.py
**Scope:** Code style, type safety, error handling, documentation

---

## Executive Summary

| Module | Grade | Critical | High | Medium | Low |
|--------|-------|----------|------|--------|-----|
| swarm/coordinator.py | B+ | 0 | 1 | 3 | 2 |
| swarm/consensus.py | B | 0 | 0 | 3 | 2 |
| swarm/manager.py | B | 0 | 1 | 2 | 2 |
| dashboard/server.py | B | 0 | 1 | 3 | 2 |
| dashboard/metrics.py | A- | 0 | 0 | 2 | 2 |
| sdk_to_mcp/compiler.py | B+ | 0 | 0 | 3 | 2 |
| sdk_to_mcp/manifest.py | A- | 0 | 0 | 2 | 2 |
| cli.py | B | 0 | 2 | 4 | 3 |
| **Overall** | **B+** | **0** | **5** | **22** | **17** |

---

## 1. swarm/coordinator.py

### 🔴 HIGH

**C1: Partial allocation rollback missing**
- Line 127-172: `allocate_task()` allocates subtasks sequentially. If subtask N fails, subtasks 1..N-1 remain allocated but the method returns `feasible=False`.
- **Impact:** Agents left in "busy" state, task state inconsistent.
- **Fix:** Implement rollback on allocation failure.

### 🟡 MEDIUM

**C2: Inline import of `math` module**
- Line 111: `import math` inside `request_bids()` method.
- **Impact:** Minor performance hit, violates PEP 8 (imports at top).
- **Fix:** Move to module-level imports.

**C3: No thread safety for shared state**
- Lines 40-43: `_agents`, `_tasks`, `_bids`, `_consensus_state` are plain dicts.
- **Impact:** Race conditions in concurrent scenarios.
- **Fix:** Use threading.Lock or asyncio.Lock.

**C4: `deregister_agent` doesn't clean up running tasks**
- Line 57-59: Agent removed from dict but if they had an active task, task state becomes orphaned.
- **Fix:** Reassign or cancel active tasks before deregistering.

### 🟢 LOW

**C5: `tuple[float, ...]` type is imprecise**
- Line 47: Position should be `tuple[float, float, float]` for 3D.

**C6: No validation of `task` dict structure in `decompose_task`**
- Line 63: Assumes `task` has expected keys without validation.

---

## 2. swarm/consensus.py

### 🟡 MEDIUM

**S1: Not real Raft — missing heartbeat/timeout**
- Line 29: Class named `RaftLikeConsensus` but lacks leader election, heartbeats, log replication.
- **Impact:** Misleading naming. Consensus can fail silently if leader crashes.
- **Fix:** Rename to `SimpleMajorityConsensus` or implement full Raft.

**S2: `vote` accepts votes from any agent_id**
- Line 73-93: No validation that voter is in `peers` list.
- **Impact:** Byzantine agents can corrupt consensus.
- **Fix:** Check `agent_id in self.peers` before accepting vote.

**S3: `check_commit` uses latest timestamp, not leader value**
- Line 101-106: Raft should commit the leader's value, not the most recent.
- **Impact:** Different values may be committed under contention.
- **Fix:** Track proposer and only commit leader proposals.

### 🟢 LOW

**S4: No log persistence**
- Proposals only in memory. Crash = lost consensus state.

**S5: `set_leader` increments term on every call**
- Line 49-50: Calling `set_leader(True)` twice doubles the term.

---

## 3. swarm/manager.py

### 🔴 HIGH

**M1: `_on_allocate_request` ignores allocation failure**
- Line 51-62: `self.allocate_task(task)` returns `Optional[str]`. If `None`, still publishes `allocate_result` with `agent_id: None`.
- **Impact:** Consumers can't distinguish failure from missing agent_id.
- **Fix:** Check return value and publish failure event.

### 🟡 MEDIUM

**M2: `allocate_task` uses first-match, not optimal**
- Line 93-107: Returns first idle agent with capabilities. No cost optimization.
- **Fix:** Sort by heuristic (queue length, distance) like coordinator does.

**M3: Missing `_do_start` lifecycle method**
- Line 13: Extends `LifecycleMixin` but only implements `_do_initialize` and `_do_stop`.
- **Fix:** Add `_do_start` for completeness or document intentional omission.

### 🟢 LOW

**M4: No agent heartbeat/timeout detection**
- Agents registered but never removed if they crash.

**M5: `allocate_task` parameter type is `dict` not `dict[str, Any]`**
- Line 93: Missing generic type parameter.

---

## 4. dashboard/server.py

### 🔴 HIGH

**D1: `detach_from_event_bus` is no-op**
- Line 78-82: Sets `_event_bus_subscription = None` but doesn't call `event_bus.unsubscribe()`.
- **Impact:** Memory leak — event handler remains subscribed.
- **Fix:** Implement proper unsubscribe.

### 🟡 MEDIUM

**D2: Wildcard topic `"#"` may not be supported**
- Line 74: Assumes EventBus supports `#` wildcard.
- **Fix:** Document requirement or verify at attach time.

**D3: `_broadcast_loop` catches all Exception silently**
- Line 132-134: Broad exception handler swallows client errors.
- **Fix:** Log exceptions at warning level.

**D4: `get_robots` assumes registry interface without type check**
- Line 105-118: Duck typing on `registry`. Will crash if wrong type passed.
- **Fix:** Add isinstance check or type annotation.

### 🟢 LOW

**D5: `_clients: set[Any]` uses `Any` type**
- Line 34: Should use WebSocket protocol type.

**D6: No rate limiting on client connections**
- `register_client` accepts any client without limit.

---

## 5. dashboard/metrics.py

### 🟡 MEDIUM

**DM1: `_trim` is not thread-safe**
- Line 164-166: List mutation without synchronization.
- **Fix:** Use threading.Lock or atomic operations.

**DM2: `get_provider_stats` doesn't compute per-provider success rate**
- Line 68-71: Tracks calls/errors but doesn't expose success_rate per provider.

### 🟢 LOW

**DM3: No metric export/persistence**
- All metrics in-memory only. Process restart = data loss.

**DM4: `avg_reward` can be NaN if all rewards are None**
- Line 114: Division by zero edge case not handled.

---

## 6. sdk_to_mcp/compiler.py

### 🟡 MEDIUM

**SC1: `compile_directory` swallows exceptions with print only**
- Line 150-151: `except Exception as exc: print(...)` — errors lost.
- **Fix:** Collect failures and return them, or log at warning level.

**SC2: `compile_robot_profile` uses getattr with default but may crash**
- Line 105-106: If `profile` is a dict not an object, `getattr(profile, "robot_id")` returns "unknown" but `getattr(profile, "capability")` may return None, then `getattr(None, "capabilities", [])` returns [] silently.
- **Fix:** Add type guard for profile type.

**SC3: `_build_input_schema` produces invalid JSON schema for non-dict params**
- Line 191: `{"type": "string", "description": str(spec)}` — if spec is int/float, type should reflect that.

### 🟢 LOW

**SC4: `export_to_json` uses `default=str` for non-serializable objects**
- Line 173-174: May produce unexpected string representations.

**SC5: No MCP schema validation**
- Generated schemas not validated against MCP protocol spec.

---

## 7. sdk_to_mcp/manifest.py

### 🟡 MEDIUM

**MB1: No uniqueness validation for tool names**
- Line 46-50: Duplicate tool names silently overwrite in the list.
- **Fix:** Check for duplicates and raise or deduplicate.

**MB2: `add_skill_tool` schema lacks `$schema` and proper JSON Schema structure**
- Line 72-89: Generated schema is minimal — missing `$schema`, `title`, proper type definitions.

### 🟢 LOW

**MB3: No file-based manifest loading**
- Can only build programmatically, no `from_file()` or `from_json()`.

**MB4: Type annotation `list[Any]` for assets is imprecise**
- Line 46: Should be `list[CompiledAsset]`.

---

## 8. cli.py

### 🔴 HIGH

**CL1: `json` module used but not imported in practice replay/export**
- Line 632: `json.loads(line)` — `json` is never imported at module level.
- **Impact:** `NameError` when running `rosclaw practice replay` or `rosclaw practice export`.
- **Fix:** Add `import json` at module level.

**CL2: `cmd_run` may crash if `Runtime` has no `is_running` property**
- Line 115: `while runtime.is_running:` — no guard for missing attribute.
- **Fix:** Use `getattr(runtime, "is_running", False)`.

### 🟡 MEDIUM

**CL3: `--firewall`, `--memory`, `--practice` defaults with `store_true` are confusing**
- Lines 712-723: `action="store_true", default=True` means `--firewall` has no effect (always True).
- **Fix:** Use `action=argparse.BooleanOptionalAction` or `--no-firewall` flags.

**CL4: `cmd_doctor` uses Unicode emoji**
- Lines 207, 212, 214, 224: May break in non-UTF-8 terminals.
- **Fix:** Use ASCII fallbacks.

**CL5: `cmd_logs --follow` advertised but not implemented**
- Line 296: Help text says "not implemented".
- **Fix:** Implement or remove from help.

**CL6: `cmd_run` imports `Runtime` inside function**
- Line 93: Should be at module level.

### 🟢 LOW

**CL7: Missing return code for unknown commands**
- Line 806-808: `parser.print_help()` returns None implicitly (actually returns 1).
- **Clarification:** Actually returns 1, so this is fine.

**CL8: No logging configuration**
- Entire CLI uses `print()` instead of structured logging.

**CL9: `cmd_init` config template hardcoded as string**
- Lines 46-77: Should be loaded from template file for maintainability.

---

## Recommendations by Priority

### Immediate (before v1.0 release)
1. **CL1**: Fix missing `json` import in cli.py practice commands
2. **C1**: Add allocation rollback in SwarmCoordinator
3. **M1**: Handle allocation failure in SwarmRuntimeManager
4. **D1**: Implement proper EventBus unsubscribe in DashboardServer

### Short-term (v1.0.x)
5. **CL2**: Add `is_running` guard in `cmd_run`
6. **CL3**: Fix CLI boolean flag semantics
7. **C2**: Move `import math` to module level
8. **S1**: Rename `RaftLikeConsensus` to reflect actual behavior
9. **SC1**: Collect compilation errors instead of printing
10. **MB1**: Validate tool name uniqueness in manifest builder

### Long-term (v1.1)
11. Add thread safety to all shared-state modules
12. Add metric persistence to DashboardMetrics
13. Implement full Raft consensus or rename
14. Add agent heartbeat/timeout in swarm
15. Add MCP schema validation in compiler
