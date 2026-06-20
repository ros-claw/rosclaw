# ROSClaw P0 Claude Code Agent Integration

This document describes the P0 integration between Claude Code and the ROSClaw physical-AI runtime. It covers the generated project files, the MCP tool surface, the adapter architecture, and how to validate an installation.

## Goals

- An agent that opens a ROSClaw project should immediately understand the runtime boundary.
- The agent has **read-only**, **simulation-only**, **validated-plan**, and **emergency** tools available.
- There is **no real-execution tool** in P0. Any request to move the real robot must be refused or routed through `validate_trajectory` with operator confirmation.
- If ROSClaw components are unavailable, tools degrade gracefully to well-formed fixture responses instead of crashing the agent.

## Generated project files

Running `rosclaw agent init claude-code` produces or updates the following files in the project root:

| File | Purpose |
|------|---------|
| `.mcp.json` | Claude Code MCP server configuration (stdio or HTTP transport) |
| `.claude/settings.json` | Workspace deny/allow rules and model context settings |
| `CLAUDE.md` | Human-readable onboarding with a managed safety-contract block |
| `ROSCLAW.md` | Extended onboarding and troubleshooting for the agent |
| `.rosclaw/agent/context.snapshot.json` | Machine-readable snapshot of detected robot, runtime, and environment |

Merge behavior:

- `CLAUDE.md` and `ROSCLAW.md` use managed blocks demarcated by `<!-- rosclaw-managed -->`. Hand-written sections outside those blocks are preserved.
- `.mcp.json` and `.claude/settings.json` are merged with conflict detection. Collisions are reported and backed up; no data is silently overwritten.

## MCP tool surface

The P0 server exposes seven tools via `src/rosclaw/mcp/server.py`:

| Tool | Safety level | Purpose |
|------|--------------|---------|
| `get_robot_state` | S0 read-only | Current body state, body sense, and readiness |
| `list_skills` | S0 read-only | Skills available to the runtime |
| `query_memory` | S0 read-only | Retrieve similar past experiences from `MemoryInterface` |
| `practice_query` | S0 read-only | Query practice episodes from `EpisodeRecorder` |
| `validate_trajectory` | S2 validated-plan | Validate a trajectory through the sandbox/firewall; never executes real motion |
| `sandbox_run` | S1 simulation-only | Run one MuJoCo simulation step |
| `emergency_stop` | S4 emergency | Publish `robot.emergency_stop` on the `EventBus` to halt all motion |

All tool responses share a common envelope defined in `src/rosclaw/mcp/schemas/common.py` with `ok`, `timestamp`, `trace_id`, and an optional structured `error` field.

## Adapter architecture

`src/rosclaw/mcp/adapters/runtime_client.py` is the single facade used by the MCP tool layer. It lazy-initializes a ROSClaw `Runtime` and builds thin adapters around each subsystem:

- `MemoryClient` — `src/rosclaw/mcp/adapters/memory_client.py`
- `PracticeClient` — `src/rosclaw/mcp/adapters/practice_client.py`
- `SandboxClient` — `src/rosclaw/mcp/adapters/sandbox_client.py`
- `SkillRegistryClient` — `src/rosclaw/mcp/adapters/skill_registry_client.py`
- `SafetyClient` — `src/rosclaw/mcp/adapters/safety_client.py`

Each adapter is small, independently unit-testable, and translates a subsystem API into the dict envelopes the MCP server expects. `RuntimeClient` caches adapters per `Runtime` instance so tests can swap in different runtimes without cross-test contamination.

### Skill registry fallback

`SkillRegistryClient` uses the `registry` attribute of the skill manager when it exists and is not `None`; otherwise it falls back to the skill manager itself. This matches production semantics and prevents test mocks from shadowing the manager with an auto-created `registry` attribute.

### Emergency stop

`SafetyClient.emergency_stop` publishes a `robot.emergency_stop` event on the `EventBus` with `EventPriority.CRITICAL`. The `EventBus` calls synchronous subscribers immediately, so any registered stop handler is invoked in-band. The MCP tool no longer calls `Runtime._on_emergency_stop` directly, removing a layering violation and making the stop path observable by other runtime subscribers.

## Usage

### Initialize a project

```bash
rosclaw agent init claude-code --check
```

Add `--check` to validate the generated files without writing them. Use `--project-root <path>` to target a directory other than the current working directory.

### Validate the installation

```bash
rosclaw agent doctor claude-code
rosclaw agent test claude-code
```

- `doctor` checks file presence, transport reachability, and safety-contract completeness.
- `test` runs a short sequence of MCP tool calls against the configured server and verifies envelope shape.

### Start the MCP server manually

```bash
# stdio (default, used by Claude Code)
rosclaw-mcp-serve --transport stdio --log-level INFO

# HTTP transport
rosclaw-mcp-serve --transport http --host 127.0.0.1 --port 9090
```

Environment variables: `ROSCLAW_MCP_TRANSPORT`, `ROSCLAW_MCP_HOST`, `ROSCLAW_MCP_PORT`, `ROSCLAW_ROBOT_ID`, `ROSCLAW_PROJECT_ROOT`, `ROSCLAW_LOG_LEVEL`.

## Safety boundaries

- `validate_trajectory` and `sandbox_run` are restricted to planning and simulation. They do not command hardware.
- `emergency_stop` is the only destructive tool and is gated at S4.
- The audit logger redacts values for keys matching `{token, password, secret, api_key, apikey, auth}` before writing to `~/.rosclaw/logs/mcp/audit.jsonl`.
- All state tools include `is_stale` and `age_ms` so the agent can judge freshness.

## Verification

```bash
# Lint
ruff check src tests

# Focused type check (project-wide mypy has pre-existing errors)
mypy src/rosclaw/mcp/adapters src/rosclaw/core/runtime.py --follow-imports=skip --ignore-missing-imports

# Tests
pytest tests/agent tests/mcp tests/security tests/practice
```

Current follow-up status:

- `ruff check src tests` — passing.
- `mypy --config-file .github/mypy-ci.ini src/rosclaw/mcp/adapters src/rosclaw/core/runtime.py` — passing.
- `pytest tests/agent tests/mcp tests/security tests/practice` — 67 passing.
- `mypy src` — pre-existing type errors across the wider codebase; changed modules compile and tests pass.

## Remaining gaps

1. ~~End-to-end MCP smoke tests over stdio/HTTP transport.~~ (Done in `tests/mcp/test_e2e.py`.)
2. ~~Live `Runtime` integration tests for all seven tools in fixture or sim mode.~~ (Done in `tests/mcp/test_runtime_integration.py`.)
3. ~~Performance/load tests for `EventBus` synchronous dispatch under emergency-stop load.~~ (Done in `tests/mcp/test_event_bus_emergency_perf.py`.)
4. ~~CI enforcement of `ruff check src tests` and a focused `mypy` gate.~~ (Done in `.github/workflows/ci.yml` and `.github/mypy-ci.ini`.)
5. Continuous documentation refresh for `CLAUDE.md` and `ROSCLAW.md` managed blocks.
