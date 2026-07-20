# ROSClaw P0 Agent Integration

This document describes the P0 integration between agent frameworks and the ROSClaw physical-AI runtime. It covers the generated project files, the MCP tool surface, the adapter architecture, and how to validate an installation.

## Goals

- An agent that opens a ROSClaw project should immediately understand the runtime boundary.
- The agent has **read-only**, **simulation-only**, **validated-plan**, and **emergency** tools available.
- There is **no real-execution tool** in P0. Any request to move the real robot must be refused or routed through `validate_trajectory` with operator confirmation.
- If ROSClaw components are unavailable, tools return a structured `ok: false`
  envelope. Synthetic responses are available only when fixture mode is
  explicitly enabled.

## Generated project files

For any agent framework, use the universal installer:

```bash
rosclaw agent install --project-root . --skip-secrets
```

Running `rosclaw agent install` produces or updates the following files in the project root:

| File | Purpose |
|------|---------|
| `.mcp.json` | MCP server configuration (stdio or HTTP transport) |
| `.codex/config.toml` | Project-scoped Codex MCP configuration with the exact P0 tool allowlist |
| `AGENTS.md` | Cross-agent safety and operating guidance for Codex and other readers |
| `.agents/skills/rosclaw/SKILL.md` | Repo-local ROSClaw skill for agent workflows |
| `.claude/settings.json` | Workspace deny/allow rules and model context settings |
| `CLAUDE.md` | Claude Code onboarding with a managed safety-contract block |
| `ROSCLAW.md` | Extended onboarding and troubleshooting for the agent |
| `.rosclaw/agent/context.snapshot.json` | Machine-readable snapshot of detected robot, runtime, and environment |

Merge behavior:

- `AGENTS.md`, `CLAUDE.md`, and `ROSCLAW.md` use managed blocks demarcated by `<!-- ROSCLAW-MANAGED-BEGIN -->` / `<!-- ROSCLAW-MANAGED-END -->`.
- If `AGENTS.md` already exists without a ROSClaw managed block, `rosclaw agent install` preserves the existing content and appends the ROSClaw section.
- Hand-written sections outside existing managed blocks are preserved on later runs.
- `.mcp.json` and `.claude/settings.json` are merged with conflict detection. Collisions are reported and backed up; no data is silently overwritten.
- `.codex/config.toml` uses a managed TOML block. Existing unrelated Codex
  settings are preserved; an unmanaged `mcp_servers.rosclaw` collision fails
  closed instead of creating duplicate TOML tables.
- OpenClaw discovers `.agents/skills/rosclaw/SKILL.md` directly. Its native
  global MCP registry remains operator-owned and can be configured with
  `openclaw mcp add`; the project installer does not silently mutate
  `~/.openclaw/openclaw.json`.

## MCP tool surface

The P0 server exposes 18 tools via `src/rosclaw/mcp/server.py`:

| Tool | Safety level | Purpose |
|------|--------------|---------|
| `get_robot_state` | S0 read-only | Current body state, body sense, and readiness |
| `list_skills` | S0 read-only | Skills available to the runtime |
| `query_memory` | S0 read-only | Retrieve similar past experiences from `MemoryInterface` |
| `practice_query` | S0 read-only | Query practice episodes from `EpisodeRecorder` |
| `get_body_profile` | S0 read-only | Static effective body profile |
| `get_body_state` | S0 read-only | Body safety state and capability matrix |
| `list_body_capabilities` | S0 read-only | Capabilities grouped by status |
| `query_body` | S0 read-only | Answer questions about the current body |
| `validate_body_action` | S0 read-only | Validate a proposed body-level action |
| `get_calibration_status` | S0 read-only | Calibration status for body components |
| `validate_trajectory` | S2 validated-plan | Validate a trajectory through the sandbox/firewall; never executes real motion |
| `sandbox_run` | S1 simulation-only | Run one real MuJoCo step; fails if physics/model loading is unavailable |
| `emergency_stop` | S4 emergency | Fan out a stop request and return dispatch, driver ACK, timeout, and physical-observation evidence |
| `get_product_status` | S0 read-only | Read the canonical release, support tiers, and evidence boundary |
| `list_product_demos` | S0 read-only | Discover official evidence-bearing product demos |
| `run_product_demo` | S1 simulation-only | Run the official MuJoCo demo and persist its receipt |
| `get_execution_receipt` | S0 read-only | Read and integrity-check a persisted execution receipt |
| `explain_execution` | S0 read-only | Explain request, policy, execution, observation, and verification for a run |

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

`SafetyClient.emergency_stop` delegates to
`Runtime.request_emergency_stop`. Runtime publishes one managed critical event,
fans out to registered drivers, waits for bounded acknowledgements, and returns
an `EmergencyStopReceipt`. `stopped=true` is emitted only when stopped state was
physically observed in the stated execution mode. Publishing a request alone is
never reported as a confirmed stop.

## Usage

### Initialize a project

```bash
rosclaw agent install --project-root . --skip-secrets
```

Use `rosclaw agent init claude-code` when you only want the legacy Claude Code file set. Use `--project-root <path>` to target a directory other than the current working directory.

### Validate the installation

```bash
rosclaw agent doctor universal
rosclaw agent test universal --quick --mcp-probe
rosclaw demo run ur5e-reach
rosclaw explain latest
```

- `doctor` checks file presence, transport reachability, and safety-contract completeness.
- `test --mcp-probe` starts the configured stdio server, requires the exact
  18-tool P0 boundary, runs the official MuJoCo demo, reloads its
  integrity-checked receipt, and verifies its explanation.
- `demo run` and `explain` expose the same receipt-bearing workflow directly to
  operators.

### Start the MCP server manually

```bash
# stdio (default, used by Claude Code)
rosclaw-mcp-serve --transport stdio --log-level INFO

# HTTP transport
rosclaw-mcp-serve --transport http --host 127.0.0.1 --port 9090

# Explicit synthetic mode for demos/tests only
rosclaw-mcp-serve --transport stdio --fixture
```

Environment variables: `ROSCLAW_MCP_TRANSPORT`, `ROSCLAW_MCP_HOST`,
`ROSCLAW_MCP_PORT`, `ROSCLAW_ROBOT_ID`, `ROSCLAW_PROJECT_ROOT`,
`ROSCLAW_LOG_LEVEL`, and explicit `ROSCLAW_MCP_FIXTURE=1`.

## Safety boundaries

- `validate_trajectory` and `sandbox_run` are restricted to planning and simulation. They do not command hardware.
- `emergency_stop` is the only real-side-effect tool in this MCP surface and is
  gated at S4. It still does not replace a certified physical E-Stop.
- The audit logger redacts values for keys matching `{token, password, secret, api_key, apikey, auth}` before writing to `~/.rosclaw/logs/mcp/audit.jsonl`.
- All state tools include `is_stale` and `age_ms` so the agent can judge freshness.

## Verification

On 2026-07-20, a separate local Codex CLI 0.144.4 process opened a trusted
project generated by `rosclaw agent install`, discovered the exact 18-tool P0
allowlist, and called `get_product_status`, `list_product_demos`,
`run_product_demo`, `get_execution_receipt`, and `explain_execution` through
MCP. Run `run_20260720T110724603844Z_92bc51fe` completed 237 MuJoCo steps with
`TASK_VERIFIED`, `has_physics=true`, and `usable_for_real_execution=false`.
This is developer-observed external-Agent evidence, not independent H5
acceptance and not real-hardware evidence.

```bash
ruff check src tests
mypy --config-file .github/mypy-ci.ini \
  src/rosclaw/kernel src/rosclaw/sandbox src/rosclaw/provider/core \
  src/rosclaw/mcp_drivers src/rosclaw/mcp/adapters src/rosclaw/agent \
  src/rosclaw/agent_runtime/mcp_hub.py src/rosclaw/core/runtime.py src/rosclaw/cli.py
pytest -q
rosclaw doctor --level verified --json
rosclaw agent test universal --project-root . --quick --mcp-probe
```

The transport tests exercise stdio and streamable HTTP. They expect synthetic
Body Sense to fail closed in normal mode and verify that MuJoCo simulation is
reported as `SIMULATION`, not `live`. Release validation also builds the wheel,
installs it outside the repository, and confirms the packaged UR5e MJCF and mesh
assets can execute a MuJoCo step without resolving files from the source tree.

## Remaining gaps

1. No real-hardware MCP execution tool is exposed by this P0 server.
2. ROS2 and DDS Body Sense collectors are still stubs and therefore return
   unavailable rather than live state.
3. Legacy physical side-effect adapters outside this P0 MCP surface are still
   being migrated to `Runtime.submit_action`.
4. Hardware E-Stop requires device-specific ACK and feedback acceptance tests.
