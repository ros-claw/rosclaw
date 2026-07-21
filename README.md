<div align="center">

# ROSClaw

### Trustworthy Physical Execution Runtime and Control Plane for Embodied Agents

**Ground actions to a body, fail closed, execute with evidence, and return an auditable receipt.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python)](https://www.python.org/)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble_|_Jazzy-FF3E00?logo=ros)](https://docs.ros.org/)
[![Simulation](https://img.shields.io/badge/Verified_Simulation-MuJoCo-black?logo=mujoco)](https://mujoco.org/)
[![MCP](https://img.shields.io/badge/Protocol-MCP-8A2BE2)](https://modelcontextprotocol.io/)
[![Status](https://img.shields.io/badge/Status-Alpha-orange)](https://github.com/ros-claw/rosclaw)

[Website](https://rosclaw.io) · [Quick Start](QUICKSTART.md) · [Architecture](ARCHITECTURE.md) · [Docs](docs/) · [Contact](mailto:ai@rosclaw.io)

</div>

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot --yes --profile offline --no-telemetry
rosclaw doctor --level verified
```

---

## What is ROSClaw?

ROSClaw is not another agent framework, a replacement for ROS 2, or a thin
LLM-to-ROS wrapper.

ROSClaw is a **trustworthy physical execution runtime and control plane for
embodied agents**. Codex, Claude Code, OpenClaw, VLA services, and other agents
are northbound clients. ROS 2, MCP, vendor SDKs, simulators, and robot
controllers are southbound systems.

Its canonical action path binds intent to a body and capability, applies policy
and authorization, arbitrates physical resources, dispatches to a driver, and
returns an evidence-bearing `ExecutionReceipt`. Memory and self-evolution are
asynchronous consumers of that evidence, not substitutes for it.

### Current maturity

ROSClaw is alpha software. The table below is the capability boundary, not a
roadmap disguised as completed work.

<!-- product-status:start -->
| Scope | Status | Evidence available today |
|---|---|---|
| UR5e tabletop reach | **Simulation verified** | Real MuJoCo execution and receipt verification are system-tested. Local Codex CLI black-box runs discovered all 22 MCP tools, completed the simulation workflow, and confirmed that rosclawd blocks unauthorized REAL actions without hardware dispatch; independent H5 acceptance remains pending. |
| rosclawd Agent/physical boundary | **Component/system verified** | Cross-UID sockets, SO_PEERCRED identity, private state, exact permits, restart recovery, DISARMED generations, Agent Sessions, Action Leases, orphan handling, watchdogs, and isolated worker restart budgets are system-tested; site-specific device ACLs, controller deadman wiring, and hardware acceptance remain pending. |
| Action contract and gateway | **Component/system verified** | Versioned action and receipt, finite deadline and lease, orphan policy, stop capability, strict acknowledgement stages, idempotency, exclusive body lease, and fail-closed executor lookup. |
| Capability-only App runtime | **Component verified** | Local and bundled App manifests, digest-locked installation, low-code authoring, and daemon-only workflow execution are tested; remote App registry and independent Agent acceptance remain pending. |
| E-Stop control path | **Component verified** | Fan-out, timeout, partial ACK, idempotency, latch, and physical-observation fields; no independent physical-stop verification. |
| Mock Sense, mock Providers, fixture Drivers | **Fixture only** | Explicit FIXTURE and SYNTHETIC data only; never valid for safety or real acceptance. |
| RealSense perception-only path | **Experimental** | A signed, commit-locked RealSense Robot Integration covers install/configure/verify/status, Body binding, daemon-side RGB-D execution, artifact hashing, and bounded MCP subprocess faults; this repository still has no independently verified hardware capture run. |
| RH56 LeRobot single-step loop | **Developer-observed; revalidation pending** | Real SerialModbusTransport, shadow reads, graded REAL actions, and fault injection were developer-observed. Loopback fault tests now prove that target-near readback is DELIVERY_INFERRED rather than a protocol ACK; independent v1 hardware revalidation and Agent black-box testing remain open. |
| ROS connectors, LeRobot, hardware MCP, real Providers | **Experimental** | Contract and component coverage varies; registration or import does not imply execution readiness. |
| ROS 2 Turtlesim guarded motion | **Not verified** | Connector contracts exist, but the ROS 2 golden path has not been run in the current validation environment. |
| Mobile base continuous-lease deadman | **Simulation verified** | A MuJoCo mobile base reaches a bounded velocity, loses a 60 ms controller lease, trips its deadman, and reaches below 0.01 m/s with bounded post-loss travel. No physical mobile-base claim is made. |
| Repository-wide real robot execution | **Revalidation pending** | RH56 has developer-run physical evidence, but independent hardware and Agent black-box acceptance are still pending; no repository-wide real-ready claim is made. |
<!-- product-status:end -->

The core package supports Python 3.11 through 3.13. The isolated LeRobot 0.6 runtime and
the bundled RH56 reference-policy plugin require Python 3.12+.

The default install keeps the first verified simulation path lightweight.
Install `rosclaw[knowledge]` when this process should also host the optional
Know/How semantic services and their model runtime.

### Execution modes

| Mode | Meaning |
|---|---|
| `FIXTURE` | Explicit synthetic data. Never verified and never valid for real execution. |
| `DRY_RUN` | Schema/static-policy evaluation; no physical simulation or dispatch. |
| `REPLAY` | Previously recorded evidence; no new physical effect. |
| `SIMULATION` | Physics-backed execution. The UR5e reach path is currently verified. |
| `SHADOW` | Contract defined; no verified golden path yet. |
| `REAL` | Contract defined and fail-closed; no repository-wide real-hardware acceptance claim. |

---

## Why Physical AI Needs Runtime Infrastructure

Large language models can plan, write code, and reason over symbols. But physical intelligence requires more than tokens.

A physical agent must know:

- What body it has;
- What sensors and actuators it owns;
- What actions are safe;
- What happened during execution;
- Why a skill failed;
- How to recover;
- How to improve without breaking safety.

Physical worlds have gravity, friction, collision, latency, sensor noise, torque limits, joint limits, and safety boundaries. ROSClaw provides the missing infrastructure between high-level AI agents and the physical world.

---

## Runtime Loop

```text
Action Intent → Body/Capability → rosclawd Permit/Policy → Resource Lease
              → Dispatch/ACK → Physical Observation → Verification → Receipt

Receipt → Trace/Practice → Memory/How/Auto/Darwin (asynchronous)
```

> **A request is not execution, dispatch is not completion, and completion
> requires evidence.**

Auto may propose changes, but it cannot approve them alone. Sandbox validation, Darwin evaluation, the promotion gate, and human approval together decide whether a change reaches the real world.

### rosclawd boundary

For Agent-driven physical work, `rosclawd` is the only supported control-plane
entry. MCP and CLI clients submit structured actions over a protected local
socket; the daemon independently checks peer/body/snapshot/capability/action
intent, Session, expiry, and use count before accepting a permit. Every action
has a finite Deadline and renewable Lease; Session loss applies the declared
orphan policy. The daemon owns the queue, isolated Adapter workers, physical
Runtime, E-Stop latch, watchdogs, and receipts. An authenticated local ledger
persists permit consumption and action transitions, restores terminal receipts
after restart, and fails closed for interrupted REAL actions pending daemon-UID
operator review. Every daemon generation starts `DISARMED` and never resumes an
old physical action.

```bash
# Development/process-boundary smoke test only
rosclawd --robot-id sim_ur5e
rosclaw daemon status --json
rosclaw daemon session-create --session-id agent-1 --actor-id codex-1 \
  --agent-framework codex --body sim_ur5e --capability sandbox.reach --json
rosclaw daemon security-check --json
# Only when status reports recovery.required=true, run as the daemon UID:
rosclaw daemon acknowledge-recovery --reason "reviewed interrupted action evidence" --json
```

Same-UID development mode is not a hardware privilege boundary. The clean-wheel
cross-UID and reference-systemd acceptance scripts verify the generic Linux
identity, socket, and private-state boundary. A deployed Agent must additionally
pin `ROSCLAW_DAEMON_UID` to the service account. REAL deployments still require
site-specific device ACLs, credential isolation, and SROS2/DDS access control in
[docs/ROSCLAWD.md](docs/ROSCLAWD.md). The local HMAC key and signed head detect
ordinary tampering and one-sided rollback; they are not a TPM or remote witness
against an owner-level attacker who can replace all daemon state.

---

## First Embodiment / Quick Start

Install the CLI and run the interactive first-boot wizard:

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot
rosclaw doctor --level verified
```

Run a local simulation demo without any hardware:

```bash
rosclaw demo run ur5e-reach
rosclaw explain latest
```

For headless or CI environments:

```bash
rosclaw firstboot --yes --profile offline --no-telemetry
```

See [QUICKSTART.md](QUICKSTART.md) for four guided paths: local simulation, agent integration, robot body setup, and developer setup.

---

## Robot Integrations

A Robot Integration is the user-facing, signed and versioned onboarding unit
above individual Body, Hardware MCP, capability, policy, calibration, and
verification assets. Its internal manifest remains `RobotPack`; it is an asset
bundle, not another execution framework. The first built-in Integration
supports perception-only RealSense D405/D435i onboarding:

```bash
rosclaw robot discover --type camera --json
rosclaw robot install ros-claw/realsense-d400
rosclaw robot verify realsense-d400 --stage contract
rosclaw robot configure realsense-d400 --instance lab-d405 --serial SERIAL
```

Add `--install-adapter` only when the host should install the Integration's native MCP
dependencies at its locked commit. Installation and offline configuration are
not hardware verification. H3 additionally requires complete live identity,
stream profiles, real RGB-D artifacts and hashes, a canonical `rosclawd`
receipt, and independent physical observation.

See [docs/ROBOT_PACKS.md](docs/ROBOT_PACKS.md) for trust, daemon loading, support
tiers, and the hardware acceptance procedure.

## Capability Apps

Apps are small capability-only task manifests. They never name device paths,
registers, ROS command topics, or Adapter APIs, and every step is submitted
through `rosclawd`:

```bash
rosclaw app install ros-claw/realsense-inspect
rosclaw app validate realsense-inspect --json
rosclaw app run realsense-inspect --body lab-d405 --mode SHADOW --json
```

The bundled `realsense-inspect` and `rh56-rps` manifests are component-tested.
This does not claim a verified camera capture or RH56 Agent run: RealSense H3
still needs independent hardware evidence, and RH56 production Robot
Integration/Worker migration is still pending. See [docs/APPS.md](docs/APPS.md).

---

## Core Runtime Modules

| Module | Responsibility |
|---|---|
| **Runtime** | Lifecycle, configuration, plugin registration, dependency injection. |
| **EventBus** | Module communication, topic routing, trace correlation. |
| **Provider** | Capability routing, schema enforcement, safety boundary. |
| **Sandbox** | Safety validation, firewall, MuJoCo pre-play. |
| **Practice** | Timeline capture, MCAP, JSONL, execution records. |
| **Memory** | Experience graph, failure/success patterns, recall. |
| **Know** | TaskCard, Pattern, EvidenceTrace, failure taxonomy. |
| **How** | Runtime intervention, injection ID, evidence-backed repair. |
| **Auto** | Proposal, patch, experiment, champion, dead-end tracking. |
| **Darwin** | Multi-seed benchmark, stress scenario, regression. |
| **Skill Registry** | Version, lineage, champion, rollback. |
| **Dashboard** | Observability, evolution trace, lineage visualization. |

---

## Structured Runtime Trace

ROSClaw records a causal span tree across Runtime, Provider, MCP, Skill, Sandbox, and Firewall
boundaries. Model/tool I/O is bounded and redacted, binary perception data is stored as references,
and private chain-of-thought fields are omitted by default in favor of structured decision summaries.

```bash
rosclaw trace list
rosclaw trace show <trace-id> --tree
rosclaw trace tail --kind VLM,MCP,SANDBOX
rosclaw dashboard --trace <trace-id>
```

The built-in Trace view is available at `/traces`.

See [docs/TRACE.md](docs/TRACE.md) for the schema, instrumentation API, capture modes, CLI, and
Dashboard endpoints.

---

## Hub & Assets

The ROSClaw Hub is a **Physical-AI Asset Hub** for skills, providers, hardware MCP servers, digital twins, and cognitive wikis. Assets can be kept entirely local or synced with a registry.

Supported asset types:

- `skill` — reusable physical-AI skill
- `provider` — runtime capability provider
- `hardware_mcp` — MCP server that wraps real hardware
- `digital_twin` — simulation asset / e-URDF twin
- `cognitive_wiki` — structured operational knowledge

```bash
rosclaw hub validate tests/fixtures/hub_assets/hardware_mcp_valid/manifest.yaml
rosclaw hub search g1
rosclaw hub install rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0 --yes
```

See [docs/ASSETS.md](docs/ASSETS.md) and [docs/hub/README.md](docs/hub/README.md).

---

## Hardware MCP Onboarding

ROSClaw can auto-install hardware MCP servers from declarative manifests and keep
them healthy. Manifests resolve in this order:

1. Local cache under `~/.rosclaw/mcp/cache/`.
2. Built-in offline registry (`unitree-g1`, `realsense-d455`, ...).
3. Remote ROSClaw Hub at `https://www.rosclaw.io/api/registry`.

If the network is unavailable, commands automatically fall back to the cache and
built-ins.

### Quick start

```bash
# Preview what an install would do (no changes written)
./rosclaw mcp install unitree-g1 --dry-run --offline

# Install a built-in hardware MCP without touching the network
./rosclaw mcp install unitree-g1 --offline

# Preview a public Hub package (remove --offline to fetch from the network)
./rosclaw mcp install ros-claw/g1-mcp --dry-run --offline

# Use a private/custom hub endpoint
ROSCLAW_MCP_HUB=https://my-hub.example.com ./rosclaw mcp install ros-claw/g1-mcp --dry-run --offline

# List installed and available servers
./rosclaw mcp list --offline
./rosclaw mcp list --offline --json

# Health-check installed servers
./rosclaw mcp health
./rosclaw mcp health unitree-g1
./rosclaw mcp health unitree-g1 --full --json
```

Use `--dry-run` to inspect the resolved manifest, version, artifact, body patch,
permissions, and Claude `.mcp.json` merge plan before writing anything. Use
`--offline` to force local-only resolution (cache + built-ins).

See [`docs/HARDWARE_MCP_ONBOARDING.md`](docs/HARDWARE_MCP_ONBOARDING.md) for the
full lifecycle, state files, permissions, and troubleshooting.

---

## Install and Configure ROSClaw for Any Agent

Paste this setup prompt into Codex, Claude Code, OpenClaw, or another
MCP-aware agent:

> Install and configure ROSClaw for this repository. Run
> `rosclaw agent install --project-root . --skip-secrets`, then read
> `ROSCLAW.md`, `AGENTS.md`, `.codex/config.toml`, and
> `.agents/skills/rosclaw/SKILL.md`. Validate
> the setup with `rosclaw agent test universal --project-root . --quick
> --mcp-probe`. For Codex, reopen this exact repository and accept workspace
> trust; for Claude Code, approve the project-scoped `rosclaw` MCP server.
> After that, use ROSClaw through its CLI and MCP tools for robot
> state, skills, memory, sandbox simulation, practice records, and safety
> checks. For physical work, use only the daemon-backed `request_action`,
> action-status, cancellation, and emergency tools; never open a device or
> publish a command topic directly.

If you are doing the setup yourself, the core install command is:

```bash
rosclaw agent install --project-root . --skip-secrets
```

This installs and configures the agent-facing ROSClaw integration files:
`.mcp.json`, `.codex/config.toml`, `AGENTS.md`, `ROSCLAW.md`, `CLAUDE.md`,
`.agents/skills/rosclaw/SKILL.md`, and
`.rosclaw/agent/context.snapshot.json`.

It does not install ROSClaw itself and does not install a native plugin for one
specific agent framework. ROSClaw remains an independent CLI, Python package,
MCP server, and robotics infrastructure layer; this command teaches agent
harnesses how to discover and use it. Claude Code receives `.mcp.json`, Codex
receives a trusted-project `.codex/config.toml`, and OpenClaw discovers the
workspace skill under `.agents/skills`.

The installer cannot bypass harness security prompts. Codex ignores the
project `.codex/config.toml` until the exact Git repository root is trusted;
check it with `rosclaw agent doctor codex --project-root .`. Claude Code shows
the project server as pending until the user approves it; check it with
`claude mcp get rosclaw`.

Run the MCP probe from the same environment and `PATH` used to launch the
Agent. The probe executes the configured `rosclaw` command and requires the
exact 22-tool boundary, so an older global CLI cannot silently pass by using
the source checkout's Python process.

---

## Persist Practice to SeekDB

Use SQLite for local development or a MySQL-compatible DSN for a real
SeekDB/OceanBase server:

When `--data-root` is omitted, ROSClaw uses `ROSCLAW_PRACTICE_DATA_ROOT` when
set, otherwise `$ROSCLAW_HOME/data/practice` (default `~/.rosclaw/data/practice`).
Container deployments can explicitly set the environment variable to
`/data/rosclaw/practice`.

```bash
# Local file
rosclaw practice ingest-seekdb <practice_id> \
  --seekdb-path ~/.rosclaw/memory/seekdb.sqlite

# Real SeekDB server
rosclaw practice ingest-seekdb <practice_id> \
  --seekdb-url mysql://root@127.0.0.1:2881/rosclaw

rosclaw practice query failures \
  --robot-id rh56 \
  --seekdb-url mysql://root@127.0.0.1:2881/rosclaw \
  --json
```

SeekDB port `2881` uses the MySQL-compatible SQL protocol. It is not an HTTP
API. Repeated Practice ingestion is idempotent by episode and evidence IDs.

---

## Safety Model

ROSClaw's core safety rule:

> **No model output should directly control a robot.**

The new canonical physical-action path is:

1. Provider produces a structured action proposal.
2. Sandbox / Firewall checks it against the effective body model and safety policy.
3. The decision is one of `ALLOW`, `MODIFY`, `BLOCK`, or `REQUIRE_HUMAN_CONFIRMATION`.
4. `rosclawd` authenticates the peer and matches an expiring, use-bounded permit
   to the Body Snapshot, explicit Capability, and exact Action Intent.
5. `ActionGateway` acquires an exclusive resource lease and dispatches a daemon-owned executor.
6. Driver ACK, observations, verification, trace, and artifacts are assembled into a receipt.
7. Practice, Memory, How, Auto, and Darwin may consume the receipt asynchronously.

Legacy execution adapters are still being migrated to this gateway. Do not
assume that every historical CLI, Skill, ROS connector, or vendor path is
already non-bypassable. `REAL` deployments must pass body-specific acceptance
tests and keep southbound credentials outside the agent process.

The known MCPHub low-level actions, standalone UR5 MCP motion tools, and ROS
connector capability execution now fail closed instead of dispatching directly.
The ROS connector still supports discovery, validation, explicit dry runs, and
daemon-backed emergency-stop request receipts while its executors are migrated.

ROSClaw is research infrastructure. It does not replace certified industrial safety systems. Always test in simulation first, keep emergency stops engaged, and use human supervision.

Read [docs/SAFETY.md](docs/SAFETY.md) for the full safety model.

---

## Documentation

- [QUICKSTART.md](QUICKSTART.md) — 5-minute quick start.
- [INSTALL.md](INSTALL.md) — Detailed installation and troubleshooting.
- [docs/FIRSTBOOT.md](docs/FIRSTBOOT.md) — Bootstrap and first boot reference.
- [docs/CLI.md](docs/CLI.md) — CLI command reference.
- [docs/SAFETY.md](docs/SAFETY.md) — Safety model and deployment rules.
- [docs/ROSCLAWD.md](docs/ROSCLAWD.md) — Agent/physical process and privilege boundary.
- [docs/ASSETS.md](docs/ASSETS.md) — Physical-AI Asset Hub.
- [ARCHITECTURE.md](ARCHITECTURE.md) — Runtime architecture.
- [CONTRIBUTING.md](CONTRIBUTING.md) — Development standards.

---

## Roadmap

| Phase | Focus |
|---|---|
| **Current / Alpha** | rosclawd process boundary and durable local control ledger, truthful action/receipt contracts, fail-closed fixtures/providers/drivers, MuJoCo UR5e reach golden path, readiness-level Doctor. |
| **Next** | Migrate every physical side-effect path to the gateway; ledger compaction/external witnessing; cancellation/preemption; ROS 2 Turtlesim observed-motion golden path. |
| **Hardware acceptance** | RealSense read-only capture, then bounded actuator tasks with ACK, feedback, stop verification, and receipts. |
| **Later** | Receipt-driven Memory/How/Auto/Darwin promotion with independent evaluation and rollback. |

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for standards, PR process, and code style.

---

## Contact

- Email: [ai@rosclaw.io](mailto:ai@rosclaw.io)
- Issues: [GitHub Issues](https://github.com/ros-claw/rosclaw/issues)
- Discussions: [GitHub Discussions](https://github.com/ros-claw/rosclaw/discussions)

---

## License

[MIT](LICENSE)
