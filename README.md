<div align="center">

# ROSClaw

### Self-Evolving Runtime Infrastructure for Physical AI & Embodied Agents

**Ground AI agents into robot bodies. Validate every action. Learn from every trace. Evolve every skill.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://www.python.org/)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble_|_Jazzy-FF3E00?logo=ros)](https://docs.ros.org/)
[![Simulation](https://img.shields.io/badge/Simulation-MuJoCo_|_Isaac--Sim-black?logo=mujoco)](https://mujoco.org/)
[![MCP](https://img.shields.io/badge/Protocol-MCP-8A2BE2)](https://modelcontextprotocol.io/)
[![Status](https://img.shields.io/badge/Release-v1.0-purple)](https://github.com/ros-claw/rosclaw/releases)

[Website](https://rosclaw.io) · [Quick Start](QUICKSTART.md) · [Architecture](ARCHITECTURE.md) · [Docs](docs/) · [Contact](mailto:ai@rosclaw.io)

</div>

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot
```

---

## What is ROSClaw?

ROSClaw is not another chatbot framework. It is not a thin LLM-to-ROS wrapper. It is not a collection of unrelated robotics scripts.

ROSClaw is a **runtime infrastructure layer for Physical AI and embodied agents**. It connects AI agents, robot embodiments, simulation sandboxes, capability routing, physical memory, praxis capture, runtime intervention, and skill evolution into one coherent operating layer.

It is built for embodied agents that must reason, act safely, remember what happened, recover from failure, and improve over time.

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
Intent → Body Context → Capability Route → Sandbox → Execution
       → Trace → Memory → Intervention → Evolution → Safer Skill
```

> **Every physical action should be grounded, validated, recorded, remembered, and improved.**

Auto may propose changes, but it cannot approve them alone. Sandbox validation, Darwin evaluation, the promotion gate, and human approval together decide whether a change reaches the real world.

---

## First Embodiment / Quick Start

Install the CLI and run the interactive first-boot wizard:

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot
rosclaw doctor
```

Run a local simulation demo without any hardware:

```bash
rosclaw sandbox run --robot sim_ur5e --world tabletop --task reach
```

For headless or CI environments:

```bash
rosclaw firstboot --yes --profile offline --no-telemetry
```

See [QUICKSTART.md](QUICKSTART.md) for four guided paths: local simulation, agent integration, robot body setup, and developer setup.

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

## Install ROSClaw Agent Integration

Paste this setup prompt into Codex, Claude Code, OpenClaw, or another
MCP-aware agent:

> Install and configure ROSClaw for this repository. Run
> `rosclaw agent install --project-root . --skip-secrets`, then read
> `ROSCLAW.md`, `AGENTS.md`, and `.agents/skills/rosclaw/SKILL.md`. Validate
> the setup with `rosclaw agent test universal --project-root . --quick
> --mcp-probe`. After that, use ROSClaw through its CLI and MCP tools for robot
> state, skills, memory, sandbox simulation, practice records, and safety
> checks.

If you are doing the setup yourself, the core install command is:

```bash
rosclaw agent install --project-root . --skip-secrets
```

This installs and configures the agent-facing ROSClaw integration files:
`.mcp.json`, `AGENTS.md`, `ROSCLAW.md`, `CLAUDE.md`,
`.agents/skills/rosclaw/SKILL.md`, and
`.rosclaw/agent/context.snapshot.json`.

It does not install ROSClaw itself and does not install a native plugin for one
specific agent framework. ROSClaw remains an independent CLI, Python package,
MCP server, and robotics infrastructure layer; this command teaches agent
harnesses how to discover and use it.

---

## Persist Practice to SeekDB

Use SQLite for local development or a MySQL-compatible DSN for a real
SeekDB/OceanBase server:

```bash
# Local file
rosclaw practice ingest-seekdb <practice_id> \
  --data-root /data/rosclaw/practice \
  --seekdb-path ~/.rosclaw/memory/seekdb.sqlite

# Real SeekDB server
rosclaw practice ingest-seekdb <practice_id> \
  --data-root /data/rosclaw/practice \
  --seekdb-url mysql://root@127.0.0.1:2881/rosclaw

rosclaw practice query failures \
  --robot-id rh56 \
  --data-root /data/rosclaw/practice \
  --seekdb-url mysql://root@127.0.0.1:2881/rosclaw \
  --json
```

SeekDB port `2881` uses the MySQL-compatible SQL protocol. It is not an HTTP
API. Repeated Practice ingestion is idempotent by episode and evidence IDs.

---

## Safety Model

ROSClaw's core safety rule:

> **No model output should directly control a robot.**

Every physical action passes through a validation pipeline:

1. Provider produces a structured action proposal.
2. Sandbox / Firewall checks it against the effective body model and safety policy.
3. The decision is one of `ALLOW`, `MODIFY`, `BLOCK`, or `REQUIRE_HUMAN_CONFIRMATION`.
4. Execution is recorded by Practice.
5. Memory and Know retain evidence for later audit.
6. How and Auto may propose improvements, but only the promotion gate can change the active skill.

ROSClaw is research infrastructure. It does not replace certified industrial safety systems. Always test in simulation first, keep emergency stops engaged, and use human supervision.

Read [docs/SAFETY.md](docs/SAFETY.md) for the full safety model.

---

## Documentation

- [QUICKSTART.md](QUICKSTART.md) — 5-minute quick start.
- [INSTALL.md](INSTALL.md) — Detailed installation and troubleshooting.
- [docs/FIRSTBOOT.md](docs/FIRSTBOOT.md) — Bootstrap and first boot reference.
- [docs/CLI.md](docs/CLI.md) — CLI command reference.
- [docs/SAFETY.md](docs/SAFETY.md) — Safety model and deployment rules.
- [docs/ASSETS.md](docs/ASSETS.md) — Physical-AI Asset Hub.
- [ARCHITECTURE.md](ARCHITECTURE.md) — Runtime architecture.
- [CONTRIBUTING.md](CONTRIBUTING.md) — Development standards.

---

## Roadmap

| Phase | Focus |
|---|---|
| **Current / v1.0** | Runtime, EventBus, Sandbox, Practice, Memory, How, MCP server, First Boot, Hub validation/search. |
| **In Progress** | Provider routing, skill execution on real bodies, Auto evolution workflow, Darwin evaluation. |
| **Research** | Multi-agent fleet coordination, continuous self-evolution, cross-robot skill transfer. |

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
