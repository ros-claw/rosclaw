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
