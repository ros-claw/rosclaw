# ROSClaw Documentation

Welcome to the ROSClaw documentation index. This directory contains all project documentation organized by category.

## Quick Navigation

| Category | Documents |
|----------|-----------|
| [User Guides](#user-guides) | Getting started, CLI, safety, assets |
| [Architecture](#architecture) | System design, engineering rules, events |
| [Integration](#integration) | ROS, MCP, OpenClaw, robot SDKs |
| [Reference](#reference) | API, event topics, schemas |
| [Specialized Topics](#specialized-topics) | Sense, benchmarks, audits |
| [Contributing](#contributing) | How to contribute |

---

## User Guides

- **[../QUICKSTART.md](../QUICKSTART.md)** — Four-path quick start: simulation, agent, robot body, developer
- **[../INSTALL.md](../INSTALL.md)** — Installation methods, platform notes, troubleshooting
- **[FIRSTBOOT.md](FIRSTBOOT.md)** — What `rosclaw firstboot` creates and how to configure it
- **[CLI.md](CLI.md)** — Complete CLI reference with Stable / Experimental / Planned / Research labels
- **[SAFETY.md](SAFETY.md)** — Safety boundary, hard rules, and deployment checklist
- **[ASSETS.md](ASSETS.md)** — Physical-AI Asset Hub: skills, providers, digital twins, e-URDF, cognitive wikis
- **[hub/README.md](hub/README.md)** — Hub asset discovery, validation, and lifecycle

---

## Architecture

- **[../ARCHITECTURE.md](../ARCHITECTURE.md)** — System architecture, engineering iron rules, execution loops, and deployment modes
- **[EVENT_TOPICS.md](EVENT_TOPICS.md)** — EventBus topic definitions
- **[BODYSENSE_SCHEMA.md](BODYSENSE_SCHEMA.md)** — BodySense schema specification

---

## Integration

- **[MCP_USAGE.md](MCP_USAGE.md)** — MCP (Model Context Protocol) usage guide
- **[OPENCLAW_INTEGRATION.md](OPENCLAW_INTEGRATION.md)** — OpenClaw integration guide
- **[ROS_CONNECTOR.md](ROS_CONNECTOR.md)** — ROS connector documentation
- **[ROS_INTEGRATION_TESTING.md](ROS_INTEGRATION_TESTING.md)** — Cross-project ROS 1 / ROS 2 integration test matrix
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** — General integration guide
- **[unitree_go2_sdk.md](unitree_go2_sdk.md)** — Unitree Go2 SDK integration

---

## Reference

- **[API_REFERENCE.md](API_REFERENCE.md)** — Complete public API reference for ROSClaw v1.0
- **[SENSE.md](SENSE.md)** — Sense system documentation
- **[G1_SENSE_DEMO.md](G1_SENSE_DEMO.md)** — G1 Sense demo documentation

---

## Specialized Topics

- **[BENCHMARK.md](BENCHMARK.md)** — Performance benchmarks (EventBus, SeekDB, SkillRegistry, FirewallValidator)
- **[AUDIT_REPORT_v1.0_POST_RELEASE.md](AUDIT_REPORT_v1.0_POST_RELEASE.md)** — Post-release architecture audit
- **[issues/rosclaw-how-match-gaps.md](issues/rosclaw-how-match-gaps.md)** — Gap analysis issue note

---

## Contributing

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for development standards, PR process, and code style guidelines.
