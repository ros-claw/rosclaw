# ROSClaw Documentation

Welcome to the ROSClaw documentation index. This directory contains all project documentation organized by category.

## Quick Navigation

| Category | Documents |
|----------|-----------|
| [User Guides](#user-guides) | Quick start, install, first boot, CLI, safety, assets |
| [Installation & First Boot](#installation--first-boot) | Bootstrap, first boot, verification, troubleshooting |
| [Architecture](#architecture) | Design decisions, reviews, audits |
| [Body / Embodiment](#body--embodiment) | e-URDF, body formats, registry, routing |
| [Practice](#practice) | Practice recording and SeekDB persistence |
| [API & Integration](#api--integration) | API reference, MCP, ROS, OpenClaw |
| [Development](#development) | Collaboration framework, contributing, benchmarks |
| [Planning](#planning) | Roadmaps, sprints, release checklist |
| [Testing](#testing) | Test reports, verification, ROS integration |

---

## User Guides

- **[QUICKSTART.md](../QUICKSTART.md)** — 5-minute quick start with four paths.
- **[INSTALL.md](../INSTALL.md)** — Detailed installation options and troubleshooting.
- **[FIRSTBOOT.md](FIRSTBOOT.md)** — Bootstrap and first boot reference.
- **[CLI.md](CLI.md)** — CLI command reference with Stable / Experimental / Planned / Research labels.
- **[SAFETY.md](SAFETY.md)** — Safety model, hard rules, and deployment checklist.
- **[ASSETS.md](ASSETS.md)** — Physical-AI Asset Hub and asset lifecycle.
- **[hub/README.md](hub/README.md)** — Hub workflows and registry setup.

---

## Installation & First Boot

- **[FIRSTBOOT.md](FIRSTBOOT.md)** — Complete bootstrap and first boot guide for end users, CI, and developers.

---

## Architecture

- **[ARCHITECTURE.md](../ARCHITECTURE.md)** — Runtime architecture and 14 Engineering Iron Rules.
- **[AUDIT_REPORT_v1.0_POST_RELEASE.md](AUDIT_REPORT_v1.0_POST_RELEASE.md)** — Post-release audit report.

---

## Body / Embodiment

- **[body/EMBODIMENT_FORMAT.md](body/EMBODIMENT_FORMAT.md)** — e-URDF / `body.yaml` / `EMBODIMENT.md` three-layer format.
- **[body/TESTING.md](body/TESTING.md)** — Body subsystem testing guide.
- **[body/MIGRATION.md](body/MIGRATION.md)** — Migration notes for body and embodiment changes.
- **[body/BODY_REGISTRY.md](body/BODY_REGISTRY.md)** — Multi-body registry, `list`/`create`/`switch`/`remove`, and `--body` routing.
- **[body/BODY_HISTORY_EXPORT.md](body/BODY_HISTORY_EXPORT.md)** — Body snapshots, `history`, `export`, and restoration workflow.
- **[body/SKILL_COMPATIBILITY.md](body/SKILL_COMPATIBILITY.md)** — Skill compatibility statuses and enforcement.
- **[body/FLEET_OPERATIONS.md](body/FLEET_OPERATIONS.md)** — Fleet-wide compatibility aggregation and fleet CLI commands.
- **[body/URI_SCHEME.md](body/URI_SCHEME.md)** — Stable `rosclaw://` references to body and e-URDF resources.
- **[BODYSENSE_SCHEMA.md](BODYSENSE_SCHEMA.md)** — Body sense schema reference.

---

## Practice

- **[practice/SEEKDB_INTEGRATION.md](practice/SEEKDB_INTEGRATION.md)** — Persist practice episodes to SeekDB via `rosclaw_practice`.

---

## API & Integration

- **[API_REFERENCE.md](API_REFERENCE.md)** — Complete public API reference for ROSClaw v1.0.
- **[MCP_USAGE.md](MCP_USAGE.md)** — Chinese-language guide to using MCP with ROSClaw.
- **[HARDWARE_MCP_ONBOARDING.md](HARDWARE_MCP_ONBOARDING.md)** — Auto-install, bind, and health-check hardware MCP servers.
- **[P0_AGENT_INTEGRATION.md](P0_AGENT_INTEGRATION.md)** — P0 agent integration guide.
- **[OPENCLAW_INTEGRATION.md](OPENCLAW_INTEGRATION.md)** — OpenClaw integration guide.
- **[ROS_CONNECTOR.md](ROS_CONNECTOR.md)** — ROS connector documentation.
- **[ROS_INTEGRATION_TESTING.md](ROS_INTEGRATION_TESTING.md)** — Cross-project ROS 1 / ROS 2 integration test matrix.
- **[SENSE.md](SENSE.md)** — Sense subsystem documentation.
- **[EVENT_TOPICS.md](EVENT_TOPICS.md)** — Event topic reference.

---

## Development

- **[BENCHMARK.md](BENCHMARK.md)** — Performance benchmarks (EventBus, SeekDB, SkillRegistry, FirewallValidator).
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** — Integration guide.
- **[G1_SENSE_DEMO.md](G1_SENSE_DEMO.md)** — Unitree G1 sense demo.

---

## Planning

- **[../ROSCLAW.md](../ROSCLAW.md)** — Project whitepaper (Chinese).
- **[../CHANGELOG.md](../CHANGELOG.md)** — Release changelog.

---

## Testing

- **[ROS_INTEGRATION_TESTING.md](ROS_INTEGRATION_TESTING.md)** — Cross-project ROS 1 / ROS 2 integration test matrix.

---

## Contributing

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for development standards, PR process, and code style guidelines.

See [../CLAUDE.md](../CLAUDE.md) for Claude Code onboarding notes.
