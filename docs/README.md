# ROSClaw Documentation

Welcome to the ROSClaw documentation index. This directory contains all project documentation organized by category.

## Quick Navigation

| Category | Documents |
|----------|-----------|
| [Installation & First Boot](#installation--first-boot) | Bootstrap, first boot, verification, troubleshooting |
| [Architecture](#architecture) | Design decisions, reviews, audits |
| [Body / Embodiment](#body--embodiment) | e-URDF, body formats, embodiment testing |
| [Practice](#practice) | Practice recording and SeekDB persistence |
| [API](#api) | API reference, improvements, end-to-end findings |
| [Development](#development) | Collaboration framework, contributing, benchmarks |
| [Security](#security) | Security audits, gap analysis |
| [Planning](#planning) | Roadmaps, sprints, release checklist |
| [Testing](#testing) | Test reports, verification, deep user tests |

---

## Installation & First Boot

- **[FIRSTBOOT.md](FIRSTBOOT.md)** — Complete bootstrap and first boot guide for end users, CI, and developers.

---

## Body / Embodiment

- **[body/EMBODIMENT_FORMAT.md](body/EMBODIMENT_FORMAT.md)** — e-URDF / `body.yaml` / `EMBODIMENT.md` three-layer format.
- **[body/TESTING.md](body/TESTING.md)** — Body subsystem testing guide.
- **[body/MIGRATION.md](body/MIGRATION.md)** — Migration notes for body and embodiment changes.

---

## Practice

- **[practice/SEEKDB_INTEGRATION.md](practice/SEEKDB_INTEGRATION.md)** — Persist practice episodes to SeekDB via `rosclaw_practice`.

---

## Architecture

- **[ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md)** — P0/P1 architecture review findings and recommendations
- **[ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md)** — Full architecture audit report
- **[ROLE_SWAP_REVIEW.md](ROLE_SWAP_REVIEW.md)** — Cross-team architecture role-swap review (score: 7.4/10)
- **[CODE_REVIEW.md](CODE_REVIEW.md)** — Code review findings and action items
- **[GAP_ANALYSIS.md](GAP_ANALYSIS.md)** — Identified gaps between design and implementation

## API

- **[API_REFERENCE.md](API_REFERENCE.md)** — Complete public API reference for ROSClaw v1.0
- **[API_IMPROVEMENTS.md](API_IMPROVEMENTS.md)** — Planned API enhancements for v1.1
- **[E2E_TEST_FINDINGS.md](E2E_TEST_FINDINGS.md)** — End-to-end test results and discovered issues

## Development

- **[COLLABORATION_FRAMEWORK.md](COLLABORATION_FRAMEWORK.md)** — Multi-agent collaboration framework specification
- **[COLLABORATION_LOG.md](COLLABORATION_LOG.md)** — Sprint-by-sprint collaboration log
- **[BENCHMARK.md](BENCHMARK.md)** — Performance benchmarks (EventBus, SeekDB, SkillRegistry, FirewallValidator)
- **[OPENCLAW_INTEGRATION.md](OPENCLAW_INTEGRATION.md)** — OpenClaw integration guide

## Security

- **[SECURITY_AUDIT.md](SECURITY_AUDIT.md)** — Security audit from stress test review
- **[GAP_ANALYSIS.md](GAP_ANALYSIS.md)** — Security and functionality gap analysis

## Planning

- **[ROADMAP_v1.1.md](ROADMAP_v1.1.md)** — v1.1 roadmap and planned features
- **[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)** — Pre-release verification checklist
- **[DESIGN_SPRINT3_5.md](DESIGN_SPRINT3_5.md)** — Sprint 3-5 design documentation
- **[SPRINT3_5_IMPLEMENTATION_REVIEW.md](SPRINT3_5_IMPLEMENTATION_REVIEW.md)** — Sprint 3-5 implementation review

## Testing

- **[DEEP_USER_TEST.md](DEEP_USER_TEST.md)** — Deep user experience test report
- **[FINAL_ACCEPTANCE.md](FINAL_ACCEPTANCE.md)** — Final acceptance criteria and results (9.2/10)
- **[FINAL_VERIFICATION.md](FINAL_VERIFICATION.md)** — Final verification checklist
- **[ROS_INTEGRATION_TESTING.md](ROS_INTEGRATION_TESTING.md)** — Cross-project ROS 1 / ROS 2 integration test matrix

---

## Design Assets

- **[design/](design/)** — Architecture diagrams and visual assets
- **[logos/](logos/)** — ROSClaw branding and logos

---

## Contributing

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for development standards, PR process, and code style guidelines.
