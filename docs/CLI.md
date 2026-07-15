# ROSClaw CLI Reference

This document lists ROSClaw CLI commands and their implementation status.

## Status Labels

- **Stable** — implemented and tested; API is unlikely to change in patch releases.
- **Experimental** — implemented, but the interface may change.
- **Planned** — documented design, not yet implemented.
- **Research** — exploration-stage workflow; may not exist as a CLI command yet.

---

## Lifecycle & Workspace

| Command | Status | Description |
|---------|--------|-------------|
| `rosclaw --version` | Stable | Show version |
| `rosclaw firstboot` | Stable | Interactive first-boot wizard |
| `rosclaw firstboot --yes` | Stable | Non-interactive first boot |
| `rosclaw doctor` | Stable | Health diagnosis |
| `rosclaw doctor --full` | Stable | Complete check |
| `rosclaw doctor --fix` | Stable | Repair safe local issues only |
| `rosclaw status` | Stable | Runtime status |
| `rosclaw config show` | Stable | Show effective config |
| `rosclaw config path` | Stable | Show config file path |
| `rosclaw profile current` | Stable | Show active profile |

---

## Runtime

| Command | Status | Description |
|---------|--------|-------------|
| `rosclaw run` | Stable | Start the ROSClaw runtime |
| `rosclaw start` | Stable | Alias for `run` |
| `rosclaw stop` | Stable | Stop the runtime |
| `rosclaw logs` | Stable | Show runtime logs |

---

## Simulation

| Command | Status | Description |
|---------|--------|-------------|
| `rosclaw sandbox run --robot <id> --world <id> --task <id>` | Stable | Run a MuJoCo simulation |
| `rosclaw robot list` | Stable | List available robots |
| `rosclaw robot inspect <id>` | Stable | Show complete robot profile |
| `rosclaw robot validate <id>` | Stable | Validate e-URDF completeness |

---

## Agent & MCP

| Command | Status | Description |
|---------|--------|-------------|
| `rosclaw agent install` | Stable | Generate cross-agent MCP config, AGENTS.md, ROSCLAW.md, and repo skill |
| `rosclaw agent init claude-code` | Stable | Generate MCP config for Claude Code |
| `rosclaw agent test universal --mcp-probe` | Stable | Test files plus live stdio MCP discovery/envelopes |
| `rosclaw agent doctor universal` | Stable | Validate agent/MCP setup |
| `rosclaw mcp serve` | Stable | Start the ROSClaw MCP server |
| `rosclaw mcp serve --transport http --port 9000` | Stable | HTTP transport |
| `rosclaw mcp install <package>` | Stable | Install a hardware MCP server |
| `rosclaw sandbox verify --case ur5e-joint-preview` | Stable | Run deterministic MuJoCo physics verification |

---

## Body / Embodiment

| Command | Status | Description |
|---------|--------|-------------|
| `rosclaw body init --robot <id>` | Stable | Initialize a body profile |
| `rosclaw body inspect` | Stable | Show effective body model |
| `rosclaw body list` | Stable | List registered bodies |
| `rosclaw body switch <id>` | Stable | Change active body |
| `rosclaw body link-eurdf --body <id> --eurdf <path>` | Stable | Link an e-URDF model |
| `rosclaw body history` | Stable | Show body snapshot history |
| `rosclaw skill check --task <id>` | Stable | Check skill compatibility for the current body |
| `rosclaw body fleet-compat --task <id>` | Stable | Fleet-wide skill compatibility |

---

## Hub

| Command | Status | Description |
|---------|--------|-------------|
| `rosclaw hub validate <manifest.yaml>` | Stable | Validate a local asset manifest |
| `rosclaw hub login --registry <url> --token <token>` | Stable | Log in to a registry |
| `rosclaw hub sync` | Stable | Sync registry metadata |
| `rosclaw hub search <term>` | Stable | Search available assets |
| `rosclaw hub verify <uri>` | Stable | Verify an asset bundle |
| `rosclaw hub publish --dry-run` | Stable | Validate before publishing |
| `rosclaw hub policy check <asset_dir>` | Stable | Check license/permission policy |
| `rosclaw hub install <uri>` | Stable | Install an asset locally |
| `rosclaw hub list --installed` | Stable | List installed assets |
| `rosclaw hub uninstall <uri>` | Stable | Remove an installed asset |
| `rosclaw hub update <uri> <asset_dir>` | Stable | Update an installed asset from a local directory |

---

## Provider

| Command | Status | Description |
|---------|--------|-------------|
| `rosclaw provider list` | Stable | List registered providers |
| `rosclaw provider health [provider_id] --json` | Stable | Show built-in provider health and capability contracts |
| `rosclaw provider route --capability <name> --json` | Stable | Explain capability routing and fallbacks |
| `rosclaw provider benchmark --dry-run --json` | Stable | Validate benchmark wiring without external model calls |
| `rosclaw provider invoke <provider_id> [input] --capability <name>` | Stable | Invoke a configured provider capability |
| `rosclaw provider diagnose --body current --json` | Stable | Diagnose provider interfaces against the active body |

---

## Practice

| Command | Status | Description |
|---------|--------|-------------|
| `rosclaw practice record --fixture <json> --out <dir> --json` | Stable | Record a deterministic fixture through RuntimeBus and PracticeRecorder |
| `rosclaw practice verify <practice_id> --strict --json` | Stable | Verify catalog, event envelopes, and artifact hashes |
| `rosclaw practice distill <practice_id> --json` | Stable | Distill failures, cognition, interventions, candidates, and sim2real deltas |
| `rosclaw practice ingest-seekdb <practice_id> --seekdb-path <file>` | Stable | Ingest into local SQLite-backed SeekDB |
| `rosclaw practice ingest-seekdb <practice_id> --seekdb-url <mysql-dsn>` | Stable | Ingest into a real SeekDB/OceanBase server |
| `rosclaw practice query <mode> [filters]` | Stable | Query episodes and distilled knowledge |
| `rosclaw practice export <practice_id> --format <parquet\|lerobot>` | Stable | Export training/evaluation artifacts |

---

## Memory

| Command | Status | Description |
|---------|--------|-------------|
| `rosclaw memory query <query>` | Stable | Query spatiotemporal memory |
| `rosclaw memory near --robot <id> --time <ts>` | Stable | Find experiences near a robot/time |
| `rosclaw memory graph --episode <id>` | Stable | Show memory subgraph |

---

## How / Know / Auto / Darwin

| Command | Status | Description |
|---------|--------|-------------|
| `rosclaw how explain --task <id>` | Stable | Explain a failure or decision |
| `rosclaw how recover --task <id>` | Stable | Recommend recovery actions |
| `rosclaw how advise --task <id> --failure <id>` | Planned | Advise on a failure |
| `rosclaw how inject --proposal <id>` | Planned | Inject a runtime intervention |
| `rosclaw know compile --episode <id>` | Planned | Compile episode into a TaskCard |
| `rosclaw know search --task <id>` | Planned | Search compiled knowledge |
| `rosclaw auto run --task <id>` | Experimental | Run auto evolution for a task |
| `rosclaw auto status` | Stable | Show evolution status |
| `rosclaw auto champion --patch <id>` | Experimental | Promote a patch to champion |
| `rosclaw auto deadends` | Experimental | List dead-end experiments |
| `rosclaw auto run --suite <suite>` | Planned | Run a named auto-evolution suite |
| `rosclaw darwin run --task-id <id> --skill-id <id>` | Research | Run a multi-seed Darwin benchmark |
| `rosclaw darwin list-scenarios --json` | Stable | List available stress scenarios |
| `rosclaw darwin history --task-id <id> --json` | Stable | Read recorded benchmark history |

---

## Exit Codes

Many ROSClaw commands use the following exit codes:

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid usage |
| `10` | Python >= 3.11 missing |
| `11` | Unsupported platform |
| `20` | Package install failed |
| `21` | PEP 668 externally-managed environment |
| `30` | Workspace permission denied |
| `40` | Existing install conflict |

---

## JSON Output

Commands that support `--json` emit structured output for scripting:

```bash
rosclaw doctor --full --json
rosclaw firstboot --yes --json
rosclaw profile current --json
```

---

## See Also

- [QUICKSTART.md](../QUICKSTART.md) — 5-minute quick start.
- [INSTALL.md](../INSTALL.md) — Installation and troubleshooting.
- [docs/SAFETY.md](SAFETY.md) — Safety model.
- [docs/ASSETS.md](ASSETS.md) — Physical-AI Asset Hub.
