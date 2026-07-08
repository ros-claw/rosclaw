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
| `rosclaw agent test claude-code --mcp-probe` | Stable | Test files plus live stdio MCP discovery/envelopes |
| `rosclaw agent doctor claude-code` | Stable | Validate agent/MCP setup |
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
| `rosclaw provider invoke <capability> [--input ...]` | Stable | Invoke a provider capability |
| `rosclaw provider init` | Planned | Initialize a provider skeleton |
| `rosclaw provider route --capability <name>` | Planned | Route a capability request |
| `rosclaw provider test <id>` | Planned | Test a provider endpoint |

---

## Practice

| Command | Status | Description |
|---------|--------|-------------|
| `rosclaw practice list` | Stable | List recorded episodes |
| `rosclaw practice show <id>` | Stable | Show episode details |
| `rosclaw practice replay <id>` | Stable | Replay episode trace |
| `rosclaw practice export <id> --format json` | Stable | Export episode metadata |
| `rosclaw practice start --sources <sources>` | Planned | Start a practice recording session |

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
| `rosclaw darwin eval --skill <id>` | Research | Evaluate a skill with Darwin |
| `rosclaw darwin benchmark --skill <id>` | Research | Run multi-seed benchmark |

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
