# First Boot

`rosclaw firstboot` initializes a local Physical-AI runtime workspace. It is the recommended first command after installing the CLI.

---

## Overview

First boot creates the workspace, generates a default configuration, checks dependencies, and optionally sets up providers, a robot body, and an MCP-compatible agent.

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot
```

For CI or headless environments:

```bash
rosclaw firstboot --yes --profile offline --no-telemetry
```

---

## What firstboot creates

```text
~/.rosclaw/
  config/
    rosclaw.yaml
    mcp.json
    profiles/
  secrets/
  bodies/
  providers/
  hub/
  artifacts/
    episodes/
  data/
    memory/
    seekdb/
  logs/
```

Key files:

| Path | Purpose |
|---|---|
| `~/.rosclaw/config/rosclaw.yaml` | Main runtime configuration |
| `~/.rosclaw/config/mcp.json` | MCP server config for agents |
| `~/.rosclaw/secrets/` | API keys and tokens (not synced to cloud by default) |
| `~/.rosclaw/bodies/` | Robot body instances |
| `~/.rosclaw/providers/` | Provider manifests |
| `~/.rosclaw/hub/` | Local asset cache |
| `~/.rosclaw/artifacts/episodes/` | Practice episode traces |

---

## Modes

| Mode | Cloud key required | Description |
|---|---|---|
| `offline` / `local-only` | No | Local runtime, local assets, local traces. Default. |
| `cloud-sync` | Yes | Sync assets and metadata with ROSClaw Cloud. |
| `hybrid` / `team-managed` | Yes | Organization-managed assets and policies. |

In local-only mode, no telemetry is sent and no cloud account is required.

---

## Provider keys

After first boot, configure provider keys as needed:

```bash
rosclaw secrets set OPENAI_API_KEY
rosclaw secrets set ANTHROPIC_API_KEY
rosclaw secrets set QWEN_API_KEY
rosclaw secrets set ROSCLAW_API_KEY
```

Secrets are stored in `~/.rosclaw/secrets/` and are not uploaded unless cloud sync is explicitly enabled.

---

## Agent setup

To connect Claude Code or another MCP-compatible agent:

```bash
rosclaw agent init claude-code
```

This generates:

- `.mcp.json`
- `CLAUDE.md`
- `ROSCLAW.md`
- `.claude/settings.json`
- `.rosclaw/agent/context.snapshot.json`

Point the agent at `~/.rosclaw/config/mcp.json` or the generated `.mcp.json`.

---

## Body setup

To initialize a robot body:

```bash
rosclaw body init --robot unitree-g1
rosclaw body link-eurdf unitree-g1
rosclaw body inspect
```

For a local simulation body:

```bash
rosclaw body init --robot sim_ur5e
rosclaw body link-eurdf ur5e
rosclaw body inspect --safety --skills
```

---

## Validation

After first boot, run the doctor to verify the workspace:

```bash
rosclaw doctor --full
```

Run a sandbox smoke test:

```bash
rosclaw sandbox run --robot sim_ur5e --world tabletop --task reach
```

---

## Non-interactive options

| Flag | Meaning |
|---|---|
| `--yes` | Accept defaults without prompting |
| `--profile offline` | Use offline/local-only mode |
| `--no-telemetry` | Disable anonymous telemetry |
| `--workspace PATH` | Use a custom workspace path |
| `--robot ROBOT` | Pre-select a robot profile |
| `--safety strict` | Set safety level |
| `--enable-sandbox` / `--disable-sandbox` | Control sandbox module |
| `--enable-mcp` / `--disable-mcp` | Control MCP config generation |
| `--dev` | Developer mode |
| `--force` | Re-run even if already initialized |
| `--dry-run` | Show what would be done without writing |

---

## Re-running firstboot

To regenerate the workspace config:

```bash
rosclaw firstboot --force
```

To re-run in dry-run mode:

```bash
rosclaw firstboot --dry-run
```

---

## Cleaning up

To remove the workspace while keeping the CLI:

```bash
rm -rf ~/.rosclaw
```

To uninstall completely:

```bash
rosclaw uninstall --purge
```

---

## Next Steps

- [Quick Start](../QUICKSTART.md)
- [Installation](../INSTALL.md)
- [CLI Reference](CLI.md)
- [Safety Model](SAFETY.md)
