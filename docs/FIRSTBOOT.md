# ROSClaw First Boot Guide

**Author**: ROSClaw Contributors  
**Date**: 2026-06-21  
**Status**: ✅ COMPLETE for v1.0

---

## Overview

ROSClaw v1.0 uses a two-stage installation model:

1. **Bootstrap** — `curl -sSL https://rosclaw.io/get | bash` installs the CLI and creates a minimal workspace.
2. **First Boot** — `rosclaw firstboot` generates your local runtime profile, MCP config, and telemetry preferences.

Both stages are **offline-first, sudo-free, and robot-safe** by default:

- No runtime is started automatically.
- No real robot is connected or commanded.
- No large models are downloaded.
- No data is uploaded to the cloud.
- No telemetry is sent unless you explicitly opt in.

---

## Quick Start

```bash
# 1. Install the CLI
curl -sSL https://rosclaw.io/get | bash

# 2. Run the interactive first-boot wizard
rosclaw firstboot

# 3. Verify the installation
rosclaw doctor
rosclaw status
```

For headless servers or CI:

```bash
rosclaw firstboot --yes --profile offline --no-telemetry
```

---

## Bootstrap Script (`scripts/get.sh`)

The bootstrapper is the only command a new user needs to run.

### What it does

1. Detects the platform (Linux, macOS, Windows WSL).
2. Finds a compatible Python (3.11+).
3. Chooses the best install backend:
   - `uv tool` if `uv` is installed
   - `pipx` if `pipx` is installed
   - Private venv at `~/.rosclaw/venv` otherwise
4. Installs the `rosclaw` package.
5. Creates the workspace skeleton at `ROSCLAW_HOME` (default `~/.rosclaw`).
6. Writes `state/install.json` with install metadata.
7. Runs `rosclaw doctor --bootstrap` if `rosclaw` is on `PATH`.
8. Prints next-step instructions.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROSCLAW_HOME` | `~/.rosclaw` | Workspace path |
| `ROSCLAW_CHANNEL` | `stable` | Package channel: `stable` or `dev` |
| `ROSCLAW_PIP_SPEC` | `rosclaw` | PyPI package or URL spec |
| `ROSCLAW_DRY_RUN` | `0` | Print actions without executing |
| `NO_COLOR` | ` ` | Disable colored output |
| `ROSCLAW_ASSUME_YES` | `0` | Non-interactive bootstrap (reserved) |

### Exit codes and recovery

| Code | Meaning | Recovery |
|------|---------|----------|
| `10` | Python >= 3.11 missing | Install Python 3.11+ or `uv` |
| `11` | Unsupported platform | Use Linux, macOS, or WSL |
| `20` | Package install failed | Check network/proxy; retry |
| `21` | PEP 668 externally-managed env | Bootstrapper auto-falls back to venv |
| `30` | Workspace permission denied | Fix ownership or set `ROSCLAW_HOME` |
| `40` | Existing install conflict | Remove workspace or force reinstall |

---

## First Boot Wizard (`rosclaw firstboot`)

### Interactive mode (default on a TTY)

The wizard asks for:

- Workspace location (`~/.rosclaw` by default)
- Operating profile: `offline`, `cloud`, or `hybrid`
- Use cases: sandbox, ROS 2, MCP, practice, memory, auto evolution, real-robot precheck
- Default robot: `sim_ur5e`, `turtlebot`, `unitree_go2`, `unitree_g1`, or custom
- Safety level: `strict`, `moderate`, or `relaxed`
- MCP config generation
- Anonymous install diagnostics opt-in

Before writing anything, a summary is shown and you must confirm.

### Non-interactive / CI mode

Use `--yes` to skip prompts. On non-TTY devices `--yes` is automatic.

```bash
rosclaw firstboot --yes \
  --profile offline \
  --robot sim_ur5e \
  --safety strict \
  --no-telemetry
```

### CLI options

| Option | Description |
|--------|-------------|
| `--yes` | Non-interactive mode |
| `--workspace PATH` | Custom workspace path |
| `--profile {offline,cloud,hybrid}` | Default operating mode |
| `--robot ID` | Default robot profile |
| `--safety {strict,moderate,relaxed}` | Safety policy level |
| `--enable-sandbox` / `--disable-sandbox` | Toggle sandbox |
| `--enable-mcp` / `--disable-mcp` | Toggle MCP config generation |
| `--enable-ros2` | Enable ROS 2 mode |
| `--enable-memory` | Enable memory module |
| `--enable-practice` | Enable practice capture |
| `--enable-auto` | Enable auto evolution / Darwin |
| `--telemetry` / `--no-telemetry` | Anonymous install diagnostics |
| `--dev` | Developer mode (reserved) |
| `--force` | Re-run even if already initialized |
| `--json` | Structured JSON output |

### Generated files

After first boot, the workspace contains:

```text
~/.rosclaw/
├── config/
│   ├── rosclaw.yaml        # Main runtime profile
│   ├── mcp.json            # MCP server config (if enabled)
│   └── telemetry.yaml      # Telemetry preferences
├── state/
│   └── install.json        # Install metadata + firstboot flags
├── logs/
├── cache/
└── backups/                # Backups of overwritten configs
```

If `rosclaw.yaml` already exists, it is **backed up** and **merged** with the new defaults; existing user keys are preserved.

---

## Profiles

### `offline` (default)

- Cloud sync disabled.
- Telemetry disabled.
- All inference and sandbox runs are local.
- Best for air-gapped robots, labs, and first-time evaluation.

### `cloud`

- Enables cloud endpoints for model serving, telemetry, and sync.
- Requires API keys to be configured separately via environment variables.

### `hybrid`

- Local sandbox and runtime by default.
- Optional cloud features can be enabled later per module.

---

## Verification

```bash
rosclaw --version
rosclaw doctor --bootstrap      # Minimal health check
rosclaw doctor --full           # Complete check
rosclaw doctor --full --json    # Structured report
rosclaw config show
rosclaw profile current
```

### Safe `doctor --fix`

`rosclaw doctor --fix` repairs only safe, local issues:

- Missing workspace directories
- Missing default configs
- Missing or outdated `mcp.json`
- Missing PATH shim
- Small schema migrations

It **never**:

- Uses `sudo`
- Installs system packages
- Connects to a real robot
- Uploads data
- Enables cloud/telemetry without opt-in

---

## Simulation Demo

After first boot, run a local sandbox demo without any hardware:

```bash
rosclaw sandbox run --robot sim_ur5e --world tabletop --task reach
```

List available robots:

```bash
rosclaw robot list
```

---

## Developer Install

To hack on ROSClaw itself:

```bash
git clone https://github.com/ros-claw/rosclaw.git
cd rosclaw
make setup
```

`make setup` creates a local venv, installs the package in editable mode, and runs `rosclaw firstboot` in dev mode.

Run tests:

```bash
PYTHONPATH=src pytest tests -q
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| `rosclaw: command not found` | PATH not updated | Add `export PATH="$HOME/.rosclaw/bin:$PATH"` to shell profile |
| `Python >= 3.11 is required` | Old Python | Install Python 3.11+ or `uv` |
| PEP 668 error | Externally-managed environment | Bootstrapper falls back to private venv automatically |
| Doctor reports missing modules | Dev install incomplete | Run `pip install -e .` from repo root |
| WSL `/mnt/c` warning | Cross-filesystem install | Use default `~/.rosclaw` inside WSL |
| Firstboot refuses to overwrite | Existing workspace | Use `--force` or backup and remove workspace |

---

## Safety Notice

ROSClaw is research infrastructure for physical AI. First Boot is intentionally read-only and simulation-oriented:

- No model output directly controls a robot during installation.
- Real-robot capabilities require explicit opt-in after first boot.
- Always test in simulation before running on hardware.
- Keep emergency stop systems engaged and use human supervision.

**ROSClaw does not replace certified industrial safety systems.**

---

## See Also

- [INSTALL.md](../INSTALL.md) — Detailed installation options.
- [QUICKSTART.md](../QUICKSTART.md) — 5-minute quick start.
- [ARCHITECTURE.md](../ARCHITECTURE.md) — 14 Engineering Iron Rules.
- [docs/ROS_INTEGRATION_TESTING.md](ROS_INTEGRATION_TESTING.md) — Cross-version ROS test notes.
- [CLAUDE.md](../CLAUDE.md) — Project onboarding for Claude Code.
