# ROSClaw v1.0 Installation Guide

## Recommended User Install

The fastest way to get ROSClaw is the official bootstrapper. It installs the CLI, creates a minimal workspace, and runs a bootstrap health check. No robot is moved, no cloud account is created, and no telemetry is sent unless you explicitly opt in later.

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot
```

After first boot, verify the installation:

```bash
rosclaw doctor
rosclaw status
```

Run a local simulation demo:

```bash
rosclaw sandbox run --robot sim_ur5e --world tabletop --task reach
```

### Non-interactive / CI / Server install

```bash
rosclaw firstboot --yes --profile offline --no-telemetry
```

You can also customize the workspace path:

```bash
export ROSCLAW_HOME=/data/rosclaw
rosclaw firstboot --yes --workspace /data/rosclaw
```

### What gets installed?

- `rosclaw` CLI command
- Minimal workspace skeleton at `~/.rosclaw`
- Full runtime profile after `rosclaw firstboot`
- Local MCP config sample at `~/.rosclaw/config/mcp.json`
- Telemetry config at `~/.rosclaw/config/telemetry.yaml` (default disabled)

### What does **not** happen by default?

- No runtime is started automatically
- No real robot is connected or commanded
- No large models are downloaded
- No data is uploaded to the cloud
- No telemetry is sent
- `sudo` is never required

---

## Developer Install from Source

If you want to modify ROSClaw itself, clone the repository and use the developer bootstrap.

```bash
git clone https://github.com/ros-claw/rosclaw.git
cd rosclaw
make setup
```

`make setup` is equivalent to:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install -e ".[dev]"
rosclaw firstboot --dev --workspace .rosclaw --profile offline --no-telemetry --yes
rosclaw doctor --full
```

Run the test suite:

```bash
PYTHONPATH=src pytest tests -q
```

---

## System Requirements

- Python 3.11+
- Linux (x86_64 or ARM64/aarch64) or macOS
- CUDA-capable GPU optional (for GPU-accelerated workloads)

---

## Bootstrapper Options

The bootstrap script respects these environment variables:

| Variable | Description |
|----------|-------------|
| `ROSCLAW_HOME` | Workspace path (default: `~/.rosclaw`) |
| `ROSCLAW_CHANNEL` | `stable` or `dev` (default: `stable`) |
| `ROSCLAW_PIP_SPEC` | PyPI package spec (default: `rosclaw`) |
| `ROSCLAW_DRY_RUN` | Print actions without executing |
| `NO_COLOR` | Disable colored output |

---

## Platform-Specific Notes

### Ubuntu 22.04 / 24.04

If Python 3.11 is not installed:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

The bootstrapper prefers `uv tool` or `pipx` when available, and falls back to a private venv at `~/.rosclaw/venv`. It never uses `--break-system-packages` by default.

### macOS

```bash
brew install python@3.12
```

### NVIDIA Spark (ARM64)

The system may have `python3` pointing to a uv-installed Python 3.11. Use the system Python 3.12 explicitly for source installs:

```bash
/usr/bin/python3 -m pip install -e . --break-system-packages
/usr/bin/python3 -m pytest tests/
```

### Windows WSL

WSL is supported. The bootstrapper detects WSL and warns if you try to install under `/mnt/c/` (slow + permission issues). Use the default `~/.rosclaw` path inside WSL.

---

See [docs/FIRSTBOOT.md](docs/FIRSTBOOT.md) for the complete bootstrap and first boot reference.

## Verify Installation

```bash
rosclaw --version
rosclaw doctor --bootstrap
rosclaw doctor --full --json
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `rosclaw: command not found` | Add `export PATH="$HOME/.rosclaw/bin:$PATH"` to your shell profile |
| Python version error | Install Python 3.11+ or use `uv` |
| PEP 668 externally-managed environment | The bootstrapper will use a private venv automatically |
| `rosclaw doctor` reports missing modules | Run `pip install -e .` from the repo root |
| MuJoCo rendering fails | `sudo apt install libglfw3 libglew2.2` |
