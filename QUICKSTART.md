# ROSClaw Quick Start

Choose the path that matches your goal. Each path starts from the same one-line install and then diverges.

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot
```

---

## Path A: Local Simulation Only

No robot required. Verify the install and run a MuJoCo demo in minutes.

```bash
# 1. Verify
rosclaw --version
rosclaw doctor

# 2. List available robots
rosclaw robot list

# 3. Run a tabletop reach demo
rosclaw sandbox run --robot sim_ur5e --world tabletop --task reach

# 4. Inspect configuration
rosclaw config show
rosclaw profile current
```

Expected: the sandbox prints scene info, runs the task, and exits cleanly.

Common failures:

- `rosclaw: command not found` — add `export PATH="$HOME/.rosclaw/bin:$PATH"` to your shell profile.
- MuJoCo rendering fails on Linux — install `libglfw3` and `libglew2.2`.

Next: read [docs/FIRSTBOOT.md](docs/FIRSTBOOT.md) and [docs/SAFETY.md](docs/SAFETY.md).

---

## Path B: Agent Integration

Connect ROSClaw to Claude Code or any MCP-compatible agent.

```bash
# 1. Generate an MCP config for Claude Code
rosclaw agent init claude-code

# 2. Review the generated config
rosclaw agent doctor

# 3. Start the MCP server manually (optional)
rosclaw mcp serve --transport stdio
```

The MCP config is written to `~/.rosclaw/config/mcp.json`. Point your agent at this file to expose read-only / simulation / emergency tools.

Expected: `rosclaw agent doctor` reports the config path and tool count.

Common failures:

- `mcp.json` missing — re-run `rosclaw firstboot --enable-mcp`.
- Agent cannot connect — check transport type (`stdio`, `http`, or `sse`).

Next: read [docs/hub/README.md](docs/hub/README.md) to install skills for the agent.

---

## Path C: Robot Body Setup

Link a real or custom robot body so the runtime knows its physical limits.

```bash
# 1. Initialize a body profile
rosclaw body init --robot unitree-g1

# 2. Link an e-URDF model
rosclaw body link-eurdf --body unitree-g1 --eurdf e-urdf-zoo/unitree-g1/main.yaml

# 3. Inspect the effective body model
rosclaw body inspect

# 4. Check skill compatibility
rosclaw skill check --task pick_cube
```

Expected: `rosclaw body inspect` shows the compiled body manual, joints, and limits.

Common failures:

- Body not found — run `rosclaw body init` first.
- e-URDF link fails — verify the e-URDF path and schema.

Next: read [docs/body/EMBODIMENT_FORMAT.md](docs/body/EMBODIMENT_FORMAT.md).

---

## Path D: Developer Setup

Clone the repo and run the test suite.

```bash
git clone https://github.com/ros-claw/rosclaw.git
cd rosclaw
make setup
PYTHONPATH=src pytest tests -q
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

Expected: all tests pass and `rosclaw doctor --full` is green.

Common failures:

- Python version error — install Python 3.11+ or use `uv`.
- PEP 668 error — the bootstrapper falls back to a private venv automatically.
- Doctor reports missing modules — run `pip install -e .` from the repo root.

Next: read [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Hub Quick Start

Discover, validate, and publish physical-AI assets through the ROSClaw Hub.

```bash
# Validate a local asset
rosclaw hub validate tests/fixtures/hub_assets/hardware_mcp_valid/manifest.yaml

# Start a local fake registry (in a separate terminal)
python -m tests.fixtures.fake_registry.server --port 8787

# Login, sync, and search
rosclaw hub login --registry http://localhost:8787 --token fake-valid-token --insecure-local
rosclaw hub sync
rosclaw hub search g1

# Install, list, and uninstall an asset
rosclaw hub install rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0 --yes
rosclaw hub list --installed
rosclaw hub uninstall rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0 --yes
```

See [docs/hub/README.md](docs/hub/README.md) for the full Hub documentation and [docs/ASSETS.md](docs/ASSETS.md) for asset definitions.

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| `rosclaw: command not found` | PATH not updated | Add `export PATH="$HOME/.rosclaw/bin:$PATH"` to shell profile |
| `Python >= 3.11 is required` | Old Python | Install Python 3.11+ or `uv` |
| PEP 668 error | Externally-managed environment | Bootstrapper falls back to private venv automatically |
| Doctor reports missing modules | Dev install incomplete | Run `pip install -e .` from repo root |
| MuJoCo rendering fails | Missing graphics libs | `sudo apt install libglfw3 libglew2.2` |
| Firstboot refuses to overwrite | Existing workspace | Use `--force` or backup and remove workspace |

---

## Next Steps

- Read [docs/FIRSTBOOT.md](docs/FIRSTBOOT.md) for the complete bootstrap and first boot reference.
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for the runtime architecture.
- Explore `rosclaw sandbox --help` for simulation options.
- Check `rosclaw doctor --full` for optional capability gaps.
- Browse [docs/hub/](docs/hub/) for the asset distribution workflow.
- Publish your first asset with `rosclaw hub publish --dry-run`.
