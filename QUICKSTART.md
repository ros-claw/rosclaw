# ROSClaw v1.0 Quick Start

## 1. Install

```bash
curl -sSL https://rosclaw.io/get | bash
```

This installs the `rosclaw` CLI and prepares a minimal workspace at `~/.rosclaw`.

## 2. First Boot

```bash
rosclaw firstboot
```

Follow the interactive wizard to choose your profile, robot, safety level, and MCP setup.

For headless/CI environments:

```bash
rosclaw firstboot --yes --profile offline --no-telemetry
```

## 3. Verify

```bash
rosclaw --version
rosclaw doctor
```

Structured doctor report:

```bash
rosclaw doctor --full --json
```

## 4. List Robots

```bash
rosclaw robot list
```

## 5. Run a Simulation Demo

```bash
rosclaw sandbox run --robot sim_ur5e --world tabletop --task reach
```

## 6. Inspect Configuration

```bash
rosclaw config show
rosclaw config path
rosclaw profile current
```

## 7. Start the MCP Hub (optional)

If you use Claude Code or another MCP-compatible agent, the MCP config was written to:

```text
~/.rosclaw/config/mcp.json
```

Point your agent at this file to expose ROSClaw physical tools.

## Hub Quick Start

Discover, install, and publish physical-AI assets through the ROSClaw Hub.

```bash
# Validate a local asset
rosclaw hub validate tests/fixtures/hub_assets/hardware_mcp_valid/manifest.yaml

# Start a local fake registry (in a separate terminal)
python -m tests.fixtures.fake_registry.server --port 8787

# Login, sync, and search
rosclaw hub login --registry http://localhost:8787 --token fake-valid-token --insecure-local
rosclaw hub sync
rosclaw hub search g1

# Install and uninstall an asset
rosclaw hub install rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0 --yes
rosclaw hub list --installed
rosclaw hub uninstall rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0 --yes
```

See [docs/hub/README.md](docs/hub/README.md) for the full Hub documentation.

## Developer Quick Start

```bash
git clone https://github.com/ros-claw/rosclaw.git
cd rosclaw
make setup
```

## Next Steps

- Read [docs/FIRSTBOOT.md](docs/FIRSTBOOT.md) for the complete bootstrap and first boot reference.
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for the 14 Engineering Iron Rules.
- Explore `rosclaw sandbox --help` for simulation options.
- Check `rosclaw doctor --full` for optional capability gaps.
- Browse [docs/hub/](docs/hub/) for the asset distribution workflow.
- Publish your first asset with `rosclaw hub publish --dry-run`.
