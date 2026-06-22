# ROSClaw Quick Start

This guide gets you running with ROSClaw in about 15 minutes. Choose the path that matches your goal:

- [Path A: Local Simulation Only](#path-a-local-simulation-only) — no robot required
- [Path B: Agent Integration](#path-b-agent-integration) — connect Claude Code or another MCP agent
- [Path C: Robot Body Setup](#path-c-robot-body-setup) — configure a real or simulated robot body
- [Path D: Developer Setup](#path-d-developer-setup) — modify ROSClaw itself

---

## Path A: Local Simulation Only

The fastest way to see ROSClaw in action.

### 1. Install

```bash
curl -sSL https://rosclaw.io/get | bash
```

This installs the `rosclaw` CLI and creates a minimal workspace at `~/.rosclaw`.

### 2. First Boot

```bash
rosclaw firstboot --yes --profile offline --no-telemetry
```

For an interactive setup, run `rosclaw firstboot` without flags.

### 3. Check Health

```bash
rosclaw doctor
```

Expected: all core modules report `HEALTHY` or a clear message if optional dependencies are missing.

### 4. Run a Sandbox Demo

```bash
rosclaw sandbox run --robot sim_ur5e --world tabletop --task reach
```

Expected output: a simulated episode result with status, steps, duration, and artifact URI.

### 5. Open Dashboard

```bash
rosclaw dashboard --open
```

Visit `http://localhost:8765` to view runtime health, traces, and registered skills.

### Common Failures

| Issue | Solution |
|---|---|
| `rosclaw: command not found` | Add `export PATH="$HOME/.rosclaw/bin:$PATH"` to your shell profile |
| `MuJoCo not installed` | Install with `pip install mujoco>=3.0.0` |
| Python version error | Use Python 3.11+ |

### Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for the system design.
- Explore `rosclaw sandbox --help` for more simulation options.
- See [docs/SAFETY.md](docs/SAFETY.md) before running on real hardware.

---

## Path B: Agent Integration

Connect ROSClaw to Claude Code or any MCP-compatible agent.

### 1. Initialize ROSClaw

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot --yes --profile offline --no-telemetry
```

### 2. Initialize Claude Code

```bash
rosclaw agent init claude-code
```

This generates `.mcp.json`, `CLAUDE.md`, `ROSCLAW.md`, and `.claude/settings.json` in the current directory.

### 3. Inspect MCP Config

```bash
rosclaw config path
# ~/.rosclaw/config/rosclaw.yaml
```

The MCP server config is written to `~/.rosclaw/config/mcp.json` by default.

### 4. Validate Agent Tools

```bash
rosclaw agent test claude-code
```

### 5. Start the MCP Server

```bash
rosclaw mcp serve --transport stdio --robot-id sim_ur5e
```

Then point your MCP client at this command.

### Common Failures

| Issue | Solution |
|---|---|
| Agent cannot see tools | Verify `~/.rosclaw/config/mcp.json` is referenced by the agent |
| `rosclaw agent test` fails | Run `rosclaw doctor` and check that the sandbox module is healthy |

### Next Steps

- Read [docs/MCP_USAGE.md](docs/MCP_USAGE.md).
- Try the tabletop grasp demo through the agent interface.
- See [docs/SAFETY.md](docs/SAFETY.md) for agent safety rules.

---

## Path C: Robot Body Setup

Configure a robot body using an e-URDF profile.

### 1. Install and First Boot

```bash
curl -sSL https://rosclaw.io/get | bash
rosclaw firstboot --yes --profile offline --no-telemetry
```

### 2. List Available Robot Profiles

```bash
rosclaw robot list
```

### 3. Initialize a Body

```bash
rosclaw body init --robot unitree-g1
```

### 4. Link an e-URDF Profile

```bash
rosclaw body link-eurdf unitree-g1
```

### 5. Inspect the Body

```bash
rosclaw body inspect --skills --safety
```

### 6. Validate in Sandbox

```bash
rosclaw sandbox run --robot sim_g1 --world tabletop --task stand_balance
```

### Common Failures

| Issue | Solution |
|---|---|
| `Robot profile not found` | Check `e-urdf-zoo/<robot>/robot.eurdf.yaml` exists |
| `No body linked` | Run `rosclaw body link-eurdf <profile>` first |
| Sandbox validation fails | Inspect `rosclaw body inspect --safety` for constraint violations |

### Next Steps

- Read [e-urdf-zoo/](e-urdf-zoo/) for available robot profiles.
- See [docs/BODYSENSE_SCHEMA.md](docs/BODYSENSE_SCHEMA.md) for body sense data.
- Calibrate the body with `rosclaw body calibration update --file calib.yaml`.

---

## Path D: Developer Setup

Work on ROSClaw itself.

### 1. Clone

```bash
git clone https://github.com/ros-claw/rosclaw.git
cd rosclaw
```

### 2. Setup

```bash
make setup
```

This creates a virtual environment, installs dependencies, and bootstraps a local workspace.

### 3. Test

```bash
make test
```

Or run pytest directly:

```bash
PYTHONPATH=src pytest tests -q
```

### 4. First Boot in Dev Mode

```bash
rosclaw firstboot --dev --workspace .rosclaw --profile offline --no-telemetry --yes
rosclaw doctor --full
```

### Common Failures

| Issue | Solution |
|---|---|
| `make setup` fails | Ensure Python 3.11+ and `pip` are available |
| Import errors | Activate the venv created by `make setup` |
| Tests fail in ROS 2 areas | Source `/opt/ros/humble/setup.bash` before running tests |

### Next Steps

- Read [CONTRIBUTING.md](CONTRIBUTING.md).
- Explore `src/rosclaw/` modules.
- Run integration tests with `make integration`.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `rosclaw: command not found` | Add `export PATH="$HOME/.rosclaw/bin:$PATH"` to your shell profile |
| Python version error | Install Python 3.11+ or use `uv` |
| PEP 668 externally-managed environment | The bootstrapper uses a private venv automatically |
| `rosclaw doctor` reports missing modules | Run `pip install -e .` from the repo root |
| MuJoCo rendering fails | `sudo apt install libglfw3 libglew2.2` |

---

## Next Steps

- [Architecture](ARCHITECTURE.md)
- [CLI Reference](docs/CLI.md)
- [Safety Model](docs/SAFETY.md)
- [Physical-AI Assets](docs/ASSETS.md)
- [First Boot Details](docs/FIRSTBOOT.md)
- [Hub](docs/hub/README.md)
