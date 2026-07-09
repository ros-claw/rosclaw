# ROSClaw × LeRobot Bridge

The ROSClaw × LeRobot bridge is an **optional** integration that lets ROSClaw
discover, install, diagnose, and call LeRobot capabilities through the ROSClaw
CLI. It is designed so that `rosclaw-core` does not depend on LeRobot, torch, or
HuggingFace packages.

## What is implemented in Round 1 (P0/P1)

- `rosclaw setup lerobot --profile core` — install or dry-run LeRobot.
- `rosclaw lerobot doctor` — report LeRobot environment status.
- `rosclaw lerobot info` — wrap `lerobot-info`.
- `rosclaw capability list` — show LeRobot capabilities.
- `rosclaw provider infer --type lerobot_policy --manifest ... --input ... --dry-run` —
  return a sample action with safety metadata.
- `rosclaw practice export --format lerobot --episode <dir> --output <dir>` —
  create a LeRobotDataset v3 skeleton.

All commands degrade gracefully when LeRobot is not installed.

## Installation

```bash
# Install rosclaw without LeRobot (core only)
pip install -e .

# Optional: install LeRobot extras
pip install -e ".[lerobot]"

# Or use the CLI setup helper (dry-run)
rosclaw setup lerobot --profile core --dry-run

# Real install
rosclaw setup lerobot --profile core
```

If HuggingFace access is restricted in your region, set the mirror before running
any command:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Usage

### Diagnose

```bash
rosclaw lerobot doctor
rosclaw lerobot doctor --json
```

### LeRobot info

```bash
rosclaw lerobot info
```

### List capabilities

```bash
rosclaw capability list
rosclaw capability list --json
```

### Provider dry-run inference

```bash
rosclaw provider infer \
  --type lerobot_policy \
  --manifest examples/lerobot/sample_policy_manifest.yaml \
  --input examples/lerobot/sample_observation.json \
  --dry-run
```

### Export practice episode to LeRobot skeleton

```bash
rosclaw practice export \
  --format lerobot \
  --episode minimal_episode \
  --output /tmp/rosclaw_lerobot_export \
  --data-root examples/practice
```

## Architecture

```text
src/rosclaw/integrations/
  registry.py              # Dependency-free integration registry
  lerobot/
    __init__.py            # Public API
    capabilities.py        # Capability registration
    cli.py                 # CLI dispatchers
    doctor.py              # Environment diagnostics
    installer.py           # Setup / dry-run installer
    provider.py            # LeRobotPolicyProvider (dry-run)
    dataset_exporter.py    # Skeleton exporter
    subprocess_runner.py   # Safe subprocess wrapper
    schemas.py             # Dataclasses
    feature_mapping.py     # ROSClaw ↔ LeRobot field maps
    profiles.py            # Load bundled profiles.yaml
    profiles.yaml          # Installation profiles
```

## Design constraints

- No top-level `import lerobot` or `import torch` in `rosclaw/integrations/lerobot/`.
- All LeRobot imports are guarded with `importlib.util.find_spec` or happen
  inside functions.
- The existing real-parquet LeRobot exporter
  (`src/rosclaw/practice/exporters/lerobot_exporter.py`) is unchanged and still
  works.

## Limitations and roadmap

Round 1 intentionally stops at the bridge surface:

- Real policy loading and inference are not implemented.
- Real video/parquet dataset writing is not implemented.
- Train, eval, rollout, and reward backends are registered as future
  capabilities only.

Future rounds will add real LeRobot policy inference, dataset materialization,
and hardware adapter integration.
