# ROSClaw × LeRobot Bridge

The ROSClaw × LeRobot bridge is an **optional** integration that lets ROSClaw
discover, install, diagnose, and call LeRobot capabilities through the ROSClaw
CLI. It is designed so that `rosclaw-core` does not depend on LeRobot, torch, or
HuggingFace packages.

## Important P0.1 principle: runtime isolation

LeRobot 0.6.1 requires **Python >= 3.12** and PyTorch >= 2.10. ROSClaw core
supports Python 3.10 / 3.11 / 3.12 and must not force your entire robot stack
to Python 3.12. P0.1 therefore treats LeRobot as an **enhanced runtime** that
can run in one of three modes:

| Mode | When to use |
|------|-------------|
| `current-env` | ROSClaw itself is already running on Python 3.12+ and you want LeRobot in the same environment. |
| `isolated` | ROSClaw is on Python 3.10/3.11; ROSClaw creates a dedicated Python 3.12 venv at `~/.rosclaw/envs/lerobot`. |
| `external` | You already have a LeRobot Python 3.12 environment; ROSClaw just registers it. |
| `auto` | Default. Picks `current-env` on Python 3.12+ or `isolated` otherwise. |

```text
ROSClaw core does not require Python 3.12.
LeRobot runtime requires Python 3.12+.
When ROSClaw runs on Python 3.10/3.11, use isolated or external LeRobot runtime.
```

## What is implemented

- `rosclaw setup lerobot --profile core --mode auto|current-env|isolated|external`
- `rosclaw lerobot doctor` — reports both the ROSClaw runtime and the LeRobot runtime.
- `rosclaw lerobot info` — runs `lerobot-info` from the configured LeRobot runtime.
- `rosclaw capability list` — shows LeRobot capabilities.
- `rosclaw provider infer --type lerobot_policy --manifest ... --input ... --dry-run` — returns a sample action with safety metadata.
- `rosclaw provider infer --type lerobot_policy --manifest ... --input ...` — performs an import smoke test, **no sample action**.
- `rosclaw practice export --format lerobot --episode <dir> --output <dir>` — creates a LeRobotDataset v3 skeleton.

All commands degrade gracefully when LeRobot is not installed.

## Installation

### Recommended for ROSClaw Python 3.11 users

```bash
rosclaw setup lerobot --profile core --mode isolated
```

This will:
1. Find a `python3.12` executable on your system.
2. Create `~/.rosclaw/envs/lerobot`.
3. Install LeRobot into the isolated runtime.
4. Write `~/.rosclaw/integrations/lerobot.yaml` with both runtimes recorded.

### Register an existing LeRobot environment

```bash
rosclaw setup lerobot --profile core --mode external \
  --python /home/user/.venv-lerobot/bin/python
```

### Install into the current Python 3.12 environment

```bash
rosclaw setup lerobot --profile core --mode current-env
```

### Dry-run any setup

```bash
rosclaw setup lerobot --profile core --mode isolated --dry-run
```

If HuggingFace access is restricted in your region, set the mirror before running
any command:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Usage

### Diagnose both runtimes

```bash
rosclaw lerobot doctor
rosclaw lerobot doctor --json
```

Example output when ROSClaw is on Python 3.11 and LeRobot is isolated:

```text
ROSClaw × LeRobot Bridge Doctor

ROSClaw Runtime
  Python executable: /home/user/rosclaw/.venv/bin/python
  Python version:    3.11.15
  In-process LeRobot import: no

LeRobot Runtime
  Mode:              isolated
  Runtime path:      /home/user/.rosclaw/envs/lerobot
  Python executable: /home/user/.rosclaw/envs/lerobot/bin/python
  Python version:    3.12.13
  LeRobot version:   0.6.1
  lerobot-info:      ok
  Torch:             2.11.0+cu128
  CUDA:              available

Bridge Capabilities
  provider_type_lerobot_policy:     enabled
  dataset_export_lerobot:           enabled
  worker_subprocess:                enabled
  worker_in_process:                disabled

Status: OK
```

### LeRobot info

```bash
rosclaw lerobot info
```

This calls `lerobot-info` from the runtime recorded in
`~/.rosclaw/integrations/lerobot.yaml`. If no runtime is configured, it falls
back to `lerobot-info` on `PATH`.

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

Returns a sample action and marks `real_inference=false` and
`not_executed=true`.

### Provider import smoke

```bash
rosclaw provider infer \
  --type lerobot_policy \
  --manifest examples/lerobot/sample_policy_manifest.yaml \
  --input examples/lerobot/sample_observation.json
```

P0.1 does **not** perform real policy inference. It only verifies that the
configured LeRobot runtime is importable and returns `action: null` with
`mode: import_smoke` and `real_inference: false`.

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
  registry.py                 # Dependency-free integration registry
  lerobot/
    __init__.py               # Public API
    capabilities.py           # Capability registration
    cli.py                    # CLI dispatchers
    config.py                 # Config read/write and v0→v1 migration
    doctor.py                 # Environment diagnostics (dual runtime)
    env_manager.py            # Isolated venv creation
    installer.py              # Setup / dry-run installer (runtime-aware)
    provider.py               # LeRobotPolicyProvider (dry-run / import smoke)
    runtime.py                # Python/LeRobot runtime discovery
    dataset_exporter.py       # Skeleton exporter
    subprocess_runner.py      # Safe subprocess wrapper
    schemas.py                # Dataclasses and error codes
    feature_mapping.py        # ROSClaw ↔ LeRobot field maps
    profiles.py               # Load bundled profiles.yaml
    profiles.yaml             # Installation profiles
```

## Design constraints

- No top-level `import lerobot` or `import torch` in `rosclaw/integrations/lerobot/`.
- All LeRobot imports are guarded with `importlib.util.find_spec` or happen
  inside functions/subprocesses.
- The existing real-parquet LeRobot exporter
  (`src/rosclaw/practice/exporters/lerobot_exporter.py`) is unchanged and still
  works.

## Limitations and roadmap

P0.1 intentionally stops at runtime isolation and surface smoke tests:

- Real policy loading and inference are not implemented.
- Real video/parquet dataset writing is not implemented.
- Train, eval, rollout, and reward backends are registered as future
  capabilities only.
- All actions returned by `--dry-run` are sample actions and are marked
  `executable: false` and `sandbox_required: true`.

Future rounds will add real LeRobot policy inference, dataset materialization,
and hardware adapter integration.

## Config file

After setup, `~/.rosclaw/integrations/lerobot.yaml` records both runtimes. A
synthetic example:

```yaml
enabled: true
integration: lerobot
profile: core
install_mode: isolated
rosclaw_runtime:
  python_executable: /home/user/rosclaw/.venv/bin/python
  python_version: 3.11.15
lerobot_runtime:
  runtime_id: default
  mode: isolated
  runtime_path: /home/user/.rosclaw/envs/lerobot
  python_executable: /home/user/.rosclaw/envs/lerobot/bin/python
  python_version: 3.12.13
  lerobot_version: 0.6.1
  torch_version: 2.11.0+cu128
  cuda_available: true
  state: ready
  subprocess_available: true
  in_process_available: false
```
