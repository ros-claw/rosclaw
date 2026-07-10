# ROSClaw × LeRobot Bridge

The ROSClaw × LeRobot bridge is an **optional** integration that lets ROSClaw
discover, install, diagnose, and call LeRobot capabilities through the ROSClaw
CLI. It is designed so that `rosclaw-core` does not depend on LeRobot, torch, or
HuggingFace packages.

## Important P0.1 principle: runtime isolation

Supported LeRobot 0.6.x releases require **Python >= 3.12**. ROSClaw core
supports Python 3.11+ and must not force your entire robot stack
to Python 3.12. P0.1 therefore treats LeRobot as an **enhanced runtime** that
can run in one of three modes:

| Mode | When to use |
|------|-------------|
| `current-env` | ROSClaw itself is already running on Python 3.12+ and you want LeRobot in the same environment. |
| `isolated` | ROSClaw is on Python 3.11; ROSClaw creates a dedicated Python 3.12 venv at `~/.rosclaw/envs/lerobot`. |
| `external` | You already have a LeRobot Python 3.12 environment; ROSClaw just registers it. |
| `auto` | Default. Picks `current-env` on Python 3.12+ or `isolated` otherwise. |

```text
ROSClaw core does not require Python 3.12.
LeRobot runtime requires Python 3.12+.
When ROSClaw runs on Python 3.11, use an isolated or external LeRobot runtime.
```

## What is implemented

- `rosclaw setup lerobot --profile core --mode auto|current-env|isolated|external`
- `rosclaw lerobot doctor` — reports both the ROSClaw runtime and the LeRobot runtime.
- `rosclaw lerobot info` — runs `lerobot-info` from the configured LeRobot runtime.
- `rosclaw capability list` — shows LeRobot capabilities.
- `rosclaw provider inspect --type lerobot_policy --manifest ... --policy.path <dir>` — read policy config/metadata without loading weights.
- `rosclaw provider load-test --type lerobot_policy --manifest ... --policy.path <dir>` — load policy weights as a runtime smoke test.
- `rosclaw provider infer --type lerobot_policy --manifest ... --input ... --policy.path <dir>` — run one real policy inference via the LeRobot worker.
- `rosclaw provider infer --type lerobot_policy --manifest ... --input ... --dry-run` — returns a sample action with safety metadata.
- `rosclaw lerobot compatibility` — show the P1.1 policy compatibility matrix.
- `rosclaw practice export --format lerobot --episode <dir> --output <dir>` — creates a LeRobotDataset v3 skeleton.

All provider commands that perform real inference still return **action proposals** only: `not_executed=true`, `requires_sandbox=true`, `executable=false`. Every proposal also carries `body_mapping_required=true`, `body_compatible=false`, and `body_name=null` to make it explicit that LeRobot actions are not mapped to a ROSClaw body in P1.1. The provider never executes actions on hardware.

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
  LeRobot version:   0.6.0
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

### Provider inspect

```bash
rosclaw provider inspect \
  --type lerobot_policy \
  --manifest examples/lerobot/sample_policy_manifest_p1.yaml \
  --policy.path tests/fixtures/lerobot_policy_minimal
```

Returns policy metadata (`policy_type`, `input_features`, `output_features`)
without loading model weights. `real_inference=false`.

### Provider load-test

```bash
rosclaw provider load-test \
  --type lerobot_policy \
  --manifest examples/lerobot/sample_policy_manifest_p1.yaml \
  --policy.path /path/to/trained_policy \
  --device cpu
```

Loads the policy weights in the configured LeRobot runtime and reports success.
`real_model_loaded=true`, `real_inference=false`.

### Provider real inference (action proposal only)

```bash
rosclaw provider infer \
  --type lerobot_policy \
  --manifest examples/lerobot/sample_policy_manifest_p1.yaml \
  --policy.path /path/to/trained_policy \
  --input examples/lerobot/sample_observation.json \
  --device cpu
```

Runs one real policy inference through the LeRobot subprocess worker. The
returned result contains a real action under `action_proposal`, but it is always
marked:

```json
{
  "mode": "real_policy_infer",
  "real_inference": true,
  "not_executed": true,
  "requires_sandbox": true,
  "action_proposal": {
    "type": "raw_lerobot_action",
    "values": [...],
    "executable": false,
    "requires_sandbox": true
  }
}
```

The provider rejects `--execute`; it never sends actions to hardware.

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

## P1.1 Real Policy Smoke Gate

After P1, the bridge can inspect, load-test, and infer through the worker, but
those capabilities still need a real policy to prove end-to-end integration.
`rosclaw lerobot smoke-policy` is the dedicated acceptance command.

### Recommended smoke policy

The default smoke policy is the LeRobot ACT policy:

```text
lerobot/act_aloha_sim_transfer_cube_human
```

It is small (~213 MB), official, and uses a simple state + single-camera
observation space:

- `observation.images.top`: `[3, 480, 640]`
- `observation.state`: `[14]`
- `action`: `[14]` (ACT may emit a chunk such as `[100, 14]`)

### Run with allow-network

```bash
rosclaw lerobot smoke-policy \
  --policy.path lerobot/act_aloha_sim_transfer_cube_human \
  --device cpu \
  --allow-network
```

This downloads the policy to `~/.rosclaw/cache/lerobot/policies/` and runs
inspect, load-test, and one inference step.

### Run with a local policy path

```bash
rosclaw lerobot smoke-policy \
  --policy.path /data/rosclaw/policies/act_aloha_sim_transfer_cube_human \
  --device cpu
```

### Smoke report and validation

A successful run writes a v1.1 report to
`~/.rosclaw/lerobot/smoke_reports/<timestamp>_<policy>.json` and updates
`latest.json`. The report schema includes `sample_observation`, `warnings`, a
`validation` block, and summarized action proposals (`preview_values` + shape,
not full tensors).

`rosclaw lerobot doctor` renders the validation state:

```text
Real Policy Smoke Validation
  Status:            validated        # or not_configured / available_not_validated / stale / failed
  Last policy:       lerobot/act_aloha_sim_transfer_cube_human
  Policy type:       act
  LeRobot version:   0.6.x
  Device:            cpu
  Action shape:      [100, 14]
  Time:              2026-07-09T12:34:56.789123Z
  Safety labels:     proposal_only, sandbox_required, body_mapping_required
```

States:

| State | Meaning |
|-------|---------|
| `not_configured` | No smoke report exists. |
| `available_not_validated` | A report exists but the policy did not pass. |
| `validated` | The latest report is `ok` and not stale. |
| `stale` | The report is older than 30 days, or the LeRobot version/Python executable changed. |
| `failed` | The latest smoke run ended with `status=error`. |

If inference, load, or the whole pipeline is slow, the report also includes
performance warnings such as `slow_one_shot_worker`, `slow_policy_load`, or
`slow_smoke_pipeline`. These are informational; the validation can still be
`validated`.

### What `smoke-policy` is not

- It is **not** a benchmark.
- It is **not** a rollout.
- It does **not** control real hardware.
- It does **not** call MCP tools.
- The inference output is always an `action_proposal` with
  `not_executed=true`, `requires_sandbox=true`, `executable=false`,
  `body_mapping_required=true`, `body_compatible=false`.

### Provider import smoke (P0.1 legacy)

P0.1 behavior (`provider infer` without `--policy.path`) is no longer the
default. In P1, a real `infer` requires `--policy.path`. If no runtime is
configured and `policy.path` is missing, the provider returns a structured
failure.

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
    compatibility.py          # Policy compatibility matrix
    config.py                 # Config read/write and v0→v1 migration
    doctor.py                 # Environment diagnostics (dual runtime)
    env_manager.py            # Isolated venv creation
    installer.py              # Setup / dry-run installer (runtime-aware)
    provider.py               # LeRobotPolicyProvider (inspect/load-test/infer)
    runtime.py                # Python/LeRobot runtime discovery
    smoke_policy.py           # Real-policy smoke workflow
    smoke_report.py           # v1.1 report persistence and validation state
    worker_main.py            # LeRobot-runtime-side worker (no rosclaw import)
    worker_runner.py          # ROSClaw-side subprocess runner
    worker_schema.py          # JSON request/response dataclasses
    observation_adapter.py    # ROSClaw input → worker observation dict
    action_adapter.py         # Worker action → ROSClaw action proposal
    policy_manifest.py        # Minimal LeRobot config.json parser
    dataset_exporter.py       # Skeleton exporter
    subprocess_runner.py      # Safe subprocess wrapper
    schemas.py                # Dataclasses and error codes
    feature_mapping.py        # ROSClaw ↔ LeRobot field maps
    profiles.py               # Load bundled profiles.yaml
    profiles.yaml             # Installation profiles
```

## Worker protocol (P1)

ROSClaw core and the LeRobot worker communicate through a one-shot JSON file
protocol. The worker is executed as:

```bash
/path/to/lerobot/python worker_main.py \
  --request-json /tmp/rosclaw_lerobot_worker_xxx/request.json \
  --output-json /tmp/rosclaw_lerobot_worker_xxx/response.json
```

Request envelope (`worker_schema.py`):

```json
{
  "schema_version": "rosclaw.lerobot.worker.v1",
  "op": "inspect|load_test|infer",
  "policy_path": "<local dir or hf repo id>",
  "revision": "main",
  "device": "cpu",
  "dtype": "auto",
  "allow_network": false,
  "timeout_sec": 120,
  "observation": {
    "task": "...",
    "observation.state": [...],
    "observation.images.front": "<image file path>"
  }
}
```

Response envelope:

```json
{
  "schema_version": "rosclaw.lerobot.worker.v1",
  "status": "ok|error",
  "op": "inspect|load_test|infer",
  "policy_path": "...",
  "real_model_loaded": true|false,
  "real_inference": true|false,
  "policy_metadata": {...},
  "action": {
    "type": "raw_lerobot_action",
    "values": [...],
    "shape": [...],
    "dtype": "float32"
  },
  "timing": {"load_time_sec": ..., "infer_time_sec": ...},
  "error": {"code": "...", "message": "...", "details": "..."}
}
```

`worker_main.py` is standalone and does **not** import `rosclaw`. It lazily
imports `torch`, `lerobot`, `numpy`, and `PIL` only inside operation functions.

## Design constraints

- No top-level `import lerobot` or `import torch` in `rosclaw/integrations/lerobot/`.
- All LeRobot imports are guarded with `importlib.util.find_spec` or happen
  inside functions/subprocesses.
- The worker script is executed by the configured LeRobot runtime Python, not
  by the ROSClaw core interpreter.
- Every real-inference result is an action proposal; execution on hardware is
  explicitly blocked.
- The existing real-parquet LeRobot exporter
  (`src/rosclaw/practice/exporters/lerobot_exporter.py`) is unchanged and still
  works.

## Limitations and roadmap

P1 implements real policy loading and one-shot inference, but still does not
execute actions:

- `provider infer` returns real action proposals, always marked
  `not_executed=true`, `requires_sandbox=true`, `executable=false`.
- `rosclaw lerobot smoke-policy` validates a real policy end-to-end, but still
  only produces action proposals.
- The explicit `--episode <directory>` bridge export is metadata-only. The
  existing practice-ID exporter still writes real Parquet data.
- Train, eval, rollout, and reward backends are registered as future
  capabilities only.
- The worker is one-shot; persistent GPU memory management is future work.
- `load_test`, `infer`, and `smoke-policy` require a local policy directory
  with a readable LeRobot config and compatible weights.

Future rounds will add dataset materialization, persistent worker daemons,
benchmark backends, and hardware adapter integration.

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
  lerobot_version: 0.6.0
  torch_version: 2.11.0+cu128
  cuda_available: true
  state: ready
  subprocess_available: true
  in_process_available: false
```
