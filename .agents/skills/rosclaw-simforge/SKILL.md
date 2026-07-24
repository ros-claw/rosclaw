---
name: rosclaw-simforge
description: Safely install, validate, diagnose, and optimize ROSClaw simulation workflows across MuJoCo/MJWarp, ROS 2 rosbridge, turtlesim, Gazebo, Isaac Sim, Isaac Lab multi-GPU training, MCP, and the signed ROSClaw Hub. Use for evidence-backed physical-AI smoke tests, 4-GPU validation, simulator integration, Hub upload/download tests, or ROSClaw verification reports without a real robot.
---

# ROSClaw SimForge

Validate the full software loop from CLI and safety gates through simulators,
MCP, evidence receipts, and asset distribution. Treat real hardware as out of
scope unless the user separately authorizes it.

## Safety boundary

- Use the exact `rosclaw` CLI from the checkout's `.venv/bin`; report a stale
  global CLI before changing it.
- Use a temporary `ROSCLAW_HOME` for smoke tests and Hub operations.
- Write raw trajectories, logs, receipts, and reports to an evidence directory
  outside the source checkout. Commit only reproducible code and tests.
- ROS graph discovery, subscription, and simulation are allowed. Never publish
  actuator commands directly, call a motion service, or send an action goal.
- Route any action request through `rosclawd request_action` / ROSClaw MCP so
  the daemon, sandbox, and firewall remain in the path.
- Do not stop unrelated GPU processes or containers. Inspect available memory
  and choose the smallest useful workload.
- A dependency failure is a failure to diagnose and install around, not a
  passing skip. Keep unsupported real-robot checks explicitly out of scope.

## Workflow

### 1. Establish provenance

Read the repository `AGENTS.md` and the task's verification document. Record:

```bash
git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
```

Fetch or compare upstream before claiming the checkout is current. Preserve
all pre-existing user changes.

### 2. Validate the ROSClaw core

Prepend the repository environment to `PATH`, use a temporary home, and run
doctor, status, the deterministic demo, receipt explanation, and the universal
agent/MCP probe. Then run the repository test groups requested by the project
verification document.

Do not claim a simulator pass from import-only evidence. Require a physics
step, a bounded task outcome, and a receipt or other machine-readable result.

### 3. Validate ROS 2 and Gazebo

Run `scripts/verify_ros2.sh` from this skill directory. It builds the Humble
rosbridge/turtlesim image, deploys the stack, discovers the live graph,
subscribes to pose, and proves direct velocity requests are blocked.

For Gazebo, run `scripts/verify_gazebo.sh`. Humble's recommended pairing is
Gazebo Fortress. The script now runs the Phase 3 GuardedBase world with real
diff-drive, odometry, laser, independent ROS bridges, and a deadman under
`launch_testing`. It injects actual process signals and requires bounded stop,
observation-loss fail-closed behavior, cancellation/recovery, and no old-action
replay. For the full MCP → rosclawd path and rosbridge-loss recovery, run:

```bash
rosclaw chaos run gazebo-guarded-base \
  --faults agent-kill,rosbridge-loss,odom-stale,worker-crash \
  --output-dir /an/external/evidence/directory
```

Raw evidence must remain outside the checkout. A `/clock` sample alone is not
enough to claim GuardedBase integration.

### 4. Validate Isaac Lab and four GPUs

Read `references/verified-stack.md`, then run
`scripts/verify_isaaclab.sh`. The script performs bounded single-GPU and
multi-GPU Cartpole training with Newton/MJWarp. Four visible devices must map
to four ranks, synchronize gradients, advance physics, and exit zero.

An Isaac Sim container that advances physics but aborts during shutdown is a
partial pass with a lifecycle defect, not a clean pass.

### 5. Validate G1 GoalForge

Use an external checkout of RoboNaldo's G1 deployment assets and keep all
episode trajectories outside ROSClaw. The backend must qualify the exact
29-joint Unitree `hg` order, ONNX shape, motion tensors, body hash, and prior
hash before physics starts.

Run the product surface from the checkout:

```bash
.venv/bin/python -m rosclaw.entrypoint simforge doctor g1-goalforge --all \
  --output /evidence/doctor/goalforge-doctor.json
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --pairs 100 --output /an/external/recovery-100.json
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile nominal-success --workers 4 \
  --output /an/external/nominal-success-30.json
.venv/bin/python -m rosclaw.entrypoint demo run g1-goalforge \
  --target-zone random --failure-to-success --live-dashboard \
  --output-dir /an/external/new/evidence/directory
.venv/bin/python -m rosclaw.entrypoint practice start \
  --task g1_penalty_kick --generation 3 \
  --output-dir /an/external/new/practice/directory
.venv/bin/python -m rosclaw.entrypoint evolution run \
  --task g1_penalty_kick --generation 10 --gpus 0,1,2,3 \
  --output-dir /an/external/new/evolution/directory
.venv/bin/python -m rosclaw.entrypoint chaos run g1-goalforge \
  --faults agent-kill,worker-crash,dds-loss,state-stale \
  --output-dir /an/external/new/chaos/directory
MUJOCO_GL=egl \
.venv/bin/python -m rosclaw.entrypoint evolution export \
  /an/external/new/evidence/directory \
  --format video \
  --output /an/external/new/video/g1-goalforge.mp4
```

The four-GPU screen is a prioritizer. Require a disjoint, balanced CPU MuJoCo
label-agreement run before accepting its labels, preserve mismatches as
counterexamples, and use CPU MuJoCo strict replay for final physical truth.
Private Holdout case rows must not enter candidate generation or public output.

Build E5 proofs only from passing source reports, then independently replay
their bundle hash and primitive causal/fault/replay fields:

```bash
.venv/bin/python -m rosclaw.entrypoint proof build g1-goalforge \
  --demo /evidence/demo/goalforge-demo.json \
  --recovery /evidence/recovery/recovery-100.json \
  --flywheel /evidence/practice/goalforge-flywheel.json \
  --memory /evidence/evolution/memory-ablation-100.json \
  --four-gpu /evidence/evolution \
  --agreement /evidence/evolution/cpu-gpu-label-agreement.json \
  --continual /evidence/evolution/continual-g0-g10.json \
  --chaos /evidence/chaos/goalforge-chaos.json \
  --output-dir /evidence/proofs
.venv/bin/python -m rosclaw.entrypoint proof replay /evidence/proofs \
  --modules body,provider,failure_router,sandbox,practice,memory,know,how,auto,darwin,registry,rosclawd
.venv/bin/python -m rosclaw.entrypoint promotion evaluate g1-goalforge \
  --doctor /evidence/doctor/goalforge-doctor.json \
  --recovery /evidence/recovery/recovery-100.json \
  --flywheel /evidence/practice/goalforge-flywheel.json \
  --four-gpu /evidence/evolution \
  --continual /evidence/evolution/continual-g0-g10.json \
  --chaos /evidence/chaos/goalforge-chaos.json \
  --proofs /evidence/proofs \
  --output /evidence/promotion-v4.json
```

The Promotion command requires a fresh simulation-only Doctor report. Treat a
Doctor/Champion/Proof Body or kick-prior hash mismatch as a hard failure, and
require all twelve GoalForge module proofs rather than only the four learning
modules.

Unitree DDS tests must use loopback, a non-default isolated domain, canonical
`rt/lowcmd` and `rt/lowstate`, simulation-only permits, and
`hardware_authorized=false`. Never start a physical G1 transport.

GoalForge video export must consume strict-replay trajectory artifacts outside
the checkout and write the MP4 and manifest outside the checkout. It is
visualization-only: never use rendered pixels or subtitles as Promotion truth.

For success-rate optimization, use `--profile nominal-success` as the CPU
MuJoCo acceptance surface. Require all 30 balanced nominal cases, at least 95%
success, a gain of at least 30 percentage points over the fixed prior, 100%
safe selection, 100% independent verification, and 100% strict replay. Keep
the two-attempt runtime recovery budget separate from the at-most-32-candidate
offline simulation search. Do not generalize that result to moving balls,
randomized mass/friction/latency/noise/disturbance, or the 0.90 m target unless
those cases pass a separately declared validation profile.

### 6. Validate MCP

First probe ROSClaw's built-in MCP server and list its tools. For the community
Isaac Sim MCP adapter, isolate installation, start its extension only in a
simulator, perform an MCP initialize/list-tools/call-tool round trip, and state
that it is community-maintained rather than an NVIDIA-official MCP.

Never expose `execute_script` or scene mutation tools to real hardware.

### 7. Validate Hub upload/download

Run `scripts/verify_hub.sh`. It uses the repository fixture key and an isolated
local HTTP registry to exercise validation, signature verification, login,
signed publish, catalog sync/search, remote dry-run download, install, list,
uninstall, and empty final state. Never reuse the fixture key for production.

### 8. Report the evidence ceiling

Separate `PASS`, `PARTIAL`, `FAIL`, and `OUT OF SCOPE`. Include exact commit,
versions, commands, exit codes, test counts, GPU mapping, and artifact paths.
The maximum claim must match the strongest verified evidence domain. Simulation
evidence can promote only from baseline to SIM and never proves real-robot safety.

## Resources

- `scripts/verify_ros2.sh` — live ROS 2/turtlesim safety loop.
- `scripts/verify_gazebo.sh` — Fortress diff-drive, odometry, laser, deadman,
  and real `launch_testing` process faults.
- `scripts/verify_isaaclab.sh` — bounded one- and multi-GPU Isaac Lab loop.
- `scripts/verify_hub.sh` — signed local Hub upload/download loop.
- `references/verified-stack.md` — tested versions, expected evidence, and
  known compatibility limits.
