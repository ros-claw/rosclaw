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
Gazebo Fortress. The script starts the server headlessly, bridges `/clock`,
and requires a ROS 2 clock sample before calling it integrated.

### 4. Validate Isaac Lab and four GPUs

Read `references/verified-stack.md`, then run
`scripts/verify_isaaclab.sh`. The script performs bounded single-GPU and
multi-GPU Cartpole training with Newton/MJWarp. Four visible devices must map
to four ranks, synchronize gradients, advance physics, and exit zero.

An Isaac Sim container that advances physics but aborts during shutdown is a
partial pass with a lifecycle defect, not a clean pass.

### 5. Validate MCP

First probe ROSClaw's built-in MCP server and list its tools. For the community
Isaac Sim MCP adapter, isolate installation, start its extension only in a
simulator, perform an MCP initialize/list-tools/call-tool round trip, and state
that it is community-maintained rather than an NVIDIA-official MCP.

Never expose `execute_script` or scene mutation tools to real hardware.

### 6. Validate Hub upload/download

Run `scripts/verify_hub.sh`. It uses the repository fixture key and an isolated
local HTTP registry to exercise validation, signature verification, login,
signed publish, catalog sync/search, remote dry-run download, install, list,
uninstall, and empty final state. Never reuse the fixture key for production.

### 7. Report the evidence ceiling

Separate `PASS`, `PARTIAL`, `FAIL`, and `OUT OF SCOPE`. Include exact commit,
versions, commands, exit codes, test counts, GPU mapping, and artifact paths.
The maximum claim must match the strongest verified evidence domain; simulation
evidence never proves real-robot safety.

## Resources

- `scripts/verify_ros2.sh` — live ROS 2/turtlesim safety loop.
- `scripts/verify_gazebo.sh` — headless Fortress-to-ROS 2 clock bridge.
- `scripts/verify_isaaclab.sh` — bounded one- and multi-GPU Isaac Lab loop.
- `scripts/verify_hub.sh` — signed local Hub upload/download loop.
- `references/verified-stack.md` — tested versions, expected evidence, and
  known compatibility limits.
