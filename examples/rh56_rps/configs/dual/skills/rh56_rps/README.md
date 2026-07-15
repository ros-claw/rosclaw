# RH56 Rock-Paper-Scissors demo

A ROSClaw skill that plays rock-paper-scissors with an **Inspire Robots RH56** dexterous hand. The robot hand counts down **3-2-1**, then reveals rock, paper, or scissors while a RealSense camera reads the human hand over a ROS 2 topic. A left-hand RH56 referee can also show the outcome.

## What it does

1. Connects to one or two RH56 hands over RS485 / Modbus RTU.
2. Streams RealSense color frames at 30 Hz for human gesture recognition.
3. Runs a 3-2-1 countdown with robot-hand gestures.
4. Reveals the robot move and resolves the round.
5. Records every round into the ROSClaw practice flywheel for closed-loop analysis.

## Supported robots

- `inspire_rh56_right` — robot hand that plays the game.
- `inspire_rh56_left` — optional referee hand that shows the result.

## Required sensors

- RealSense D435i / D405 color camera publishing `/camera/camera/color/image_raw`.
- `robot_state` from the RH56 hand (joint positions, force, temperature, status).

## Required providers

- `local_rule_planner` for skill parameters.
- `rh56_serial_provider` / `rh56_state_provider` for hand control and telemetry.
- `realsense_color_provider` for camera frames.

## Safety constraints

See `safety.yaml`. Key limits:

- Target angles clamped to **0–1000 raw**.
- Per-joint force ≤ **500 g** (preferred **300 g**).
- Joint speed ≤ **60%** to keep motion visible and safe.
- Stop if any joint temperature approaches **65 °C** or STATUS reports protection.
- Always start from a known safe pose; keep a software emergency stop available.

## How to run

```bash
cd examples/rh56_rps
./run_rosclaw_rps.sh --mode full --rounds 5
```

Validate the skill package:

```bash
rosclaw skill validate examples/rh56_rps/configs/dual/skills/rh56_rps
```

## Coordinate conventions

| DOF order | Name | Meaning |
|-----------|------|---------|
| 0 | little | little finger |
| 1 | ring | ring finger |
| 2 | middle | middle finger |
| 3 | index | index finger |
| 4 | thumb | thumb MCP flexion |
| 5 | thumb_rot | thumb opposition / rotation |

Raw angle space: **0 = closed / bent**, **1000 = open / extended**.

## Evaluation evidence

- Closed-loop sessions are recorded under `~/.rosclaw/practice/runs/rh56_rps/`.
- `rosclaw practice verify --strict` passes for the latest session.
- Camera stream runs at 30 Hz with p99 frame latency under 50 ms.

## Version history

### 1.0.0

- Initial ROSClaw skill package for the RH56 RPS demo.
- Dual-hand support with left-hand referee.
- Real-time 30 Hz camera + telemetry closed loop.
- Explicit `body_id` and canonical outcome mapping for SeekDB.

## Known limitations

- Camera/inference frame drops can still occur under high CPU load; MediaPipe inference runs in a separate thread.
- The left-hand referee is optional; the demo falls back to UI-only outcome display.
- Serial adapters must be FTDI or CH340 with a stable USB topology; see `safety.yaml` for fault policy.
