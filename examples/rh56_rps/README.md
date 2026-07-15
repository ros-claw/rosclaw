# Inspire RH56 Hand Gesture Countdown

A ROSClaw skill that demonstrates dexterous-hand gestures on the **Inspire Robots RH56 series** hands over RS485 / Modbus RTU.

It runs a countdown sequence — **five → four → three → two → one → OK** — and ends with a force-feedback-verified OK pose. v2.1 fixes the countdown rhythm, moves thumb and index together into the final OK pose, and promotes a repeatable, safe OK contact policy.

A fun **Rock-Paper-Scissors (RPS)** example is also included: see `examples/rh56_rps/`. The **right hand** counts down **3 → 2 → 1** and reveals its move while MediaPipe reads the human hand over a ROS 2 camera topic. The optional **left hand** can act as a referee: thumb-up when the human wins, pinkie when the human loses, and open palm on a draw.

## What it does

1. Connects to the RH56 hand(s) on `/dev/ttyUSB0` (right, slave id 2) and optionally `/dev/ttyUSB1` (left, slave id 1), 115200 baud, Modbus RTU.
2. Sends raw target-angle commands to 6 actuators / 12 coupled joints per hand.
3. Runs the countdown with configurable hold/pause intervals.
4. Closes into an OK pose from the **one** pose by moving **thumb and index together**.
5. Returns the hand to a safe open palm.

## Supported robots

- `inspire_rh56_right` — Inspire Robots RH56 right dexterous hand.
- `inspire_rh56_left` — Inspire Robots RH56 left dexterous hand.

## Required providers

- `local_rule_planner` for skill parameters (or a direct body-command bridge to the hand).
- Hardware access to the RS485 / USB adapter(s) and `/dev/ttyUSB0` (right) plus `/dev/ttyUSB1` (left, for dual-hand RPS).
- `rosclaw` generic force-feedback types (`PhysicalFeedbackFrame`, `ForceModel`, `ContactEvent`, `BodyCognition`, `PromotionGateResult`) or the local `rosclaw-rh56-runtime` fallback.

## Required sensors

- `robot_state` (joint positions, temperatures, force/current feedback).

## Safety constraints

See `safety.yaml`. Key hand-specific limits:

- Target angles are clamped to **0–1000 raw**.
- Per-joint force limits default to **≤500 g**; use **300 g** for gentle contact.
- Speed **≤500** (~1.6 s full range) to avoid violent motion.
- Pause/dwell **≥1.5 s** for visible, stable poses.
- Stop and cool if any joint temperature approaches its limit or STATUS reports current protection / locked rotor.
- Always start from a known safe pose and keep a software emergency stop available.

## How to run

```bash
# Validate the skill definition
rosclaw skill validate inspire_rh56_hand_gestures

# Run the standalone example directly on real hardware
rosclaw skill eval inspire_rh56_hand_gestures
```

The standalone example is also runnable without the orchestrator:

```bash
cd ~/.rosclaw/skills/inspire_rh56_hand_gestures
PYTHONDONTWRITEBYTECODE=1 ~/.rosclaw-rh56-runtime/.venv/bin/python -B examples/countdown_ok.py
```

### Rock-Paper-Scissors with left-hand referee

Two launchers are provided:

- `run_ui.sh` — legacy demo path (custom `RoundLogger`, no ROSClaw runtime).
- `run_rosclaw_rps.sh` — ROSClaw-native path using EventBus / RuntimeBus,
  `PracticeCoordinator` + `PracticeRecorder`, and `SkillExecutor`.

```bash
cd /home/nvidia/workspace/rosclaw/rosclaw_test/examples/rh56_rps

# Legacy mock mode (no hardware)
./run_ui.sh --mode mock --auto --rounds 5

# ROSClaw-native mock mode (no hardware)
./run_rosclaw_rps.sh --mode mock --auto --rounds 5

# ROSClaw-native hand-test (requires working serial adapters)
./run_rosclaw_rps.sh --mode hand-test

# ROSClaw-native full game with camera and left-hand referee
./run_rosclaw_rps.sh --mode full
```

The left-hand referee is controlled by the `referee:` section in `configs/dual/rps_demo.yaml`.
Set `enabled: true` and `controller: rh56` to enable it; keep it `false` for right-hand-only play.

#### Data flywheel (ROSClaw path)

The ROSClaw-native run writes every game event to:

```
~/.rosclaw/practice/runs/rh56_rps/sessions/<practice_id>/raw/events.jsonl
~/.rosclaw/practice/runs/rh56_rps/indexes/practice_catalog.sqlite
```

To enable the SQLite-backed SeekDB knowledge plane, edit `configs/dual/rps_rosclaw.yaml`:

```yaml
memory:
  backend: sqlite
  db_path: ~/.rosclaw/practice/runs/rh56_rps/seekdb.sqlite
```

## Coordinate and command conventions

| DOF order | ROSClaw name | Meaning |
|-----------|--------------|---------|
| 0 | little | little finger |
| 1 | ring | ring finger |
| 2 | middle | middle finger |
| 3 | index | index finger |
| 4 | thumb | thumb MCP flexion |
| 5 | thumb_rot | thumb opposition / rotation |

- Raw angle space: **0 = closed / bent**, **1000 = open / extended**.
- Speed units: approximate. `speed=1000` ≈ 800 ms full range; `speed=500` ≈ 1.6 s.
- Force units: **grams (g)**.
- Current units: **mA**, but it reads **0 when static**; use force feedback for contact detection.

## Feedback channels

| Register | Meaning | Use case |
|----------|---------|----------|
| `POS_ACT` | current raw angle | verify reached pose |
| `ANGLE_SET` | commanded raw angle | last target |
| `FORCE_ACT` | force in grams | **best contact detector** |
| `CURRENT` | motor current mA | only nonzero during motion |
| `STATUS` | state / fault | bit0=running, bit1=in_position, bit2=current_protection, bit3=force_protection, bit4=temp_protection |
| `TEMP` | actuator temperature | thermal safety |

## OK pose calibration notes (v2.1 promoted policy)

After force-closed-loop self-evolution, the promoted safe OK contact policy is:

```yaml
policy_id: ok_contact_safe_v2
version: 0.3.0
target_pose:
  thumb: 420
  index: 410
  thumb_rot: 300
execution:
  path: thumb_index_together
  speed: 200
  force: 250
  dwell_s: 1.0
  start_from_open: false   # start from the countdown "one" pose
force_window:
  thumb: [80, 180]
  index: [80, 200]
  thumb_rot: [40, 160]
evidence:
  repeatability_rounds: 5
  contact_detected_rate: 1.0
  thumb_mean_force_net_g: 98.2
  index_mean_force_net_g: 150.8
  thumb_rot_mean_force_net_g: 100.2
  ok_shape_score_mean: 4
```

Key v2.1 findings:

- The final OK gesture must move **thumb and index together** starting from the **one** pose (index extended, thumb closed). Moving them together from open palm causes thumb-only or over-contact.
- `thumb=420` (slightly more open than 430) is required for balanced contact when paired with `index=410`.
- Index force decay (~42 g with `thumb_first`) is **path-dependent** and largely disappears with the together path.
- URDF/analytical predictions (650/450/1000) do not match the real contact region (~992/995/991 raw); always use measured body cognition.

## Lessons learned on real hardware

1. **Clear Python bytecode** — stale `.pyc` files can override fresh source edits. Run with `PYTHONDONTWRITEBYTECODE=1 python -B`.
2. **Calibration overrides code** — `~/.rosclaw-rh56/body/calibration.yaml` can overwrite planner targets. Keep both in sync.
3. **Force, not current, detects contact** — current drops to 0 at steady state; force stays elevated while fingers press each other.
4. **Thumb rotation deadband matters** — widened the post-check deadband to ±30 raw units for `thumb_rot`.
5. **Temperature is the real endurance limit** — repeated full-range motions heat the hand; insert cool-downs.
6. **Speed too low looks broken** — `speed=80` took 3.5–4 s to reach the OK pose and triggered timeouts.
7. **Operator rhythm matters** — hold/pause intervals are configurable; the final OK must look synchronized.

## Force-closed-loop v2.1 (self-evolution)

This skill includes a second layer that lets the hand learn its own contact behavior:

```text
open-palm force baseline → OK coordinate/asymmetric search → contact event classification
→ repeatability test → countdown validation → body cognition / sim2real delta
→ promoted safe OK pose
```

Key v2.1 files:

| File | Purpose |
|------|---------|
| `examples/countdown_ok_v2.py` | Countdown using the promoted force-regulated OK pose, with configurable rhythm |
| `examples/force_baseline_calibration.py` | Calibrate open-palm FORCE_ACT baseline |
| `examples/ok_force_contact_search.py` | Safe asymmetric search for OK contact |
| `examples/ok_force_repeatability.py` | Repeatability test from arbitrary start gesture |
| `examples/practice_flywheel_v2.py` | Aggregate practice episodes and update cognition |
| `tests/test_force_model.py` | Unit tests for force baseline / net / thresholds |
| `tests/test_contact_event_detector.py` | Unit tests for contact/stall/protection events |

### v2.1 workflow

```bash
# 1. Calibrate force baseline (30s open palm)
PYTHONDONTWRITEBYTECODE=1 ~/.venv/bin/python -B examples/force_baseline_calibration.py

# 2. Search for a safe OK contact pose
PYTHONDONTWRITEBYTECODE=1 ~/.venv/bin/python -B examples/ok_force_contact_search.py

# 3. Repeatability test
PYTHONDONTWRITEBYTECODE=1 ~/.venv/bin/python -B examples/ok_force_repeatability.py --rounds 5 --start-gesture one

# 4. Countdown validation
PYTHONDONTWRITEBYTECODE=1 ~/.venv/bin/python -B examples/countdown_ok_v2.py --rounds 10 --policy policies/ok_contact_safe_v2.yaml

# 5. Aggregate practice data and update body cognition
PYTHONDONTWRITEBYTECODE=1 ~/.venv/bin/python -B examples/practice_flywheel_v2.py
```

### Contact event taxonomy

The v2.1 detector classifies each frame into a mutually-exclusive primary event:

- `desired_contact` — net force inside the desired window
- `no_contact` — target reached but force too low
- `over_contact` — net force above the hard limit
- `early_contact` — contact before reaching target
- `self_collision` / `motion_blocked` — force while still far from target
- `position_stall` — commanded to move but angle is not changing
- `hardware_protection` — STATUS protection bits or ERROR nonzero
- `temperature_limited` — temperature above warning threshold
- `force_sensor_drift` — unexpected baseline shift
- `thumb_only_contact` / `index_only_contact` — asymmetric OK contact

### Force windows (default)

| Level | Threshold |
|-------|-----------|
| soft contact | ≥ 50 g |
| desired contact | 80–180 g (thumb), 80–200 g (index), 40–160 g (thumb_rot) |
| hard contact | ≥ 250 g |
| emergency contact | ≥ 350 g |

These are starting values; the calibration/search scripts estimate them from real data.

## Example

`examples/countdown_ok.py` contains the original self-contained countdown script.

`examples/countdown_ok_v2.py` uses the promoted force-regulated OK pose, configurable rhythm, and the v2.1 sandbox.

## Evaluation evidence

- Real-robot execution logs from repeated countdown + OK runs are recorded in the project memory.
- Final OK pose verified by elevated `FORCE_ACT` on thumb and index.
- v2.1 search episodes, repeatability tests, and countdown validations are stored in `data/episodes/`.
- See `evidence/reports/` for any formal eval reports generated by `rosclaw skill eval`.

## Version history

### 0.3.0

- Fixed countdown rhythm and made the final OK gesture move thumb and index together.
- Added `thumb_index_together` path and `start_from_open: false` starting from the `one` pose.
- Promoted `ok_contact_safe_v2` policy: `thumb=420, index=410, thumb_rot=300`.
- Added asymmetric OK contact search, force-curve decay analysis, and path-dependence notes.
- Synced examples and tests with the latest `rosclaw` generic force-feedback types.
- Added countdown gate validation (10/10 pass) and operator feedback integration.

### 0.2.0

- Added force-closed-loop v2 layer:
  - `ForceModel` with signed force baselines and net-force thresholds.
  - `ContactEventDetector` for desired/over/early/no contact, stall, and hardware protection.
  - `ReactiveSandboxV2` using `FORCE_ACT` as the primary contact signal.
  - `OKForceContactSearch` safe coordinate-descent script.
  - `BodyCognitionStore` and `ForceInterventionEngine` for memory/how.
  - Practice flywheel v2 with parquet summary.
- Updated skill metadata, safety, and eval specs for v2 metrics.

### 0.1.0

- Initial draft.
- Added validated countdown + OK example.
- Documented RH56 RS485/Modbus conventions, feedback channels, calibration caveats, and safety limits.

## Known limitations

- Promoted to `validated` for `ok_contact_safe_v2`; other gestures remain experimental.
- Force baseline is pose-dependent; recalibrate if the hand is reoriented.
