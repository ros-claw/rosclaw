# ROSClaw × LeRobot Bridge v1.0 — Final Report (P5-E)

Date: 2026-07-17
Branch: `feat/lerobot-bridge-real-hardware`
Body under test: `rh56_left_01` (Inspire RH56 left, FTDI RS485 @115200, Modbus slave 1)

## Definition of Done — 11/11

| # | DoD item | Status | Evidence |
|---|---|---|---|
| 1 | SerialModbusTransport 真机实现通过 | ✅ | exp0 audit: hand_id/voltage/read_state/write+read-back/restore/exception/reopen; golden fixtures; 27 tests |
| 2 | 真机 calibration validated | ✅ | `configs/rh56_left_01_calibration.yaml` (validated, mock=false, 5/5 probe rounds, measured thresholds) |
| 3 | 真实 shadow 1000 steps 通过 | ✅ | S2 gate 10/10 @5Hz OBSERVED, 0 deadline misses, 0 disconnects; practice strict pass |
| 4 | No-op / micro / gesture / OK 达标 | ✅ | no-op 20/20, micro 10/10, motion 10/10, gesture 10/10, OK contact 10/10; protection 0, over-contact 0 |
| 5 | 真机 fault injection 通过 | ✅ | 8/8 scenarios (S1-S7 suite + S8 sysfs USB cycle) |
| 6 | Practice verify --strict 通过 | ✅ | S1/S2 shadow + exp3 formal practices all pass |
| 7 | 真机数据可导出标准 LeRobotDataset | ✅ | S2 shadow practice → 1000 frames/1 episode, observation.state+action, Load OK + Index OK |
| 8 | Clean-room 安装验证通过 | ✅ | `rosclaw setup lerobot --reference-policy rh56` on fresh runtime path: venv + lerobot 0.6.0 + plugin + worker smoke + proposal-only 5/5 + preflight 4/4 |
| 9 | 全量测试回归通过 | ✅ | full pytest suite (see §Regression) |
| 10 | 最终报告和兼容矩阵完成 | ✅ | this file + `COMPATIBILITY_MATRIX.md` |
| 11 | rosclaw-lerobot-bridge-v1.0 tag | ✅ | created at this branch's head |

## What was built (P5-D real-hardware phase)

1. **Real serial transport** — vendored Modbus-RTU frame layer (0x03/0x06/0x10,
   CRC16) + IO discipline from the 7×24 runs (one transaction per port under
   RLock, two-stage reads, stale-ACK drain, pyserial-level reopen).
2. **Real calibration** — measured on the physical hand: identity, force
   baseline 49 g, per-actuator ranges/tolerances, speed ladder to 800,
   thermal curve; thresholds labeled `measured_conservative`.
3. **REAL execution path** — `SingleStepExecutor` REAL+REAL pairing (all other
   pairings fail closed), `RH56RealStepExecutor` registered in the Runtime
   ActionGateway as capability `rh56.single_step`: one ActionEnvelope = one
   single-step command = one PHYSICALLY_OBSERVED receipt, permit-bound to the
   exact artifact hash set.
4. **Worker environment** — isolated py3.12 runtime with lerobot 0.6.0 +
   `lerobot-policy-rosclaw-rh56` plugin, installed by
   `rosclaw setup lerobot --reference-policy rh56` (clean-room proven).

## Real-hardware findings that changed the code (P5-D)

1. **Grouped STATUS/TEMP registers** (3 per block, per actuator pair, on this
   firmware) — reading 6 bleeds TEMP into status; profile declares the width,
   transport expands pair-wise.
2. **STATUS bit semantics** — running(0x01)/in_position(0x02) are healthy
   informational bits; only protection bits (0x04/0x08/0x10) are faults.  The
   old verifier would have revoked every real motion permit.
3. **termios.error EIO escape** — a vanished adapter raises termios.error,
   NOT an OSError; now converted to TransportIOError (found by the USB-unplug
   fault injection: EXECUTOR_ERROR → proper COMMUNICATION_LOST + permit revoked).
4. **Zero-delta rewrite coast dip** — commanding setpoint ≈ current coasts the
   servo one cycle (~15-17 raw on gravity joints); unchanged-setpoint rewrites
   are free.  Fix: 5-raw setpoint hold band in RH56Executor.
5. **Steady-state + tracking lag** — per-actuator tolerances re-measured
   (fingers 12, thumb 40, thumb_rot 40); verifier tolerance = steady-state +
   in-motion lag, documented as measured values.
6. **Left-hand OK contact geometry** — index≈400, thumb_rot≈250, thumb≈210-260,
   ~12 g/raw gradient; two-phase approach; either FORCE_ACT channel may
   register; right-hand pose does not transfer.
7. **FTDI cycle wedge** — ~6+ unbind/bind cycles wedge the adapter firmware
   (descriptor read -110); only physical replug recovers.  Fault-injection
   cycles are now budgeted.

## Regression

- Full pytest suite: **4370 passed / 10 failed / 50 skipped** (18m44s).
  The 10 failures (`tests/storage/test_seekdb_native.py`,
  `tests/test_bench_realsense_*`) are **pre-existing on main** — verified by
  running the identical tests on the base commit 54ad2bf (same failures:
  aarch64 pyseekdb embedded persistence + RealSense camera absent).  This
  branch introduces **zero regressions**; all 149 body/lerobot tests pass.
- ruff + mypy clean on all changed files.
- Two pre-existing ruff F601s in `src/rosclaw/memory/seekdb_client.py`
  (PR-MEM-1 on main, not this branch) — left untouched.

## Boundaries (unchanged from plan)

- CAN RH56 execution: fail-closed stub, not in v1.0.
- Open-loop action chunks: forbidden by construction (single-step only).
- Multiple active sessions: 1.
- Unattended execution: operator-armed permits + estop confirmation required.
- The mock `P5_RH56_REAL_SHADOW_REPORT.md` (misleading "REAL" name) is
  superseded by the OBSERVED shadow report; it never existed in this repo.

## Post-v1.0 scope

Maintenance only: bug fixes, safety fixes, LeRobot 0.6.x compatibility,
RH56 reference regression, docs.  Training/DAgger/Reward/Hub/Data-Flywheel
are separate projects.
