# P5-D Graded REAL Execution Report (Exp 3)

- Execution mode: `REAL` (Runtime ActionGateway, capability `rh56.single_step`)
- Trust level: `VERIFIED` (PHYSICALLY_OBSERVED evidence per receipt)
- Body: `rh56_left_01` (Inspire RH56 left, /dev/ttyUSB1 @115200, Modbus slave 1)
- Transport profile: `rh56_left_rs485_v1`
- Practice: `prac_20260717T202040Z_19fec5` (`practice verify --strict` PASS)
- Runner: `scripts/experiments/exp3_graded_execution.py`
- Date: 2026-07-17

## Gate results (DoD §4)

| Level | Trials | Required | Result | Pass |
|---|---|---|---|---|
| no-op (hold open pose) | 20/20 | 20/20 | ✅ | ✅ |
| micro (index ±20 raw) | 10/10 | 10/10 | ✅ | ✅ |
| motion (index ±50 raw) | 10/10 | 10/10 | ✅ | ✅ |
| gesture (multi-finger half-close, non-contact) | 10/10 | 10/10 | ✅ | ✅ |
| OK contact (FORCE_ACT criterion) | 10/10 | ≥9/10 | ✅ | ✅ |
| hardware protection events | 0 | 0 | ✅ | ✅ |
| emergency over-contact | 0 | 0 | ✅ | ✅ |

- Hardware actions executed through the gateway: **1954** (one envelope =
  one single-step command = one feedback-verified receipt).
- Permits revoked during the formal run: **0**.
- Every level issued its own REAL permit (operator-armed, estop-confirmed)
  bound to the exact hash set (policy contract / body / calibration /
  mapping / transport profile).

## Execution chain per step

```
observe (read_state ~75 ms)
  → sandbox preflight (range/calibration/step-delta, same snapshot)
  → ActionEnvelope (authorization approved, scope rh56.single_step,
    body_snapshot_hash = calibration hash,
    verification required evidence PHYSICALLY_OBSERVED, fail-closed)
  → ActionGateway.submit (resource lease, receipt, idempotent)
  → RH56RealStepExecutor → SingleStepExecutor
      (ARMED check → watchdog → permit validate (5 hashes) →
       freshness <= 300 ms → one command → feedback verify)
  → RH56Executor (setpoint hold band, step-delta guard, one 0x10 write,
    read-back)
  → ExecutionReceipt (COMPLETED + PHYSICALLY_OBSERVED)
```

## Real-hardware findings that shaped this run (P5-D)

1. **Grouped STATUS/TEMP registers** — this firmware groups STATUS/TEMP into
   3 registers per block (per actuator pair); reading 6 bleeds TEMP into
   status slots (decoded as false protection bits).  Profile declares
   `status_registers: 3` / `temperature_registers: 3`, expanded pair-wise.
2. **STATUS bit semantics** — 0x01 running / 0x02 in_position are
   informational (healthy); only 0x04/0x08/0x10 (current/force/temp
   protection) are hard faults.  The verifier previously faulted on any
   non-zero status — every real motion would have revoked the permit.
3. **Zero-delta rewrite coast dip** — commanding a setpoint ≈ the current
   position makes the firmware coast one servo cycle (~15-17 raw dip on
   gravity-loaded joints, any FORCE_SET); rewriting an UNCHANGED setpoint
   does not re-plan.  Fix: setpoint hold band 5 raw in RH56Executor —
   near-zero deltas keep the existing setpoint; real motion always re-plans.
4. **Steady-state + tracking lag** — thumb/thumb_rot show up to ~31 raw
   position error under load; calibrated per-actuator tolerances updated
   from measurements at 300/500/700/850/950 (fingers 12, thumb 40,
   thumb_rot 40).  Verifier tolerance = steady-state + in-motion lag.
5. **OK contact geometry (left hand)** — contact region index≈400,
   thumb_rot≈250, thumb≈210-260, gradient ~12 g/raw; EITHER contact channel
   may register the press (f_th 69-463 or f_idx 76-463).  Two-phase
   approach (coarse 10-raw to thumb 400, fine 10-raw steps, contact ≥70 g,
   over-contact abort 250 g).  The right-hand promoted pose does not
   transfer.  During the geometry search (direct transport probes, outside
   the bridge), thumb force peaked at 463 g for <2 s before backing off —
   no protection trip, no damage; the bridge trials never exceed 250 g.
6. **Sandbox/waypoint snapshot discipline** — the sandbox must check the
   step delta against the SAME observation the waypoint was computed from
   (a settling joint moves between reads); waypoints march from the last
   commanded value and stay within the permit's step window of the current
   actual.

## Trial evidence sample (OK contact)

Trial 0 of the formal run: contact latched at FORCE_ACT[thumb] = 100 g
(threshold 70 g, abort 250 g, hard 300 g), retreat to open pose completed,
receipt COMPLETED + PHYSICALLY_OBSERVED.  Full per-step envelopes and
receipts are in practice `prac_20260717T202040Z_19fec5`
(`~/.rosclaw/practice/runs/lerobot_bridge`).
