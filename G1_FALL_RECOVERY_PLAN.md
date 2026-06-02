# Plan: G1 Gait Fall Recovery Mechanism (P0)

## Context

Edge-case audit revealed a **Critical gap**: G1's `WalkState` has fall detection (`check_fall`) but **zero recovery path**. Once `fall_detected=True`, simulation terminates immediately with no retry, no get-up sequence, and no state reset. This makes long-distance gait demos brittle — any stumble is fatal.

Goal: Add a minimal but robust `get_up_policy` + `reset_fall()` so the robot can recover from falls and resume walking.

---

## Design Decisions

### 1. Recovery State Machine (Simple, Not Over-Engineered)
- **4 phases**: `crouch` → `lift` → `extend` → `stabilize`
- Each phase is a target joint pose + duration
- Interpolation is linear in joint space (reuses existing `unitree_g1.py::reset()` pattern)
- After successful recovery, `reset_fall()` clears flags and resumes main loop
- Max 3 recovery attempts; on 4th fall, terminate (avoid infinite loops)

### 2. Get-Up Keyframes (Free-Floating 12-DOF Model)

Joint order: `[yaw_L, roll_L, pitch_L, knee_L, ankle_pitch_L, ankle_roll_L, yaw_R, roll_R, pitch_R, knee_R, ankle_pitch_R, ankle_roll_R]`

| Phase | Duration | Pose | Rationale |
|-------|----------|------|-----------|
| **crouch** | 0.8s | knees 0.9, hips -0.4, ankles -0.2 | Compact, stable, low COM — hardest to topple |
| **lift** | 1.0s | knees 0.5, hips -0.2, ankles -0.1 | Partial extension, raise COM gradually |
| **extend** | 1.0s | knees 0.05, hips 0.0, ankles 0.0 | Near-standing pose |
| **stabilize** | 0.7s | active balance correction | Re-use `compute_gait_control` with phase=0 for 0.7s |

Total recovery time: ~3.5s (similar to `UnitreeG1.reset()` duration)

### 3. Where to Hook

**File**: `rosclaw-v1.0/src/rosclaw/examples/g1_free_floating_walk.py`

| Location | Change |
|----------|--------|
| `WalkState` dataclass (~line 153) | Add `recovery_phase`, `recovery_attempts`, `max_recovery_attempts` |
| After `check_fall()` (~line 430) | Replace immediate `break` with recovery branch |
| New function `reset_fall()` | Clear `fall_detected`, `fall_reason`, increment `recovery_attempts` |
| New function `get_up_policy()` | Return phase → (target_ctrl, duration) mapping |
| New function `execute_recovery()` | Run keyframe interpolation, monitor pelvis height, return success/fail |

### 4. Success Criteria for Recovery
- Pelvis height > `min_pelvis_height * 1.3` (0.39m) after extend phase
- Tilt angle < `max_tilt_angle_deg * 0.6` (27°) after stabilize phase
- If either check fails, mark recovery as failed and count toward max attempts

### 5. Test Plan

**File**: `rosclaw-v1.0/tests/test_g1_edge_cases.py`

Add tests:
- `test_reset_fall_clears_state`: verify `reset_fall()` resets flags and increments counter
- `test_get_up_sequence_recovers_height`: force fallen state, run recovery, assert height recovered
- `test_max_recovery_attempts`: simulate repeated falls, assert termination after 3 attempts
- `test_recovery_resumes_walking`: single fall → recovery → robot reaches target

### 6. Verification

```bash
cd /home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0
PYTHONPATH=src python3 -m pytest tests/test_g1_free_floating.py tests/test_g1_edge_cases.py -v
```

All existing tests must still pass; new recovery tests must pass.

---

## Critical Files to Modify

1. `/home/ubuntu/rosclaw/rosclaw-v1.0/src/rosclaw/examples/g1_free_floating_walk.py` — core recovery logic
2. `/home/ubuntu/rosclaw/rosclaw-v1.0/tests/test_g1_edge_cases.py` — new test cases

## Reuse Patterns

- `UnitreeG1.reset()` interpolation pattern (lerp over time steps)
- `compute_gait_control()` for stabilize phase active balance
- Existing `check_fall()` for recovery validation
