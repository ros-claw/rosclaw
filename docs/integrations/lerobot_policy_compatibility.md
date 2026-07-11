# LeRobot Policy Compatibility Matrix

This document describes how ROSClaw classifies LeRobot policy families for the
P1.1 bridge.  The matrix answers the question: *"I have a policy of type X —
what can the ROSClaw LeRobot bridge do with it today?"*

## Compatibility levels

Levels are ordered from least to most supported:

| Level | Meaning |
|-------|---------|
| `unsupported` | The bridge cannot even inspect this policy family safely. |
| `listed` | The policy family is known but not yet tested. |
| `inspect_ok` | `rosclaw provider inspect` can read config/metadata without loading weights. |
| `load_ok` | `rosclaw provider load-test` can load the policy weights. |
| `infer_ok` | `rosclaw provider infer` can produce a real action proposal. |
| `validated` | A real `rosclaw lerobot smoke-policy` run succeeded for this policy type on this runtime. |

A level is *not* a quality score.  Even `validated` policies still return
**action proposals only** — they are never executed on hardware in P1.1.

## P1.1 matrix

Run `rosclaw lerobot compatibility` to see the live matrix for your runtime:

```bash
rosclaw lerobot compatibility
rosclaw lerobot compatibility --policy-type act
rosclaw lerobot compatibility --json
```

| Policy type | Inspect | Load | Infer | Notes |
|-------------|:-------:|:----:|:-----:|-------|
| `act`       | yes     | yes  | yes   | ALOHA-style ACT policies; action chunks such as `[100, 14]` are supported. |
| `diffusion` | yes     | no   | no    | Pending real smoke; diffusion observation/action preprocessing is not validated in P1.1. |
| `vqbet`     | yes     | no   | no    | Pending real smoke; VQ-BeT loading and inference are not validated in P1.1. |
| `tdmpc`     | yes     | no   | no    | Pending real smoke; TDMPC loading and observation preprocessing are not validated in P1.1. |

Unknown policy families default to `inspect_ok`: the bridge can read their
config and report feature shapes, but loading/inference are not guaranteed.

## How the matrix is used

1. **Provider dispatch**: `LeRobotPolicyProvider` does not reject policies based
   on the matrix.  It attempts the requested operation and reports structured
   errors if the worker fails.

2. **Doctor/validation**: `rosclaw lerobot doctor` shows whether the latest
   smoke report matches the current LeRobot runtime.  If the report is older
   than 30 days, or was created with a different LeRobot version or Python
   executable, the state becomes `stale`.

3. **Compatibility CLI**: `rosclaw lerobot compatibility` cross-checks the
   matrix against the latest smoke report.  A successful smoke report upgrades
   its policy type from `infer_ok` to `validated` for the current runtime.

## Body mapping safety contract

Every entry in the matrix has:

- `body_mapping_required: true`
- `body_compatible: false`

This means a LeRobot action tensor cannot be sent directly to a robot.  A
body-specific mapping layer (future work) must convert LeRobot action indices
into the current effective body's joint/command space before any sandbox or
hardware execution.

## What "validated" does not mean

- It does **not** mean the policy solves a task.
- It does **not** mean the policy is safe for the current robot.
- It does **not** enable hardware execution.
- It only means: *inspect, load-test, and one-step inference all passed through
  the isolated worker and returned a valid action proposal.*

## Extending the matrix

To add a new policy family:

1. Add an entry to `POLICY_COMPATIBILITY_MATRIX` in
   `src/rosclaw/integrations/lerobot/compatibility.py`.
2. Set `inspect`, `load_test`, and `infer` based on what the worker protocol can
   exercise today.
3. Add a test in `tests/integrations/test_lerobot_compatibility.py`.
4. Run `rosclaw lerobot smoke-policy` against a real checkpoint of that family
   and verify the level upgrades to `validated`.

## See also

- [`lerobot_bridge.md`](lerobot_bridge.md) — general bridge usage and worker protocol.
- `rosclaw lerobot doctor` — runtime and validation state.
- `rosclaw lerobot smoke-policy` — how validation reports are produced.
