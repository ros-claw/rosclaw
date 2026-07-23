# rh56_reference_policy_v1

Deterministic LeRobot reference policy for the Inspire RH56 dexterous hand
(RS485/Modbus-RTU, 6 actuators, 0-1000 raw device units).

This fixture is owned by and co-located with the
`lerobot_policy_rosclaw_rh56` worker plugin. RosClaw distributions bundle it
for offline smoke tests and contract validation.

This artifact validates the **LeRobot → ROSClaw → RH56** deployment loop
without any training.  It runs inside the real persistent worker, goes through
the real `policy_preprocessor.json` / `policy_postprocessor.json` pipelines,
and emits 6-dim `joint_position` actions with **authoritative** semantics from
`policy_contract.yaml`.

## Files

| File | Role |
|------|------|
| `config.json` | LeRobot config (`type: rosclaw_rh56_reference`, declares `plugin_module`) |
| `model.safetensors` | Placeholder state dict (no trainable weights) |
| `policy_preprocessor.json` | rename → to_batch → device (identity, raw units) |
| `policy_postprocessor.json` | device → cpu (identity, raw units) |
| `policy_contract.yaml` | **Authoritative** action contract (names, representation, unit) |

## Tasks

`hold_current`, `open_hand`, `micro_index_flex`, `half_close`, `return_open`,
`countdown_pose`, `ok_pose_safe` — all receding-horizon single-step, bounded by
`max_step_delta` (default 20 raw units/step).

## Worker plugin

Requires the `lerobot-policy-rosclaw-rh56` plugin installed **in the LeRobot
worker environment only** (never in ROSClaw core):

```bash
.venv-lerobot/bin/python -m pip install -e worker_plugins/lerobot_policy_rosclaw_rh56 --no-deps
```

## Action order (RS485 canonical)

`[little, ring, middle, index, thumb, thumb_rot]` — 0 = closed, 1000 = open.
Do NOT use with the CAN 2.0B interface (11 joints, 0-65535).
