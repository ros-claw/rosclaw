# P5 RH56 Observed Shadow Report

- Execution mode: `SHADOW`
- Trust level: `OBSERVED`
- Verified for real execution: `false`
- Transport profile: `rh56_left_rs485_v1`
- Calibration hash: `sha256:115e008d78ae21f20487c652b6556e263e7774ef13a417778ac5707777ed7aed`
- Policy contract: `rh56_reference_policy_v1`
- Practice id: `prac_20260717T163604Z_d2045a`
- Stop reason: `completed`
- Gate passed: **True**

## 1000-step metrics

| Check | Value | Required | Pass |
|---|---|---|---|
| steps_completed | 1000 | >= 1000 | ✅ |
| hardware_actions_executed | 0 | == 0 | ✅ |
| unknown_action_semantics | 0 | == 0 | ✅ |
| incompatible_mapping | 0 | == 0 | ✅ |
| nan_inf | 0 | == 0 | ✅ |
| required_observation_stale | 0 | == 0 | ✅ |
| serial_disconnect_count | 0 | == 0 | ✅ |
| worker_restart_count | 0 | == 0 | ✅ |
| effective_hz | 5.0 | >= 4.8 | ✅ |
| deadline_miss_rate | 0.0 | < 0.01 | ✅ |

## Latency distribution

```json
{
  "step_latency_ms": {
    "count": 1000,
    "min_ms": 67.839,
    "max_ms": 108.365,
    "mean_ms": 76.333,
    "p95_ms": 83.475
  },
  "inference_latency_ms": {
    "count": 1000,
    "min_ms": 0.451,
    "max_ms": 1.404,
    "mean_ms": 0.767,
    "p95_ms": 0.982
  },
  "mapping_latency_ms": {
    "count": 1000,
    "min_ms": 0.013,
    "max_ms": 0.044,
    "mean_ms": 0.021,
    "p95_ms": 0.029
  },
  "sandbox_latency_ms": {
    "count": 1000,
    "min_ms": 0.01,
    "max_ms": 0.159,
    "mean_ms": 0.016,
    "p95_ms": 0.023
  },
  "deadline_miss_ms": {
    "count": 0,
    "min_ms": 0.0,
    "max_ms": 0.0,
    "mean_ms": 0.0,
    "p95_ms": 0.0
  },
  "effective_control_hz": 5.0
}
```

## Serial health

```json
{
  "read_count": 1000,
  "read_failure_count": 0,
  "disconnect_count": 0,
  "connected": false
}
```

## Provenance (real hardware)

This report supersedes the earlier fixture/mock shadow evidence (whose file
name misleadingly contained "REAL").  All observations above come from the
physical RH56 left hand via the real Modbus-RTU transport:

- Transport: `SerialModbusTransport` (`hardware_serial_modbus_rtu`),
  `/dev/ttyUSB1` @ 115200 8N1, Modbus slave id 1 (HAND_ID read-back = 1).
- Serial read latency over 1000 reads: mean 74.8 ms, p95 82.0 ms, max 82.8 ms;
  0 read failures, 0 disconnects.
- Worker: `.venv-lerobot` (python 3.12.3, lerobot 0.6.0, torch 2.11.0+cu130,
  plugin `lerobot-policy-rosclaw-rh56` 0.1.0), inference mean 0.77 ms.
- S1 smoke (preceding stage): 100 steps @ 2 Hz — joint order
  `[little, ring, middle, index, thumb, thumb_rot]`, positions within raw
  range [0, 1000], force/current/temp/status channels present, 0 deadline
  misses; all non-threshold gate criteria held.
- Practice: S2 session `prac_20260717T163604Z_d2045a` and S1 session
  `prac_20260717T163204Z_2488ca` both pass
  `rosclaw practice verify --strict`.
- Runner: `scripts/experiments/exp2_real_shadow.py` on branch
  `feat/lerobot-bridge-real-hardware`.
