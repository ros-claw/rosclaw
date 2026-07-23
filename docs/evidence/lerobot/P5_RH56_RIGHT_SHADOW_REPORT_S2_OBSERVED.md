# P5 RH56 Observed Shadow Report

- Execution mode: `SHADOW`
- Trust level: `OBSERVED`
- Verified for real execution: `false`
- Transport profile: `rh56_right_rs485_v1`
- Calibration hash: `sha256:951f480584f4afbeda0a7628ffd4ebd61d85c6031de01fecf57bf1f3c6e8301f`
- Policy contract: `rh56_reference_policy_v1`
- Practice id: `prac_20260723T083806Z_0eea70`
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
    "min_ms": 68.008,
    "max_ms": 85.109,
    "mean_ms": 76.719,
    "p95_ms": 83.853
  },
  "inference_latency_ms": {
    "count": 1000,
    "min_ms": 0.468,
    "max_ms": 1.611,
    "mean_ms": 0.822,
    "p95_ms": 1.079
  },
  "mapping_latency_ms": {
    "count": 1000,
    "min_ms": 0.011,
    "max_ms": 0.047,
    "mean_ms": 0.022,
    "p95_ms": 0.027
  },
  "sandbox_latency_ms": {
    "count": 1000,
    "min_ms": 0.007,
    "max_ms": 0.147,
    "mean_ms": 0.016,
    "p95_ms": 0.02
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
physical RH56 hand via the real Modbus-RTU transport:

- Transport: `SerialModbusTransport` (`hardware_serial_modbus_rtu`),
  `/dev/ttyUSB1` @ 115200 8N1, Modbus slave id 1 (HAND_ID read-back = 1).
- Serial read latency over 1000 reads: mean 75.1 ms,
  p95 82.21 ms, max 83.34 ms.
- Worker: `.venv-lerobot` (python 3.12, lerobot 0.6.x, plugin
  `lerobot-policy-rosclaw-rh56`).
- Runner: `scripts/experiments/exp2_real_shadow.py`.
