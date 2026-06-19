# BodySense Schema Reference

This document describes the data schemas used by `rosclaw.sense`.

## Overview

The Sense module produces four main artifacts:

| Artifact | Purpose | Frequency |
|----------|---------|-----------|
| `BodyState` | Raw dynamic state from sensors and telemetry | High (10-1000 Hz from collectors) |
| `BodyRiskSummary` | Normalized per-subsystem risk levels | Per tick |
| `BodyReadiness` | Per-task/capability readiness evaluation | Per tick or on demand |
| `BodySense` | Semantic, human/Agent-readable summary | Low (1 Hz default) |
| `BodyEvent` | Discrete events detected by estimators | On detection |

## BodyState

```python
from rosclaw.sense.schemas import BodyState

state = BodyState(robot_id="g1_lab_01", timestamp=time.time())
```

### Sub-states

| Field | Type | Description |
|-------|------|-------------|
| `energy` | `EnergyState` | Battery, voltage, current, runtime |
| `joints` | `dict[str, JointState]` | Per-joint position, velocity, torque, temperature |
| `imu` | `IMUState` | Orientation and angular velocity |
| `contact` | `dict[str, FootContactState]` | End-effector contact and slip risk |
| `communication` | `CommunicationState` | DDS latency, packet loss, heartbeat |
| `perception` | `PerceptionHealth` | Camera FPS, detector confidence, obstruction |
| `balance` | `BalanceState` | Support margin, CoM projection, stability |
| `compute` | `ComputeState` | CPU/GPU/memory usage and temperature |
| `raw` | `dict` | Collector-specific raw fields |

## BodyRiskSummary

Risk levels per subsystem:

- `unknown` — no data
- `low` — nominal
- `medium` — attention recommended
- `high` — action required
- `critical` — stop or emergency intervention

## BodyReadiness

```python
readiness = runtime.sense.get_readiness(task="g1_kick_ball")
item = readiness.capabilities["g1_kick_ball"]
print(item.status)  # ready | degraded | not_ready
```

Each `ReadinessItem` contains:

- `failed_requirements`: list of `FailedRequirement` with `current` and `required`
- `reasons`: short human-readable strings
- `allowed_alternatives`: safe fallback actions

## BodySense

Primary output consumed by agents. Contains:

- `overall_status`
- `blocked_capabilities`
- `degraded_capabilities`
- `main_reasons`
- `recommended_actions`
- `natural_language_summary`
- `evidence`: numeric snapshot for explanations

## BodyEvent

Discrete event schema with fields:

- `event_id`, `robot_id`, `timestamp`
- `type` (e.g. `joint_hot`, `low_battery`)
- `severity` (`info` | `low` | `medium` | `high` | `critical`)
- `affected_parts`, `measurement`, `thresholds`, `recommended_actions`
