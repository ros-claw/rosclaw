# ROSClaw Sense Module

`rosclaw.sense` gives ROSClaw agents dynamic **proprioception**: the ability to
feel the robot's current body state and decide whether a task can safely run.

## Quick Start

```bash
# No collector configured: show unavailable telemetry, not a healthy mock
rosclaw sense now

# Check if G1 can kick a ball
rosclaw sense readiness --task kick_ball --mock kick_not_ready

# Watch an explicitly synthetic example
rosclaw sense watch --mock hot_knee --interval 1
```

```python
from rosclaw.core.runtime import Runtime, RuntimeConfig

rt = Runtime(RuntimeConfig(
    robot_id="g1_lab_01", enable_sense=True, sense_collector="mock",
))  # Explicit synthetic example; this does not connect a robot.
rt.initialize()
print(rt.sense.get_body_sense().natural_language_summary)
rt.stop()
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `rosclaw sense now` | Current BodySense snapshot |
| `rosclaw sense state` | Detailed raw BodyState |
| `rosclaw sense readiness --task <task>` | Task readiness with failed requirements |
| `rosclaw sense watch` | Live stream of sense updates |
| `rosclaw sense events` | Recent BodyEvents |
| `rosclaw sense explain --task <task>` | Explain why a task is blocked |

All commands accept `--mock <scenario>` to request synthetic data explicitly.
`now`, `state`, `readiness`, and `events` also accept `--json`.

SenseConfig, SenseInterface, RuntimeConfig and the CLI default to an
`unconfigured` collector. It opens no robot connection and leaves telemetry
unknown. Existing callers that need the previous synthetic example must select
`collector="mock"`, `sense_collector="mock"`, or `--mock normal` explicitly.
The packaged `ros2` and `dds` collectors remain stubs; selecting them does not
establish live observation or a configured coverage executor.

BodySense evidence includes `collector_source` and `observation_kind`:
`synthetic`, `historical_replay`, `unavailable`, or `unverified`. These labels
describe the supplied evidence; they do not authorize robot actions. Mock
readiness describes a synthetic scenario. File replay preserves the recorded
timestamp, prefixes the recorded source with `file_replay:`, and records the
separate read time in `raw.replay.replayed_at`. Readiness carries that source
and the input timestamp, so replay does not become current sensor evidence.
Low estimated risk alone cannot promote unknown readiness to ready.
The existing MCP `get_robot_state` rejects unconfigured telemetry and empty
replay as `STATE_PROVENANCE_UNAVAILABLE`, even if transport freshness is good.
A nonempty replay remains `REPLAY` / `RECORDED` and cannot be used for REAL
execution.

## Mock Scenarios

| Scenario | Description |
|----------|-------------|
| `normal` | Nominal body state |
| `hot_knee` | Right knee at 78.2°C |
| `kick_not_ready` | Hot knee + low confidence + unstable support |
| `low_battery` | Battery at 20% |
| `critical_battery` | Battery at 8% |
| `camera_degraded` | Low FPS and target confidence |
| `dds_latency_high` | High communication latency |
| `compute_overload` | CPU/memory overload |

Fresh telemetry establishes the reported values and their original source ages.
The S0 MCP `get_robot_state` projection keeps those values readable, while
reporting `execution_mode: UNKNOWN` and `usable_for_real_execution: false`.
Source names, receive freshness and self-reported binding flags do not establish
a physical evidence domain, an immutable Body binding or action-specific safety.
Nonempty replay retains `REPLAY` / `RECORDED`. Canonical motion admission remains
responsible for the current Body, permissions and each action's required facts;
this telemetry projection does not grant execution authority.

## MCP Tools

When the runtime is running, these MCP tools are exposed:

- `get_body_sense()`
- `get_body_readiness(task)`
- `explain_body_block(task)`

## Event Topics

| Constant | Topic | Description |
|----------|-------|-------------|
| `EventTopics.SENSE_STATE_UPDATED` | `rosclaw.sense.state.updated` | New raw BodyState |
| `EventTopics.SENSE_BODY_UPDATED` | `rosclaw.sense.body.updated` | New BodySense summary |
| `EventTopics.SENSE_EVENT_DETECTED` | `rosclaw.sense.event.detected` | New BodyEvent |
| `EventTopics.SENSE_READINESS_UPDATED` | `rosclaw.sense.readiness.updated` | Readiness changed |
| `EventTopics.SENSE_CAPABILITY_BLOCKED` | `rosclaw.sense.capability.blocked` | Capability blocked |
| `EventTopics.SENSE_CAPABILITY_DEGRADED` | `rosclaw.sense.capability.degraded` | Capability degraded |

## Architecture

```text
Collector -> BodyState
                |
                v
HealthEstimator / RiskEstimator -> BodyRiskSummary + BodyEvents
                |
                v
ReadinessEvaluator -> BodyReadiness
                |
                v
SenseExplainer -> BodySense
```

## Integration with Skill Execution

Skills can declare body-sense requirements in their metadata:

```yaml
name: g1_kick_ball
metadata:
  requires_body_sense:
    battery_percent_min: 40
    max_leg_joint_temp_c: 72
    target_detector_confidence_min: 0.80
```

If the body is not ready, the SkillExecutor returns:

```json
{"status": "blocked", "reason": "blocked_by_body_sense"}
```

## Configuration

RuntimeConfig fields:

| Field | Default | Description |
|-------|---------|-------------|
| `enable_sense` | `True` | Enable SenseRuntime |
| `sense_collector` | `"mock"` | Collector backend |
| `sense_update_hz` | `1.0` | Tick frequency |
| `sense_robot_profile` | `None` | Robot family for thresholds |
| `sense_thresholds_path` | `None` | YAML override path |
| `sense_replay_path` | `None` | JSONL replay path |
