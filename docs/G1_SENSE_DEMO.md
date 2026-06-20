# ROSClaw Sense — G1 Mock Demo

This demo shows how `rosclaw.sense` reasons about a Unitree G1 robot using
synthetic mock states. It is safe to run anywhere because it uses the
`mock` collector and never talks to real hardware.

## Run the demo

```bash
python examples/sense/g1_mock_demo.py
```

## What it demonstrates

1. **Nominal state** (`normal`) — all evaluated capabilities are ready.
2. **Hot knee** (`hot_knee`) — right knee at 78.2°C.
   - `kick_ball` is blocked because leg joints must stay below 72°C.
   - `walk_slow` remains available.
   - Recommended action: `cooldown`.
3. **Kick not ready** (`kick_not_ready`) — hot knee + unstable support + low
target confidence.
   - Multiple capabilities are blocked or degraded.
   - Operational envelope reports `sandbox_only` and `cooldown_required`.
4. **Operational envelope** — dynamic motion constraints derived from the
   BodySense, including `max_velocity_scale` and `cooldown_required`.

## Example output

```text
=== G1 Mock Sense Demo ===
Scenario: normal
  overall_status: ready
  blocked: []
  degraded: []
  summary: g1_lab_01 is ready for all evaluated capabilities.
  envelope: {'sandbox_only': False, 'cooldown_required': False, 'max_velocity_scale': 1.0, ...}

Scenario: hot_knee
  overall_status: not_ready
  blocked: ['kick_ball']
  degraded: []
  summary: g1_lab_01 status: not_ready (some capabilities blocked).
  envelope: {'sandbox_only': True, 'cooldown_required': True, 'max_velocity_scale': 0.0, ...}

Scenario: kick_not_ready
  overall_status: not_ready
  blocked: ['kick_ball', 'high_power_motion']
  degraded: ['walk_slow']
  summary: g1_lab_01 status: not_ready (some capabilities blocked, others degraded). ...
  envelope: {'sandbox_only': True, 'cooldown_required': True, 'max_velocity_scale': 0.0, ...}
```

## Dashboard integration

When the dashboard is running, BodySense snapshots are published to
`rosclaw.sense.body.updated` and surfaced at:

```text
GET http://localhost:8765/sense
```

The response contains:

```json
{
  "available": true,
  "robot_id": "g1_lab_01",
  "overall_status": "not_ready",
  "blocked_capabilities": ["kick_ball"],
  "degraded_capabilities": [],
  "risk_summary": {...},
  "recommended_actions": ["cooldown", "stabilize", "re_detect_target"],
  "main_reasons": [...]
}
```

## Safety note

This demo uses read-only / simulation tools only. It never commands motion.
Real-motion requests must go through `validate_trajectory` and operator
confirmation per the ROSClaw P0 safety contract.
