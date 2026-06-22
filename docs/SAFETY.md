# ROSClaw Safety Model

## Core Rule

> **No model output should directly control a robot.**

ROSClaw is research infrastructure for physical AI and embodied agents. It reduces risk through validation, replay, and runtime guards. It does not guarantee absolute safety and does not replace certified industrial safety systems.

---

## Safety Pipeline

Every physical action must pass through:

```text
Agent Intent
  ↓
Provider Schema
  ↓
Embodiment Constraints
  ↓
Sandbox Validation
  ↓
Runtime Guard
  ↓
Robot Controller
```

Each layer narrows what is allowed:

1. **Provider Schema** — the capability request must match a registered, typed schema.
2. **Embodiment Constraints** — the action must respect e-URDF limits: joint ranges, workspace, velocity, force.
3. **Sandbox Validation** — the action is simulated and checked for collisions, instability, and policy violations.
4. **Runtime Guard** — runtime monitors enforce rate limits, emergency stop, and approval gates.
5. **Robot Controller** — the certified controller receives the final, validated command.

---

## Sandbox Decisions

The Sandbox firewall can return one of four decisions:

| Decision | Meaning |
|---|---|
| `ALLOW` | Action passed validation; may proceed to runtime guard. |
| `MODIFY` | Action passed after adjusting parameters to stay within constraints. |
| `BLOCK` | Action violates safety constraints and must not execute. |
| `REQUIRE_HUMAN_CONFIRMATION` | Action is borderline; requires explicit operator approval. |

Example blocked action:

```json
{
  "decision": "BLOCK",
  "risk_score": 0.92,
  "reason": "Predicted collision between wrist_link and table",
  "violated_constraints": ["collision", "workspace_boundary"],
  "replay_id": "sandbox://replays/firewall_00042"
}
```

---

## Hard Rules

- **VLA outputs are proposals, not raw motor commands.** A vision-language-action model may suggest a target pose or trajectory; the runtime decides whether and how to execute it.
- **World models are previews, not safety proofs.** Neural simulators predict outcomes but cannot certify safety.
- **MCP is an agent tool interface, not a real-time control bus.** Tool calls are request/response, not hard real-time loops.
- **Auto-generated skills must pass sandbox validation before execution.** No generated code runs on a real robot without validation.
- **Code patches require human review before production use.** Auto may propose patches; humans must approve them.
- **Safety configuration changes require explicit approval.** Changes to safety limits, workspace, or firewalls must be reviewed.
- **Every promoted skill must be versioned and rollback-safe.** Champion skills can be reverted to a previous known-good version.

---

## Human Approval Gates

The following actions require explicit human approval in production mode:

- First execution on real hardware
- Safety configuration changes
- Code patches from Auto evolution
- Skill promotion from simulation to real robot
- Installation of unknown or unverified assets
- Capability degradation or enable after fault

---

## What ROSClaw Does Not Guarantee

ROSClaw is designed to **reduce risk** and **increase observability**, not to eliminate all hazards:

- It does not guarantee zero collision.
- It does not guarantee perfect sensor perception.
- It does not replace certified industrial safety systems.
- It does not remove the need for human supervision during real-robot deployment.

Operators remain responsible for emergency stops, workspace boundaries, safety-rated controllers, and risk assessment.

---

## Recommended Deployment Practice

Before running on a real robot:

1. Start in local simulation.
2. Validate in a digital twin that matches the real robot.
3. Use low-speed hardware tests with a human operator present.
4. Enable emergency stop and verify it works.
5. Record all traces with Practice.
6. Promote skills gradually: sim → sandbox → low-speed real → full-speed real.
7. Review Auto-proposed patches before applying them.
8. Keep safety configuration under version control.

---

## Emergency Procedures

- Use `rosclaw ros emergency-stop` to halt motion immediately.
- Maintain a physical emergency stop near the robot.
- Test e-stop behavior in simulation before real deployment.
- Document recovery steps after an e-stop event.

---

## Failure Evidence

When a safety event occurs, ROSClaw captures:

- The original agent intent
- The provider output
- Sandbox decision and risk score
- Embodiment constraints that were checked
- Runtime guard logs
- Sensor data and robot state at the time
- Replay artifact ID

This evidence supports root-cause analysis and skill improvement without requiring the failure to be reproduced.
