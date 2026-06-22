# ROSClaw Safety Model

ROSClaw is research infrastructure for Physical AI. It adds validation, memory, and governance between AI agents and physical robots, but it does **not** replace certified industrial safety systems.

---

## Core Rule

> **No model output should directly control a robot.**

Every physical action must be:

1. Proposed in a structured, inspectable form.
2. Validated against the effective body model and safety policy.
3. Either allowed, modified, blocked, or escalated to a human.
4. Recorded for audit and learning.

---

## Safety Pipeline

```
Agent Intent
    ↓
Provider produces structured action proposal
    ↓
Sandbox / Firewall validation
    ↓
Decision: ALLOW / MODIFY / BLOCK / REQUIRE_HUMAN_CONFIRMATION
    ↓
Runtime execution (if allowed)
    ↓
Practice captures execution trace
    ↓
Memory and Know retain evidence
    ↓
How / Auto may propose improvements
    ↓
Promotion gate decides whether to update active skills
```

---

## Sandbox Decisions

The Sandbox evaluates every action against:

- The effective body model (joint limits, torque limits, collision geometry).
- The active safety level (`strict`, `moderate`, or `relaxed`).
- The declared skill manifest and capability requirements.
- Historical failure patterns from Memory.

Possible decisions:

| Decision | Meaning |
|----------|---------|
| `ALLOW` | Proceed as proposed. |
| `MODIFY` | Proceed with a bounded correction (e.g., clamped velocity). |
| `BLOCK` | Refuse execution and log the reason. |
| `REQUIRE_HUMAN_CONFIRMATION` | Pause and wait for operator approval. |

---

## Human Approval Gates

The following actions require explicit human confirmation by default:

- Switching from simulation to a real-robot body for the first time.
- Running an unverified skill on hardware.
- Applying a skill patch promoted by Auto before Darwin evaluation.
- Disabling the sandbox or lowering the safety level.
- Executing an emergency-stop override.

---

## What ROSClaw Does Not Guarantee

- ROSClaw cannot prove that a physical action is safe in all possible worlds.
- ROSClaw does not replace functional safety hardware (e.g., physical e-stops, light curtains, force limits).
- ROSClaw does not certify robots or skills for regulatory compliance.
- ROSClaw cannot prevent misuse if the operator disables gates or feeds malicious configuration.

---

## Recommended Deployment Practice

1. **Always start in simulation.** Use `rosclaw sandbox run` before any hardware command.
2. **Keep emergency stops engaged.** Human supervision is required for real-robot operation.
3. **Use `strict` safety level for new bodies.** Relax only after extensive validation.
4. **Run `rosclaw doctor --full` before each session.** Verify workspace, config, and body model.
5. **Review the effective body model.** Run `rosclaw body inspect` and confirm limits.
6. **Test skills in sandbox first.** Use `rosclaw skill check` before hardware dispatch.
7. **Inspect practice records.** Review episodes with `rosclaw practice show` and `rosclaw practice replay`.
8. **Audit promoted skills.** Never promote an Auto patch before Darwin evaluation.
9. **Disable cloud features by default.** Use `--profile offline` unless cloud sync is required.
10. **Report security issues privately.** Email [ai@rosclaw.io](mailto:ai@rosclaw.io).

---

## Fail-Closed Defaults

- If the effective body model is missing or invalid, skill execution is blocked.
- If the Sandbox cannot reach a decision, it defaults to `BLOCK`.
- If the Skill Registry cannot verify a skill manifest, execution is blocked.
- If the active body changes without confirmation, real-robot execution is paused.

---

## See Also

- [QUICKSTART.md](../QUICKSTART.md) — Quick start.
- [ARCHITECTURE.md](../ARCHITECTURE.md) — Architecture and module boundaries.
- [docs/CLI.md](CLI.md) — CLI command reference.
- [docs/FIRSTBOOT.md](FIRSTBOOT.md) — First boot safety defaults.
