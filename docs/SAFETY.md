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
rosclawd validates peer/body/snapshot/capability/action-intent permit
    ↓
rosclawd durably records the action and consumes the exact REAL permit
    ↓
ActionGateway acquires resource lease
    ↓
Runtime ActionGateway dispatches a registered executor (if allowed)
    ↓
Driver ACK + physical observation + verification + ExecutionReceipt
    ↓
Practice / Memory / Know consume the receipt asynchronously
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
- The local rosclawd HMAC ledger cannot prevent a root/daemon-state owner from
  forging or jointly rolling back the database and signed head. It is not a TPM,
  monotonic counter, or remote transparency log.

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

For REAL actions, run the Agent and `rosclawd` as different Unix users. The
Agent user must not have serial/CAN device groups, vendor credentials, or ROS 2
command-topic permissions. See [rosclawd Control Plane](ROSCLAWD.md) for the
reference systemd and SROS2 deployment boundary.

Before allowing new REAL work after an unclean daemon restart, inspect
`rosclaw daemon status --json`. If `recovery.required` is true, retain external
controller logs, inspect the robot and interrupted receipt, and keep the
physical E-Stop engaged. Only the daemon service UID may then run
`rosclaw daemon acknowledge-recovery --reason <review-record>`. This records
review but does not clear the E-Stop latch or prove the physical outcome.

---

## Fail-Closed Defaults

- If the effective body model is missing or invalid, skill execution is blocked.
- If the Sandbox cannot reach a decision, it defaults to `BLOCK`.
- If the Skill Registry cannot verify a skill manifest, execution is blocked.
- If the active body changes without confirmation, real-robot execution is paused.
- If rosclawd is unavailable, all Agent-side physical action requests fail.
- Agent-facing physical action requests default to `SHADOW`; `REAL` must be
  explicit and independently authorized by rosclawd.
- Caller-provided `authorization.approved` values never create a REAL permit.
- Official Permit issuance requires the daemon service UID, a healthy durable
  ledger, an armed generation, an active target-UID Session, and a registered
  exact-capability REAL executor.
- A permit never accepts a wildcard Capability and cannot be reused with
  substituted arguments or execution constraints.
- Permit consumption, immutable action IDs, and terminal receipts are persisted
  before or at their safety-relevant transitions. Ledger integrity or write
  failure blocks new actions.
- An interrupted REAL action is never automatically retried. rosclawd requests
  E-Stop, records an unknown outcome, and blocks new REAL work pending review.
- A software E-stop dispatch or driver ACK is not reported as a successful
  physical stop without physical stop observation.

---

## See Also

- [QUICKSTART.md](../QUICKSTART.md) — Quick start.
- [ARCHITECTURE.md](../ARCHITECTURE.md) — Architecture and module boundaries.
- [docs/CLI.md](CLI.md) — CLI command reference.
- [docs/FIRSTBOOT.md](FIRSTBOOT.md) — First boot safety defaults.
