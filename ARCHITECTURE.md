# ROSClaw 1.0 Architecture

## Mission

**Trustworthy Physical Execution Runtime and Control Plane for Embodied Agents**

Ground Agent intent into a body, authorize and execute through a daemon-owned
control plane, verify physical outcomes, and return auditable receipts.

## Engineering Identity

```text
ROSClaw 1.0 =
  rosclawd Physical Control Plane
  + Physical Runtime
  + Capability Provider
  + Sandbox Safety Gate
  + Praxis Capture
  + Spatiotemporal Memory
  + Runtime Intervention
  + Knowledge Compiler
  + Self-Evolution Control Plane
  + Skill Registry
  + Darwin Evaluation
```

## Engineering Iron Rules

1. **Runtime owns lifecycle.**
2. **Event Bus owns module communication.**
3. **SeekDB owns structured knowledge.**
4. **Object Store owns heavy artifacts.**
5. **e-URDF owns physical embodiment.**
6. **Provider owns capabilities, not raw model calls.**
7. **Sandbox owns safety validation.**
8. **Practice owns factual execution records.**
9. **Memory owns long-term embodied recall.**
10. **Know owns compiled engineering priors.**
11. **How owns minimal runtime intervention.**
12. **Auto owns self-evolution orchestration.**
13. **Darwin owns evaluation pressure.**
14. **Skill Registry owns promoted capabilities.**

> **Auto 可以提出改变，但不能独自批准改变。Sandbox、Darwin、Promotion Gate 和 Human Approval 共同决定改变是否进入真实世界。**

## Module Boundaries

| Module | Owns | Must NOT Do |
|--------|------|-------------|
| rosclawd | Southbound privileges, action queue, permits, leases, E-Stop latch, receipts | Trust Agent approval flags, expose raw device/ROS writes |
| Runtime | Lifecycle, config, plugin registration, dependency injection | Bypass sandbox, allow unapproved code patches |
| EventBus | Module communication, topic routing, trace correlation | Hold business logic, mutate payloads |
| Provider | Capability routing, schema, safety boundary | Direct robot control, raw model inference |
| Sandbox | Safety validation, firewall, MuJoCo pre-play | Approve patches, promote skills |
| Practice | Timeline, MCAP, JSONL, PraxisEvent | Intervene in runtime, modify skills |
| Memory | Experience graph, failure/success patterns, recall | Compile knowledge, generate patches |
| Know | TaskCard, Pattern, EvidenceTrace, failure taxonomy | Direct intervention, real-time control |
| How | Runtime intervention, injection_id, evidence | Large-scale research, skill promotion |
| Auto | Proposal, Patch, Experiment, Champion, DeadEnd | Direct robot control, bypass sandbox |
| Darwin | Multi-seed benchmark, stress scenario, regression | Approve patches, override safety |
| Skill Registry | Version, lineage, champion, rollback | Execute skills, validate safety |
| Dashboard | Observability, evolution trace, lineage viz | Mutate state, bypass gates |

## Execution Loop

```text
1. Agent receives task and inspects Body/Capability.
2. Agent submits a structured ActionEnvelope through MCP/CLI/SDK.
3. rosclawd authenticates the Unix peer and matches an expiring, use-bounded
   permit to the Body Snapshot, explicit Capability, and exact Action Intent.
4. Sandbox/Firewall decides ALLOW/BLOCK/MODIFY/REQUIRE_CONFIRMATION.
5. ActionGateway acquires the physical resource lease.
6. A daemon-registered executor dispatches and obtains Driver ACK.
7. Observation and task verification produce an ExecutionReceipt.
8. Practice, Memory, Know, How, Auto, and Darwin consume receipts asynchronously.

The Agent process must not construct a second physical Runtime or load a
southbound driver. See [docs/ROSCLAWD.md](docs/ROSCLAWD.md).

This is the target REAL contract. The current base `rosclawd` deliberately
loads no hardware pack, REAL executor, permit issuer, or pack-specific policy
validator, so REAL fails closed until those daemon-side pieces are installed
and accepted for a specific body.
```

## Self-Evolution Loop

```text
PraxisFailedEvent
  → FailureCase
  → Diagnosis
  → Hypothesis
  → Proposal
  → Patch
  → Sandbox Experiment
  → Darwin Evaluation
  → Promotion Gate
  → Champion / DeadEnd
  → Skill Registry Update
  → How / Know / Memory Evidence Update
```

## SeekDB Collections

```text
robots, providers, skills, skill_versions, tasks, runs, episodes
praxis_events, failures, memory_nodes, memory_edges
knowledge_patterns, task_cards, embodiment_cards, verifier_cards
interventions, evidence_traces
auto_proposals, auto_patches, auto_experiments, auto_results
champions, dead_ends, darwin_benchmarks, artifacts
```

## Core Events

```text
TaskSubmittedEvent
ProviderInferenceCompletedEvent
SandboxActionBlockedEvent
SandboxEpisodeFinishedEvent
RuntimeExecutionStartedEvent
RuntimeExecutionCompletedEvent
RuntimeExecutionFailedEvent
PraxisEventCreatedEvent
MemoryWriteCompletedEvent
HowInterventionIssuedEvent
HowFeedbackReceivedEvent
KnowAssetPublishedEvent
AutoProposalCreatedEvent
AutoPatchCreatedEvent
AutoExperimentStartedEvent
AutoExperimentCompletedEvent
ChampionPromotedEvent
DeadEndRegisteredEvent
DarwinBenchmarkCompletedEvent
HumanApprovalRequiredEvent
```

## Event Envelope

```json
{
  "event_id": "evt_...",
  "event_type": "rosclaw.auto.proposal.created",
  "timestamp": "2026-06-04T00:00:00Z",
  "trace_id": "trace_...",
  "run_id": "run_...",
  "task_id": "task_...",
  "robot_id": "ur5e",
  "skill_id": "pick_cube_v1.4",
  "source": "rosclaw-auto",
  "payload": {}
}
```
