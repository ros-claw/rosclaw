# ROSClaw Architecture

## Mission

**Self-Evolving Runtime Infrastructure for Physical AI & Embodied Agents**

Ground AI agents into robot bodies. Validate every action. Learn from every trace. Evolve every skill.

## Engineering Identity

```text
ROSClaw =
  Physical Runtime
  + Capability Provider
  + Sandbox Safety Gate
  + Praxis Capture
  + Spatiotemporal Memory
  + Runtime Intervention
  + Knowledge Compiler
  + Self-Evolution Control Plane
  + Skill Registry
  + Darwin Evaluation
  + Physical-AI Asset Hub
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
15. **Hub owns asset discovery and distribution; Runtime owns asset execution.**

> **Auto can propose changes, but cannot approve them alone. Sandbox, Darwin, the promotion gate, and human approval together decide whether a change reaches the real world.**

## Runtime Principles

- **EventBus-only communication**: All modules communicate exclusively through the EventBus. No direct module-to-module calls. This ensures complete decoupling and enables any agent to connect without hardware-specific knowledge.
- **Embodiment-first**: Every physical action is grounded through robot embodiment, capability schemas, safety limits, and runtime context.
- **Sandbox-before-reality**: No model output reaches a real robot without passing through simulation-based validation.
- **Practice-as-fact**: Execution traces are first-class evidence for failure analysis, memory recall, and skill evolution.
- **Human-in-the-loop for safety**: Code patches, safety configuration changes, and skill promotion to real hardware require explicit human approval.

## System Overview

```text
┌──────────────────────────────────────────────────────────────┐
│           External Cognitive Brains                          │
│     OpenClaw / Claude / GPT / Qwen / Custom Agents           │
└───────────────────────────┬──────────────────────────────────┘
                            │ MCP / SDK / AgentContext
                            ▼
┌──────────────────────────────────────────────────────────────┐
│           ROSClaw Runtime                                    │
│  AgentContext │ TaskContext │ SkillContext │ Trace           │
└───────────────────────────┬──────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Provider    │   │   Sandbox     │   │    Darwin     │
│  Capability   │   │  e-URDF /     │   │  Benchmark /  │
│   Router      │   │  MuJoCo /     │   │  Regression / │
│               │   │  Firewall     │   │  Evaluation   │
└───────┬───────┘   └───────┬───────┘   └───────────────┘
        │                   │
        └───────────┬───────┘
                    ▼
┌──────────────────────────────────────────────────────────────┐
│           Physical World / Simulator                         │
│        UR5e / G1 / Go2 / RealSense / IoT / MuJoCo            │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│           Practice Capture                                   │
│   Unified Timeline / MCAP / JSONL / Video / Events           │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│           SeekDB Knowledge Plane                             │
│  Robot │ Skill │ Provider │ Episode │ Failure │ Evidence    │
└───────────────┬────────────────────────────┬─────────────────┘
                │                            │
                ▼                            ▼
┌───────────────────────┐      ┌───────────────────────────────┐
│     Memory            │      │         Know                  │
│  Spatiotemporal       │      │  Physical-AI Knowledge        │
│  Failure / Success    │      │  Compiler                     │
│  Pattern / Causal     │      │  TaskCard / Pattern / Evidence│
└───────────┬───────────┘      └───────────────┬───────────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │      How  ←→  Auto           │
            │  Runtime Intervention        │
            │  Self-Evolution Control      │
            │  Proposal / Patch / Champion │
            └──────────────┬───────────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │  Skill Registry │
                 │  Versioned /    │
                 │  Champion /     │
                 │  Rollback-safe  │
                 └─────────────────┘
```

## Module Boundaries

| Module | Owns | Must NOT Do |
|--------|------|-------------|
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
| Hub | Asset discovery, validation, distribution | Execute assets, bypass runtime gates |

## Execution Loop

```text
1. Agent receives task
2. Runtime constructs AgentContext
3. Provider routes capabilities
4. Provider outputs structured results
5. Sandbox enters firewall mode
6. Sandbox decides ALLOW / BLOCK / MODIFY / REQUIRE_HUMAN_CONFIRMATION
7. Runtime executes on physical robot or simulator
8. Practice captures execution trace
9. Critic judges success/failure
10. Memory writes experience
11. How decides intervention
12. Auto decides self-evolution
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

A skill is not overwritten blindly. It is **versioned, evaluated, promoted, and rollback-safe**:

```text
pick_cube@v1.0.0  baseline_champion
    ↓
pick_cube@candidate_0001  sandbox_passed
    ↓
pick_cube@v1.1.0  sim_champion
    ↓
pick_cube@v1.1.0  sandbox_champion
    ↓
pick_cube@v1.1.0  real_candidate
    ↓
pick_cube@v1.1.0  real_champion
```

## Event Bus

All modules publish and subscribe through a unified EventBus. Topics are namespaced by module and trace correlation.

### Core Events

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

### Event Envelope

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

## Agent Context

The AgentContext carries everything an MCP-compatible agent needs to interact with ROSClaw without knowing hardware details:

- Task description and history
- Connected robot body profile
- Available capabilities and providers
- Sandbox decisions and constraints
- Memory recall results
- Safety level and approval status

## Embodiment Context

ROSClaw treats robot embodiment as a first-class system primitive. An e-URDF profile defines:

- Robot structure
- Joints, links, sensors, actuators
- Safety envelopes
- Tool frames
- Workspace limits
- Capabilities
- Simulation assets
- Benchmark metadata

This allows the same skill to be adapted, validated, and transferred across different robot bodies.

## Capability Provider

The Provider layer turns heterogeneous models and algorithms into routable physical capabilities:

- LLM for planning and reasoning
- VLM for visual grounding
- VLA for vision-language-action proposals
- VLN for navigation
- World model for scene understanding
- Skill policy for execution
- Critic for success/failure judgment
- Embedding for semantic memory retrieval

Providers produce structured outputs, not raw motor commands.

## Sandbox / Firewall

The Sandbox module provides simulation-first validation. Its firewall mode can block risky actions before they reach hardware.

Possible decisions:

```text
ALLOW
BLOCK
MODIFY
REQUIRE_HUMAN_CONFIRMATION
```

Example result:

```json
{
  "decision": "BLOCK",
  "risk_score": 0.92,
  "reason": "Predicted collision between wrist_link and table",
  "violated_constraints": ["collision", "workspace_boundary"],
  "replay_id": "sandbox://replays/firewall_00042"
}
```

## Practice Timeline

Practice captures a unified physical timeline including:

- Robot state snapshots
- Sensor data
- Action traces
- Provider traces
- Sandbox decisions
- Skill execution metadata
- Critic results
- MCAP / JSONL artifacts
- Replay files
- Failure reports

## Memory Schema

SeekDB collections:

```text
robots, providers, skills, skill_versions, tasks, runs, episodes
praxis_events, failures, memory_nodes, memory_edges
knowledge_patterns, task_cards, embodiment_cards, verifier_cards
interventions, evidence_traces
auto_proposals, auto_patches, auto_experiments, auto_results
champions, dead_ends, darwin_benchmarks, artifacts
```

## How Intervention

`rosclaw-how` acts as a runtime reflex layer. When an agent is stuck, unsafe, invalid-heavy, or regressing, it provides minimal, evidence-backed interventions such as:

- Safety constraints
- Feasibility repair
- Stabilizing hints
- Next experiment suggestions
- Recovery instructions

## Know TaskCard

`rosclaw-know` compiles unstructured engineering knowledge into structured TaskCards:

- Papers and documentation
- Code and logs
- Trajectories and benchmark traces
- Failures and causal evidence

## Auto Evolution

`rosclaw-auto` turns repeated failures into structured improvement cycles:

```text
FailureCase
    ↓
Diagnosis
    ↓
Hypothesis
    ↓
Proposal
    ↓
Patch
    ↓
Sandbox Experiment
    ↓
Darwin Evaluation
    ↓
Champion / DeadEnd
```

## Darwin Evaluation

`rosclaw-darwin` provides the evaluation pressure that gates skill promotion:

- Multi-seed validation
- Stress scenarios
- Regression tests
- Safety event tracking
- Success-rate thresholds

Promotion gates:

| Gate | Check |
|------|-------|
| Success Improvement | Candidate success rate > baseline + threshold |
| Safety Regression | No increase in collision or safety events |
| Multi-Seed Validation | Passes on seeds [0, 1, 2, ...] |
| Sandbox Clearance | Firewall decision == ALLOW |
| Regression Suite | No degradation on existing tasks |
| Human Approval | Required for code patches and safety config |

## Hub Asset Lifecycle

Physical-AI assets flow through the Hub with explicit lifecycle states:

```text
Create
  → Validate
  → Sign
  → Publish
  → Sync
  → Install
  → Activate
  → Evaluate
  → Update / Rollback
  → Uninstall
```

Asset types:

| Type | Description |
|---|---|
| `skill` | Task policy, recovery logic, skill graph, parameters, evaluation metadata |
| `provider` | LLM / VLM / VLA / VLN / world model / critic / embedding / robotics algorithm |
| `hardware_mcp` | MCP-compatible robot, sensor, or device interface |
| `digital_twin` | Simulation world, robot asset, replay scene, validation environment |
| `e_urdf` | Robot embodiment profile and safety envelope |
| `cognitive_wiki` | TaskCard, failure taxonomy, constraints, evidence, and repair knowledge |

## Safety Boundary

The safety pipeline:

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

Hard rules:

- VLA outputs are proposals, not raw motor commands.
- World models are previews, not safety proofs.
- MCP is an agent tool interface, not a real-time control bus.
- Auto-generated skills must pass sandbox validation before execution.
- Code patches require human review before production use.
- Safety configuration changes require explicit approval.
- Every promoted skill must be versioned and rollback-safe.

See [docs/SAFETY.md](docs/SAFETY.md) for the complete safety model.

## Deployment Modes

| Mode | Description | Cloud Key Required |
|---|---|---|
| **Simulation-only (offline)** | Local runtime, local assets, local traces. No real robot. | No |
| **Simulation-to-Reality** | Firewall-enabled; all real-robot actions validated in sandbox first. | Optional |
| **Production** | All gates active, human approval required for patches and promotion. | Yes |

## Data Flow

```text
Agent → Runtime → EventBus
              ├── Provider → Sandbox → Robot/Simulator → Practice → Memory
              ├── How ←→ Auto ←→ Darwin → Skill Registry
              └── Dashboard ←→ Hub
```

## Repository Structure

```text
rosclaw/
├── src/rosclaw/              # Core runtime, schemas, CLI, MCP gateway
│   ├── core/                 # Runtime, EventBus, lifecycle
│   ├── schemas/              # Unified canonical dataclasses
│   ├── provider/             # Capability provider layer
│   ├── sandbox/              # MuJoCo simulation & firewall
│   ├── practice/             # Timeline capture & MCAP
│   ├── memory/               # Spatiotemporal memory
│   ├── how/                  # Runtime intervention
│   ├── know/                 # Knowledge compiler
│   ├── auto/                 # Self-evolution control plane
│   ├── darwin/               # Benchmark & evaluation arena
│   ├── forge/                # Asset compiler
│   ├── dashboard/            # Observability & WebSocket
│   └── mcp/                  # MCP server implementation
├── e-urdf-zoo/               # Physical DNA registry
├── docs/                     # Architecture, RFCs, usage guides
├── examples/                 # Robot and simulation examples
├── tests/                    # Unit, integration, E2E, safety tests
├── benchmarks/               # Benchmark and evaluation tasks
├── scripts/                  # Install and utility scripts
├── rosclaw.yaml              # Default runtime config
├── ARCHITECTURE.md           # This file
├── QUICKSTART.md             # Quick start guide
└── INSTALL.md                # Installation details
```

## References

- [Quick Start](QUICKSTART.md)
- [Installation](INSTALL.md)
- [Safety Model](docs/SAFETY.md)
- [Physical-AI Assets](docs/ASSETS.md)
- [CLI Reference](docs/CLI.md)
- [Hub](docs/hub/README.md)
