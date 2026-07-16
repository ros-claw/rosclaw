# ROSClaw Trace

ROSClaw Trace is the structured decision and runtime evidence layer for physical AI. It records
what the runtime observed and did, which model/tool/safety boundary was involved, how long each
operation took, and what evidence supports the result.

It does **not** expose private model chain-of-thought by default. The `standard` capture mode omits
fields such as `cot`, `reasoning`, and `chain_of_thought`; use `DecisionSummary` for auditable goals,
observations, constraints, candidates, decisions, short reasons, confidence, and evidence refs.

## Data flow

```text
Runtime / Provider / MCP / Skill / Sandbox / Firewall
                         │
                    TraceSpan
                         │
          ┌──────────────┴──────────────┐
          │                             │
  EventBus live events        bounded writer queue
          │                             │
 events/live.jsonl              traces/live.jsonl
          │                             │
          └──────── Dashboard / CLI ────┘
```

Every completed span follows `rosclaw.trace.v1` and has a `trace_id`, `span_id`, optional
`parent_span_id`, timing, status, kind, redacted input/output, attributes, and evidence references.
Binary data, arrays, images, and point clouds are represented as metadata or artifact references.

## Use it

Tracing is enabled by default for `Runtime`. Records are stored at
`$ROSCLAW_HOME/traces/live.jsonl` (normally `~/.rosclaw/traces/live.jsonl`).

```bash
rosclaw trace list
rosclaw trace tail --kind LLM,VLM,MCP,SANDBOX
rosclaw trace show <trace-id> --tree
rosclaw trace show <trace-id> --provider
rosclaw trace explain <event-id>
rosclaw trace replay <trace-id>
rosclaw trace export <trace-id> --format json --output trace.json
rosclaw dashboard
# open http://localhost:8765/traces
```

Dashboard APIs:

```text
GET /api/traces
GET /api/traces?trace_id=...&kind=VLM,MCP&status=ERROR
GET /api/traces/{trace_id}
GET /api/traces/events/{event_id}
```

## Instrument an integration

Components sharing an `EventBus` also share a tracer and causal context:

```python
from rosclaw.observability import DecisionSummary, get_tracer

tracer = get_tracer(event_bus)

async with tracer.start_span(
    "camera.detect",
    "PERCEPTION",
    source="front_camera",
    operation="object_detection",
) as span:
    span.set_input({"image": frame})  # array data is omitted; metadata is retained
    detections = await detector(frame)
    span.set_output(detections)
    span.add_evidence("artifact://episode-7/front/1842.jpg")

summary = DecisionSummary(
    goal="navigate to the door",
    observations=["person on the right"],
    constraints=["minimum clearance 0.8m"],
    candidates=[{"action": "left bypass", "score": 0.83}],
    decision="left bypass",
    reason_summary="highest-clearance candidate",
    confidence=0.83,
)
```

Exceptions mark spans as `ERROR`. Safety rejections should explicitly call
`span.set_status("BLOCKED", reason)`. Trace failures are swallowed and never change the provider,
sandbox, skill, or robot-control result.

## Runtime configuration

```python
RuntimeConfig(
    enable_tracing=True,
    trace_capture_mode="standard",  # minimal | standard | research
    trace_queue_size=4096,
    trace_rotate_mb=64.0,
    trace_home=None,
)
```

`ROSCLAW_TRACE_MODE` can set the default capture mode. Secrets remain redacted in every mode.
`research` permits explicit reasoning fields for controlled laboratory use; it does not disable
credential redaction. The exporter queue is bounded and non-blocking. If saturated, normal spans
may be dropped; an error or blocked span displaces a normal queued record.

## Current instrumentation

- Runtime closed-loop missions and structured planning decisions
- Every `Runtime.execute()` mission includes a nested `PLANNER` decision summary, including when
  provider planning is unavailable and the deterministic requested action is retained
- KNOW provider preflight as a `CONTEXT` span, with capability, safety-limit, risk-pattern, and
  `know://...` evidence references. Runtime performs this query once and marks the subsequent
  provider event so the event-driven KNOW fallback does not query it again.
- Bounded Memory retrieval before inference. Prior experiences are passed to the Provider as
  summaries without persisted private `cot_trace`, and are linked with
  `memory://experience/<id>` references.
- Provider requests receive a `context.grounding` envelope containing KNOW, Memory, and evidence
  references. KNOW safety limits are also placed in Provider constraints, so grounding is part of
  the real model request rather than a dashboard-only annotation.
- Provider routing, real inference input/output, model metadata, token usage, fallback, and errors
- MCP tool arguments/results, side-effect classification, robot/session identity, timeout/error state
- Skill execution input/output and block/error state
- Runtime sandbox validation, digital-twin rollout, firewall layers, violations, and replay refs
- Explicit `ROBOT_ACTION` and `ROBOT_STATE` spans for Runtime trajectory simulation and its observed
  result. These spans carry `physical_actuation=false` / `physical_observation=false`; they do not
  pretend a digital-twin run reached hardware.
- Critic evaluation, Memory experience write-back, and KNOW usage write-back in the same mission
  tree. Stored experience metadata links the decision summary, originating trace, retrieved
  experiences, and recovery evidence.
- A blocked execution gets one Runtime-owned HOW recovery span. Heuristic, Memory analogy, and KNOW
  analogy sources use `how://`, `memory://`, and `know://` references; duplicate EventBus recovery
  calls for the same closed-loop failure are suppressed.
- EventBus envelope propagation of `trace_id`, `span_id`, and `parent_span_id`

For example, a navigation/inspection dry run now produces a causal tree like:

```text
MISSION  runtime.execute
├── CONTEXT       knowledge.preflight
├── MEMORY        memory.retrieve_experiences
├── VLM           provider.invoke
│   └── VLM       provider.inference
├── PLANNER       agent.execution_decision
├── SANDBOX       sandbox.validate_action
├── ROBOT_ACTION  robot.simulate_trajectory
│   └── SANDBOX   sandbox.validate_trajectory
├── ROBOT_STATE   robot.observe_execution_result
├── CRITIC        critic.evaluate_execution
├── MEMORY        memory.store_experience
└── CONTEXT       knowledge.record_usage
```

The failure branch adds `RECOVERY how.generate_recovery_hint`. A real robot bridge or MCP driver
should add its own `ROBOT_ACTION` command/ACK and physical `ROBOT_STATE` feedback spans; Runtime's
simulation spans remain clearly labeled until those integrations are present.

High-frequency robot state should continue to use Practice/MCAP. Trace is for causal semantic
operations; it should reference large physical artifacts rather than embed them.
