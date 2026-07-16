"""ROSClaw Trace schema, propagation, persistence, and runtime integration tests."""

from __future__ import annotations

import json

import pytest

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.observability.context import current_trace_context
from rosclaw.observability.exporters.jsonl import JsonlTraceExporter
from rosclaw.observability.redaction import TraceRedactor
from rosclaw.observability.schema import ObservabilityConfig
from rosclaw.observability.store import TraceStore
from rosclaw.observability.tracer import Tracer
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.core.router import CapabilityRouter


class _ArrayLike:
    shape = (480, 640, 3)
    dtype = "uint8"


class _Provider(Provider):
    name = "trace-provider"
    version = "1.2.3"
    capabilities = ["vlm.scene_understanding"]

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        if request.inputs.get("scenario") == "corridor_patrol":
            return ProviderResponse(
                request_id=request.request_id,
                provider=self.name,
                capability=request.capability,
                result={
                    "objects": [
                        {
                            "label": "right-side pedestrian at 0.65 m",
                            "artifact_ref": "artifact://patrol/camera/front/1842.jpg",
                        },
                        {"label": "conference-room door to the north-east"},
                    ],
                    "constraints": [
                        "minimum pedestrian clearance 0.8 m",
                        "maximum linear speed 0.5 m/s",
                    ],
                    "candidates": [
                        {
                            "action": "pass on the right",
                            "score": 0.31,
                            "risk": "clearance below threshold",
                        },
                        {
                            "action": "slow down and pass on the left",
                            "score": 0.86,
                            "risk": "adds about 3 seconds",
                        },
                    ],
                    "decision": "slow down and pass on the left",
                    "reason_summary": (
                        "The left detour preserves pedestrian clearance with the lowest risk."
                    ),
                    "evidence_refs": ["artifact://patrol/local_costmap/301.json"],
                    "chain_of_thought": "private scratch reasoning must not be exposed",
                },
                confidence=0.86,
                evidence=[{"artifact_ref": "artifact://patrol/mocap/9921.json"}],
                model_info={"model": "test-navigation-vlm"},
                trace={"usage": {"input_tokens": 128, "output_tokens": 47}},
            )
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={"objects": ["door"]},
            model_info={"model": "test-vlm"},
            trace={"usage": {"input_tokens": 12, "output_tokens": 3}},
        )


class _GroundingAwareProvider(Provider):
    """Test provider that proves Runtime grounding reaches model inputs."""

    name = "grounding-aware-provider"
    version = "1.0.0"
    capabilities = ["vlm.scene_understanding"]
    requests: list[ProviderRequest] = []

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        grounding = request.context.get("grounding", {})
        knowledge = grounding.get("knowledge") or {}
        memory = grounding.get("memory") or {}
        experiences = memory.get("experiences") or []
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={
                "objects": [{"label": "conference-room door"}],
                "candidates": [
                    {"action": "continue patrol", "score": 0.82, "risk": "low"},
                    {"action": "stop patrol", "score": 0.18, "risk": "missed inspection"},
                ],
                "decision": "continue patrol",
                "reason_summary": (
                    "The declared safety limits and a successful prior patrol support the route."
                ),
                "constraints": [
                    f"knowledge_has_capability={knowledge.get('has_capability')}",
                    f"retrieved_experiences={len(experiences)}",
                ],
                "evidence_refs": grounding.get("evidence_refs", []),
            },
            confidence=0.82,
            model_info={"model": "grounding-aware-test-vlm"},
        )


def test_redactor_removes_secrets_and_large_artifacts():
    redactor = TraceRedactor(max_text_chars=8)
    result = redactor.redact(
        {
            "api_key": "super-secret",
            "input_tokens": 42,
            "authorization": "Bearer abc.def",
            "chain_of_thought": "private scratch work",
            "reason_summary": "auditable reason",
            "image": _ArrayLike(),
            "blob": b"binary",
            "prompt": "0123456789",
        }
    )

    assert result["api_key"] == "[REDACTED]"
    assert result["input_tokens"] == 42
    assert result["authorization"] == "[REDACTED]"
    assert result["chain_of_thought"] == "[PRIVATE_REASONING_OMITTED]"
    assert result["reason_summary"]["text"] == "auditabl"
    assert result["image"]["artifact"] == "array-omitted"
    assert result["blob"]["artifact"] == "inline-binary-omitted"
    assert result["prompt"]["truncated"] is True


def test_span_tree_event_context_and_nonblocking_export(tmp_path):
    path = tmp_path / "traces" / "live.jsonl"
    exporter = JsonlTraceExporter(output_path=path, queue_size=8, rotate_mb=0)
    bus = EventBus()
    events = []
    bus.subscribe("#", events.append)
    tracer = Tracer(
        event_bus=bus,
        config=ObservabilityConfig(capture_mode="standard"),
        exporters=[exporter],
    )

    with tracer.start_span("mission", "MISSION", trace_id="trace-test") as root:
        root.set_input({"password": "do-not-store", "goal": "inspect"})
        emitted = Event(topic="custom.runtime.event", payload={"state": "running"})
        bus.publish(emitted)
        with tracer.start_span("model", "VLM") as child:
            child.set_output({"answer": "door"})

    tracer.close()
    records = TraceStore(path=path).read(trace_id="trace-test")

    assert len(records) == 2
    by_name = {record["name"]: record for record in records}
    assert by_name["model"]["parent_span_id"] == by_name["mission"]["span_id"]
    assert by_name["mission"]["input"]["password"] == "[REDACTED]"
    assert emitted.trace_id == "trace-test"
    assert emitted.span_id == root.span_id
    assert {event.topic for event in events} >= {
        "rosclaw.trace.span.started",
        "rosclaw.trace.span.completed",
    }


@pytest.mark.asyncio
async def test_trace_context_crosses_sync_async_worker_boundary():
    from rosclaw.core.async_utils import run_sync

    async def read_context():
        return current_trace_context()

    tracer = Tracer()
    with tracer.start_span("root", "AGENT", trace_id="cross-thread") as root:
        propagated = run_sync(read_context())

    assert propagated is not None
    assert propagated.trace_id == "cross-thread"
    assert propagated.span_id == root.span_id


@pytest.mark.asyncio
async def test_provider_router_emits_real_nested_provider_trace(tmp_path):
    bus = EventBus()
    registry = ProviderRegistry(event_bus=bus)
    manifest = ProviderManifest.from_dict(
        {
            "name": "trace-provider",
            "version": "1.2.3",
            "type": "vlm",
            "capabilities": ["vlm.scene_understanding"],
        }
    )
    registry.register(manifest, lambda item: _Provider(item), auto_load=False)
    registry.set_provider_health("trace-provider", ok=True)
    path = tmp_path / "live.jsonl"
    exporter = JsonlTraceExporter(output_path=path, rotate_mb=0)
    tracer = Tracer(event_bus=bus, exporters=[exporter])
    router = CapabilityRouter(registry, tracer=tracer)

    response = await router.invoke(
        ProviderRequest(
            request_id="provider-request",
            capability="vlm.scene_understanding",
            inputs={"image": _ArrayLike(), "text": "find the door"},
        )
    )
    tracer.close()

    assert response.is_ok
    topics = [event.topic for event in bus.get_history(limit=100)]
    assert "rosclaw.provider.inference.requested" in topics
    assert "rosclaw.provider.inference.completed" in topics
    trace = TraceStore(path=path).get_trace("provider-request")
    assert trace["span_count"] == 2
    assert trace["tree"][0]["name"] == "provider.invoke"
    inference = trace["tree"][0]["children"][0]
    assert inference["name"] == "provider.inference"
    assert inference["input"]["image"]["artifact"] == "array-omitted"
    assert inference["attributes"]["token.usage"]["input_tokens"] == 12


@pytest.mark.asyncio
async def test_mcp_runtime_provider_chain_shares_one_trace_tree(tmp_path):
    from rosclaw.agent_runtime.mcp_hub import MCPHub
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    bus = EventBus()
    path = tmp_path / "live.jsonl"
    exporter = JsonlTraceExporter(output_path=path, rotate_mb=0)
    tracer = Tracer(event_bus=bus, exporters=[exporter])
    bus._rosclaw_tracer = tracer

    registry = ProviderRegistry(event_bus=bus)
    manifest = ProviderManifest.from_dict(
        {
            "name": "trace-provider",
            "version": "1.2.3",
            "type": "vlm",
            "capabilities": ["vlm.scene_understanding"],
        }
    )
    registry.register(manifest, lambda item: _Provider(item), auto_load=False)
    registry.set_provider_health("trace-provider", ok=True)
    router = CapabilityRouter(registry, tracer=tracer)

    runtime = Runtime(
        RuntimeConfig(enable_provider=False, enable_tracing=False),
        event_bus=bus,
    )
    runtime._capability_router = router
    bus.subscribe("agent.capability.request", runtime._on_capability_request)
    hub = MCPHub(event_bus=bus, robot_id="test-robot", runtime=runtime, tracer=tracer)
    hub.initialize()

    result = await hub.handle_tool_call(
        "observe_scene",
        {"image_topic": "/camera/front", "query": "find the door"},
    )
    hub.stop()
    tracer.close()

    assert result["status"] == "ok"
    traces = TraceStore(path=path).list_traces()
    assert len(traces) == 1
    trace = TraceStore(path=path).get_trace(traces[0]["trace_id"])
    assert [span["name"] for span in trace["spans"]] == [
        "mcp.call_tool",
        "provider.invoke",
        "provider.inference",
    ]
    root = trace["tree"][0]
    assert root["name"] == "mcp.call_tool"
    assert root["children"][0]["name"] == "provider.invoke"
    assert root["children"][0]["children"][0]["name"] == "provider.inference"


def test_runtime_closed_loop_persists_replayable_trace(tmp_path):
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(
        RuntimeConfig(
            robot_id="trace-test-robot",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_swarm=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=True,
            trace_home=str(tmp_path),
        )
    )
    runtime.initialize()
    plan = runtime.plan_action(
        "inspect the door",
        {"confidence": 0.8, "result": {"objects": [{"label": "door"}]}},
    )
    result = runtime.execute(
        {
            "request_id": "mission-123",
            "instruction": "dry-run test",
            "skill_name": "inspect",
            "trajectory": [[0.0] * 6],
        }
    )
    runtime.stop()

    assert plan["decision_summary"]["goal"] == "inspect the door"
    assert plan["decision_summary"]["observations"] == ["door"]
    assert "chain_of_thought" not in plan["decision_summary"]
    assert result["status"] == "ok"
    assert result["trace_id"] == "mission-123"
    trace = TraceStore(home=tmp_path).get_trace("mission-123")
    assert {span["span_kind"] for span in trace["spans"]} >= {
        "MISSION",
        "PLANNER",
        "SANDBOX",
    }
    assert trace["tree"][0]["name"] == "runtime.execute"
    assert trace["tree"][0]["output"]["status"] == "ok"
    planner = next(span for span in trace["spans"] if span["span_kind"] == "PLANNER")
    assert planner["parent_span_id"] == trace["tree"][0]["span_id"]
    decision = planner["output"]["decision_summary"]
    assert decision["goal"] == "dry-run test"
    assert decision["constraints"]
    assert decision["candidates"]
    assert decision["decision"]
    assert decision["reason_summary"]
    assert "chain_of_thought" not in decision


def test_runtime_mission_links_provider_decision_and_sandbox(tmp_path):
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(
        RuntimeConfig(
            robot_id="trace-test-robot",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_swarm=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=True,
            trace_home=str(tmp_path),
        )
    )
    runtime.initialize()
    registry = ProviderRegistry(event_bus=runtime.event_bus)
    manifest = ProviderManifest.from_dict(
        {
            "name": "trace-provider",
            "version": "1.2.3",
            "type": "vlm",
            "capabilities": ["vlm.scene_understanding"],
        }
    )
    registry.register(manifest, lambda item: _Provider(item), auto_load=False)
    registry.set_provider_health("trace-provider", ok=True)
    runtime._capability_router = CapabilityRouter(registry, tracer=runtime.tracer)

    result = runtime.execute(
        {
            "request_id": "mission-provider-123",
            "instruction": "find the door and inspect it",
            "skill_name": "inspect",
            "capability": "vlm.scene_understanding",
            "parameters": {"image": _ArrayLike(), "text": "find the door"},
            "trajectory": [[0.0] * 6],
        }
    )
    runtime.stop()

    assert result["status"] == "ok"
    trace = TraceStore(home=tmp_path).get_trace("mission-provider-123")
    spans = {span["name"]: span for span in trace["spans"]}
    mission = spans["runtime.execute"]
    provider = spans["provider.invoke"]
    inference = spans["provider.inference"]
    planner = spans["agent.execution_decision"]
    sandbox = spans["sandbox.validate_action"]

    assert provider["parent_span_id"] == mission["span_id"]
    assert inference["parent_span_id"] == provider["span_id"]
    assert planner["parent_span_id"] == mission["span_id"]
    assert sandbox["parent_span_id"] == mission["span_id"]
    assert inference["input"]["image"]["artifact"] == "array-omitted"
    assert inference["output"]["result"] == {"objects": ["door"]}
    decision = planner["output"]["decision_summary"]
    assert decision["goal"] == "find the door and inspect it"
    assert decision["observations"] == ["door"]
    assert decision["constraints"] == [
        "safety_level=MODERATE",
        "sandbox_validation=required",
    ]
    assert decision["decision"] == {
        "skill_name": "inspect",
        "capability": "vlm.scene_understanding",
        "parameters": {
            "image": {
                "artifact": "array-omitted",
                "shape": [480, 640, 3],
                "dtype": "uint8",
                "type": "_ArrayLike",
            },
            "text": "find the door",
        },
    }
    assert result["decision_summary"]["reason_summary"]


def test_runtime_navigation_decision_preserves_auditable_model_evidence(tmp_path):
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(
        RuntimeConfig(
            robot_id="trace-navigation-robot",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_swarm=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=True,
            trace_home=str(tmp_path),
        )
    )
    runtime.initialize()
    registry = ProviderRegistry(event_bus=runtime.event_bus)
    manifest = ProviderManifest.from_dict(
        {
            "name": "trace-provider",
            "version": "1.2.3",
            "type": "vlm",
            "capabilities": ["vlm.scene_understanding"],
        }
    )
    registry.register(manifest, lambda item: _Provider(item), auto_load=False)
    registry.set_provider_health("trace-provider", ok=True)
    runtime._capability_router = CapabilityRouter(registry, tracer=runtime.tracer)

    result = runtime.execute(
        {
            "request_id": "mission-navigation-123",
            "instruction": "patrol the corridor and inspect the conference-room door",
            "skill_name": "navigate_and_inspect",
            "capability": "vlm.scene_understanding",
            "parameters": {
                "scenario": "corridor_patrol",
                "api_key": "must-not-leak",
            },
            "trajectory": [[0.0] * 6, [0.3, 0.1, 0.0, 0.0, 0.0, 0.0]],
        }
    )
    runtime.stop()

    assert result["status"] == "ok"
    trace = TraceStore(home=tmp_path).get_trace("mission-navigation-123")
    spans = {span["name"]: span for span in trace["spans"]}
    decision = spans["agent.execution_decision"]["output"]["decision_summary"]

    assert decision["observations"] == [
        "right-side pedestrian at 0.65 m",
        "conference-room door to the north-east",
    ]
    assert decision["constraints"] == [
        "safety_level=MODERATE",
        "sandbox_validation=required",
        "minimum pedestrian clearance 0.8 m",
        "maximum linear speed 0.5 m/s",
    ]
    assert decision["candidates"] == [
        {
            "action": "pass on the right",
            "score": 0.31,
            "risk": "clearance below threshold",
        },
        {
            "action": "slow down and pass on the left",
            "score": 0.86,
            "risk": "adds about 3 seconds",
        },
    ]
    assert decision["confidence"] == 0.86
    assert decision["reason_summary"] == (
        "The left detour preserves pedestrian clearance with the lowest risk."
    )
    assert decision["evidence_refs"] == [
        "artifact://patrol/camera/front/1842.jpg",
        "artifact://patrol/local_costmap/301.json",
        "artifact://patrol/mocap/9921.json",
    ]
    assert decision["decision"]["selected_candidate"] == "slow down and pass on the left"
    assert decision["decision"]["executed_action"]["parameters"]["api_key"] == "[REDACTED]"
    assert "chain_of_thought" not in decision
    assert spans["provider.inference"]["output"]["result"]["chain_of_thought"] == (
        "[PRIVATE_REASONING_OMITTED]"
    )


def test_runtime_trace_closes_knowledge_memory_decision_feedback_loop(tmp_path):
    """A patrol mission retrieves grounding before inference and records its outcome."""

    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(
        RuntimeConfig(
            robot_id="ur5e",
            enable_firewall=False,
            enable_memory=True,
            enable_practice=False,
            enable_swarm=False,
            enable_skill_manager=False,
            enable_knowledge=True,
            enable_how=True,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=True,
            trace_home=str(tmp_path),
            seekdb_backend="memory",
        )
    )
    runtime.initialize()
    assert runtime.memory is not None
    runtime.memory.store_experience(
        event_id="prior-patrol-success",
        event_type="praxis",
        instruction="patrol the corridor and inspect the conference-room door",
        cot_trace=["private historical reasoning must not enter a new model request"],
        outcome="success",
        tags=["navigate_and_inspect", "success"],
        metadata={"reason_summary": "The left route maintained safe clearance."},
    )
    assert runtime.knowledge is not None
    original_preflight = runtime.knowledge.query_for_provider_selection
    preflight_calls = 0

    def counted_preflight(*args, **kwargs):
        nonlocal preflight_calls
        preflight_calls += 1
        return original_preflight(*args, **kwargs)

    runtime.knowledge.query_for_provider_selection = counted_preflight
    runtime.start()

    _GroundingAwareProvider.requests.clear()
    registry = ProviderRegistry(event_bus=runtime.event_bus)
    manifest = ProviderManifest.from_dict(
        {
            "name": "grounding-aware-provider",
            "version": "1.0.0",
            "type": "vlm",
            "capabilities": ["vlm.scene_understanding"],
        }
    )
    registry.register(manifest, lambda item: _GroundingAwareProvider(item), auto_load=False)
    registry.set_provider_health("grounding-aware-provider", ok=True)
    runtime._capability_router = CapabilityRouter(registry, tracer=runtime.tracer)

    result = runtime.execute(
        {
            "request_id": "mission-grounded-patrol",
            "task_id": "patrol-42",
            "instruction": "patrol the corridor and inspect the conference-room door",
            "skill_name": "navigate_and_inspect",
            "capability": "vlm.scene_understanding",
            "parameters": {"scenario": "corridor_patrol"},
            "trajectory": [[0.0] * 6, [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]],
        }
    )
    runtime.stop()

    assert result["status"] == "ok"
    assert preflight_calls == 1
    assert len(runtime.event_bus.get_history("rosclaw.knowledge.pre_check")) == 1
    assert len(_GroundingAwareProvider.requests) == 1
    request = _GroundingAwareProvider.requests[0]
    grounding = request.context["grounding"]
    assert grounding["knowledge"]["robot_id"] == "ur5e"
    assert grounding["knowledge"]["safety_limits"]
    assert grounding["memory"]["experiences"][0]["id"] == "prior-patrol-success"
    assert "cot_trace" not in grounding["memory"]["experiences"][0]
    assert request.constraints["knowledge_safety_limits"]

    trace = TraceStore(home=tmp_path).get_trace("mission-grounded-patrol")
    spans = {span["name"]: span for span in trace["spans"]}
    expected = {
        "runtime.execute": "MISSION",
        "knowledge.preflight": "CONTEXT",
        "memory.retrieve_experiences": "MEMORY",
        "provider.invoke": "VLM",
        "provider.inference": "VLM",
        "agent.execution_decision": "PLANNER",
        "sandbox.validate_action": "SANDBOX",
        "robot.simulate_trajectory": "ROBOT_ACTION",
        "sandbox.validate_trajectory": "SANDBOX",
        "robot.observe_execution_result": "ROBOT_STATE",
        "critic.evaluate_execution": "CRITIC",
        "memory.store_experience": "MEMORY",
        "knowledge.record_usage": "CONTEXT",
    }
    assert {name: spans[name]["span_kind"] for name in expected} == expected
    mission_span_id = spans["runtime.execute"]["span_id"]
    for name in expected.keys() - {
        "runtime.execute",
        "provider.inference",
        "sandbox.validate_trajectory",
    }:
        assert spans[name]["parent_span_id"] == mission_span_id
    assert spans["provider.inference"]["parent_span_id"] == spans["provider.invoke"]["span_id"]
    assert (
        spans["sandbox.validate_trajectory"]["parent_span_id"]
        == spans["robot.simulate_trajectory"]["span_id"]
    )
    assert spans["robot.simulate_trajectory"]["attributes"]["physical_actuation"] is False
    assert spans["memory.retrieve_experiences"]["evidence_refs"] == [
        "memory://experience/prior-patrol-success"
    ]
    decision = spans["agent.execution_decision"]["output"]["decision_summary"]
    assert "retrieved_experiences=1" in decision["constraints"]
    assert "memory://experience/prior-patrol-success" in decision["evidence_refs"]
    assert "chain_of_thought" not in decision


def test_runtime_trace_records_single_evidence_backed_how_recovery(tmp_path):
    """A safety block uses one HOW call and links its source experience."""

    from rosclaw.core.runtime import Runtime, RuntimeConfig
    from rosclaw.firewall.validator import (
        ValidationLayer,
        ValidationResponse,
        ViolationDetail,
    )

    runtime = Runtime(
        RuntimeConfig(
            robot_id="ur5e",
            enable_firewall=False,
            enable_memory=True,
            enable_practice=False,
            enable_swarm=False,
            enable_skill_manager=False,
            enable_knowledge=True,
            enable_how=True,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=True,
            trace_home=str(tmp_path),
            seekdb_backend="memory",
        )
    )
    runtime.initialize()
    assert runtime.memory is not None
    runtime.memory.store_experience(
        event_id="prior-firewall-recovery",
        event_type="praxis",
        instruction="recover from an unusual safety interlock",
        outcome="failure",
        error_details="firewall blocked",
        tags=["firewall", "blocked"],
        metadata={"recovery_hint": "Re-localize, reduce speed, and request a fresh safety check."},
    )

    class _BlockingFirewall:
        def validate(self, request):
            return ValidationResponse(
                request_id=request.request_id,
                is_safe=False,
                layers_checked=[ValidationLayer.SEMANTIC_SAFETY],
                violations=[
                    ViolationDetail(
                        layer=ValidationLayer.SEMANTIC_SAFETY,
                        severity="HIGH",
                        joint_index=None,
                        description="unclassified safety interlock",
                    )
                ],
            )

    runtime._firewall = _BlockingFirewall()
    assert runtime._how is not None
    original_generate = runtime._how.generate_recovery_hint
    recovery_calls = 0

    async def counted_generate(*args, **kwargs):
        nonlocal recovery_calls
        recovery_calls += 1
        return await original_generate(*args, **kwargs)

    runtime._how.generate_recovery_hint = counted_generate
    result = runtime.execute(
        {
            "request_id": "mission-blocked-recovery",
            "instruction": "continue the inspection through the interlock",
            "skill_name": "navigate_and_inspect",
            "capability": "vlm.scene_understanding",
            "trajectory": [[0.0] * 6],
        }
    )
    runtime.stop()

    assert result["status"] == "blocked"
    assert recovery_calls == 1
    assert result["recovery_hint"]["source"] == "memory_analogy"
    trace = TraceStore(home=tmp_path).get_trace("mission-blocked-recovery")
    spans = {span["name"]: span for span in trace["spans"]}
    recovery = spans["how.generate_recovery_hint"]
    assert recovery["span_kind"] == "RECOVERY"
    assert recovery["status"] == "OK"
    assert recovery["output"]["source"] == "memory_analogy"
    assert recovery["evidence_refs"] == [
        "how://rule/analogy_prior-firewall-recovery",
        "memory://experience/prior-firewall-recovery",
    ]
    assert spans["robot.simulate_trajectory"]["status"] == "BLOCKED"
    assert spans["critic.evaluate_execution"]["output"]["status"] == "BLOCKED"


def test_trace_store_ignores_malformed_lines(tmp_path):
    path = tmp_path / "live.jsonl"
    path.write_text(
        "not-json\n"
        + json.dumps(
            {
                "trace_id": "t1",
                "span_id": "s1",
                "parent_span_id": None,
                "started_at": 1.0,
                "ended_at": 2.0,
                "status": "OK",
                "span_kind": "AGENT",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert TraceStore(path=path).list_traces() == [
        {
            "trace_id": "t1",
            "started_at": 1.0,
            "ended_at": 2.0,
            "duration_ms": 1000.0,
            "span_count": 1,
            "status": "OK",
            "kinds": ["AGENT"],
        }
    ]
