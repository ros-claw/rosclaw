"""Fixture replay, fault classification, semantic readiness and boundary gates."""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.connectors.ros.compiler import CapabilityManifest, CapabilityManifestCompiler
from rosclaw.connectors.ros.context import compile_agent_summary
from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.intelligence import RosSystemModel, build_system_model
from rosclaw.connectors.ros.mission import compile_mission
from rosclaw.connectors.ros.resolver import resolve_capabilities, resolve_task

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/ros_expert"
NOW = datetime(2026, 10, 3, tzinfo=UTC)


@pytest.fixture
def healthy():
    def load(name):
        return json.loads((FIXTURES / "nav2" / f"{name}.json").read_text())

    return build_system_model(
        RosGraphSnapshot.from_dict(load("graph")),
        robot_id="fixture_base",
        body=load("body"),
        native=load("native"),
    )


def nav(model):
    return next(
        c
        for c in resolve_capabilities(model, now=NOW)
        if c["semantic_id"] == "navigation.navigate_to_pose"
    )


def test_model_roundtrip_and_tamper_detection(healthy):
    assert RosSystemModel.from_dict(healthy.to_dict()).snapshot_hash == healthy.snapshot_hash
    assert healthy.seal().snapshot_hash == healthy.snapshot_hash
    payload = healthy.to_dict()
    payload["navigation"]["costmaps_fresh"] = False
    with pytest.raises(ValueError, match="hash mismatch"):
        RosSystemModel.from_dict(payload)


def test_unknown_version_and_naive_time_rejected(healthy):
    payload = healthy.to_dict()
    payload["schema_version"] = "rosclaw.ros_system_model.v9"
    with pytest.raises(Exception, match="unsupported major"):
        RosSystemModel.from_dict(payload)
    with pytest.raises(ValueError, match="timezone"):
        RosSystemModel(robot_id="r", snapshot_id="s", captured_at="2026-10-03T00:00:00")


def test_healthy_and_interface_only_readiness(healthy):
    assert diagnose(healthy, now=NOW)["status"] == "HEALTHY"
    assert nav(healthy)["status"] == "AVAILABLE"
    graph = RosGraphSnapshot.from_dict(healthy.graph)
    graph.captured_at = NOW.isoformat()
    graph_only = build_system_model(graph)
    assert nav(graph_only)["status"] == "UNKNOWN"
    assert diagnose(graph_only, now=NOW)["status"] == "UNKNOWN"
    assert not nav(healthy)["usable_for_real_execution"]


FAULTS = [
    ("ROS_TF_001", lambda m: setattr(m, "transforms", m.transforms[1:])),
    ("ROS_TF_002", lambda m: setattr(m, "transforms", [m.transforms[0], m.transforms[2]])),
    ("ROS_TF_003", lambda m: setattr(m, "transforms", m.transforms[:2])),
    ("ROS_TF_004", lambda m: setattr(m.transforms[0], "age_ms", 2000)),
    (
        "ROS_TF_005",
        lambda m: m.transforms.append(m.transforms[0].model_copy(update={"parent": "other"})),
    ),
    (
        "ROS_TF_006",
        lambda m: m.transforms.append(
            m.transforms[0].model_copy(update={"parent": "odom", "child": "map"})
        ),
    ),
    (
        "ROS_TF_007",
        lambda m: m.transforms.append(
            m.transforms[0].model_copy(update={"parent": "other", "child": "unconnected"})
        ),
    ),
    ("ROS_TOPIC_001", lambda m: m.body["required_topics"].append("/absent")),
    ("ROS_TOPIC_002", lambda m: setattr(m.signals[0], "last_message_age_ms", 2000)),
    ("ROS_TOPIC_003", lambda m: setattr(m.signals[0], "rate_hz", 1)),
    ("ROS_TOPIC_004", lambda m: setattr(m.signals[0], "publisher_count", 0)),
    (
        "ROS_QOS_001",
        lambda m: m.qos.update(
            endpoints=[
                {"topic": "/scan", "kind": "publisher", "reliability": "BEST_EFFORT"},
                {"topic": "/scan", "kind": "subscriber", "reliability": "RELIABLE"},
            ]
        ),
    ),
    ("ROS_TIME_001", lambda m: m.observations["node_use_sim_time"].update({"/amcl": False})),
    ("ROS_TIME_002", lambda m: setattr(m.transforms[0], "age_ms", -2000)),
    ("ROS_TIME_003", lambda m: m.observations.update(clock_advancing=False)),
    ("ROS_LIFECYCLE_001", lambda m: setattr(m.lifecycle[0], "state", "INACTIVE")),
    ("NAV2_LOCALIZATION_001", lambda m: m.navigation.update(localization_ready=False)),
    ("NAV2_COSTMAP_001", lambda m: m.navigation.update(obstacle_source_configured=False)),
    ("NAV2_COSTMAP_002", lambda m: m.navigation.update(costmaps_fresh=False)),
    ("NAV2_CONTROLLER_001", lambda m: m.navigation.update(controller_stable=False)),
    ("NAV2_CONTROLLER_002", lambda m: m.navigation.update(controller_progress=False)),
    ("NAV2_COLLISION_001", lambda m: m.navigation.update(collision_monitor_ready=False)),
    ("COVERAGE_PLAN_001", lambda m: m.navigation.update(coverage_plan_valid=False)),
    ("COVERAGE_VERIFY_001", lambda m: m.navigation.update(coverage_complete=False)),
    ("ROS_GRAPH_001", lambda m: setattr(m, "captured_at", NOW - timedelta(seconds=10))),
    ("ROS_ENV_001", lambda m: m.errors.append("read-only lifecycle RPC timeout")),
]


@pytest.mark.parametrize("code,inject", FAULTS, ids=[x[0] for x in FAULTS])
def test_fault_taxonomy_is_evidence_first(healthy, code, inject):
    inject(healthy)
    diagnosis = diagnose(healthy, now=NOW)
    issue = next(i for i in diagnosis["issues"] if i["issue_code"] == code)
    assert issue["evidence"][0]["source"] == healthy.snapshot_id
    assert issue["evidence"][0]["observation"]
    assert issue["confidence"] == 1
    assert issue["runtime_mutation_required"] is False


@pytest.mark.parametrize("inject", [FAULTS[0][1], FAULTS[9][1], FAULTS[15][1], FAULTS[18][1]])
def test_navigation_blocks_on_observed_failures(healthy, inject):
    inject(healthy)
    assert nav(healthy)["status"] == "BLOCKED"


def test_multiple_implementations_do_not_hide_a_ready_one(healthy):
    healthy.graph["actions"].insert(
        0, {"name": "/missing/navigate_to_pose", "action_type": "nav2_msgs/action/NavigateToPose"}
    )
    assert nav(healthy)["status"] == "AVAILABLE"
    assert nav(healthy)["interface"]["name"] == "/navigate_to_pose"


def test_semantics_roundtrip_keeps_legacy_ids(healthy):
    manifest = CapabilityManifestCompiler(robot_id="robot").compile(
        RosGraphSnapshot.from_dict(healthy.graph)
    )
    restored = CapabilityManifest.from_dict(manifest.to_dict())
    assert [c.id for c in manifest.capabilities] == [c.id for c in restored.capabilities]
    assert any(c.semantic_id == "navigation.navigate_to_pose" for c in restored.capabilities)
    ros1 = RosGraphSnapshot.from_dict(json.loads((FIXTURES / "ros1/graph.json").read_text()))
    manifest1 = CapabilityManifestCompiler(robot_id="ros1").compile(ros1)
    assert any(c.semantic_id == "navigation.navigate_to_pose" for c in manifest1.capabilities)


def test_resolve_chinese_cleaning_selects_reuse(healthy):
    result = resolve_task(healthy, "把这个房间完整清扫一遍，并避开移动障碍。", now=NOW)
    assert result["task_class"] == "complete_area_cleaning"
    assert "coverage.execute" in result["missing"]
    assert "opennav_coverage" in [c["id"] for c in result["recommended"]]
    assert result["configured"] is False
    assert "cleaning.enable" in result["missing"]
    with pytest.raises(ValueError, match="unsupported task"):
        resolve_task(healthy, "sing a song", now=NOW)


def test_mission_uses_existing_contract_without_dispatch(healthy):
    graph = compile_mission(healthy, "清扫房间", mission_id="m1")
    graph.validate_dag()
    assert graph.schema_version == "rosclaw.task_graph.v1"
    assert all(
        n.kind.value == "request_action"
        for n in graph.nodes
        if n.required_capabilities
        in [["cleaning.enable"], ["coverage.execute"], ["cleaning.disable"]]
    )
    assert all(n.status.value == "PENDING" for n in graph.nodes)


def test_context_is_compact_and_identifies_freshness(healthy):
    summary = compile_agent_summary(healthy, now=NOW)
    assert len(summary) < 12000
    assert healthy.snapshot_hash in summary
    assert "coverage.execute=MISSING" in summary


def test_probe_has_only_diagnostic_publishers_and_read_rpcs():
    tree = ast.parse((ROOT / "integrations/ros_probe/ros2/probe.py").read_text())
    clients, publishers = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {
                "set_parameters",
                "send_goal_async",
                "create_action_client",
            }
            if node.func.attr == "create_publisher":
                publishers.append(ast.unparse(node.args[1]))
            if node.func.attr == "read_rpc":
                clients.append(ast.unparse(node.args[1]))
    assert publishers == ["PROBE_TOPIC", "'/rosclaw_probe/status'"]
    assert set(clients) <= {"GetState", "GetParameters", "ListParameters"}


def test_context_compiler_integration_is_deterministic_and_fails_stale(healthy):
    from rosclaw.agentd.context.compiler import ContextCompiler, StaleSourceError
    from rosclaw.agentd.context.sources import SelfFacts, SourceBundle
    from rosclaw.agentd.runtime_sources import (
        ConfigConsentSource,
        EmptyMemorySource,
        NullOrgSource,
        SimBodySource,
        StaticCapabilitySource,
    )
    from rosclaw.contracts.agent.mission import BodyBinding, Goal, MissionSessionV1

    class FixedSelf:
        def get_self(self, body_id):
            return SelfFacts(
                self_snapshot_hash="fixed", sequence=1, observed_at=NOW, summary="fixture self"
            )

    body = SimBodySource("fixture_base")
    sources = SourceBundle(
        constitution_text="Guarded actions only",
        body=body,
        self_source=FixedSelf(),
        capabilities=StaticCapabilitySource(["request_action"]),
        memory=EmptyMemorySource(),
        organization=NullOrgSource(),
        consent=ConfigConsentSource(),
        extra={"ros_system_model": healthy},
    )
    mission = MissionSessionV1(
        mission_id="m1",
        owner_principal="user:fixture",
        goal=Goal(text="清扫"),
        body_binding=BodyBinding(body_id="fixture_base", effective_body_hash=body.body_hash),
        created_at=NOW.isoformat(),
        updated_at=NOW.isoformat(),
    )
    graph = compile_mission(healthy, "清扫房间", mission_id="m1")
    compiler = ContextCompiler(sources)
    first = compiler.compile(mission, graph, [], context_revision=1, now=NOW, context_id="c1")
    second = compiler.compile(mission, graph, [], context_revision=1, now=NOW, context_id="c1")
    assert first.bundle_hash == second.bundle_hash
    assert healthy.snapshot_hash in first.layers.dynamic_self.inline_summary
    assert first.body_binding.effective_body_hash == body.body_hash
    with pytest.raises(StaleSourceError, match="ROS observations"):
        compiler.compile(mission, graph, [], context_revision=2, now=NOW + timedelta(seconds=10))


def test_mcp_four_composite_tools_and_tamper_refusal(healthy):
    from rosclaw.connectors.ros.mcp import register_ros_tools

    class Registry:
        def __init__(self):
            self.tools = {}

        def tool(self, description=""):
            def register(func):
                self.tools[func.__name__] = func
                return func

            return register

    registry = Registry()
    register_ros_tools(registry)
    for name in [
        "ros_inspect_system",
        "ros_diagnose_system",
        "ros_resolve_task",
        "ros_verify_mission",
    ]:
        assert name in registry.tools
    assert not any(
        name in registry.tools for name in ["ros_topic_pub", "ros_service_call", "ros_param_set"]
    )
    result = registry.tools["ros_diagnose_system"](snapshot=healthy.to_dict(), profile="tf")
    assert result["ok"]
    corrupted = healthy.to_dict()
    corrupted["body"]["body_type"] = "forged"
    assert registry.tools["ros_inspect_system"](snapshot=corrupted)["ok"] is False


def test_cli_replay_and_unknown_profile_errors(healthy, tmp_path, capsys):
    import argparse

    from rosclaw.connectors.ros.cli.ros_cli import add_ros_subparser, dispatch_ros_command

    parser = argparse.ArgumentParser()
    add_ros_subparser(parser.add_subparsers(dest="command"))
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(healthy.to_dict()))
    arguments = parser.parse_args(
        ["ros", "resolve", "--snapshot", str(snapshot), "--task", "清扫房间", "--json"]
    )
    assert dispatch_ros_command(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["task_class"] == "complete_area_cleaning"
    assert result["configured"] is False


def test_practice_keeps_unverified_observations_out_of_success(healthy):
    from rosclaw.connectors.ros.practice import RosPracticeAdapter
    from rosclaw.core.event_bus import Event

    class Bus:
        def __init__(self):
            self.events = []

        def publish(self, event):
            self.events.append(event)

    bus = Bus()
    adapter = RosPracticeAdapter(bus)
    adapter._on_expert_event(
        Event(
            topic="rosclaw.ros.verification.completed",
            payload={
                "snapshot_id": healthy.snapshot_id,
                "verification": {"verification_status": "NOT_VERIFIED"},
            },
        )
    )
    assert bus.events[-1].payload["outcome"] == "observed"
    assert bus.events[-1].payload["raw"]["verification"]["verification_status"] == "NOT_VERIFIED"


def test_canonical_mcp_registration_exposes_exactly_four_annotated_tools():
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from rosclaw.connectors.ros.mcp.tools import register_ros_expert_tools

    server = FastMCP("ros-expert-test")
    register_ros_expert_tools(server)
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {
        "ros_inspect_system",
        "ros_diagnose_system",
        "ros_resolve_task",
        "ros_verify_mission",
    }
    assert all(t.annotations.readOnlyHint and not t.annotations.destructiveHint for t in tools)


def test_diagnosis_flows_through_existing_practice_event_adapter(healthy):
    from rosclaw.connectors.ros.practice import RosPracticeAdapter

    class LoopbackBus:
        def __init__(self):
            self.events, self.handlers = [], {}

        def subscribe(self, topic, handler):
            self.handlers.setdefault(topic, []).append(handler)

        def unsubscribe(self, topic, handler):
            self.handlers[topic].remove(handler)

        def publish(self, event):
            self.events.append(event)
            for handler in self.handlers.get(event.topic, []):
                handler(event)

    bus = LoopbackBus()
    adapter = RosPracticeAdapter(bus)
    adapter.initialize()
    diagnose(healthy, now=NOW, event_bus=bus)
    captured = next(e for e in bus.events if e.topic == "praxis.recorded")
    assert captured.payload["outcome"] == "observed"
    assert captured.payload["raw"]["diagnosis"]["snapshot_hash"] == healthy.snapshot_hash
    adapter.stop()


def test_unknown_tf_clock_is_not_misdiagnosed_as_missing_topology(healthy):
    for edge in healthy.transforms:
        if not edge.static:
            edge.age_ms = None
    report = diagnose(healthy, now=NOW)
    assert report["status"] == "UNKNOWN"
    assert "tf_timing" in report["unknown_checks"]
    assert not {"ROS_TF_001", "ROS_TF_002"} & {i["issue_code"] for i in report["issues"]}
    assert nav(healthy)["status"] == "UNKNOWN"


def test_know_and_how_keep_readiness_and_diagnostic_evidence(healthy):
    from rosclaw.connectors.ros.how.ros_recovery_rules import diagnostic_intervention
    from rosclaw.connectors.ros.know.ros_knowledge_seed import seed_ros_system

    class Knowledge:
        def __init__(self):
            self.triples = []

        def add_triple(self, **triple):
            self.triples.append(triple)

    healthy.lifecycle[0].state = "INACTIVE"
    healthy.seal()
    knowledge = Knowledge()
    assert seed_ros_system(knowledge, healthy, now=NOW) == len(knowledge.triples)
    navigation = [t for t in knowledge.triples if t["obj"] == "navigation.navigate_to_pose"]
    assert not any(t["predicate"] == "has_capability" for t in navigation)
    assert any(t["predicate"] == "has_ros_interface" for t in knowledge.triples)
    assert all(healthy.snapshot_id in t["source"] for t in knowledge.triples)
    issue = next(
        i for i in diagnose(healthy, now=NOW)["issues"] if i["issue_code"] == "ROS_LIFECYCLE_001"
    )
    intervention = diagnostic_intervention(issue)
    assert intervention["evidence"] == issue["evidence"]
    assert intervention["success_count"] == 0
    assert not intervention["runtime_mutation_required"]
    with pytest.raises(ValueError, match="evidence"):
        diagnostic_intervention({"issue_code": "ROS_LIFECYCLE_001"})
