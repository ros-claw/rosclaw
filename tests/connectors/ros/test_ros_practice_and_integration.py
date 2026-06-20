"""Tests for ROS practice capture adapter and Phase 9 integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rosclaw.connectors.ros.practice import RosPracticeAdapter
from rosclaw.connectors.ros.provider import RosCapabilityProvider
from rosclaw.provider.core.manifest import ProviderManifest


@dataclass
class FakeEventBus:
    """In-memory EventBus for testing."""

    events: list[Any] = field(default_factory=list)

    def publish(self, event: Any) -> None:
        self.events.append(event)

    def subscribe(self, topic: str, handler: Any) -> None:
        pass

    def unsubscribe(self, topic: str, handler: Any) -> None:
        pass


@dataclass
class FakeEvent:
    topic: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = ""


def test_ros_practice_adapter_publishes_praxis_recorded():
    bus = FakeEventBus()
    adapter = RosPracticeAdapter(bus)
    adapter.initialize()

    adapter._on_practice_event_created(FakeEvent(
        topic="rosclaw.practice.event.created",
        payload={
            "trace_id": "ros_abc123",
            "robot_id": "turtlesim",
            "capability_id": "turtlesim.base.velocity_command",
            "ros_name": "/turtle1/cmd_vel",
            "ros_kind": "topic",
            "result": {"ok": True},
        },
    ))

    recorded = [e for e in bus.events if getattr(e, "topic", "") == "praxis.recorded"]
    assert len(recorded) == 1
    assert recorded[0].payload["outcome"] == "success"
    assert recorded[0].payload["capability_id"] == "turtlesim.base.velocity_command"


def test_ros_practice_adapter_publishes_failure_as_recorded():
    bus = FakeEventBus()
    adapter = RosPracticeAdapter(bus)
    adapter.initialize()

    adapter._on_sandbox_episode_failed(FakeEvent(
        topic="rosclaw.sandbox.episode.failed",
        payload={
            "trace_id": "ros_def456",
            "robot_id": "turtlesim",
            "capability_id": "turtlesim.base.velocity_command",
            "error": "Blocked by safety contract",
            "sandbox_blocked": True,
        },
    ))

    recorded = [e for e in bus.events if getattr(e, "topic", "") == "praxis.recorded"]
    assert len(recorded) == 1
    assert recorded[0].payload["outcome"] == "blocked"


def test_ros_capability_provider_publishes_firewall_blocked_event():
    bus = FakeEventBus()
    from rosclaw.connectors.ros.compiler import (
        CapabilityManifest,
        RosCapability,
        RosCapabilityRisk,
        RosInterface,
        SafetyContractCompiler,
    )

    cap = RosCapability(
        id="turtlesim.base.velocity_command",
        kind="actuation",
        interface=RosInterface(ros_kind="topic", name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist"),
        risk=RosCapabilityRisk(
            level="high", read_only=False, destructive=True,
            requires_sandbox=True, requires_runtime_guard=True, requires_stop_guard=True, max_duration_sec=1.0,
        ),
        safety={"constraints": {"linear.x": [-0.2, 0.2]}},
    )
    manifest = CapabilityManifest(robot_id="turtlesim", capabilities=[cap])

    provider_manifest = ProviderManifest(
        name="ros_capability_provider",
        version="0.1.0",
        type="ros",
        runtime={"endpoint": "ws://127.0.0.1:9090"},
        extra={"robot_id": "turtlesim", "dry_run": True, "auto_discover": False, "event_bus": bus},
    )
    provider = RosCapabilityProvider(provider_manifest)
    provider._manifest = manifest
    provider._contract = SafetyContractCompiler().compile(manifest)
    provider.capabilities = [cap.id]

    import asyncio

    from rosclaw.provider.core.request import ProviderRequest

    request = ProviderRequest(
        request_id="test_block",
        capability="turtlesim.base.velocity_command",
        inputs={"linear": {"x": 1.0}, "duration": 0.5},
        context={},
    )
    response = asyncio.run(provider.infer(request))

    assert response.status == "blocked"
    blocked_events = [e for e in bus.events if getattr(e, "topic", "") == "firewall.action_blocked"]
    assert len(blocked_events) == 1
    assert blocked_events[0].payload["capability_id"] == "turtlesim.base.velocity_command"


def test_ros_recovery_rules_seed():
    class FakeSeekDB:
        def __init__(self):
            self.rows: list[dict] = []

        def insert(self, table: str, record: dict) -> str:
            self.rows.append({"table": table, **record})
            return record.get("id", "")

    from rosclaw.connectors.ros.how.ros_recovery_rules import seed_ros_recovery_rules

    db = FakeSeekDB()
    count = seed_ros_recovery_rules(db)
    assert count > 0
    assert all(r["table"] == "heuristic_rules" for r in db.rows)
    conditions = {r["condition"] for r in db.rows}
    assert "rosbridge connection refused" in conditions


def test_ros_knowledge_seed():
    class FakeKnowledgeInterface:
        def __init__(self):
            self.triples: list[dict] = []

        def add_triple(self, **kwargs):
            self.triples.append(kwargs)

    from rosclaw.connectors.ros.know.ros_knowledge_seed import seed_ros_capabilities

    know = FakeKnowledgeInterface()
    count = seed_ros_capabilities(know, "turtlesim", ["turtlesim.base.velocity_command", "turtlesim.observe.pose"])
    assert count == 2
    assert know.triples[0]["subject"] == "turtlesim"
    assert know.triples[0]["predicate"] == "has_capability"
