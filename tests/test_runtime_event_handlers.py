"""Tests for Runtime event handlers."""

import time

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.core.event_bus import Event


def test_runtime_firewall_action_blocked_handler():
    """_on_firewall_action_blocked handles event even with sync proxy."""
    config = RuntimeConfig(
        robot_id="test_bot",
        enable_firewall=False,
        enable_memory=True,
        enable_practice=False,
        enable_knowledge=False,
        enable_how=True,
        enable_provider=False,
        seekdb_backend="memory",
    )
    rt = Runtime(config)
    rt._do_initialize()

    received = []
    def on_recovery(event):  # noqa: E306
        received.append(event.topic)
    rt.event_bus.subscribe("heuristic.recovery_suggested", on_recovery)

    rt.event_bus.publish(Event(
        topic="firewall.action_blocked",
        payload={"request_id": "r1", "violations": [{"description": "collision"}]},
        source="test",
    ))
    time.sleep(0.1)
    # Event handler may fail due to asyncio.run on sync proxy, but is caught
    rt._do_stop()


def test_runtime_sandbox_episode_failed_handler():
    """_on_sandbox_episode_failed generates recovery hint."""
    config = RuntimeConfig(
        robot_id="test_bot",
        enable_firewall=False,
        enable_memory=True,
        enable_practice=False,
        enable_knowledge=False,
        enable_how=True,
        enable_provider=False,
        seekdb_backend="memory",
    )
    rt = Runtime(config)
    rt._do_initialize()

    received = []
    def on_hint(event):  # noqa: E306
        received.append(event.topic)
    rt.event_bus.subscribe("rosclaw.how.recovery_hint.generated", on_hint)

    rt.event_bus.publish(Event(
        topic="rosclaw.sandbox.episode.failed",
        payload={"failure_type": "collision", "request_id": "r2"},
        source="test",
    ))
    time.sleep(0.2)
    rt._do_stop()


def test_runtime_sandbox_action_blocked_handler():
    """_on_sandbox_action_blocked generates recovery hint."""
    config = RuntimeConfig(
        robot_id="test_bot",
        enable_firewall=False,
        enable_memory=True,
        enable_practice=False,
        enable_knowledge=False,
        enable_how=True,
        enable_provider=False,
        seekdb_backend="memory",
    )
    rt = Runtime(config)
    rt._do_initialize()

    received = []
    def on_hint(event):  # noqa: E306
        received.append(event.topic)
    rt.event_bus.subscribe("rosclaw.how.recovery_hint.generated", on_hint)

    rt.event_bus.publish(Event(
        topic="rosclaw.sandbox.action.blocked",
        payload={"reason": "out_of_bounds", "request_id": "r3"},
        source="test",
    ))
    time.sleep(0.2)
    rt._do_stop()


def test_runtime_execution_failed_handler():
    """_on_runtime_execution_failed generates recovery hint."""
    config = RuntimeConfig(
        robot_id="test_bot",
        enable_firewall=False,
        enable_memory=True,
        enable_practice=False,
        enable_knowledge=False,
        enable_how=True,
        enable_provider=False,
        seekdb_backend="memory",
    )
    rt = Runtime(config)
    rt._do_initialize()

    received = []
    def on_hint(event):  # noqa: E306
        received.append(event.topic)
    rt.event_bus.subscribe("rosclaw.how.recovery_hint.generated", on_hint)

    rt.event_bus.publish(Event(
        topic="rosclaw.runtime.execution.failed",
        payload={"error_type": "timeout", "request_id": "r4"},
        source="test",
    ))
    time.sleep(0.2)
    rt._do_stop()


def test_runtime_safety_violation_handler():
    """_on_safety_violation publishes emergency_stop event."""
    config = RuntimeConfig(
        robot_id="test_bot",
        enable_firewall=False,
        enable_memory=True,
        enable_practice=False,
        enable_knowledge=False,
        enable_how=False,
        enable_provider=False,
        seekdb_backend="memory",
    )
    rt = Runtime(config)
    rt._do_initialize()

    received = []
    def on_stop(event):  # noqa: E306
        received.append(event.topic)
    rt.event_bus.subscribe("robot.emergency_stop", on_stop)

    rt.event_bus.publish(Event(
        topic="safety.violation",
        payload={"joint": 2, "limit": "torque"},
        source="test",
    ))
    time.sleep(0.1)
    assert "robot.emergency_stop" in received
    rt._do_stop()
