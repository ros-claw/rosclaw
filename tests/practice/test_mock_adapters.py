"""Tests for mock source adapters."""

from __future__ import annotations

from rosclaw.practice.adapters.mock_agent_adapter import MockAgentAdapter
from rosclaw.practice.adapters.mock_runtime_adapter import MockRuntimeAdapter
from rosclaw.practice.config import PracticeSession


class _FakeSession:
    practice_id = "prac_test"


def test_mock_agent_adapter_generates_trace():
    adapter = MockAgentAdapter("r1", task="pick cup")
    adapter.start(_FakeSession())
    events = []
    for _ in range(5):
        events.extend(adapter.poll())
    adapter.stop()

    types = [e.event_type for e in events]
    assert "agent.task_received" in types
    assert "agent.plan_created" in types
    assert "agent.tool_call_started" in types
    assert "agent.tool_call_finished" in types
    assert all(e.practice_id == "prac_test" for e in events)


def test_mock_runtime_adapter_generates_actions():
    adapter = MockRuntimeAdapter("r1")
    adapter.start(_FakeSession())
    events = []
    for _ in range(4):
        events.extend(adapter.poll())
    adapter.stop()

    types = [e.event_type for e in events]
    assert "runtime.action_proposed" in types
    assert "runtime.action_executed" in types
    assert "runtime.reward_observed" in types


def test_adapter_health():
    adapter = MockAgentAdapter("r1")
    health = adapter.health()
    assert health.source == "agent"
    assert health.healthy is True
