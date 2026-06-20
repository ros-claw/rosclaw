"""Unit tests for the RuntimeClient facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from rosclaw.core.event_bus import EventBus
from rosclaw.mcp.adapters.runtime_client import RuntimeClient


class _FakeSense:
    def __init__(self) -> None:
        self.is_stale = False
        self.state_age_ms = 42

    def get_latest_state(self) -> dict[str, Any]:
        return {"joint_positions": [0.1] * 6}

    def get_body_sense(self) -> dict[str, Any]:
        return {"temperature": "normal"}

    def get_readiness(self) -> dict[str, Any]:
        return {"overall": "READY"}


class _FakeRuntime:
    """Minimal runtime double for RuntimeClient wiring tests."""

    def __init__(self) -> None:
        self.event_bus = EventBus()
        self.sense = _FakeSense()
        self.memory = MagicMock()
        self.memory.find_similar_experiences.return_value = []
        self.skill_manager = MagicMock()
        self.skill_manager.registry = None
        self.skill_manager.list_skills.return_value = []
        self.sandbox = MagicMock()
        self.sandbox.validate_trajectory.return_value = {
            "is_safe": True,
            "risk_score": 0.0,
            "reason": "ok",
            "violations": [],
            "replay_id": "replay-1",
        }
        self.sandbox.simulate_step.return_value = {"qpos": [0.2] * 6}
        self.episode_recorder = MagicMock()
        self.episode_recorder.list_episodes.return_value = []


@pytest.fixture
def client_with_runtime() -> RuntimeClient:
    client = RuntimeClient(
        project_root=Path("/tmp/rosclaw-test"),
        robot_id="test_bot",
        runtime_profile={},
    )
    client._runtime = _FakeRuntime()
    client._adapter_cache = None
    return client


async def test_get_robot_state_returns_live(client_with_runtime: RuntimeClient) -> None:
    response = await client_with_runtime.get_robot_state()
    assert response["robot_id"] == "test_bot"
    assert response["mode"] == "live"
    assert response["body_state"]["joint_positions"] == [0.1] * 6
    assert response["age_ms"] == 42


async def test_get_robot_state_fixture_when_sense_missing() -> None:
    client = RuntimeClient(
        project_root=Path("/tmp/rosclaw-test"),
        robot_id="fixture_bot",
        runtime_profile={},
        fixture_mode=True,
    )
    response = await client.get_robot_state()
    assert response["mode"] == "fixture"
    assert response["robot_id"] == "fixture_bot"


async def test_list_skills_delegates_to_adapter(client_with_runtime: RuntimeClient) -> None:
    client_with_runtime._runtime.skill_manager.list_skills.return_value = [
        MagicMock(to_dict=lambda: {"name": "pick"}),
    ]
    response = await client_with_runtime.list_skills(skill_type="manipulation")
    assert response["count"] == 1
    assert response["skills"][0]["name"] == "pick"


async def test_query_memory_delegates_to_adapter(client_with_runtime: RuntimeClient) -> None:
    client_with_runtime._runtime.memory.find_similar_experiences.return_value = [
        {"instruction": "pick", "outcome": "success"},
    ]
    response = await client_with_runtime.query_memory("pick", limit=1)
    assert response["count"] == 1
    assert response["experiences"][0]["outcome"] == "success"


async def test_validate_trajectory_delegates_to_sandbox(client_with_runtime: RuntimeClient) -> None:
    response = await client_with_runtime.validate_trajectory([[0.0] * 6, [0.1] * 6])
    assert response["is_safe"] is True
    assert response["replay_id"] == "replay-1"


async def test_sandbox_run_delegates_to_sandbox(client_with_runtime: RuntimeClient) -> None:
    response = await client_with_runtime.sandbox_run([0.1] * 6)
    assert response["mode"] == "live"
    assert response["physics_state"]["qpos"] == [0.2] * 6


async def test_practice_query_delegates_to_recorder(client_with_runtime: RuntimeClient) -> None:
    client_with_runtime._runtime.episode_recorder.list_episodes.return_value = [
        {"episode_id": "ep-1"},
    ]
    response = await client_with_runtime.practice_query(limit=5)
    assert response["count"] == 1
    assert response["episodes"][0]["episode_id"] == "ep-1"


async def test_emergency_stop_publishes_via_event_bus(client_with_runtime: RuntimeClient) -> None:
    received: list[Any] = []
    client_with_runtime._runtime.event_bus.subscribe("robot.emergency_stop", received.append)
    response = await client_with_runtime.emergency_stop("integration test")
    assert response["stopped"] is True
    assert response["mode"] == "live"
    assert len(received) == 1


async def test_emergency_stop_degraded_without_runtime() -> None:
    client = RuntimeClient(
        project_root=Path("/tmp/rosclaw-test"),
        robot_id="test_bot",
        runtime_profile={},
        fixture_mode=True,
    )
    response = await client.emergency_stop("no runtime")
    assert response["stopped"] is True
    assert response["mode"] == "degraded"
    assert "physical E-stop" in response["note"]
