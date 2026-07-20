"""Unit tests for thin MCP subsystem adapters."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from rosclaw.mcp.adapters.memory_client import MemoryClient
from rosclaw.mcp.adapters.practice_client import PracticeClient
from rosclaw.mcp.adapters.safety_client import SafetyClient
from rosclaw.mcp.adapters.sandbox_client import SandboxClient
from rosclaw.mcp.adapters.skill_registry_client import SkillRegistryClient


class _FakeEntry:
    def __init__(self, name: str) -> None:
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


def test_memory_client_returns_experiences() -> None:
    memory = MagicMock()
    memory.find_similar_experiences.return_value = [
        {"instruction": "pick cup", "outcome": "success"},
    ]
    client = MemoryClient(memory)
    response = client.find_similar_experiences("pick cup", limit=3, outcome_filter="success")
    assert response["mode"] == "live"
    assert response["count"] == 1
    assert response["experiences"][0]["outcome"] == "success"
    memory.find_similar_experiences.assert_called_once_with(
        instruction="pick cup",
        limit=3,
        outcome_filter="success",
    )


def test_practice_client_by_episode_id() -> None:
    recorder = MagicMock()
    recorder.get_episode.return_value = {"episode_id": "ep-1"}
    client = PracticeClient(recorder)
    response = client.query(episode_id="ep-1")
    assert response["count"] == 1
    assert response["episodes"][0]["episode_id"] == "ep-1"


def test_practice_client_lists_recent() -> None:
    recorder = MagicMock()
    recorder.list_episodes.return_value = [{"episode_id": "ep-2"}, {"episode_id": "ep-3"}]
    client = PracticeClient(recorder)
    response = client.query(limit=1)
    assert response["count"] == 1
    assert response["episodes"][0]["episode_id"] == "ep-2"


def test_sandbox_client_validates_dict_result() -> None:
    sandbox = MagicMock()
    sandbox.validate_trajectory.return_value = {
        "is_safe": True,
        "risk_score": 0.1,
        "reason": "ok",
        "violations": [],
        "replay_id": "r1",
    }
    client = SandboxClient(sandbox)
    response = client.validate_trajectory([[0.0] * 6, [0.1] * 6], safety_level="STRICT")
    assert response["is_safe"] is True
    assert response["replay_id"] == "r1"


def test_sandbox_client_normalizes_non_dict_result() -> None:
    sandbox = MagicMock()
    sandbox.validate_trajectory.return_value = object()
    client = SandboxClient(sandbox)
    response = client.validate_trajectory([[0.0] * 6])
    assert response["is_safe"] is False
    assert "invalid_validation_result" in response["violations"]


def test_sandbox_client_simulate_step() -> None:
    sandbox = MagicMock()
    sandbox.has_physics = True
    sandbox.simulate_step.return_value = {"qpos": [0.1] * 6}
    client = SandboxClient(sandbox)
    response = client.simulate_step([0.1] * 6)
    assert response["mode"] == "simulation"
    assert response["execution_mode"] == "SIMULATION"
    assert response["has_physics"] is True
    assert response["physics_state"]["qpos"] == [0.1] * 6


def test_sandbox_client_simulate_step_unavailable_when_empty() -> None:
    sandbox = MagicMock()
    sandbox.has_physics = False
    sandbox.simulate_step.return_value = {}
    client = SandboxClient(sandbox)
    response = client.simulate_step([0.1] * 6)
    assert response["mode"] == "unavailable"
    assert response["has_physics"] is False
    assert response["physics_state"] == {}


def test_skill_registry_client_uses_registry_when_present() -> None:
    registry = MagicMock()
    registry.list_skills.return_value = [_FakeEntry("grasp")]
    skill_manager = MagicMock()
    skill_manager.registry = registry
    client = SkillRegistryClient(skill_manager)
    response = client.list_skills(skill_type="manipulation", full_ids=True)
    assert response["count"] == 1
    assert response["skills"][0] == {"name": "grasp"}
    registry.list_skills.assert_called_once_with(
        skill_type="manipulation",
        return_entries=True,
        full_ids=True,
    )


def test_skill_registry_client_falls_back_to_skill_manager() -> None:
    skill_manager = MagicMock()
    skill_manager.registry = None
    skill_manager.list_skills.return_value = ["skill-a"]
    client = SkillRegistryClient(skill_manager)
    response = client.list_skills()
    assert response["count"] == 1
    assert response["skills"] == ["skill-a"]


def test_safety_client_calls_daemon_emergency_path() -> None:
    daemon = MagicMock()
    daemon.emergency_stop.return_value = {
        "request_id": "stop-1",
        "reason": "collision imminent",
        "request_dispatched": True,
        "driver_acknowledged": True,
        "physical_stop_observed": False,
        "stopped": False,
        "final_status": "ACKNOWLEDGED",
    }
    client = SafetyClient(daemon)
    response = client.emergency_stop("collision imminent")
    assert response["stopped"] is False
    assert response["final_status"] == "ACKNOWLEDGED"
    daemon.emergency_stop.assert_called_once_with(
        "collision imminent",
        source="mcp.emergency_stop",
    )


def test_safety_client_does_not_call_private_handler() -> None:
    daemon = MagicMock()
    daemon.emergency_stop.return_value = {
        "stopped": False,
        "final_status": "UNVERIFIED",
    }
    client = SafetyClient(daemon)
    client.emergency_stop("test")
    daemon._on_emergency_stop.assert_not_called()
