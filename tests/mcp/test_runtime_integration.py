"""Live Runtime integration tests for all seven P0 MCP tools.

These tests instantiate a real ``Runtime`` with mock/empty dependencies and
inject it into ``RuntimeClient`` so that every P0 tool exercises the actual
subsystem adapters rather than fixture fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.mcp.adapters.runtime_client import RuntimeClient


@pytest.fixture
def live_runtime_client(tmp_path: Path) -> RuntimeClient:
    """Build a RuntimeClient backed by a fully initialized Runtime."""
    config = RuntimeConfig(
        robot_id="integration_test_bot",
        enable_firewall=False,
        enable_memory=True,
        enable_practice=True,
        enable_knowledge=False,
        enable_how=False,
        enable_provider=False,
        enable_auto=False,
        seekdb_backend="memory",
        sense_collector="mock",
        sense_update_hz=1.0,
        timeline_output_dir=str(tmp_path / "practice_data"),
    )
    rt = Runtime(config)
    rt.initialize()

    client = RuntimeClient(
        project_root=tmp_path,
        robot_id="integration_test_bot",
        runtime_profile={},
    )
    client._runtime = rt
    client._adapter_cache = None

    yield client

    rt.stop()


async def test_get_robot_state_live(live_runtime_client: RuntimeClient) -> None:
    response = await live_runtime_client.get_robot_state()
    assert response["robot_id"] == "integration_test_bot"
    assert response["mode"] in {"live", "stale"}
    assert "readiness" in response
    assert response["readiness"].get("overall_status") == "ready"
    assert "is_stale" in response
    assert "age_ms" in response


async def test_list_skills_live(live_runtime_client: RuntimeClient) -> None:
    response = await live_runtime_client.list_skills(skill_type=None, full_ids=False)
    assert response["mode"] == "live"
    assert isinstance(response["skills"], list)
    assert response["count"] == len(response["skills"])


async def test_query_memory_live(live_runtime_client: RuntimeClient) -> None:
    response = await live_runtime_client.query_memory(
        "pick an object",
        limit=3,
        outcome_filter=None,
    )
    assert response["mode"] == "live"
    assert isinstance(response["experiences"], list)
    assert response["count"] == len(response["experiences"])


async def test_practice_query_live(live_runtime_client: RuntimeClient) -> None:
    response = await live_runtime_client.practice_query(limit=5)
    assert response["mode"] == "live"
    assert isinstance(response["episodes"], list)
    assert response["count"] == len(response["episodes"])


async def test_validate_trajectory_live(live_runtime_client: RuntimeClient) -> None:
    trajectory = [[0.0] * 6, [0.05] * 6, [0.1] * 6]
    response = await live_runtime_client.validate_trajectory(
        trajectory,
        safety_level="MODERATE",
    )
    assert "is_safe" in response
    assert "risk_score" in response
    assert "violations" in response
    assert "replay_id" in response
    assert isinstance(response["is_safe"], bool)


async def test_sandbox_run_live(live_runtime_client: RuntimeClient) -> None:
    response = await live_runtime_client.sandbox_run([0.0] * 6)
    assert response["mode"] == "live"
    assert isinstance(response["physics_state"], dict)


async def test_emergency_stop_live(live_runtime_client: RuntimeClient) -> None:
    rt = live_runtime_client._runtime
    received: list[Any] = []
    rt.event_bus.subscribe("robot.emergency_stop", received.append)

    response = await live_runtime_client.emergency_stop("integration test halt")
    assert response["stopped"] is True
    assert response["mode"] == "live"
    assert response["reason"] == "integration test halt"
    assert len(received) == 1
    assert received[0].topic == "robot.emergency_stop"
    assert received[0].priority.name == "CRITICAL"
