"""Smoke tests for P0 MCP tools in fixture mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.mcp.tools import (
    emergency_stop,
    get_robot_state,
    list_skills,
    practice_query,
    query_memory,
    sandbox_run,
    set_client,
    validate_trajectory,
)


@pytest.fixture(autouse=True)
def _fixture_client() -> None:
    client = RuntimeClient(
        project_root=Path("/tmp/rosclaw-test"),
        robot_id="test_bot",
        runtime_profile={},
        fixture_mode=True,
    )
    set_client(client)


def _envelope(result: str) -> dict:
    payload = json.loads(result)
    assert "ok" in payload
    assert payload["schema_version"].startswith("p0.")
    assert "trace_id" in payload
    assert "timestamp" in payload
    return payload


async def test_get_robot_state() -> None:
    payload = _envelope(await get_robot_state())
    assert payload["ok"] is True
    assert payload["data"]["robot_id"] == "test_bot"
    assert payload["data"]["mode"] == "fixture"


async def test_list_skills() -> None:
    payload = _envelope(await list_skills(skill_type=None))
    assert payload["ok"] is True
    assert payload["data"]["skills"] == []
    assert payload["data"]["mode"] == "fixture"


async def test_query_memory() -> None:
    payload = _envelope(await query_memory(instruction="pick cup", limit=3))
    assert payload["ok"] is True
    assert payload["data"]["experiences"] == []
    assert payload["data"]["count"] == 0


async def test_practice_query() -> None:
    payload = _envelope(await practice_query(limit=5))
    assert payload["ok"] is True
    assert payload["data"]["episodes"] == []


async def test_validate_trajectory_unsafe_when_runtime_missing() -> None:
    payload = _envelope(await validate_trajectory(trajectory=[[0.0] * 6, [0.1] * 6]))
    assert payload["ok"] is True
    assert payload["data"]["is_safe"] is False
    assert "runtime_unavailable" in payload["data"]["violations"]


async def test_sandbox_run_fixture() -> None:
    payload = _envelope(await sandbox_run(joint_positions=[0.1] * 6))
    assert payload["ok"] is True
    assert payload["data"]["mode"] == "fixture"


async def test_emergency_stop_degraded_acknowledgment() -> None:
    payload = _envelope(await emergency_stop(reason="test halt"))
    assert payload["ok"] is True
    assert payload["data"]["stopped"] is True
    assert payload["data"]["reason"] == "test halt"
    assert payload["data"]["mode"] == "degraded"
